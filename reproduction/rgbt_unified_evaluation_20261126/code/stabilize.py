"""Camera-compensated windows for the Level-Variation-History decomposition (LVH).

DeepPro-LV takes the median of each *fixed image pixel* over the window. With a moving
camera (e.g. DJI_0293_1 at 1.68 px/frame, DJI_0237_2 at 1.01 px/frame) a target that is
stationary relative to the ground moves through many pixels, so the fixed-pixel median no
longer separates persistent from transient content. Near-stationary targets are defined
with camera-compensated relative speed (profile_motion.py), so the decomposition has to
be computed in the same compensated frame.

This module warps every frame of a window into the coordinates of a reference frame (the
window's middle frame) with the per-pair homographies written by profile_motion.py:
    results/profile/homographies/<clip>.npy  (N-1, 3, 3), H[i] maps frame i -> frame i+1

It also builds the long-term History field B: the median of frames *outside* the window,
warped to the same reference. A ground hot spot is present in B; a target that arrived,
or will leave, is not.

Only numpy and OpenCV are used, so the data path runs in DataLoader workers.
"""
import os

import cv2
import numpy as np

HOMOGRAPHY_DIR = os.environ.get('HOMOGRAPHY_DIR', 'results/profile/homographies')


def windows(n, T=40):
    """Inference windows of infer.windows (copied so that numpy-only scripts avoid torch)."""
    import math
    cat = int(T * 0.1)
    num = math.ceil((n - cat) / (T - cat))
    out = []
    for i in range(num):
        last = min(n, (i + 1) * (T - cat) + cat)
        out.append((max(0, last - T), last))
    return out


def reference_frame(a, b):
    """Reference (middle) frame of the window [a, b)."""
    return a + (b - a) // 2


def load_video_u8(clip, scene=None, cache_dir='cache/video'):
    """uint8 (N,H,W) frames, reusing train.py's memmap cache when it exists."""
    path = os.path.join(cache_dir, f'{clip}.npy')
    if os.path.exists(path):
        return np.load(path, mmap_mode='r')
    if scene is None:
        import vtdata
        scene = vtdata.Scene(clip)
    return np.stack([np.clip(scene.image(i), 0, 255).astype(np.uint8) for i in range(len(scene))])


# ---------------------------------------------------------------- homography chains
def chain_from_pairs(pairs):
    """(N-1,3,3) consecutive homographies -> (N,3,3) cumulative maps frame 0 -> frame t."""
    pairs = np.asarray(pairs, dtype=np.float64).reshape(-1, 3, 3)
    out = np.empty((len(pairs) + 1, 3, 3))
    out[0] = np.eye(3)
    for t, H in enumerate(pairs):
        M = H @ out[t]
        out[t + 1] = M / M[2, 2]
    return out


def load_chain(clip, n_frames=None, directory=HOMOGRAPHY_DIR):
    path = os.path.join(directory, f'{clip}.npy')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} is missing; run code/profile_motion.py first '
                                f'or set HOMOGRAPHY_DIR')
    chain = chain_from_pairs(np.load(path))
    if n_frames is not None and len(chain) != n_frames:
        raise ValueError(f'{clip}: {len(chain)} homography frames for {n_frames} video frames')
    return chain


def ref_to(chain, ref, t):
    """Homography mapping reference-frame coordinates to frame-t coordinates."""
    M = chain[t] @ np.linalg.inv(chain[ref])
    return M / M[2, 2]


def apply_h(M, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    p = np.c_[pts, np.ones(len(pts))] @ M.T
    return p[:, :2] / p[:, 2:3]


def translation(dx, dy):
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def max_displacement(chain, frames, ref, shape, grid=5):
    """Largest movement (px) of a grid of image points between the reference and any frame."""
    h, w = shape
    pts = np.stack(np.meshgrid(np.linspace(0, w - 1, grid), np.linspace(0, h - 1, grid)), -1).reshape(-1, 2)
    d = 0.0
    for t in frames:
        q = apply_h(ref_to(chain, ref, t), pts)
        d = max(d, float(np.linalg.norm(q - pts, axis=1).max()))
    return d


def gate_chain(chain, frames, ref, tau, shape):
    """Motion-gated compensation.

    A static camera still gets small, noisy homographies from feature tracking (about
    0.03-0.2 px/frame on the static VT-Tiny-MOT clips), and warping with them both jitters
    edges and resamples small targets bilinearly. When no frame in `frames` moves more than
    `tau` pixels from the reference, return an identity chain so the frames are used as
    recorded. tau <= 0 disables gating. -> (chain, gated)
    """
    if tau <= 0 or not len(frames):
        return chain, False
    if max_displacement(chain, frames, ref, shape) <= tau:
        return np.broadcast_to(np.eye(3), chain.shape).copy(), True
    return chain, False


# ---------------------------------------------------------------- warping
def _grid_valid(M, h, w, src_h, src_w):
    """Which destination pixels of an (h,w) grid map (by M: dst -> src) inside the source."""
    yy, xx = np.mgrid[0:h, 0:w]
    p = np.stack([xx.ravel(), yy.ravel(), np.ones(h * w)], 0)
    q = M @ p
    x, y = q[0] / q[2], q[1] / q[2]
    ok = (x >= 0) & (x <= src_w - 1) & (y >= 0) & (y <= src_h - 1)
    return ok.reshape(h, w)


def warp_frames(video, chain, frames, ref, r0=0, c0=0, size=None):
    """Warp video[frames] into reference coordinates.

    The output pixel (u, v) is the reference-frame location (c0 + u, r0 + v), so a crop
    of the stabilized window costs only size[0] * size[1] samples per frame.
    Returns (len(frames), h, w) float32 and a validity mask of the same shape.
    """
    src_h, src_w = video.shape[1:3]
    h, w = size if size is not None else (src_h, src_w)
    out = np.zeros((len(frames), h, w), np.float32)
    valid = np.zeros((len(frames), h, w), bool)
    offset = translation(c0, r0)
    for k, t in enumerate(frames):
        M = ref_to(chain, ref, t) @ offset  # destination crop pixel -> source frame t pixel
        src = np.asarray(video[t], dtype=np.float32)
        out[k] = cv2.warpPerspective(src, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_REPLICATE)
        valid[k] = _grid_valid(M, h, w, src_h, src_w)
    return out, valid


def history_frames(n, a, b, length, step, mode='both'):
    """Frames outside the window [a, b) used for the History field.

    mode 'past' uses [a - length, a); mode 'both' also uses [b, b + length).
    The default evaluation is offline (windows already use later frames), so 'both'
    is the matching choice; 'past' is the causal variant.
    """
    if length <= 0:
        return []
    past = list(range(max(0, a - length), a, step))
    if mode == 'past':
        return past
    if mode != 'both':
        raise ValueError(mode)
    return past + list(range(b, min(n, b + length), step))


def masked_median(stack, valid, min_count=1):
    """Per-pixel median over valid samples. Returns (median, count); median 0 where count < min_count."""
    if len(stack) == 0:
        h, w = stack.shape[1:3]
        return np.zeros((h, w), np.float32), np.zeros((h, w), np.int32)
    x = np.where(valid, stack, np.nan)
    count = valid.sum(0).astype(np.int32)
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            med = np.nanmedian(x, axis=0)
    med = np.where(count >= min_count, med, 0.0).astype(np.float32)
    return med, count


def history_field(video, chain, a, b, ref, length, step, mode='both', r0=0, c0=0, size=None,
                  min_count=3, return_frames=False):
    """Median of out-of-window frames warped to the reference. -> (B, available mask).

    With return_frames=True also returns the history frame indices and their per-pixel
    validity (len(frames), h, w).
    """
    frames = history_frames(len(video), a, b, length, step, mode)
    src_h, src_w = video.shape[1:3]
    h, w = size if size is not None else (src_h, src_w)
    if not frames:
        out = np.zeros((h, w), np.float32), np.zeros((h, w), bool)
        return out + ([], np.zeros((0, h, w), bool)) if return_frames else out
    stack, valid = warp_frames(video, chain, frames, ref, r0, c0, (h, w))
    med, count = masked_median(stack, valid, min_count)
    if return_frames:
        return med, count >= min_count, frames, valid
    return med, count >= min_count


def warp_back(ref_maps, chain, frames, ref, fill):
    """Map per-frame score maps from reference coordinates back to each frame.

    ref_maps: (len(frames), H, W) scores defined on reference-frame pixels.
    Returns (len(frames), H, W) scores on frame pixels; pixels that fall outside the
    reference field get `fill`.
    """
    h, w = ref_maps.shape[1:3]
    out = np.empty(ref_maps.shape, np.float32)
    for k, t in enumerate(frames):
        M = ref_to(chain, ref, t)  # reference pixel -> frame-t pixel (forward map)
        out[k] = cv2.warpPerspective(np.asarray(ref_maps[k], np.float32), M, (w, h),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=float(fill))
    return out


def map_boxes(boxes, M):
    """Move (x, y, w, h, cat, track) boxes by M, keeping their size (small-rotation regime)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 6)
    if len(boxes) == 0:
        return boxes.copy()
    # Pixel-centre convention of vtdata.gaussian_mask: the centre pixel is x + w/2 - 0.5.
    c = boxes[:, :2] + boxes[:, 2:4] / 2.0 - 0.5
    c2 = apply_h(M, c)
    out = boxes.copy()
    out[:, :2] = c2 + 0.5 - boxes[:, 2:4] / 2.0
    return out


# ---------------------------------------------------------------- local contrast
def local_contrast(img, inner=1, outer=(4, 7)):
    """Mean of a (2*inner+1)^2 centre minus the mean of the ring between outer radii.

    NaNs in img are treated as missing samples.
    """
    img = np.asarray(img, np.float32)
    ok = np.isfinite(img).astype(np.float32)
    z = np.where(ok > 0, img, 0.0).astype(np.float32)

    def box_sum(a, r):
        k = 2 * r + 1
        return cv2.boxFilter(a, -1, (k, k), normalize=False, borderType=cv2.BORDER_CONSTANT)

    s_in, n_in = box_sum(z, inner), box_sum(ok, inner)
    s_lo, n_lo = box_sum(z, outer[0] - 1), box_sum(ok, outer[0] - 1)
    s_hi, n_hi = box_sum(z, outer[1]), box_sum(ok, outer[1])
    ring_s, ring_n = s_hi - s_lo, n_hi - n_lo
    with np.errstate(all='ignore'):
        c = s_in / n_in - ring_s / ring_n
    c[(n_in < 1) | (ring_n < 4)] = np.nan
    return c
