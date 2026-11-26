"""Multi-rate temporal profiles (MRTP) with stationarity-adaptive coordinates.

Problem: the velocity blind zone of temporal-profile detectors
    DeepPro and DeepPro-LV judge each pixel by its profile x_p(t) over a window of T
    samples. A target of extent s moving at v px/frame covers a pixel for s / v frames, or
    s / (k v) samples when the window takes every k-th frame. Two limits follow:
      * dwell above T / 2 samples: the window median absorbs the target, which then looks
        like background (blind zone), i.e. k v < 2 s / T;
      * dwell below one sample: the target can skip the pixel between samples (temporal
        aliasing), i.e. k v > s.
    One rate therefore detects speeds in the band 2 s / (k T) <= v <= s / k, a band T / 2
    wide on a log scale. With T = 40 and s = 6 px, rate 1 starts at 0.3 px/frame. On the
    five static-camera validation clips DeepPro-LV recall is 79 % at 0.3-0.5 px/frame,
    22 % at 0.2-0.3 and below 13 % under 0.2 (recording-matched 1 FA/frame, seed 46).
    DeepPro's authors report the same limitation ("It is hard to detect stationary
    objects from the temporal profile").

Method
    1. Rates. The same T-sample network runs on phase-interleaved windows
       {phi, phi + k, ..., phi + (T - 1) k}, phi = 0..k-1, for k in a small set such as
       {1, 2, 4}. Sampled at rate k, a target at speed v has the sample-domain profile of a
       target at speed k v sampled at rate 1, so the weights are shared across rates (rate
       equivariance of the target). Background dynamics are not rate equivariant, which is
       what rate-augmented training (train.py --rates) is for. Each rate costs about as
       much as rate 1: about n / 36 windows of T samples per clip.
    2. Coordinates. A temporal profile also assumes a stationary background. Per window
       and rate we take the coordinate frame (recorded pixels, or pixels compensated with
       the camera homographies) in which the background changes least between the two
       halves of the window (stationarity residual). This replaces always warping
       (DeepPro-LVH) and the displacement gate, which never fired on these clips.
    3. Fusion. A longer span lets the background change more, so the rates differ in
       clutter. Each rate's logits are calibrated against their own local clutter level,
       CFAR-style (the m-th largest candidate score per frame, pooled over +-half frames),
       before the maximum over rates.

Everything here is numpy / OpenCV; the network is passed in as a callable (infer_mr.py).
"""
import math

import cv2
import numpy as np
from scipy import ndimage

from . import stabilize as st

FLOOR = -12.0   # evaluate.FLOOR: candidate peaks need a logit above this
NMS = 9         # evaluate.NMS: 9 x 9 local maxima


# ---------------------------------------------------------------- rates and windows
def detectable_band(s, k=1, T=40):
    """Speeds (px/frame) a T-sample profile at rate k can represent for a target of extent s."""
    return 2.0 * s / (k * T), s / float(k)


def plan_windows(n, k, T=40):
    """Windows of the rate-k pass over n frames: list of (phase, frame indices).

    Rate 1 is the DeepPro inference scheme (infer.windows). For k > 1 the frames of each
    phase phi = 0..k-1 form a subsequence phi, phi + k, ...; the rate-1 scheme runs on each
    subsequence, so every frame is scored once per rate. Needs n >= k T so that every
    window holds T samples; otherwise the rate is not used for the clip (empty plan).
    """
    if k < 1:
        raise ValueError(k)
    if k == 1:
        return [(0, np.arange(a, b)) for a, b in st.windows(n, T)]
    if n < k * T:
        return []
    plan = []
    for phi in range(k):
        idx = np.arange(phi, n, k)
        for a, b in st.windows(len(idx), T):
            plan.append((phi, idx[a:b]))
    return plan


def rate_starts(n, k, T=40):
    """Number of training windows s, s + k, ..., s + (T - 1) k that fit in n frames."""
    return max(0, n - ((T - 1) * k + 1) + 1)


def parse_rates(text):
    rates = sorted({int(r) for r in str(text).split(',') if r.strip()})
    if not rates or rates[0] < 1:
        raise ValueError(f'bad rates {text!r}')
    return rates


def rate_probs(rates, text=None):
    """Sampling probabilities of the training rates; default 1/2 for rate 1, rest equal."""
    if text:
        p = np.array([float(v) for v in text.split(',')], float)
        if len(p) != len(rates) or (p < 0).any() or p.sum() <= 0:
            raise ValueError(f'--rate-probs {text!r} does not match rates {rates}')
        return p / p.sum()
    if len(rates) == 1:
        return np.ones(1)
    p = np.full(len(rates), 0.5 / (len(rates) - 1))
    if rates[0] == 1:
        p[0] = 0.5
    else:
        p[:] = 1.0 / len(rates)
    return p / p.sum()


# ---------------------------------------------------------------- background stationarity
def lowres(frames, f=4):
    """(K,H,W) frames -> (K, H // f, W // f) float32 block means (INTER_AREA)."""
    K = len(frames)
    H, W = np.asarray(frames[0]).shape
    h, w = H // f, W // f
    out = np.empty((K, h, w), np.float32)
    for i in range(K):
        fr = np.asarray(frames[i], np.float32)[:h * f, :w * f]
        out[i] = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
    return out


def scale_homography(M, f):
    """Full-resolution map M -> the same map on f x f block means (block centres)."""
    A = np.array([[f, 0.0, (f - 1) / 2.0], [0.0, f, (f - 1) / 2.0], [0.0, 0.0, 1.0]])
    S = np.linalg.inv(A) @ M @ A
    return S / S[2, 2]


def _half_change(stack, valid, mask, edges, q):
    """Change of the background between the two halves of a window.

    Per pixel the median of each half (over valid samples) is taken; their difference has
    its global offset removed (sensor gain drift) and its q-th percentile over edge pixels
    is returned. Medians of half windows suppress sensor noise, so the value is driven by
    structure that moves: camera drift, parallax, registration error, waves.
    """
    h = len(stack) // 2
    m1, c1 = st.masked_median(stack[:h], valid[:h])
    m2, c2 = st.masked_median(stack[h:], valid[h:])
    ok = mask & (c1 > 0) & (c2 > 0)
    if ok.sum() < 16:
        return math.nan
    d = m1 - m2
    d = d - np.median(d[ok])
    sel = ok & edges
    if sel.sum() < 16:
        sel = ok
    return float(np.percentile(np.abs(d[sel]), q))


def stationarity(low, chain, frames, f=4, cover_frac=0.75, q=90.0, edge_frac=0.25):
    """Stationarity residuals of one window in recorded and compensated coordinates.

    low: (K,h,w) block means of the window's frames, in their own coordinates.
    chain: homography chain of the clip (stabilize.load_chain) at full resolution.
    Both residuals use the same pixels: those valid in >= cover_frac of the compensated
    samples and among the top edge_frac of gradient magnitude in the reference frame.
    -> dict(raw, warp, cover) with cover the fraction of pixels kept by the validity test.
    """
    frames = [int(t) for t in frames]
    K, h, w = low.shape
    ref = frames[K // 2]
    warped = np.empty_like(low)
    valid = np.empty(low.shape, bool)
    for i, t in enumerate(frames):
        M = scale_homography(st.ref_to(chain, ref, t), f)
        warped[i] = cv2.warpPerspective(low[i], M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                        borderMode=cv2.BORDER_REPLICATE)
        valid[i] = st._grid_valid(M, h, w, h, w)
    mask = valid.mean(0) >= cover_frac
    cover = float(mask.mean())
    if cover <= 0:
        return dict(raw=math.nan, warp=math.nan, cover=0.0)
    g = np.hypot(cv2.Sobel(low[K // 2], cv2.CV_32F, 1, 0), cv2.Sobel(low[K // 2], cv2.CV_32F, 0, 1))
    edges = g >= np.percentile(g[mask], 100 * (1 - edge_frac))
    ones = np.ones(low.shape, bool)
    return dict(raw=_half_change(low, ones, mask, edges, q),
                warp=_half_change(warped, valid, mask, edges, q), cover=cover)


def choose_coords(policy, stats, k, margin=0.1, abs_tol=0.5, min_cover=0.1):
    """'raw' | 'warp' | 'skip' for one window.

    policy 'raw' always uses recorded frames; 'warp' always compensates; 'auto'
    compensates only when that lowers the stationarity residual by more than
    max(margin * raw, abs_tol) grey levels. A rate > 1 window whose compensated field
    keeps less than min_cover of the pixels (the camera swept past most of the view) is
    skipped; at rate 1 such a window is compensated, as in DeepPro-LVH.
    """
    if policy == 'raw':
        return 'raw'
    if policy not in ('warp', 'auto'):
        raise ValueError(policy)
    if stats.get('cover', 1.0) < min_cover:
        return 'warp' if k == 1 else 'skip'
    if policy == 'warp':
        return 'warp'
    r, w = stats.get('raw', math.nan), stats.get('warp', math.nan)
    if not (np.isfinite(r) and np.isfinite(w)):
        return 'raw'
    return 'warp' if (r - w) > max(margin * r, abs_tol) else 'raw'


# ---------------------------------------------------------------- one rate over a clip
def assemble(n, shape, plan, decide, score):
    """Per-frame logits of one rate: maximum over the windows that contain each frame.

    decide(frames) -> (coords, stats); score(frames, coords) -> (len(frames), H, W)
    logits in frame coordinates. Frames covered only by skipped windows stay -inf.
    -> (n, H, W) float32, list of per-window records.
    """
    out = np.full((n,) + tuple(shape), -np.inf, np.float32)
    meta = []
    for phase, frames in plan:
        coords, stats = decide(frames)
        meta.append(dict(phase=int(phase), first=int(frames[0]), last=int(frames[-1]), coords=coords,
                         **{key: float(v) for key, v in stats.items()}))
        if coords == 'skip':
            continue
        lg = np.asarray(score(frames, coords), np.float32)
        out[frames] = np.maximum(out[frames], lg)
    return out, meta


# ---------------------------------------------------------------- CFAR calibration and fusion
def peak_scores(logit, floor=FLOOR, size=NMS, topk=300):
    """Scores of the candidate peaks of one frame (evaluate.peaks without centroids)."""
    lg = np.asarray(logit, np.float32)
    fin = np.isfinite(lg)
    if not fin.any():
        return np.zeros(0, np.float32)
    lg = np.where(fin, lg, np.float32(-1e30))
    mx = ndimage.maximum_filter(lg, size=size, mode='nearest')
    s = lg[(lg >= mx) & (lg > floor)]
    if len(s) > topk:
        s = np.partition(s, len(s) - topk)[len(s) - topk:]
    return np.sort(s)[::-1]


def cfar_reference(peaks, m=10, half=18, floor=FLOOR):
    """Local clutter level per frame: the (m * w)-th largest candidate score over the w
    frames within +-half, or `floor` where fewer candidates exist (a clean stretch)."""
    n = len(peaks)
    ref = np.full(n, floor, np.float32)
    for t in range(n):
        lo, hi = max(0, t - half), min(n, t + half + 1)
        pool = np.concatenate([np.asarray(peaks[i], np.float32) for i in range(lo, hi)])
        need = m * (hi - lo)
        if len(pool) >= need:
            ref[t] = np.partition(pool, len(pool) - need)[len(pool) - need]
    return ref


def calibrate(logits, ref):
    """Subtract each frame's clutter level: (n,H,W) - (n,)."""
    return np.asarray(logits, np.float32) - np.asarray(ref, np.float32)[:, None, None]


def fuse_max(arrays):
    out = np.asarray(arrays[0], np.float32).copy()
    for a in arrays[1:]:
        np.maximum(out, np.asarray(a, np.float32), out=out)
    return out
