"""Rough Gaussian masks for RGBT-Tiny, written to the DeepPro paper.

What the paper specifies (DeepPro, arXiv:2506.12766, Sec. 4.6, RGBT-Tiny):

    "For training, we generated rough Gaussian masks as labels according to the
    bounding boxes provided by its authors."

That sentence is the whole specification: no sigma, no formula, no citation, and neither
the masks nor the code that made them were released. This module implements the
sentence and nothing more. For every box (x, y, w, h) of the RGBT-Tiny annotations
(COCO convention: (x, y) is the top-left corner, pixel (u, v) covers [u, u+1) x [v, v+1),
its centre is (u + 0.5, v + 0.5)):

    g(u, v) = exp( -(u + 0.5 - cx)^2 / (2 sx^2) - (v + 0.5 - cy)^2 / (2 sy^2) ),
    cx = x + w / 2,  cy = y + h / 2,  sx = w / k,  sy = h / k,

a Gaussian with peak 1 at the box centre and spread proportional to the box. The mask
of a frame is the per-pixel maximum over its boxes.

Choices the sentence leaves open are named parameters, so that a paper can state them:
    k        box-to-sigma ratio, sigma = w / k (not in the paper)
    storage  how the float mask became the 8-bit image the loader reads (not in the paper):
             'round' = rint(255 g), 'floor' = uint8(255 g)
    ignored  whether boxes flagged 'ignore' in the annotations are drawn (not in the paper)

Two things are fixed by the authors' released code rather than by the paper:
    * The loaders (TrainDataLoader.py, TestDataLoader.py, dataset 'RGB-T') read the mask as
      an 8-bit image, divide by 255 and set every value > 0 to 1, at the native 640 x 512
      size. What a network trains on is therefore the support of the stored Gaussian: an
      ellipse with semi-axes a w / k and a h / k, where a = sqrt(2 ln 510) = 3.53 for
      'round' and sqrt(2 ln 255) = 3.33 for 'floor'.
    * Evaluation (ShootingRules.py) counts the targets of a frame as the 8-connected
      components of that binary label. The per-clip counts in the authors' released log
      (eval_ShootingRules_LocLen2_4.txt) therefore constrain k, storage and ignored;
      code/check_mask_protocol.py tests every combination against them.

Boxes with w <= 0 or h <= 0 are skipped. A box under about 2 px can lose every pixel in
8-bit storage, because its Gaussian is narrower than a pixel; it then has no label, as
the sentence implies. export_masks.py and check_mask_protocol.py count such boxes.
"""
import math
import re

import numpy as np

FORMULA = ('g(u, v) = exp(-(u + 0.5 - cx)^2 / (2 sx^2) - (v + 0.5 - cy)^2 / (2 sy^2)), cx = x + w/2, '
           'cy = y + h/2, sx = w/k, sy = h/k (COCO boxes, pixel centres at +0.5); per-pixel maximum over the '
           "frame's boxes; 8-bit storage as given; the loaders binarise with > 0")
CUT = 6.0  # per-axis cut in sigmas: factors below exp(-18) = 1.5e-8 cannot reach an 8-bit image

# Targets per test clip in the authors' released RGB-T log
# (log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/eval_ShootingRules_LocLen2_4.txt).
AUTHOR_TEST_COUNTS = {
    'DJI_0049_2': 602, 'DJI_0109_1': 90, 'DJI_0119_2': 360, 'DJI_0129_2': 590, 'DJI_0155_2': 410,
    'DJI_0159_2': 250, 'DJI_0173_2': 450, 'DJI_0221_2': 697, 'DJI_0237_2': 2947, 'DJI_0255_1': 600,
    'DJI_0257_2': 1126, 'DJI_0277_2': 985, 'DJI_0303_2': 985, 'DJI_0307_2': 161, 'DJI_0309_4': 1050,
}
AUTHOR_TEST_TOTAL = 11303


def heatmap(boxes, H, W, k):
    """Float mask (H, W) in [0, 1]: per-pixel maximum of the boxes' Gaussians."""
    if not k > 0:
        raise ValueError(f'k must be positive, got {k}')
    m = np.zeros((H, W), np.float64)
    for b in boxes:
        x, y, w, h = (float(v) for v in b[:4])
        if not (w > 0 and h > 0):
            continue
        cx, cy, sx, sy = x + w / 2.0, y + h / 2.0, w / k, h / k
        u0, u1 = max(int(math.floor(cx - CUT * sx - 0.5)), 0), min(int(math.ceil(cx + CUT * sx - 0.5)) + 1, W)
        v0, v1 = max(int(math.floor(cy - CUT * sy - 0.5)), 0), min(int(math.ceil(cy + CUT * sy - 0.5)) + 1, H)
        if u0 >= u1 or v0 >= v1:
            continue
        gx = np.exp(-(np.arange(u0, u1) + 0.5 - cx) ** 2 / (2.0 * sx * sx))
        gy = np.exp(-(np.arange(v0, v1) + 0.5 - cy) ** 2 / (2.0 * sy * sy))
        np.maximum(m[v0:v1, u0:u1], gy[:, None] * gx[None, :], out=m[v0:v1, u0:u1])
    return m


def store(g, storage):
    """Float mask -> the 8-bit image a mask file holds."""
    if storage == 'round':
        return np.rint(255.0 * g).astype(np.uint8)
    if storage == 'floor':
        return np.floor(255.0 * g).astype(np.uint8)
    raise ValueError(storage)


def loader_label(img):
    """The authors' loaders: uint8 image / 255, every value > 0 set to 1."""
    lab = np.asarray(img, np.uint8) / 255.0
    if lab.ndim == 3:
        lab = lab[:, :, 0]
    lab[lab > 0] = 1.0
    return lab.astype(np.float32)


def label(boxes, H, W, k, storage='round'):
    """Binary training and evaluation label of one frame, as the authors' code sees it."""
    return loader_label(store(heatmap(boxes, H, W, k), storage))


def semi_axis(storage):
    """Support half-width of a stored Gaussian in sigmas (where g reaches the first 8-bit level)."""
    return math.sqrt(2 * math.log({'round': 510.0, 'floor': 255.0}[storage]))


def count_targets(binary):
    """Targets in one frame as ShootingRules.py counts them: 8-connected components."""
    try:
        from skimage import measure
        return int(measure.label(np.asarray(binary) > 0, connectivity=2).max())
    except ImportError:
        from scipy import ndimage
        return int(ndimage.label(np.asarray(binary) > 0, structure=np.ones((3, 3), int))[1])


def vanished(boxes, H, W, k, storage='round'):
    """Number of drawable boxes whose own stored Gaussian has no pixel above 0."""
    n = 0
    for b in boxes:
        if b[2] > 0 and b[3] > 0 and not store(heatmap([b], H, W, k), storage).any():
            n += 1
    return n


def parse_eval_log(text):
    """Per-clip target counts from a DeepPro evaluation log ('<clip>results:' blocks)."""
    counts, cur = {}, None
    for line in text.splitlines():
        m = re.match(r'^(\S+?)results:\s*$', line.strip())
        if m:
            cur = m.group(1)
            continue
        if line.strip().endswith(':') and not line.startswith('Th_Seg'):
            cur = None  # 'Final results:'
            continue
        m = re.search(r'PD:\[(\d+)/(\d+)', line)
        if m and cur:
            counts[cur] = int(m.group(2))
    return counts
