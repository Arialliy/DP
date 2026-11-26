"""Evaluation.

Primary (object level, threshold-free):
  * detections = local maxima of the logit map (9x9 window, one per plateau), score = logit
  * a detection matches a GT box if its distance to the box (0 inside) <= TOL px;
    greedy one-to-one assignment in descending score order (nearest box centre first)
  * FROC: Pd versus false alarms per frame; scalar = CPM, the mean Pd at
    FA/frame in {1/8, 1/4, 1/2, 1, 2, 4} (competition performance metric, LUNA16 style)
  * per-GT record of the matched detection score -> exact Pd for any threshold and stratum
Secondary (authors' protocol): pixel ShootingRules on reconstructed Gaussian masks
  (TP if a positive pixel lies within +-1 px of a target pixel; false pixels = positives
  farther than 4 px from every target pixel; Fa = false pixels / all pixels).
"""
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
import vtdata  # noqa: E402

TOL = 2.0
NMS = 9
FLOOR = -12.0  # score floor for candidate peaks (logit -12, p ~ 6e-6)
CPM_FA = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)


def peaks(logit, floor=FLOOR, size=NMS, topk=300):
    mx = ndimage.maximum_filter(logit, size=size, mode='nearest')
    m = (logit >= mx) & (logit > floor)
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros((0, 3))
    idx = np.arange(1, n + 1)
    cy = ndimage.mean(np.indices(logit.shape)[0], lab, idx)
    cx = ndimage.mean(np.indices(logit.shape)[1], lab, idx)
    sc = ndimage.maximum(logit, lab, idx)
    d = np.stack([cx, cy, sc], 1)
    d = d[np.argsort(-d[:, 2])][:topk]
    return d


def box_dist(px, py, boxes):
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    dx = np.maximum(np.maximum(x - px, 0), px - (x + w - 1))
    dy = np.maximum(np.maximum(y - py, 0), py - (y + h - 1))
    return np.hypot(dx, dy)


def match_frame(det, boxes, tol=TOL):
    """-> gt_score (len boxes; -inf if unmatched), det_is_tp (len det)."""
    g = np.full(len(boxes), -np.inf)
    tp = np.zeros(len(det), bool)
    if len(boxes) == 0 or len(det) == 0:
        return g, tp
    cxy = boxes[:, :2] + boxes[:, 2:4] / 2.0 - 0.5
    used = np.zeros(len(boxes), bool)
    for i, (px, py, s) in enumerate(det):
        d = box_dist(px, py, boxes)
        ok = (d <= tol) & ~used
        if ok.any():
            c = np.where(ok)[0]
            j = c[np.argmin(np.hypot(cxy[c, 0] - px, cxy[c, 1] - py))]
            used[j] = True
            g[j] = s
            tp[i] = True
    return g, tp


def eval_clip(logits, scene):
    """-> dict(gt=list of (t, track, cat, score), fp_scores=array, frames=int)"""
    gt, fps = [], []
    for t in range(len(scene)):
        det = peaks(logits[t].astype(np.float32))
        b = scene.boxes[t]
        g, tp = match_frame(det, b)
        for k in range(len(b)):
            gt.append((t, int(b[k, 5]), int(b[k, 4]), float(g[k])))
        fps.append(det[~tp, 2])
    return dict(gt=gt, fp=np.concatenate(fps) if fps else np.zeros(0), frames=len(scene))


def froc(gt_scores, fp_scores, frames):
    gt_scores = np.asarray(gt_scores, float)
    fp_scores = np.sort(np.asarray(fp_scores, float))[::-1]
    thr = np.unique(np.concatenate([gt_scores[np.isfinite(gt_scores)], fp_scores]))[::-1]
    gs = np.sort(gt_scores[np.isfinite(gt_scores)])[::-1]
    tp = np.searchsorted(-gs, -thr, side='right')
    fp = np.searchsorted(-fp_scores, -thr, side='right')
    return thr, tp / max(len(gt_scores), 1), fp / max(frames, 1)


def pd_at(gt_scores, fp_scores, frames, fa):
    """Max Pd with FA/frame <= fa; also returns the threshold achieving it."""
    thr, pd, far = froc(gt_scores, fp_scores, frames)
    ok = far <= fa
    if not ok.any():
        return 0.0, np.inf
    i = np.where(ok)[0][-1]
    return float(pd[i]), float(thr[i])


def cpm(gt_scores, fp_scores, frames, fas=CPM_FA):
    return float(np.mean([pd_at(gt_scores, fp_scores, frames, f)[0] for f in fas]))


def ap(gt_scores, fp_scores):
    """Average precision of point detections (all-point interpolation)."""
    gs = np.asarray(gt_scores, float)
    tps = gs[np.isfinite(gs)]
    s = np.concatenate([tps, fp_scores])
    lab = np.concatenate([np.ones(len(tps)), np.zeros(len(fp_scores))])
    o = np.argsort(-s, kind='stable')
    lab = lab[o]
    ctp = np.cumsum(lab)
    cfp = np.cumsum(1 - lab)
    rec = ctp / max(len(gs), 1)
    prec = ctp / np.maximum(ctp + cfp, 1)
    mprec = np.maximum.accumulate(prec[::-1])[::-1]
    r = np.concatenate([[0], rec])
    return float(np.sum((r[1:] - r[:-1]) * mprec))


# ---------------------------------------------------------------- authors' protocol
def shooting_rules(prob, mask, th):
    """Vectorised re-implementation of ShootingRules.forward for one frame."""
    pos = prob >= th
    lab, n = ndimage.label(mask > 0, structure=np.ones((3, 3)))
    if n:
        near = ndimage.binary_dilation(pos, structure=np.ones((3, 3)))
        hit = ndimage.maximum(near, lab, np.arange(1, n + 1))
        true = int(np.sum(hit > 0))
        protect = ndimage.binary_dilation(mask > 0, structure=np.ones((9, 9)))
    else:
        true, protect = 0, np.zeros_like(pos)
    false = int(np.count_nonzero(pos & ~protect))
    return false, true, n


def author_metrics(logits, scene, ths=(0.5, 0.99), rule='paper_k6_floor'):
    out = {th: [0, 0, 0] for th in ths}
    for t in range(len(scene)):
        m = vtdata.gaussian_mask(scene.boxes[t], scene.H, scene.W, rule)
        p = 1.0 / (1.0 + np.exp(-logits[t].astype(np.float32)))
        for th in ths:
            f, tr, n = shooting_rules(p, m, th)
            out[th][0] += f
            out[th][1] += tr
            out[th][2] += n
    pix = len(scene) * scene.H * scene.W
    return {th: dict(false_px=v[0], tp=v[1], targets=v[2], pixels=pix) for th, v in out.items()}
