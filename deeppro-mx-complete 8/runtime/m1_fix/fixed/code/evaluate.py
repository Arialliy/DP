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
def shooting_rules(prob, mask, th, edges='official'):
    """Vectorised re-implementation of ShootingRules.forward for one frame.

    edges 'official' (default) keeps a quirk of ShootingRules.py: it slices
    [i - r : i + r + 1] around every target pixel, and for i < r the negative start makes
    the slice empty. A target pixel in the first row or column therefore cannot be hit,
    and one in the first 4 rows or columns protects nothing from false alarms. 'clipped'
    clips the neighbourhoods at the image edge instead (the earlier behaviour here).
    """
    pos = prob >= th
    tgt = mask > 0
    lab, n = ndimage.label(tgt, structure=np.ones((3, 3)))
    if n:
        if edges == 'official':
            rr, cc = np.indices(tgt.shape)
            hit_src, prot_src = tgt & (rr >= 1) & (cc >= 1), tgt & (rr >= 4) & (cc >= 4)
        elif edges == 'clipped':
            hit_src = prot_src = tgt
        else:
            raise ValueError(edges)
        near = ndimage.binary_dilation(pos, structure=np.ones((3, 3))) & hit_src
        hit = ndimage.maximum(near, lab, np.arange(1, n + 1))
        true = int(np.sum(hit > 0))
        protect = ndimage.binary_dilation(prot_src, structure=np.ones((9, 9)))
    else:
        true, protect = 0, np.zeros_like(pos)
    false = int(np.count_nonzero(pos & ~protect))
    return false, true, n


# test.py's Th_Seg: the authors' 27 thresholds of the Pd / Fa curve
TH_SEG = (0, 1e-20, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.3, .35, 0.4, .45, 0.5, .55,
          0.6, .65, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1)


def shooting_rules_sweep(prob, mask, ths=TH_SEG, edges='official'):
    """shooting_rules for every threshold at once, plus test.py's IoU counts.

    -> arrays over ths: false (prob >= th outside the protection), true (targets hit),
    targets (8-connected mask components, the same for every th), and intersection / union
    of prob > th with the mask (test.py's IoU, a strict comparison).
    Comparisons are made in float32, as the authors' float32 maps against their thresholds
    under NumPy 1 (and against Python floats under NumPy 2): a probability that is
    float32(1e-6) counts as >= 1e-6.
    """
    ths = np.asarray(ths, np.float32)
    prob = np.asarray(prob, np.float32)
    tgt = mask > 0
    lab, n = ndimage.label(tgt, structure=np.ones((3, 3)))
    true = np.zeros(len(ths), np.int64)
    if n:
        if edges == 'official':
            rr, cc = np.indices(tgt.shape)
            hit_src, prot_src = tgt & (rr >= 1) & (cc >= 1), tgt & (rr >= 4) & (cc >= 4)
        elif edges == 'clipped':
            hit_src = prot_src = tgt
        else:
            raise ValueError(edges)
        protect = ndimage.binary_dilation(prot_src, structure=np.ones((9, 9)))
        near = ndimage.maximum_filter(prob, size=3, mode='constant', cval=-np.inf)  # 3 x 3 inside the image
        best = np.full(n, -np.inf, np.float32)
        np.maximum.at(best, lab[hit_src] - 1, near[hit_src])
        true = (best[None, :] >= ths[:, None]).sum(1)
    else:
        protect = np.zeros_like(tgt)
    outside = np.sort(prob[~protect])
    false = len(outside) - np.searchsorted(outside, ths, side='left')              # prob >= th
    pin = np.sort(prob[tgt])
    pout = np.sort(prob[~tgt])
    inter = len(pin) - np.searchsorted(pin, ths, side='right')                      # prob > th
    union = int(tgt.sum()) + len(pout) - np.searchsorted(pout, ths, side='right')
    return false, true, n, inter, union


def author_metrics(logits, scene, ths=TH_SEG, rule='paper_k6_floor', edges='official'):
    """The authors' test.py protocol on fused logits: ShootingRules Pd / Fa at each
    threshold (prob >= th) and, as test.py, IoU of prob > th against the binary mask."""
    k = len(ths)
    acc = np.zeros((5, k), np.int64)
    for t in range(len(scene)):
        m = vtdata.gaussian_mask(scene.boxes[t], scene.H, scene.W, rule)
        p = 1.0 / (1.0 + np.exp(-logits[t].astype(np.float32)))
        f, tr, n, i, u = shooting_rules_sweep(p, m, ths, edges)
        acc += np.stack([f, tr, np.full(k, n), i, u])
    pix = len(scene) * scene.H * scene.W
    return {th: dict(false_px=int(acc[0, j]), tp=int(acc[1, j]), targets=int(acc[2, j]), pixels=pix,
                     intersection=int(acc[3, j]), union=int(acc[4, j])) for j, th in enumerate(ths)}


def official_auc(fa, pd):
    """sklearn.metrics.auc(Fa, Pd) as write_results.writeMIRST_ROC computes it: the
    trapezoid area over the thresholds' points, x monotonic in either direction."""
    fa, pd = np.asarray(fa, np.float64), np.asarray(pd, np.float64)
    d = np.diff(fa)
    if np.all(d <= 0):
        direction = -1.0
    elif np.all(d >= 0):
        direction = 1.0
    else:
        raise ValueError('Fa is not monotonic over the thresholds')
    trapezoid = getattr(np, 'trapezoid', None) or np.trapz
    return float(direction * trapezoid(pd, fa))
