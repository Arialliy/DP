"""M0: training-free test of the motion-exemplar hypothesis behind DeepPro-MX.

Hypothesis
    A near-stationary target and a ground hot spot are both bright, persistent blobs in the
    window level L, so local contrast (what DeepPro-LV's level pathway mostly uses) cannot
    separate them: on the validation clips the hot spots often have more contrast. A
    near-stationary target does, however, look like the moving targets of the same clip,
    which the temporal profile detects well. If that is true, similarity to moving
    exemplars separates near-stationary targets from hot spots better than contrast does.

Test, per inference window (DeepPro windows, coordinates by the MRTP rule at rate 1)
    positives  near-stationary annotations (rel_speed < 0.2 px/frame) resident in the
               window (annotated in >= half of its frames), each annotation once (closest
               window), at its reference-frame position
    negatives  local maxima of |local contrast of L| at least NEG_SEP px from every
               annotation of the window: the persistent spots the level pathway reports
    exemplars  up to K moving GT boxes (rel_speed >= 0.5) of the same clip from frames
               outside the window (the training rule of DeepPro-MX), patches of the
               recorded frames; optionally also peaks of a cached run (--det-run), the
               exemplars inference would use
    scores     contrast    local contrast of L (3 x 3 maximum)
               ncc_L       max over exemplars of the normalised cross-correlation of the
                           S x S patch of L with the exemplar patch
               absncc_L    the same with |NCC| (polarity-free: a parked car can be colder
                           than a moving one)
               ncc_med     median over every `step`-th window frame of the max NCC of that
                           frame's patch (what DeepPro-MX computes, with a learned encoder)
               ncc_det     ncc_L with the detection exemplars of --det-run
               Every score is the maximum over shifts of up to --shift px (default 2):
               positives sit on box centres, negatives on contrast peaks, and neither
               should lose similarity to a small offset.
Per clip the ROC AUC of each score (positives vs negatives) is computed on the windows that
have exemplars, so all scores see the same samples. Pooled AUCs are not reported because
two clips hold most near-stationary annotations.

A raw AUC far below 0.5 is also information (hot spots brighter than targets) that a
learned level pathway could use. The decision therefore asks whether exemplar similarity
adds separability to what contrast already gives: two logistic regressions with balanced
class weights, held out by window parity (2 folds),
    base  [contrast, contrast^2]
    full  [contrast, contrast^2, ncc_L, absncc_L, ncc_med]
and gain = held-out AUC(full) - held-out AUC(base).

Decision (pre-registered; --delta, --frac, --key-clips)
    GO if gain >= 0.10 on every key clip present and on at least 60 % of the clips with
    >= --min-pos positives and negatives. Otherwise exemplar similarity does not carry the
    missing information and DeepPro-MX should not be trained.

M0c (calib_diag.py) reuses process_clip with band negatives and the exemplar contrasts of
the motion calibration (DeepPro-MXC); without those options the output above is unchanged.

Usage (repository root, VT_ROOT set, profile_motion.py already run)
    python code/exemplar_diag.py --split val
    python code/exemplar_diag.py --split fold0/val fold1/val --det-run official_rgbt
Writes results/exemplar_diag/{positives.csv, negatives.csv, summary.json}.
Exit status 0 on GO, 3 on NO-GO (1 on errors).
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
import mx_data as mxd  # noqa: E402
import stabilize as st  # noqa: E402
from diagnose_history import auc, closest_windows, sample_max  # noqa: E402

NEG_SEP = 6.0        # Chebyshev px from every annotation centre of the window (diagnose_history)
NEG_FLOOR = 3.0      # minimum |contrast| (grey levels) of a negative peak
NEG_PER_WINDOW = 50
SCORES = ('contrast', 'ncc_L', 'absncc_L', 'ncc_med', 'ncc_det')
CAL = ('cal_R', 'cal_absR', 'cal_D', 'cal_P')                 # mx_data.calibrated_features
CAL_DET = tuple(k.replace('cal_', 'det_') for k in CAL)       # the same with --det-run exemplars
CAL_SELF = tuple(k.replace('cal_', 'self_') for k in CAL)     # the same with the no-motion bank (mx_data.self_bank)
EXEMPLAR = ('ncc_L', 'absncc_L', 'ncc_med')    # features added to contrast in the decision
KEY_CLIPS = ('DJI_0117_2', 'DJI_0319_2')


def ncc_matrix(Q, E):
    """Q (n,S,S), E (K,S,S) -> (n,K) normalised cross-correlation; 0 for flat patches."""
    def unit(P):
        P = np.asarray(P, np.float64).reshape(len(P), -1)
        P = P - P.mean(1, keepdims=True)
        nrm = np.linalg.norm(P, axis=1, keepdims=True)
        return np.where(nrm > 1e-6, P / np.maximum(nrm, 1e-12), 0.0)
    if len(Q) == 0 or len(E) == 0:
        return np.zeros((len(Q), len(E)))
    return unit(Q) @ unit(E).T


def window_view(video, chain, frames, coords):
    """Stack of the window in its coordinates, validity, and per-frame maps frame -> ref."""
    frames = list(frames)
    ref = st.reference_frame(frames[0], frames[-1] + 1)
    if coords == 'warp':
        stack, valid = st.warp_frames(video, chain, frames, ref)
        to_ref = {t: st.ref_to(chain, t, ref) for t in frames}
    else:
        stack = np.stack([np.asarray(video[t], np.float32) for t in frames])
        valid = np.ones(stack.shape, bool)
        to_ref = {t: np.eye(3) for t in frames}
    return stack, valid, to_ref


def choose(video, chain, frames, policy):
    if policy == 'raw' or chain is None:
        return 'raw'
    import mr_data  # numpy; imported here so the raw path needs no homographies
    return mr_data.decide(video, chain, frames, policy, 1)[0]


def shifted_patches(img, points, S, r):
    """(n, (2r+1)^2, S, S) patches of img around every point shifted by -r..r px."""
    d = range(-r, r + 1)
    return np.stack([np.stack([mxd.patch2d(img, x + dx, y + dy, S) for dy in d for dx in d])
                     for x, y in points])


def best_match(img, points, E, S, r):
    """(n, K) NCC with each exemplar, maximised over shifts of up to r px."""
    Q = shifted_patches(img, points, S, r)
    n, m = Q.shape[:2]
    return ncc_matrix(Q.reshape(n * m, S, S), E).reshape(n, m, len(E)).max(1)


def sample_absmax(c, x, y, r):
    """Signed value of largest magnitude of c within +-r px of (x, y); NaN if none is finite."""
    h, w = c.shape
    xi, yi = int(round(x)), int(round(y))
    p = c[max(yi - r, 0):min(yi + r + 1, h), max(xi - r, 0):min(xi + r + 1, w)]
    if not p.size or not np.isfinite(p).any():
        return math.nan
    return float(p.ravel()[np.nanargmax(np.abs(p).ravel())])


def exemplar_contrasts(video, rows, shift):
    """Signed local contrast (grey levels) of each exemplar row (t, x, y, extra), from a
    patch just large enough for the contrast ring, as mx.exemplar_contrast."""
    if not len(rows):
        return np.zeros(0)
    S = 2 * (7 + shift) + 1
    return mxd.exemplar_contrast(np.stack([mxd.patch2d(video[int(t)], x, y, S) for t, x, y, _ in rows]), shift)


def calibration(points, cL, ce, r, floor='relative'):
    """Signed contrast of largest magnitude within +-r px of each point (contrast_abs) and its
    (n,4) calibrated features against the exemplar contrasts ce (mx_data.calibrated_features)."""
    c = np.array([sample_absmax(cL, x, y, r) for x, y in points]) if len(points) else np.zeros(0)
    return c, mxd.calibrated_features(c, ce, floor)


def query_scores(points, Lfill, cL, stack, valid, idx, E, E_det, S, r=2):
    """Scores of query points (x, y) in reference coordinates -> dict of arrays.

    Every score is maximised over a (2r+1)^2 neighbourhood: positives sit on box centres
    and negatives on contrast peaks, and a dense detector would take the local maximum too,
    so neither kind should lose similarity to a small offset.
    """
    n = len(points)
    out = {k: np.full(n, np.nan) for k in SCORES}
    if n == 0:
        return out
    out['contrast'] = np.array([sample_max(cL, x, y, r) for x, y in points])
    if len(E):
        c_pos = best_match(Lfill, points, E, S, r)        # max over shifts of NCC
        c_neg = best_match(-Lfill, points, E, S, r)       # max over shifts of -NCC (inverted polarity)
        out['ncc_L'] = c_pos.max(1)
        out['absncc_L'] = np.maximum(c_pos, c_neg).max(1)
        per = np.full((len(idx), n), np.nan)
        for j, i in enumerate(idx):
            ok = np.array([mxd.patch2d(valid[i].astype(np.float32), x, y, S).mean() >= 0.5 for x, y in points])
            if ok.any():
                pts = [p for p, k in zip(points, ok) if k]
                per[j, ok] = best_match(stack[i], pts, E, S, r).max(1)
        with np.errstate(all='ignore'):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                out['ncc_med'] = np.nanmedian(per, axis=0)
    if len(E_det):
        out['ncc_det'] = best_match(Lfill, points, E_det, S, r).max(1)
    return out


def process_clip(clip, video, boxes, speed, chain, args, det=None, band=0, calib=False):
    """One clip -> (positive rows, negative rows, window rows). video (n,H,W) grey;
    boxes per frame (x, y, w, h, cat, track); speed {(clip, t, track): rel_speed};
    chain: homography chain or None (recorded coordinates only); det: (m,4) detection
    candidates (t, x, y, logit) or None.

    M0c options: calib adds to every row the signed contrast of largest magnitude
    (contrast_abs) and the calibrated features against the window's GT exemplars (CAL),
    detection exemplars (CAL_DET) and the no-motion bank of the clip's strongest blobs
    (CAL_SELF, mx_data.self_bank); the window rows get the exemplar contrasts. band > 0
    adds up to that many band negatives per window: peaks whose |contrast| lies within the
    10-90 % range of the GT exemplars' |contrast| (clutter that looks like this scene's
    targets by contrast), drawn at random among those not already taken (kind 'band';
    the strongest peaks are kind 'top')."""
    n = len(video)
    rng = np.random.default_rng(args.seed + sum(map(ord, clip)))
    band_rng = np.random.default_rng(args.seed + 7919 + sum(map(ord, clip)))   # keeps rng's exemplar draws
    self_peaks = mxd.SelfPeaks(video, args.k) if calib else None
    moving = mxd.moving_table(clip, boxes, speed, mxd.MOVING)
    ws = st.windows(n, args.seqlen)
    owner = closest_windows(n, ws)
    S = args.patch
    pos_rows, neg_rows, win_rows = [], [], []
    for k, (a, b) in enumerate(ws):
        frames = range(a, b)
        coords = choose(video, chain, frames, args.coords)
        stack, valid, to_ref = window_view(video, chain, frames, coords)
        L, count = st.masked_median(stack, valid)
        L = np.where(count > 0, L, np.nan)
        cL = st.local_contrast(L)
        Lfill = np.where(np.isfinite(L), L, np.nanmedian(L) if np.isfinite(L).any() else 0.0)
        ex = mxd.gt_bank(rng, moving, a, b - 1, 'outside', args.k)
        E = np.stack([mxd.patch2d(video[int(t)], x, y, S) for t, x, y, _ in ex]) if len(ex) else np.zeros((0, S, S))
        exd = mxd.select_bank(det, a, b - 1, 'outside', args.k) if det is not None else np.zeros((0, 4))
        E_det = (np.stack([mxd.patch2d(video[int(t)], x, y, S) for t, x, y, _ in exd])
                 if len(exd) else np.zeros((0, S, S)))
        idx = list(range(0, b - a, args.step))
        if calib:
            ce = exemplar_contrasts(video, ex, args.shift)
            ce_det = exemplar_contrasts(video, exd, args.shift)
            ce_self = exemplar_contrasts(video, mxd.self_bank(self_peaks, n, a, b - 1, args.k), args.shift)
        present = defaultdict(int)
        centres = []
        for t in frames:
            mb = st.map_boxes(boxes[t], to_ref[t])
            centres.append(mxd.box_centres(mb))
            for row in boxes[t]:
                present[int(row[5])] += 1
        centres = np.concatenate(centres) if centres else np.zeros((0, 2))
        # positives: near-stationary annotations owned by this window, resident in it
        pts, meta = [], []
        for t in np.flatnonzero(owner == k):
            t = int(t)
            mb = st.map_boxes(boxes[t], to_ref[t])
            for row, (x, y) in zip(boxes[t], mxd.box_centres(mb)):
                tid = int(row[5])
                v = speed.get((clip, t, tid), math.nan)
                presence = present[tid] / (b - a)
                if np.isfinite(v) and v < mxd.NEAR and presence >= 0.5:
                    pts.append((x, y))
                    meta.append(dict(clip=clip, t=t, track=tid, cat=int(row[4]), rel_speed=v,
                                     window_presence=presence))
        sc = query_scores(pts, Lfill, cL, stack, valid, idx, E, E_det, S, args.shift)
        if calib:
            add_calibration(sc, pts, cL, ce, ce_det, args.shift, getattr(args, 'floor', 'relative'), ce_self)
        for i, m in enumerate(meta):
            pos_rows.append(dict(m, window_start=a, coords=coords, exemplars=len(ex), det_exemplars=len(exd),
                                 x=float(pts[i][0]), y=float(pts[i][1]),
                                 **{key: float(sc[key][i]) for key in sc}))
        # negatives: strongest persistent spots away from every annotation of the window
        c = np.abs(np.nan_to_num(cL, nan=0.0))
        peak = (c == ndimage.maximum_filter(c, size=9, mode='nearest')) & (c > NEG_FLOOR)
        ys, xs = np.nonzero(peak)
        if len(xs) and len(centres):
            d = np.max(np.abs(np.stack([xs, ys], 1)[:, None, :] - centres[None]), axis=2).min(1)
            xs, ys = xs[d > NEG_SEP], ys[d > NEG_SEP]
        order = np.argsort(-c[ys, xs], kind='stable')
        top = order[:NEG_PER_WINDOW]
        kinds = ['top'] * len(top)
        n_band = 0
        if band > 0 and calib and np.isfinite(ce).sum() >= 2:
            lo, hi = np.nanpercentile(np.abs(ce), [10, 90])
            rest = order[NEG_PER_WINDOW:]
            inside = rest[(c[ys[rest], xs[rest]] >= lo) & (c[ys[rest], xs[rest]] <= hi)]
            pick = np.sort(band_rng.permutation(inside)[:band]) if len(inside) else inside
            n_band = len(pick)
            top = np.concatenate([top, pick]).astype(int)
            kinds += ['band'] * n_band
        npts = [(float(xs[i]), float(ys[i])) for i in top]
        sc = query_scores(npts, Lfill, cL, stack, valid, idx, E, E_det, S, args.shift)
        if calib:
            add_calibration(sc, npts, cL, ce, ce_det, args.shift, getattr(args, 'floor', 'relative'), ce_self)
        for i, (x, y) in enumerate(npts):
            row = dict(clip=clip, window_start=a, coords=coords, exemplars=len(ex),
                       det_exemplars=len(exd), x=x, y=y, **{key: float(sc[key][i]) for key in sc})
            if calib:
                row['kind'] = kinds[i]
                row['peak_abs_contrast'] = float(c[int(y), int(x)])
            neg_rows.append(row)
        w = dict(clip=clip, window_start=a, window_end=b, coords=coords, exemplars=len(ex),
                 exemplar_frames=[int(t) for t in ex[:, 0]], det_exemplars=len(exd),
                 positives=len(pts), negatives=len(npts))
        if calib:
            w.update(band_negatives=n_band, exemplar_contrast=[round(float(v), 3) for v in ce],
                     det_exemplar_contrast=[round(float(v), 3) for v in ce_det],
                     self_exemplar_contrast=[round(float(v), 3) for v in ce_self])
        win_rows.append(w)
    return pos_rows, neg_rows, win_rows


def add_calibration(sc, points, cL, ce, ce_det, r, floor='relative', ce_self=()):
    """Adds contrast_abs, CAL, CAL_DET and CAL_SELF to a query_scores dict, in place. M0c
    compares every feature set on contrast_abs, the contrast the calibration uses."""
    cabs, f = calibration(points, cL, ce, r, floor)
    _, fd = calibration(points, cL, ce_det, r, floor)
    _, fs = calibration(points, cL, ce_self, r, floor)
    for j, key in enumerate(CAL_SELF):
        sc[key] = fs[:, j] if len(fs) else np.zeros(0)
    sc['contrast_abs'] = cabs
    for j, key in enumerate(CAL):
        sc[key] = f[:, j] if len(f) else np.zeros(0)
    for j, key in enumerate(CAL_DET):
        sc[key] = fd[:, j] if len(fd) else np.zeros(0)
    if not len(ce_det):
        for key in CAL_DET:
            sc[key] = np.full(len(points), np.nan)


def logistic_fit(Z, y, l2=1e-2, iters=50):
    """Newton steps of an L2 logistic regression with balanced class weights.
    Z (n,d) standardised features, y (n,) in {0,1} -> weights (d + 1,), the last the intercept."""
    X = np.c_[Z, np.ones(len(Z))]
    cw = np.where(y == 1, 0.5 / max(y.mean(), 1e-9), 0.5 / max(1 - y.mean(), 1e-9))
    reg = np.full(X.shape[1], l2)
    reg[-1] = 0.0
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (cw * (p - y)) / len(y) + reg * w
        hess = (X * (cw * p * (1 - p))[:, None]).T @ X / len(y) + np.diag(reg + 1e-9)
        step = np.linalg.solve(hess, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def heldout_auc(X, y, fold):
    """AUC of 2-fold held-out logistic scores (fold ids 0/1); nan if a training fold lacks a class."""
    X = np.asarray(X, np.float64).reshape(len(y), -1)
    pred = np.full(len(y), np.nan)
    for f in (0, 1):
        tr, te = fold != f, fold == f
        if not te.any() or len(np.unique(y[tr])) < 2:
            return math.nan
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        w = logistic_fit((X[tr] - mu) / sd, y[tr])
        pred[te] = np.c_[(X[te] - mu) / sd, np.ones(int(te.sum()))] @ w
    return auc(pred[y == 1], pred[y == 0])


def separability(pos, neg):
    """Held-out AUCs of [contrast, contrast^2] and of the same plus exemplar similarity.
    Folds alternate over the windows, so a window's samples stay in one fold."""
    rows = [r for r in pos if np.isfinite(r['contrast'])] + [r for r in neg if np.isfinite(r['contrast'])]
    if not rows:
        return math.nan, math.nan
    y = np.array([1.0] * sum(np.isfinite(r['contrast']) for r in pos) +
                 [0.0] * sum(np.isfinite(r['contrast']) for r in neg))
    starts = sorted({r['window_start'] for r in rows})
    fold = np.array([starts.index(r['window_start']) % 2 for r in rows])
    c = np.array([r['contrast'] for r in rows])
    base = np.c_[c, c * c]
    ex = np.array([[r[k] for k in EXEMPLAR] for r in rows], float)
    for j in range(ex.shape[1]):                      # ncc_med is nan where the patch left the field
        col = ex[:, j]
        col[~np.isfinite(col)] = np.nanmedian(col) if np.isfinite(col).any() else 0.0
    return heldout_auc(base, y, fold), heldout_auc(np.c_[base, ex], y, fold)


def summarize(pos, neg, windows, args):
    """Per-clip AUCs on the windows with GT exemplars, and the decision inputs."""
    p = [r for r in pos if r['exemplars'] > 0]
    q = [r for r in neg if r['exemplars'] > 0]
    s = dict(positives=len(pos), negatives=len(neg), positives_with_exemplars=len(p),
             negatives_with_exemplars=len(q), windows=len(windows),
             windows_without_exemplars=sum(w['exemplars'] == 0 for w in windows),
             mean_exemplars=float(np.mean([w['exemplars'] for w in windows])) if windows else math.nan,
             warped_windows=sum(w['coords'] == 'warp' for w in windows))
    for key in SCORES:
        s[f'auc_{key}'] = auc([r[key] for r in p], [r[key] for r in q])
    s['auc_lr_base'], s['auc_lr_full'] = separability(p, q)
    s['gain'] = s['auc_lr_full'] - s['auc_lr_base']
    s['eligible'] = bool(len(p) >= args.min_pos and len(q) >= args.min_pos and np.isfinite(s['gain']))
    return s


def decide(per_clip, args):
    """GO / NO-GO from the per-clip summaries (rule in the module docstring)."""
    eligible = [c for c, s in per_clip.items() if s['eligible']]
    passed = [c for c in eligible if per_clip[c]['gain'] >= args.delta]
    keys = [c for c in args.key_clips if c in per_clip]
    key_ok = {c: bool(per_clip[c]['eligible'] and per_clip[c]['gain'] >= args.delta) for c in keys}
    frac = len(passed) / len(eligible) if eligible else math.nan
    go = bool(eligible and all(key_ok.values()) and frac >= args.frac)
    return dict(go=go, eligible_clips=sorted(eligible), passed_clips=sorted(passed), fraction_passed=frac,
                key_clips=key_ok, rule=f'gain >= {args.delta} on every key clip present and on >= '
                                       f'{args.frac:.0%} of eligible clips (>= {args.min_pos} positives and negatives)')


def detection_candidates(run, clip, n, min_score, root='cache/preds'):
    """Peaks of a cached run's logits as exemplar candidates (t, x, y, logit)."""
    import evaluate
    lg = np.load(os.path.join(root, run, f'{clip}.npy'), mmap_mode='r')
    if len(lg) != n:
        raise ValueError(f'{run}/{clip}: {len(lg)} frames, clip has {n}')
    rows = [mxd.peak_rows(lg[t], t, evaluate.peaks, min_score) for t in range(n)]
    return np.concatenate(rows) if rows else np.zeros((0, 4))


def _one(task):
    clip, args, speed = task
    import vtdata
    sc = vtdata.Scene(clip)
    mxd.require_speeds({clip: sc.boxes}, speed, args.targets)
    video = st.load_video_u8(clip, sc)
    chain = None if args.coords == 'raw' else st.load_chain(clip, len(sc))
    det = detection_candidates(args.det_run, clip, len(sc), args.det_min_score) if args.det_run else None
    return clip, process_clip(clip, video, sc.boxes, speed, chain, args, det)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', nargs='*', default=['val'])
    ap.add_argument('--clips', default=None, help='comma-separated clips (overrides --split)')
    ap.add_argument('--targets', default='results/profile/targets.csv')
    ap.add_argument('--seqlen', type=int, default=40)
    ap.add_argument('--coords', choices=('raw', 'warp', 'auto'), default='auto')
    ap.add_argument('--patch', type=int, default=11, help='odd patch size for NCC')
    ap.add_argument('--k', type=int, default=16, help='exemplars per window')
    ap.add_argument('--step', type=int, default=4, help='frame step of ncc_med')
    ap.add_argument('--shift', type=int, default=2, help='every score is maximised over +-shift px')
    ap.add_argument('--det-run', dest='det_run', default=None, help='cached run whose peaks give ncc_det')
    ap.add_argument('--det-min-score', dest='det_min_score', type=float, default=0.0)
    ap.add_argument('--min-pos', dest='min_pos', type=int, default=20)
    ap.add_argument('--delta', type=float, default=0.10, help='minimum held-out AUC gain per clip')
    ap.add_argument('--frac', type=float, default=0.6)
    ap.add_argument('--key-clips', dest='key_clips', default=','.join(KEY_CLIPS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results/exemplar_diag')
    ap.add_argument('--nproc', type=int, default=4)
    args = ap.parse_args(argv)
    args.key_clips = [c for c in args.key_clips.split(',') if c]
    if args.patch % 2 != 1:
        raise SystemExit('--patch must be odd')
    if args.clips:
        clips = args.clips.split(',')
    else:
        import splits
        clips = [c for s in args.split for c in splits.clips(s)]
    speed = mxd.read_speeds(args.targets, clips)
    os.makedirs(args.out, exist_ok=True)
    tasks = [(c, args, {key: v for key, v in speed.items() if key[0] == c}) for c in clips]
    per_clip, all_pos, all_neg, all_win = {}, [], [], []
    with Pool(max(1, min(args.nproc, len(tasks)))) as pool:
        for clip, (pos, neg, win) in pool.imap_unordered(_one, tasks):
            per_clip[clip] = summarize(pos, neg, win, args)
            all_pos += pos
            all_neg += neg
            all_win += win
            s = per_clip[clip]
            print(f"{clip:12s} pos={s['positives_with_exemplars']:5d} neg={s['negatives_with_exemplars']:5d} "
                  f"AUC contrast={s['auc_contrast']:.3f} ncc_L={s['auc_ncc_L']:.3f} "
                  f"|ncc|={s['auc_absncc_L']:.3f} ncc_med={s['auc_ncc_med']:.3f} gain={s['gain']:+.3f}", flush=True)
    for name, rows in (('positives.csv', all_pos), ('negatives.csv', all_neg)):
        if rows:
            with open(os.path.join(args.out, name), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    verdict = decide(per_clip, args)
    summary = dict(settings={k: v for k, v in vars(args).items()}, per_clip=per_clip, decision=verdict,
                   windows=all_win)
    json.dump(summary, open(os.path.join(args.out, 'summary.json'), 'w'), indent=1, default=float)
    print('\nclip           pos   neg  contrast  ncc_L  |ncc|  ncc_med  ncc_det  lr_base  lr_full    gain')
    for c in sorted(per_clip):
        s = per_clip[c]
        print(f"{c:12s} {s['positives_with_exemplars']:5d} {s['negatives_with_exemplars']:5d} "
              f"{s['auc_contrast']:9.3f} {s['auc_ncc_L']:6.3f} {s['auc_absncc_L']:6.3f} {s['auc_ncc_med']:8.3f} "
              f"{s['auc_ncc_det']:8.3f} {s['auc_lr_base']:8.3f} {s['auc_lr_full']:8.3f} {s['gain']:+7.3f}"
              f"{'' if s['eligible'] else '  (not eligible)'}")
    print(f"\n{'GO' if verdict['go'] else 'NO-GO'}: {verdict['rule']}; passed {len(verdict['passed_clips'])}/"
          f"{len(verdict['eligible_clips'])} eligible clips; key clips {verdict['key_clips']}")
    return summary


if __name__ == '__main__':
    # Exit status 0 on GO and 3 on NO-GO, so that a pipeline (run_mx.sh) stops before training.
    sys.exit(0 if main()['decision']['go'] else 3)
