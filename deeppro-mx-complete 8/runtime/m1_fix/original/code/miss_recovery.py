"""M1: can single frames recover the targets a temporal detector misses, taught by its own detections?

Question (the advisor's claim, made measurable)
    A target that the temporal profile misses in some frames still looks, in each single
    frame, like the targets the same detector finds in the same video: often the very same
    track, detected in other windows. On the official RGB-T test set the misses of
    DeepPro-Plus are concentrated in time (the first 60 frames of the clips hold 41 % of the
    misses at 1e-5; the same tracks are missed 26 % of the time there and 10 % later), not
    in near-stationary targets (isolated near-stationary boxes are missed as often as
    isolated moving ones). If single-frame appearance, calibrated by the detector's own
    confident detections, separates the missed targets from the clutter of the frame, a
    single-frame appearance pathway taught at test time by the temporal pathway can recover
    them, whatever made the temporal evidence weak.

What is measured, per clip of --split, from a cached run (no training, no test-label fitting
of the teacher)
    misses     a GT box is missed at threshold th if no probability >= th lies on its mask
               (vtdata.gaussian_mask, paper_k6_floor) or its 3 x 3 neighbourhood, as
               ShootingRules hits a target (per box, not per merged component)
    teacher    peaks of the run's logits with probability >= --teacher-th (0.5, the authors'
               operating point), frames outside the window (the training rule of DeepPro-MX),
               greedily separated, up to --k: the network's own confident detections
    single     every frame on its own: candidates are local maxima of |local contrast| of the
    frame      frame (mx_data.frame_peaks, top --per-frame), labelled target within --r-pos px
               of a GT centre, clutter beyond --r-neg px of every GT centre (others dropped)
    features   of a target (its centre, largest |contrast| within +-2 px) and of a candidate,
               against the teacher of its window: the calibrated contrast R, |R|, D, P
               (mx_data.calibrated_features, relative floor) and the NCC with the teacher
               patches (largest, and largest |NCC|), all from the single frame
    score      a logistic regression of those features, fitted leave-one-recording-out on
               the other clips' target vs clutter candidates (every clip and class weighs
               the same); a window without teacher detections cannot recover anything
    budget     one threshold over all clips that admits --budgets clutter candidates per
               frame (pooled), and per-clip thresholds (clip)
Reported per group (all; frames before / after --early; category; target speed band;
camera group; clip): targets, misses, misses recovered (score >= threshold), hits covered,
and recall official vs official-or-recovered (box level; the added false alarms are the
budget, in candidates per frame, not the pixel Fa of the authors' protocol).
Per window also the background stationarity residuals of multirate.stationarity (raw and
compensated, annotation-free homographies) and the window's miss rate, to see whether the
windows with many misses are the ones whose background changes.

The LORO fit uses the labels of the other evaluated clips: this is a diagnostic of what is
recoverable, not a test result. A test-free version fits on --fit-split clips scored by the
same run (e.g. a cross-validation run's predictions on its own validation folds).

Usage
    python code/miss_recovery.py --run official_rgbt --split test
Writes results/miss_recovery/<run>/<split>/{targets.csv, windows.csv, summary.json}.
"""
import argparse
import csv
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
import calib_diag as cd  # noqa: E402
import exemplar_diag as ed  # noqa: E402
import mx_data as mxd  # noqa: E402
import stabilize as st  # noqa: E402
from diagnose_history import closest_windows  # noqa: E402

FEATURES = ('R', 'absR', 'D', 'P', 'ncc', 'absncc')
BANDS = (('near', 0.0, 0.2), ('mid', 0.2, 0.5), ('moving', 0.5, math.inf))


def logit(p):
    return float(np.log(p / (1.0 - p)))


def box_hits(prob, boxes, H, W, ths, rule='paper_k6_floor'):
    """(n_boxes, len(ths)) bool: some probability >= th on the box's mask or next to it."""
    import vtdata
    out = np.zeros((len(boxes), len(ths)), bool)
    for i, b in enumerate(boxes):
        m = vtdata.gaussian_mask(np.asarray(b, float)[None], H, W, rule) > 0
        if not m.any():
            continue
        m = ndimage.binary_dilation(m, structure=np.ones((3, 3)))
        best = float(prob[m].max())
        out[i] = [best >= th for th in ths]
    return out


def ncc_features(frame, points, E, S):
    """Largest NCC and largest |NCC| of each point's S x S patch with the teacher patches E."""
    if not len(points) or not len(E):
        return np.zeros(len(points)), np.zeros(len(points))
    Q = np.stack([mxd.patch2d(frame, x, y, S) for x, y in points])
    c = ed.ncc_matrix(Q, E)
    return c.max(1), np.abs(c).max(1)


def features(frame, cmap, points, ce, E, S, shift):
    """(n, 6) FEATURES of points of one frame against a teacher (contrasts ce, patches E)."""
    if not len(points):
        return np.zeros((0, len(FEATURES)))
    c = np.array([ed.sample_absmax(cmap, x, y, shift) for x, y in points])
    cal = mxd.calibrated_features(c, ce) if len(ce) else np.zeros((len(points), 4))
    ncc, absncc = ncc_features(frame, points, E, S)
    out = np.c_[cal, ncc, absncc]
    out[~np.isfinite(out)] = 0.0
    return out


def process_clip(clip, video, boxes, logits, args, chain=None, cam_group='static', speed=None):
    """-> (target rows, candidate rows, window rows) of one clip. video (n,H,W) grey,
    boxes per frame (x, y, w, h, cat, track), logits (n,H,W) of the run."""
    import multirate
    n, H, W = video.shape
    speed = speed or {}
    ths = list(args.thresholds)
    teacher = np.concatenate([mxd.peak_rows(np.asarray(logits[t], np.float32), t, _peaks, logit(args.teacher_th))
                              for t in range(n)]) if n else np.zeros((0, 4))
    ws = st.windows(n, args.seqlen)
    owner = closest_windows(n, ws)
    S, shift = args.patch, args.shift
    banks = []
    for a, b in ws:
        rows = mxd.select_bank(teacher, a, b - 1, 'outside', args.k)
        ce = ed.exemplar_contrasts(video, rows, shift)
        E = (np.stack([mxd.patch2d(video[int(t)], x, y, S) for t, x, y, _ in rows])
             if len(rows) else np.zeros((0, S, S)))
        banks.append((rows, ce, E))
    low = multirate.lowres(video, 4) if chain is not None else None
    targets, cands, wins = [], [], []
    for t in range(n):
        frame = np.asarray(video[t], np.float32)
        cmap = st.local_contrast(frame)
        rows, ce, E = banks[owner[t]]
        prob = 1.0 / (1.0 + np.exp(-np.asarray(logits[t], np.float32)))
        b = np.asarray(boxes[t], float).reshape(-1, 6)
        cen = mxd.box_centres(b) if len(b) else np.zeros((0, 2))
        hits = box_hits(prob, b, H, W, ths) if len(b) else np.zeros((0, len(ths)), bool)
        f_t = features(frame, cmap, [tuple(p) for p in cen], ce, E, S, shift)
        for i, row in enumerate(b):
            v = speed.get((clip, t, int(row[5])), math.nan)
            targets.append(dict(clip=clip, t=t, track=int(row[5]), cat=int(row[4]), x=float(cen[i, 0]),
                                y=float(cen[i, 1]), rel_speed=v, camera=cam_group, window=int(owner[t]),
                                teacher=len(rows), **{f'hit_{th:g}': bool(hits[i, j]) for j, th in enumerate(ths)},
                                **{k: float(f_t[i, j]) for j, k in enumerate(FEATURES)}))
        pk = mxd.frame_peaks(frame, t, args.per_frame)
        pts = []
        for _, x, y, _ in pk:
            d = np.max(np.abs(cen - [x, y]), axis=1).min() if len(cen) else np.inf
            if d <= args.r_pos:
                continue                      # target candidates: the GT centres above stand for them
            if d > args.r_neg:
                pts.append((float(x), float(y)))
        f_c = features(frame, cmap, pts, ce, E, S, shift)
        for i, (x, y) in enumerate(pts):
            cands.append(dict(clip=clip, t=t, x=x, y=y, window=int(owner[t]), teacher=len(rows),
                              **{k: float(f_c[i, j]) for j, k in enumerate(FEATURES)}))
    for k, (a, b) in enumerate(ws):
        frames = list(range(a, b))
        stats = (multirate.stationarity(low[frames], chain, frames, 4) if chain is not None
                 else dict(raw=math.nan, warp=math.nan, cover=math.nan))
        mine = [r for r in targets if r['window'] == k]
        th0 = f'hit_{ths[0]:g}'
        wins.append(dict(clip=clip, window=k, first=a, last=b - 1, teacher=len(banks[k][0]),
                         targets=len(mine), misses=sum(not r[th0] for r in mine),
                         stationarity_raw=stats.get('raw', math.nan), stationarity_warp=stats.get('warp', math.nan),
                         cover=stats.get('cover', math.nan)))
    return targets, cands, wins


def _peaks(lg):
    import evaluate
    return evaluate.peaks(lg)


def loro_scores(targets, cands):
    """Logistic scores of every target and candidate, fitted on the other recordings'
    targets (1) and clutter candidates (0). Rows without a teacher score -inf."""
    rows = targets + cands
    y = np.array([1.0] * len(targets) + [0.0] * len(cands))
    X = np.array([[r[k] for k in FEATURES] for r in rows], float).reshape(len(rows), len(FEATURES))
    clips = np.array([r['clip'] for r in rows])
    groups = np.array([cd.recording(c) for c in clips])
    has = np.array([r['teacher'] > 0 for r in rows])
    score = np.full(len(rows), -np.inf)
    for c in sorted(set(clips)):
        te = (clips == c) & has
        tr = (groups != groups[clips == c][0]) & has
        if not te.any() or len(np.unique(y[tr])) < 2:
            continue
        w = cd.clip_weights(clips[tr], y[tr])
        mu = (w[:, None] * X[tr]).sum(0)
        sd = np.sqrt((w[:, None] * (X[tr] - mu) ** 2).sum(0)) + 1e-9
        beta = cd.weighted_logistic((X[tr] - mu) / sd, y[tr], w)
        score[te] = np.c_[(X[te] - mu) / sd, np.ones(int(te.sum()))] @ beta
    for r, s in zip(rows, score):
        r['score'] = float(s)


def thresholds(cands, frames, budget, mode):
    """Score thresholds that admit `budget` clutter candidates per frame: one for all clips
    (pooled) or one per clip."""
    def thr(scores, n):
        s = np.sort(np.asarray([v for v in scores if np.isfinite(v)]))[::-1]
        k = int(math.floor(budget * n))
        if k <= 0:
            return math.inf
        return s[k - 1] if k <= len(s) else -math.inf
    if mode == 'pooled':
        t = thr([r['score'] for r in cands], sum(frames.values()))
        return {c: t for c in frames}
    return {c: thr([r['score'] for r in cands if r['clip'] == c], frames[c]) for c in frames}


def summarize(targets, thr, th, early, cat_of=None):
    """Recovery per group at one official threshold th and per-clip score thresholds thr."""
    key = f'hit_{th:g}'

    def band(v):
        for name, lo, hi in BANDS:
            if lo <= v < hi:
                return name
        return 'unknown'
    groups = {'all': lambda r: True,
              f'frames < {early}': lambda r: r['t'] < early, f'frames >= {early}': lambda r: r['t'] >= early}
    for name, _, _ in BANDS:
        groups[f'speed {name}'] = (lambda nm: (lambda r: band(r['rel_speed']) == nm))(name)
    for cam in ('static', 'moving'):
        groups[f'camera {cam}'] = (lambda cm: (lambda r: r['camera'] == cm))(cam)
    for cat in sorted({r['cat'] for r in targets}):
        groups[f'category {cat}'] = (lambda ct: (lambda r: r['cat'] == ct))(cat)
    for clip in sorted({r['clip'] for r in targets}):
        groups[f'clip {clip}'] = (lambda cl: (lambda r: r['clip'] == cl))(clip)
    out = {}
    for name, sel in groups.items():
        rows = [r for r in targets if sel(r)]
        if not rows:
            continue
        hit = np.array([r[key] for r in rows])
        rec = np.array([r['score'] >= thr[r['clip']] for r in rows])
        out[name] = dict(targets=len(rows), misses=int((~hit).sum()), recovered=int((~hit & rec).sum()),
                         hits_covered=int((hit & rec).sum()), recall=float(hit.mean()),
                         recall_with_recovery=float((hit | rec).mean()),
                         recovered_fraction=float((~hit & rec).sum() / max((~hit).sum(), 1)))
    return out


def _one(task):
    clip, args, speed, cam_group = task
    import vtdata
    sc = vtdata.Scene(clip)
    video = st.load_video_u8(clip, sc)
    logits = np.load(os.path.join(args.preds, args.run, f'{clip}.npy'), mmap_mode='r')
    if len(logits) != len(sc):
        raise ValueError(f'{args.run}/{clip}: {len(logits)} frames, clip has {len(sc)}')
    try:
        chain = st.load_chain(clip, len(sc))
    except FileNotFoundError:
        chain = None
    return clip, process_clip(clip, video, sc.boxes, logits, args, chain, cam_group, speed)


def write_rows(path, rows):
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, help='cached run under --preds (the teacher and the misses)')
    ap.add_argument('--split', default='test')
    ap.add_argument('--preds', default='cache/preds')
    ap.add_argument('--targets', default='results/profile/targets.csv', help='rel_speed for the speed bands')
    ap.add_argument('--thresholds', type=float, nargs='+', default=[1e-5, 0.5], help='official thresholds')
    ap.add_argument('--teacher-th', dest='teacher_th', type=float, default=0.5)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--patch', type=int, default=11)
    ap.add_argument('--shift', type=int, default=2)
    ap.add_argument('--per-frame', dest='per_frame', type=int, default=100, help='candidates per frame')
    ap.add_argument('--r-pos', dest='r_pos', type=float, default=3.0)
    ap.add_argument('--r-neg', dest='r_neg', type=float, default=8.0)
    ap.add_argument('--budgets', type=float, nargs='+', default=[0.25, 1.0, 2.0])
    ap.add_argument('--early', type=int, default=60)
    ap.add_argument('--seqlen', type=int, default=40)
    ap.add_argument('--out', default='results/miss_recovery')
    ap.add_argument('--nproc', type=int, default=4)
    args = ap.parse_args(argv)
    import splits
    clips = splits.clips(args.split)
    speed = mxd.read_speeds(args.targets, clips) if os.path.exists(args.targets) else {}
    cam = json.load(open(splits.SPLIT_FILE)).get('camera_speed_px_per_frame', {})
    tasks = [(c, args, {k: v for k, v in speed.items() if k[0] == c},
              'moving' if cam.get(c, 0.0) >= 0.3 else 'static') for c in clips]
    targets, cands, wins = [], [], []
    with Pool(max(1, min(args.nproc, len(tasks)))) as pool:
        for clip, (t, c, w) in pool.imap_unordered(_one, tasks):
            targets += t
            cands += c
            wins += w
            print(f'{clip:12s} targets={len(t):6d} candidates={len(c):7d} windows={len(w)}', flush=True)
    order = {c: i for i, c in enumerate(clips)}
    for rows in (targets, cands, wins):
        rows.sort(key=lambda r: (order[r['clip']], r.get('t', r.get('first', 0))))
    loro_scores(targets, cands)
    nframes = {}
    for r in targets + cands:
        nframes.setdefault(r['clip'], set()).add(r['t'])
    frames = {c: len(nframes.get(c, ())) for c in clips}
    out = os.path.join(args.out, args.run, args.split.replace('/', '_'))
    os.makedirs(out, exist_ok=True)
    write_rows(os.path.join(out, 'targets.csv'), targets)
    write_rows(os.path.join(out, 'windows.csv'), wins)
    result = {}
    for mode in ('pooled', 'clip'):
        for budget in args.budgets:
            thr = thresholds(cands, frames, budget, mode)
            for th in args.thresholds:
                key = f'{mode} budget {budget:g} / official th {th:g}'
                result[key] = summarize(targets, thr, th, args.early)
    json.dump(dict(settings=vars(args), frames=frames, candidates=len(cands), results=result),
              open(os.path.join(out, 'summary.json'), 'w'), indent=1, default=float)
    for key, groups in result.items():
        if not key.startswith('pooled'):
            continue
        print(f'\n== {key} (candidates per frame admitted; box level)')
        print(f"{'group':24s} {'targets':>8s} {'misses':>7s} {'recovered':>9s} {'frac':>6s} "
              f"{'recall':>7s} {'+rec':>7s}")
        for name, s in groups.items():
            if name.startswith('clip ') and s['misses'] == 0:
                continue
            print(f"{name:24s} {s['targets']:8d} {s['misses']:7d} {s['recovered']:9d} {s['recovered_fraction']:6.2f} "
                  f"{s['recall']:7.3f} {s['recall_with_recovery']:7.3f}")
    return result


if __name__ == '__main__':
    main()
