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
    single     every frame on its own: actual local maxima of |local contrast|, top --per-frame
    frame      candidates only. Candidates within --r-pos px of a GT centre are positive;
               beyond --r-neg of every centre are clutter; others are excluded from fitting.
    features   calibrated contrast R, |R|, D, P and NCC / |NCC|, sampled at each actual
               candidate location. Ground-truth centres are never detection proposals.
    score      logistic regression fitted leave-one-recording-out on actual positive vs
               clutter candidates, balanced by clip and class. All candidates of the held-out
               clip are scored without using their labels in the fit. No teacher means no recovery.
    recovery   finite scored candidates are assigned to boxes one-to-one, in descending score
               order, nearest eligible centre first. The geometric candidate ceiling uses
               maximum matching, ignoring scores and budgets, among the misses of each group.
    budget     conservative thresholds admit at most --budgets labelled clutter candidates per
               frame (pooled or per clip). Ties are excluded when needed; actual rates are reported.
Reported per group (all; frames before / after --early; category; target speed band;
camera group; clip): targets, misses, misses recovered (score >= threshold), hits covered,
and recall official vs official-or-recovered (box level; the added false alarms are the
budget, in candidates per frame, not the pixel Fa of the authors' protocol).
Per window also the background stationarity residuals of multirate.stationarity (raw and
compensated, annotation-free homographies) and the window's miss rate, to see whether the
windows with many misses are the ones whose background changes.

The LORO fit uses the labels of the other evaluated clips: this is a diagnostic of what is
recoverable, not a test result. Thresholds also use the evaluated clips' labelled clutter. Neither fitting nor threshold
calibration is a test-free performance result.

Usage
    python code/miss_recovery.py --run official_rgbt --split test
Writes results/miss_recovery/<run>/<split>/{targets.csv, candidates.csv, windows.csv, summary.json}.
The summary includes the predeclared pooled 1/frame, baseline 1e-5 gate: early recovery
>=30% and overall >=20%, with early fixed to frames 0..59. It remains diagnostic only.
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
    """Per-box ShootingRules hit, including official first-row/column eligibility."""
    import vtdata
    out = np.zeros((len(boxes), len(ths)), bool)
    near = ndimage.maximum_filter(np.asarray(prob, np.float32), size=3,
                                  mode='constant', cval=-np.inf)
    thresholds_ = np.asarray(ths, np.float32)
    for i, b in enumerate(boxes):
        eligible = vtdata.gaussian_mask(np.asarray(b, float)[None], H, W, rule) > 0
        eligible[0, :] = False
        eligible[:, 0] = False
        if eligible.any():
            out[i] = near[eligible].max() >= thresholds_
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
    """Actual single-frame candidates, target metadata, and owned-window diagnostics.

    GT centres are used only to label/match candidates. They are never scored as
    detection locations. Ambiguous candidates are scored but excluded from fitting.
    """
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
        with np.errstate(over='ignore'):
            prob = 1.0 / (1.0 + np.exp(-np.asarray(logits[t], np.float32)))
        b = np.asarray(boxes[t], float).reshape(-1, 6)
        cen = mxd.box_centres(b) if len(b) else np.zeros((0, 2))
        hits = box_hits(prob, b, H, W, ths) if len(b) else np.zeros((0, len(ths)), bool)
        pk = mxd.frame_peaks(frame, t, args.per_frame)
        pts = pk[:, 1:3]
        distance = (np.max(np.abs(pts[:, None, :] - cen[None, :, :]), axis=2)
                    if len(cen) else np.zeros((len(pts), 0)))
        f_c = features(frame, cmap, [tuple(p) for p in pts], ce, E, S, shift)
        for i, row in enumerate(b):
            ids = np.flatnonzero(distance[:, i] <= args.r_pos).tolist()
            targets.append(dict(clip=clip, t=t, box_index=i, track=int(row[5]), cat=int(row[4]),
                                x=float(cen[i, 0]), y=float(cen[i, 1]),
                                rel_speed=speed.get((clip, t, int(row[5])), math.nan),
                                camera=cam_group, window=int(owner[t]), teacher=len(rows),
                                candidate_ids=ids, candidate_count=len(ids), score=-math.inf,
                                **{f'hit_{th:g}': bool(hits[i, j]) for j, th in enumerate(ths)}))
        for i, (x, y) in enumerate(pts):
            d = float(distance[i].min()) if len(cen) else math.inf
            matched = np.flatnonzero(distance[i] <= args.r_pos).tolist()
            label = 1 if d <= args.r_pos else 0 if d > args.r_neg else -1
            cands.append(dict(clip=clip, t=t, candidate_id=i, x=float(x), y=float(y),
                              window=int(owner[t]), teacher=len(rows), label=label,
                              target_indices=matched, target_distances=distance[i, matched].tolist(),
                              **{k: float(f_c[i, j]) for j, k in enumerate(FEATURES)}))
    for k, (a, b) in enumerate(ws):
        frames = list(range(a, b))
        stats = (multirate.stationarity(low[frames], chain, frames, 4) if chain is not None
                 else dict(raw=math.nan, warp=math.nan, cover=math.nan))
        mine = [r for r in targets if r['window'] == k]
        th0 = f'hit_{ths[0]:g}'
        wins.append(dict(clip=clip, window=k, first=a, last=b - 1, teacher=len(banks[k][0]),
                         targets=len(mine), misses=sum(not r[th0] for r in mine),
                         owned_first=min((r['t'] for r in mine), default=None),
                         owned_last=max((r['t'] for r in mine), default=None),
                         stationarity_raw=stats.get('raw', math.nan), stationarity_warp=stats.get('warp', math.nan),
                         cover=stats.get('cover', math.nan)))
    return targets, cands, wins


def _peaks(lg):
    import evaluate
    return evaluate.peaks(lg)


def loro_scores(targets, cands):
    """Fit actual positive/clutter candidates on other recordings; score all candidates.

    Candidate labels of the held-out recording never enter the fit or normalization.
    Target scores are assigned afterwards by deterministic one-to-one spatial matching.
    """
    if cands:
        y = np.array([r['label'] for r in cands], float)
        X = np.array([[r[k] for k in FEATURES] for r in cands], float)
        clips = np.array([r['clip'] for r in cands])
        groups = np.array([cd.recording(c) for c in clips])
        has = np.array([r['teacher'] > 0 for r in cands])
        score = np.full(len(cands), -np.inf)
        for c in sorted(set(clips)):
            te = (clips == c) & has
            tr = (groups != cd.recording(c)) & has & (y >= 0)
            if not te.any() or len(np.unique(y[tr])) < 2:
                continue
            w = cd.clip_weights(clips[tr], y[tr])
            mu = (w[:, None] * X[tr]).sum(0)
            sd = np.sqrt((w[:, None] * (X[tr] - mu) ** 2).sum(0)) + 1e-9
            beta = cd.weighted_logistic((X[tr] - mu) / sd, y[tr], w)
            score[te] = np.c_[(X[te] - mu) / sd, np.ones(int(te.sum()))] @ beta
        for r, value in zip(cands, score):
            r['score'] = float(value)
    assign_candidates(targets, cands)


def thresholds(cands, frames, budget, mode):
    """Conservative threshold: finite labelled clutter admissions never exceed budget.

    Boundary ties are excluded together when including them would exceed the cap.
    Candidate labels are used for diagnostic threshold calibration, not deployment.
    """
    def thr(rows, n):
        s = np.sort(np.asarray([r['score'] for r in rows
                                if r['label'] == 0 and r['teacher'] > 0
                                and np.isfinite(r['score'])], float))[::-1]
        k = int(math.floor(budget * n))
        if k <= 0 or not len(s):
            return math.inf if k <= 0 else -math.inf
        if k >= len(s):
            return -math.inf
        return float(np.nextafter(s[k], math.inf))
    if mode == 'pooled':
        t = thr(cands, sum(frames.values()))
        return {c: t for c in frames}
    return {c: thr([r for r in cands if r['clip'] == c], frames[c]) for c in frames}


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
        rec = np.array([r['teacher'] > 0 and np.isfinite(r['score'])
                        and r['score'] >= thr[r['clip']] for r in rows], bool)
        ceiling = candidate_ceiling([r for r in rows if not r[key]])
        if int((~hit & rec).sum()) > ceiling:
            raise AssertionError('Recovery exceeds the actual candidate matching ceiling')
        out[name] = dict(targets=len(rows), misses=int((~hit).sum()), recovered=int((~hit & rec).sum()),
                         candidate_ceiling=ceiling,
                         candidate_ceiling_fraction=ceiling / max(int((~hit).sum()), 1),
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
    return clip, len(sc), process_clip(clip, video, sc.boxes, logits, args, chain, cam_group, speed)


def write_rows(path, rows):
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)



def assign_candidates(targets, cands):
    """Greedy score-descending matching, one real candidate and target at most once."""
    from collections import defaultdict
    by_frame, target_map = defaultdict(list), {}
    for r in targets:
        r['score'], r['matched_candidate'] = -math.inf, None
        target_map[(r['clip'], r['t'], r['box_index'])] = r
    for r in cands:
        if r['teacher'] > 0 and np.isfinite(r.get('score', -math.inf)):
            by_frame[(r['clip'], r['t'])].append(r)
    for key, rows in by_frame.items():
        used = set()
        for r in sorted(rows, key=lambda v: (-v['score'], v['candidate_id'])):
            choices = [(distance, box) for box, distance in zip(r['target_indices'], r['target_distances'])
                       if box not in used and (*key, box) in target_map]
            if choices:
                _, box = min(choices)
                target = target_map[(*key, box)]
                if target['teacher'] > 0:
                    target['score'], target['matched_candidate'] = r['score'], r['candidate_id']
                    used.add(box)


def candidate_ceiling(targets):
    """Maximum one-to-one matches among these misses before score/FA constraints.

    This is a geometric ceiling, independently evaluated within each reported group.
    A teacher is required. Augmenting paths prevent greedy matching lowering the ceiling.
    """
    from collections import defaultdict
    frames = defaultdict(list)
    for r in targets:
        if r['teacher'] > 0:
            frames[(r['clip'], r['t'])].append(r.get('candidate_ids', []))
    total = 0
    for edges in frames.values():
        matched = {}
        def visit(target, seen):
            for candidate in edges[target]:
                if candidate in seen:
                    continue
                seen.add(candidate)
                if candidate not in matched or visit(matched[candidate], seen):
                    matched[candidate] = target
                    return True
            return False
        total += sum(visit(i, set()) for i in range(len(edges)))
    return total


def clutter_admissions(cands, frames, thr, budget, mode):
    admitted = {c: sum(r['clip'] == c and r['label'] == 0 and r['teacher'] > 0
                        and np.isfinite(r['score']) and r['score'] >= thr[c] for r in cands)
                for c in frames}
    if mode == 'pooled':
        ok = sum(admitted.values()) <= math.floor(budget * sum(frames.values()))
    else:
        ok = all(admitted[c] <= math.floor(budget * frames[c]) for c in frames)
    if not ok:
        raise AssertionError('Realized clutter exceeds the diagnostic budget')
    return dict(admitted=sum(admitted.values()), frames=sum(frames.values()),
                per_frame=sum(admitted.values()) / max(sum(frames.values()), 1),
                per_clip=admitted, thresholds=thr, within_budget=ok)


def decision(results, observed, early=60):
    """The predeclared gate, fixed before reading data: pooled 1/frame, p=1e-5."""
    key, budget_key = 'pooled budget 1 / official th 1e-05', 'pooled budget 1'
    groups = results.get(key, {})
    all_, start = groups.get('all', {}), groups.get('frames < 60', {})
    measured = observed.get(budget_key, {})
    available = early == 60 and bool(all_) and bool(start) and bool(measured)
    enough = available and all_.get('misses', 0) > 0 and start.get('misses', 0) > 0
    all_fraction, early_fraction = all_.get('recovered_fraction'), start.get('recovered_fraction')
    budget_ok = bool(measured.get('within_budget', False)) and measured.get('per_frame', math.inf) <= 1.0
    go = bool(enough and budget_ok and early_fraction >= 0.30 and all_fraction >= 0.20)
    return dict(go=go, diagnostic_only=True, applicable=bool(enough), mode='pooled', budget=1.0,
                baseline_threshold=1e-5, early_frames=60,
                early_minimum=0.30, overall_minimum=0.20,
                early_recovered_fraction=early_fraction, overall_recovered_fraction=all_fraction,
                realized_clutter_per_frame=measured.get('per_frame'), budget_satisfied=budget_ok,
                existing_hits_retained='Logical OR with baseline; appearance hits_covered is reported separately',
                reason=('Thresholds met' if go else 'Required gate outputs/misses absent' if not enough
                        else 'Realized clutter exceeds budget' if not budget_ok else 'Recovery thresholds not met'))


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
    targets, cands, wins, frames = [], [], [], {}
    with Pool(max(1, min(args.nproc, len(tasks)))) as pool:
        for clip, nframes, (t, c, w) in pool.imap_unordered(_one, tasks):
            frames[clip] = nframes
            targets += t
            cands += c
            wins += w
            print(f'{clip:12s} targets={len(t):6d} candidates={len(c):7d} windows={len(w)}', flush=True)
    order = {c: i for i, c in enumerate(clips)}
    for rows in (targets, cands, wins):
        rows.sort(key=lambda r: (order[r['clip']], r.get('t', r.get('first', 0))))
    loro_scores(targets, cands)
    out = os.path.join(args.out, args.run, args.split.replace('/', '_'))
    os.makedirs(out, exist_ok=True)
    write_rows(os.path.join(out, 'targets.csv'), targets)
    write_rows(os.path.join(out, 'windows.csv'), wins)
    write_rows(os.path.join(out, 'candidates.csv'), cands)
    result, budgets_observed = {}, {}
    for mode in ('pooled', 'clip'):
        for budget in args.budgets:
            thr = thresholds(cands, frames, budget, mode)
            budgets_observed[f'{mode} budget {budget:g}'] = clutter_admissions(cands, frames, thr, budget, mode)
            for th in args.thresholds:
                key = f'{mode} budget {budget:g} / official th {th:g}'
                result[key] = summarize(targets, thr, th, args.early)
    gate = decision(result, budgets_observed, args.early)
    json.dump(dict(settings=vars(args), frames=frames, candidates=len(cands), results=result,
                   candidate_labels={'positive': 1, 'clutter': 0, 'ambiguous': -1},
                   measured_clutter=budgets_observed, decision=gate,
                   metric='Per-box official mask hits; recovery from actual contrast peaks with one-to-one matching',
                   fitting='LORO actual positive/clutter candidates; threshold uses evaluated labelled clutter; diagnostic only'),
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
    print('M1 decision:', 'GO' if gate['go'] else 'NO-GO', gate, flush=True)
    return result


if __name__ == '__main__':
    main()
