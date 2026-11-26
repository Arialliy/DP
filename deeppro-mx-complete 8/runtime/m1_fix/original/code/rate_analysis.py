"""Recall against target speed, apparent speed and dwell ratio (MRTP evidence).

For each scored run, each recording gets its own threshold at the false-alarm budget
(mode clip; mode pooled uses one threshold for all listed recordings), and every
annotated box counts as hit or missed. Boxes are then grouped:
  speed        relative speed v (px/frame, camera-compensated, from --speeds)
  apparent     k v for a run made at rate k (label=run@k): if the target is rate
               equivariant, the per-rate curves fall on one curve on this axis
  dwell        r = k v T / s with s = sqrt(w h) of the box (--sizes, needs VT_ROOT):
               the blind-zone law predicts the recall cliff at r ~ 2 for every clip
plus per-clip recall and its mean over clips (macro), with and without --exclude-clip.
A run with no candidates at all on a clip (a rate longer than the clip) skips that clip.

Usage
  python code/rate_analysis.py --split val \
      --score-root results/lvh_performance_s46/scores --score-root results/lvh_decoupled_s46/scores \
      --runs base=gpu_lvfull_s46 focal=gpu_lvfull_focal_s46 lvh=gpu_lvhfull_s46 dec=gpu_lvhdec_s46 \
      --speeds results/lvh_performance_s46/validation_relative_speed.csv --exclude-clip DJI_0319_2
  python code/rate_analysis.py --split cv --score-root results/scores \
      --runs base=lvfocal_f{fold}_s46__r1@1 mr=mrlv_f{fold}_s46__mrcal@1 --speeds results/profile/targets.csv
  python code/rate_analysis.py --split val --score-root results/scores --sizes \
      --runs r1=focal_mr__r1@1 r2=focal_mr__r2@2 r4=focal_mr__r4@4 cal=focal_mr__mrcal@1 \
      --speeds results/profile/targets.csv
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import compare_groups as cg  # noqa: E402
import splits  # noqa: E402

SPEED_BINS = (0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, math.inf)
DWELL_BINS = (0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, math.inf)
BANDS = ((0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, math.inf))


def find_run(roots, run, split):
    for r in roots:
        if os.path.exists(os.path.join(r, run, split, 'summary.json')):
            return r
    raise SystemExit(f'{run}/{split} not found under {roots}')


def load_spec(roots, run, split):
    """Scores of one run. split 'cv' merges the K held-out folds: the run name may hold
    '{fold}' (e.g. mrlv_f{fold}_s46__mrcal), scored with score.py --split fold<i>/val."""
    if split != 'cv':
        return cg.load_run(find_run(roots, run, split), run, split)
    k = len(json.load(open(splits.FOLD_FILE))['folds'])
    frames, gt, fp = {}, {}, {}
    for i in range(k):
        r, sp = run.format(fold=i), f'fold{i}/val'
        f_, g_, p_ = cg.load_run(find_run(roots, r, sp), r, sp)
        frames.update(f_)
        gt.update(g_)
        fp.update(p_)
    return frames, gt, fp


def split_clips(split):
    if split != 'cv':
        return splits.clips(split)
    return sorted(c for f in json.load(open(splits.FOLD_FILE))['folds'] for c in f)


def box_sizes(clips):
    import vtdata
    out = {}
    for c in clips:
        sc = vtdata.Scene(c)
        for t, b in enumerate(sc.boxes):
            for row in b:
                out[(c, t, int(row[5]))] = math.sqrt(max(row[2], 1.0) * max(row[3], 1.0))
    return out


def hits_of(run_data, clips, budget, mode):
    """-> {clip: [(t, track, hit)]}, skipped clips."""
    frames, gt, fp = run_data
    used = [c for c in clips if len(fp[c]) or np.isfinite([s for *_, s in gt[c]]).any()]
    thr = {}
    if mode == 'clip':
        for c in used:
            thr[c] = cg.threshold([s for *_, s in gt[c]], fp[c], frames[c], budget)
    else:
        t = cg.threshold([s for c in used for *_, s in gt[c]], np.concatenate([fp[c] for c in used]),
                         sum(frames[c] for c in used), budget)
        thr = {c: t for c in used}
    out = {c: [(t, tid, bool(np.isfinite(s) and s >= thr[c])) for t, tid, s in gt[c]] for c in used}
    return out, [c for c in clips if c not in used]


def binned(values, hits, bins):
    v = np.asarray(values, float)
    h = np.asarray(hits, bool)
    ok = np.isfinite(v)
    idx = np.searchsorted(np.asarray(bins), v[ok], side='right') - 1
    n = np.bincount(idx, minlength=len(bins) - 1)[:len(bins) - 1]
    k = np.bincount(idx, weights=h[ok], minlength=len(bins) - 1)[:len(bins) - 1]
    return k.astype(int), n.astype(int)


def label(lo, hi):
    return f'[{lo:g},{hi:g})' if np.isfinite(hi) else f'>={lo:g}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True, help='label=run or label=run@rate')
    ap.add_argument('--split', default='val', help="split, foldK/val, or 'cv' for all held-out folds")
    ap.add_argument('--score-root', dest='roots', action='append', default=None)
    ap.add_argument('--speeds', default='results/profile/targets.csv')
    ap.add_argument('--sizes', action='store_true', help='read box sizes from VT_ROOT for the dwell axis')
    ap.add_argument('--seqlen', type=int, default=40)
    ap.add_argument('--budgets', nargs='+', type=float, default=[0.25, 1.0])
    ap.add_argument('--mode', choices=('clip', 'pooled'), default='clip')
    ap.add_argument('--exclude-clip', dest='exclude', action='append', default=[])
    ap.add_argument('--out', default='results/rate_analysis')
    args = ap.parse_args()
    roots = args.roots or ['results/scores']
    specs = []
    for r in args.runs:
        lab, run = r.split('=', 1)
        run, _, k = run.partition('@')
        specs.append((lab, run, int(k) if k else 1))
    clips = split_clips(args.split)
    cam = json.load(open(splits.SPLIT_FILE))['camera_speed_px_per_frame']
    group = {c: 'moving' if cam.get(c, 0.0) >= cg.MOVING_CAMERA else 'static' for c in clips}
    speed = cg.read_speeds(args.speeds)
    sizes = box_sizes(clips) if args.sizes else {}
    data = {lab: load_spec(roots, run, args.split) for lab, run, _ in specs}
    rows = []
    for budget in args.budgets:
        print(f'\n===== {args.mode}-matched thresholds, {budget:g} FA/frame (recall %, hits/boxes)')
        res = {}
        for lab, run, k in specs:
            hits, skipped = hits_of(data[lab], clips, budget, args.mode)
            if skipped:
                print(f'  note: {lab} has no candidates on {", ".join(skipped)} (rate not used there); skipped')
            res[lab] = (hits, k)
        # speed-like axes, per camera group
        for gname in ('static', 'moving', 'all'):
            gclips = [c for c in clips if gname == 'all' or group[c] == gname]
            for axis, bins in (('speed', SPEED_BINS), ('apparent', SPEED_BINS), ('dwell', DWELL_BINS)):
                if axis == 'dwell' and not sizes:
                    continue
                if axis == 'apparent' and all(k == 1 for *_, k in specs):
                    continue
                table = {}
                for lab, _, _ in specs:
                    hits, k = res[lab]
                    vals, hs = [], []
                    for c in gclips:
                        for t, tid, h in hits.get(c, []):
                            v = speed.get((c, t, tid), math.nan)
                            if axis == 'apparent':
                                v = k * v
                            elif axis == 'dwell':
                                v = k * v * args.seqlen / sizes.get((c, t, tid), math.nan)
                            vals.append(v)
                            hs.append(h)
                    table[lab] = binned(vals, hs, bins)
                    for i in range(len(bins) - 1):
                        rows.append(dict(mode=args.mode, budget=budget, group=gname, axis=axis,
                                         bin=label(bins[i], bins[i + 1]), model=lab,
                                         hits=int(table[lab][0][i]), boxes=int(table[lab][1][i])))
                print(f'\n  {gname} clips ({len(gclips)}), recall by {axis}')
                print('  ' + f'{"bin":>12s}' + ''.join(f'{lab:>16s}' for lab, _, _ in specs))
                for i in range(len(bins) - 1):
                    cells = []
                    for lab, _, _ in specs:
                        h, n = table[lab][0][i], table[lab][1][i]
                        cells.append(f'{100 * h / n:5.1f}% {h:4d}/{n:<4d}' if n else '               -')
                    if any(table[lab][1][i] for lab, _, _ in specs):
                        print('  ' + f'{label(bins[i], bins[i + 1]):>12s}' + ''.join(f'{x:>16s}' for x in cells))
        # bands on static clips and per-clip macro
        static = [c for c in clips if group[c] == 'static']
        print('\n  static-camera recall in speed bands (px/frame)')
        print('  ' + f'{"band":>12s}' + ''.join(f'{lab:>16s}' for lab, _, _ in specs))
        for lo, hi in BANDS:
            cells = []
            for lab, _, _ in specs:
                hs = [h for c in static for t, tid, h in res[lab][0].get(c, [])
                      if lo <= speed.get((c, t, tid), math.nan) < hi]
                cells.append(f'{100 * np.mean(hs):5.1f}% {sum(hs):4d}/{len(hs):<4d}' if hs else '-')
                rows.append(dict(mode=args.mode, budget=budget, group='static', axis='band', bin=label(lo, hi),
                                 model=lab, hits=int(sum(hs)), boxes=len(hs)))
            print('  ' + f'{label(lo, hi):>12s}' + ''.join(f'{x:>16s}' for x in cells))
        print('\n  per-clip recall and macro mean over clips')
        print('  ' + f'{"clip":>12s} {"cam":>6s}' + ''.join(f'{lab:>10s}' for lab, _, _ in specs))
        per = defaultdict(dict)
        for c in clips:
            cells = []
            for lab, _, _ in specs:
                hs = [h for *_, h in res[lab][0].get(c, [])]
                per[lab][c] = np.mean(hs) if hs else math.nan
                cells.append(f'{100 * per[lab][c]:9.1f}%' if hs else '         -')
                rows.append(dict(mode=args.mode, budget=budget, group=c, axis='clip', bin='all', model=lab,
                                 hits=int(sum(hs)), boxes=len(hs)))
            print('  ' + f'{c:>12s} {cam.get(c, math.nan):6.2f}' + ''.join(cells))
        for name, sel in (('macro', clips), ('macro static', static),
                          ('macro moving', [c for c in clips if group[c] == 'moving'])) + \
                         ((('macro w/o ' + '+'.join(args.exclude), [c for c in clips if c not in args.exclude]),)
                          if args.exclude else ()):
            vals = [np.nanmean([per[lab][c] for c in sel]) if sel else math.nan for lab, _, _ in specs]
            print('  ' + f'{name:>19s}' + ''.join(f'{100 * v:9.1f}%' for v in vals))
            for (lab, _, _), v in zip(specs, vals):
                rows.append(dict(mode=args.mode, budget=budget, group=name, axis='macro', bin='all', model=lab,
                                 hits='', boxes='', recall=v))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.split.replace('/', '_')}_{args.mode}.csv")
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['mode', 'budget', 'group', 'axis', 'bin', 'model', 'hits', 'boxes', 'recall'])
        w.writeheader()
        for r in rows:
            r.setdefault('recall', r['hits'] / r['boxes'] if r.get('boxes') else '')
            w.writerow(r)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
