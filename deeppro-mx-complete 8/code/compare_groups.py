"""Compare scored runs by camera motion and by near-stationary history category.

Pooled thresholds mix two effects: how well a model detects inside each recording, and
how consistent its scores are across recordings. With seed 46 on validation, a pooled
1 FA/frame budget favoured lvh_full by 17 points, while recording-matched budgets showed
+11.7 points on moving-camera clips and -4.7 points on static-camera clips. This script
reports both, per group:

  camera group   moving if the clip's mean camera speed (configs/splits.json) >= 0.3 px/frame,
                 the threshold of the split rule; static otherwise
  speed group    near-stationary (< 0.2 px/frame relative speed) or not
  history group  for near-stationary boxes, from the whole clip's annotations:
                   static_whole_clip  the track is annotated from the first to the last frame
                                      and never moves (>= 0.5 px/frame): History holds it
                   other              it arrives, leaves or moves somewhere in the clip

Threshold modes
  clip    each recording gets its own threshold at the budget (detection inside a recording)
  pooled  one threshold over all listed recordings (adds cross-recording score consistency)

Usage
  python code/compare_groups.py --runs baseline=gpu_lvfull_s46 lvh=gpu_lvhfull_s46 \
      --score-root results/lvh_performance_s46/scores \
      --speeds results/lvh_performance_s46/validation_relative_speed.csv
Writes results/group_comparison/<split>.csv and prints a table.
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
import evaluate as ev  # noqa: E402
import splits  # noqa: E402

NEAR, MOVING_TARGET, MOVING_CAMERA = 0.2, 0.5, 0.3


def load_run(root, run, split):
    d = os.path.join(root, run, split)
    summary = json.load(open(os.path.join(d, 'summary.json')))
    frames = {c: int(v['frames']) for c, v in summary['per_clip'].items()}
    gt = defaultdict(list)
    with open(os.path.join(d, 'gt.csv'), newline='') as f:
        for r in csv.DictReader(f):
            gt[r['clip']].append((int(r['t']), int(r['track']), float(r['score'])))
    with np.load(os.path.join(d, 'fp.npz')) as z:
        fp = {c: z[c].astype(np.float64) for c in z.files}
    return frames, gt, fp


def read_speeds(path):
    out = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            v = r.get('rel_speed', '')
            out[(r['scene'], int(r['t']), int(r['track']))] = float(v) if v not in ('', 'nan') else math.nan
    return out


def categories(gt, frames, speed):
    """(clip, t, track) -> 'nonstationary' | 'unknown' | 'near_static_whole_clip' | 'near_other'."""
    cat = {}
    for clip, rows in gt.items():
        tracks = defaultdict(list)
        for t, tid, _ in rows:
            tracks[tid].append(t)
        n = frames[clip]
        info = {}
        for tid, ts in tracks.items():
            moving = [t for t in ts if speed.get((clip, t, tid), math.nan) >= MOVING_TARGET]
            info[tid] = (min(ts), max(ts), bool(moving))
        for t, tid, _ in rows:
            v = speed.get((clip, t, tid), math.nan)
            if not np.isfinite(v):
                cat[(clip, t, tid)] = 'unknown'
            elif v >= NEAR:
                cat[(clip, t, tid)] = 'nonstationary'
            else:
                first, last, moved = info[tid]
                whole = first == 0 and last == n - 1 and not moved
                cat[(clip, t, tid)] = 'near_static_whole_clip' if whole else 'near_other'
    return cat


def threshold(scores, fps, n_frames, budget):
    return ev.pd_at(np.asarray(scores, float), np.asarray(fps, float), n_frames, budget)[1]


def evaluate_run(frames, gt, fp, cat, cam_group, budget, mode, clips):
    """-> {(camera_group, target_group): [hits, n]} plus per-clip overall rows."""
    thr = {}
    if mode == 'clip':
        for c in clips:
            thr[c] = threshold([s for *_, s in gt[c]], fp[c], frames[c], budget)
    else:
        allt = threshold([s for c in clips for *_, s in gt[c]], np.concatenate([fp[c] for c in clips]),
                         sum(frames[c] for c in clips), budget)
        thr = {c: allt for c in clips}
    counts = defaultdict(lambda: [0, 0])
    fa = defaultdict(lambda: [0, 0])
    for c in clips:
        g = cam_group[c]
        nfa = int((fp[c] >= thr[c]).sum()) if np.isfinite(thr[c]) else 0
        for grp in (g, 'all'):
            fa[grp][0] += nfa
            fa[grp][1] += frames[c]
        for t, tid, s in gt[c]:
            hit = int(np.isfinite(s) and s >= thr[c])
            k = cat[(c, t, tid)]
            tgroups = ['overall', k] + (['near_stationary'] if k.startswith('near_') else [])
            for grp in (g, 'all', c):
                for tg in tgroups:
                    counts[(grp, tg)][0] += hit
                    counts[(grp, tg)][1] += 1
    return counts, fa


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True, help='label=run_name')
    ap.add_argument('--split', default='val')
    ap.add_argument('--score-root', dest='score_root', default='results/scores')
    ap.add_argument('--speeds', default='results/profile/targets.csv',
                    help='CSV with scene, t, track, rel_speed')
    ap.add_argument('--budgets', nargs='+', type=float, default=[0.25, 1.0])
    ap.add_argument('--exclude-clip', dest='exclude', action='append', default=[])
    ap.add_argument('--out', default='results/group_comparison')
    args = ap.parse_args()
    cam = json.load(open(splits.SPLIT_FILE))['camera_speed_px_per_frame']
    speed = read_speeds(args.speeds)
    runs = [r.split('=', 1) for r in args.runs]
    loaded = {label: load_run(args.score_root, run, args.split) for label, run in runs}
    frames0, gt0, _ = next(iter(loaded.values()))
    clips = [c for c in splits.clips(args.split) if c not in args.exclude]
    for label, (frames, gt, _) in loaded.items():
        if set(clips) - set(frames):
            raise SystemExit(f'{label}: missing clips {sorted(set(clips) - set(frames))}')
        if {c: len(gt[c]) for c in clips} != {c: len(gt0[c]) for c in clips}:
            raise SystemExit(f'{label}: annotation counts differ from the first run')
    cat = categories({c: gt0[c] for c in clips}, frames0, speed)
    cam_group = {c: 'moving' if cam[c] >= MOVING_CAMERA else 'static' for c in clips}
    rows = []
    tgroups = ('overall', 'nonstationary', 'near_stationary', 'near_other', 'near_static_whole_clip')
    for mode in ('clip', 'pooled'):
        for budget in args.budgets:
            res = {label: evaluate_run(*loaded[label], cat, cam_group, budget, mode, clips)
                   for label, _ in runs}
            print(f'\n== thresholds per {mode}, {budget:g} FA/frame  (hits / boxes)')
            print(f"{'group':38s}" + ''.join(f'{label:>18s}' for label, _ in runs))
            for cg in ('moving', 'static', 'all'):
                n_frames = res[runs[0][0]][1][cg][1]
                for tg in tgroups:
                    cells = []
                    for label, _ in runs:
                        h, n = res[label][0][(cg, tg)]
                        f, nf = res[label][1][cg]
                        rows.append(dict(mode=mode, budget=budget, camera_group=cg, target_group=tg,
                                         model=label, hits=h, boxes=n,
                                         pd=h / n if n else math.nan,
                                         false_alarms=f, frames=nf))
                        cells.append(f'{h:6d}/{n:<5d}' + (f'{100 * h / n:5.1f}%' if n else '     -'))
                    if res[runs[0][0]][0][(cg, tg)][1]:
                        print(f'{cg + " " + tg:38s}' + ''.join(f'{c:>18s}' for c in cells))
                print(f"{cg + ' FA/frame':38s}" + ''.join(
                    f"{res[label][1][cg][0] / max(n_frames, 1):18.3f}" for label, _ in runs))
            if mode == 'clip':
                for c in clips:
                    for label, _ in runs:
                        h, n = res[label][0][(c, 'overall')]
                        hn, nn = res[label][0][(c, 'near_stationary')]
                        rows.append(dict(mode=mode, budget=budget, camera_group=c, target_group='overall',
                                         model=label, hits=h, boxes=n, pd=h / n if n else math.nan,
                                         false_alarms='', frames=frames0[c]))
                        rows.append(dict(mode=mode, budget=budget, camera_group=c, target_group='near_stationary',
                                         model=label, hits=hn, boxes=nn, pd=hn / nn if nn else math.nan,
                                         false_alarms='', frames=frames0[c]))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.split.replace('/', '_')}.csv")
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
