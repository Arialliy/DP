"""Training-free test of the Level-Variation-History hypothesis.

Question 1 (can History help at all?)
    For every near-stationary annotation (camera-compensated relative speed < 0.2 px/frame)
    classify its track by what the out-of-window frames can see:
      static_whole_clip : present from the first to the last frame and never moving
                          (rel_speed >= MOVING everywhere absent) -> History cannot separate it
      arrives / leaves / moves : the track is absent or moving somewhere in the clip
    and record how far (in frames) the annotation is from the track's first and last frame.

Question 2 (does L - B separate targets from persistent bright spots?)
    Per inference window, in camera-compensated reference coordinates:
      L  = median of the 40 window frames            (DeepPro-LV level, stabilized)
      B  = median of frames outside the window        (History)
      D  = L - B
    Local contrast (3x3 centre minus ring r = 4..7) of L and of D is sampled at
      * every annotation assigned to the window (positives), and
      * local maxima of the L contrast that lie away from all annotations of the window
        (negatives: persistent bright spots, the kind the level branch reports).
    Separation is the ROC AUC of contrast between resident near-stationary positives (annotated
    in at least half of the window) and negatives, computed per clip where History is available.

No network is involved, so this decides whether the History input is worth training.

Usage (from the repository root, VT_ROOT set, profile_motion.py already run):
    python code/diagnose_history.py --split val test
    python code/diagnose_history.py --clips DJI_0293_1,DJI_0319_2 --hist-len 120
Outputs results/history_diag/{targets.csv, negatives.csv, summary.json} and a table.
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
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(__file__))
import splits  # noqa: E402
import stabilize as st  # noqa: E402
import vtdata  # noqa: E402

NEAR = 0.2          # near-stationary threshold (README / diagnose_occupancy)
MOVING = 0.5        # a track counts as moving at frames with rel_speed >= MOVING
NEG_SEP = 6.0       # negatives keep this Chebyshev distance (px) from every annotation
NEG_FLOOR = 3.0     # minimum L contrast (grey levels) for a negative peak
NEG_PER_WINDOW = 50


def read_motion(path, clips):
    keep = set(clips)
    speed = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            if row['scene'] in keep:
                v = row['rel_speed']
                speed[(row['scene'], int(row['t']), int(row['track']))] = float(v) if v not in ('', 'nan') else math.nan
    return speed


def track_table(scene, speed):
    """track -> dict(first, last, moving_frames sorted array)."""
    frames = defaultdict(list)
    for t, b in enumerate(scene.boxes):
        for row in b:
            frames[int(row[5])].append(t)
    out = {}
    for tid, ts in frames.items():
        mv = [t for t in ts if speed.get((scene.name, t, tid), math.nan) >= MOVING]
        out[tid] = dict(first=min(ts), last=max(ts), moving=np.array(sorted(mv), int))
    return out


def history_category(info, t, n):
    moved_before = bool(len(info['moving']) and info['moving'][0] < t)
    moved_after = bool(len(info['moving']) and info['moving'][-1] > t)
    arrives, leaves = info['first'] > 0, info['last'] < n - 1
    if not (moved_before or moved_after or arrives or leaves):
        return 'static_whole_clip'
    if arrives or moved_before:
        return 'arrives_or_moved_before'
    return 'leaves_or_moves_after'


def sample_max(c, x, y, r=1):
    h, w = c.shape
    xi, yi = int(round(x)), int(round(y))
    x0, x1, y0, y1 = max(xi - r, 0), min(xi + r + 1, w), max(yi - r, 0), min(yi + r + 1, h)
    if x0 >= x1 or y0 >= y1:
        return math.nan
    patch = c[y0:y1, x0:x1]
    return float(np.nanmax(patch)) if np.isfinite(patch).any() else math.nan


def closest_windows(n, ws):
    owner, best = np.full(n, -1), np.full(n, np.inf)
    for k, (a, b) in enumerate(ws):
        d = np.abs(np.arange(a, b) - (a + b - 1) / 2)
        use = d < best[a:b]
        owner[a:b][use] = k
        best[a:b][use] = d[use]
    return owner


def process(task):
    clip, args, speed = task
    scene = vtdata.Scene(clip)
    n = len(scene)
    video = st.load_video_u8(clip, scene)
    chain = st.load_chain(clip, n)
    tracks = track_table(scene, speed)
    ws = st.windows(n, args.seqlen)
    owner = closest_windows(n, ws)
    pos_rows, neg_rows = [], []
    full_chain = chain
    for k, (a, b) in enumerate(ws):
        ref = st.reference_frame(a, b)
        frames_all = list(range(a, b)) + st.history_frames(n, a, b, args.hist_len, args.hist_step, args.mode)
        chain, _ = st.gate_chain(full_chain, frames_all, ref, getattr(args, 'motion_gate', 0.0),
                                 video.shape[1:3])
        stack, valid = st.warp_frames(video, chain, range(a, b), ref)
        L, count = st.masked_median(stack, valid)
        L = np.where(count > 0, L, np.nan)
        B, avail = st.history_field(video, chain, a, b, ref, args.hist_len, args.hist_step,
                                    args.mode, min_count=args.min_hist)
        cL = st.local_contrast(L)
        cD = st.local_contrast(np.where(avail, L - B, np.nan))
        # annotations of the whole window in reference coordinates (for negative exclusion)
        allc = [st.map_boxes(scene.boxes[t], st.ref_to(chain, t, ref)) for t in range(a, b)]
        allc = np.concatenate([q[:, :2] + q[:, 2:4] / 2 - 0.5 for q in allc if len(q)] or [np.zeros((0, 2))])
        present = defaultdict(int)
        for t in range(a, b):
            for row in scene.boxes[t]:
                present[int(row[5])] += 1
        for t in np.flatnonzero(owner == k):
            m = st.map_boxes(scene.boxes[t], st.ref_to(chain, t, ref))
            for row, mrow in zip(scene.boxes[t], m):
                tid = int(row[5])
                x, y = mrow[0] + mrow[2] / 2 - 0.5, mrow[1] + mrow[3] / 2 - 0.5
                v = speed.get((clip, int(t), tid), math.nan)
                info = tracks[tid]
                pos_rows.append(dict(
                    clip=clip, t=int(t), track=tid, cat=int(row[4]), rel_speed=v,
                    near=int(np.isfinite(v) and v < NEAR), window_start=a, window_end=b,
                    contrast_L=sample_max(cL, x, y), contrast_D=sample_max(cD, x, y),
                    history_available=int(avail[min(max(int(round(y)), 0), avail.shape[0] - 1),
                                                min(max(int(round(x)), 0), avail.shape[1] - 1)]),
                    category=history_category(info, int(t), n),
                    window_presence=present[tid] / (b - a),
                    frames_since_first=int(t) - info['first'], frames_until_last=info['last'] - int(t)))
        # negatives: persistent bright spots away from every annotation of the window
        c = np.nan_to_num(cL, nan=-np.inf)
        peak = (c == ndimage.maximum_filter(c, size=9, mode='nearest')) & (c > NEG_FLOOR)
        ys, xs = np.nonzero(peak)
        if len(xs) and len(allc):
            d = np.max(np.abs(np.stack([xs, ys], 1)[:, None, :] - allc[None]), axis=2).min(1)
            keep = d > NEG_SEP
            xs, ys = xs[keep], ys[keep]
        order = np.argsort(-c[ys, xs])[:NEG_PER_WINDOW]
        for i in order:
            neg_rows.append(dict(clip=clip, window_start=a, window_end=b, x=int(xs[i]), y=int(ys[i]),
                                 contrast_L=float(cL[ys[i], xs[i]]), contrast_D=float(cD[ys[i], xs[i]]),
                                 history_available=int(avail[ys[i], xs[i]])))
    return clip, n, pos_rows, neg_rows


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie) (Mann-Whitney). NaN when a side is empty."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return math.nan
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def false_spots_at_half_recall(pos, neg, windows):
    """Negatives per window above the threshold that keeps half of the positives."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not windows:
        return math.nan
    thr = np.median(pos)
    return float((neg >= thr).sum() / windows)


def summarize(clip_rows, neg_rows, n_windows):
    near = [r for r in clip_rows if r['near']]
    cats = defaultdict(int)
    for r in near:
        cats[r['category']] += 1
    # Resident: annotated in at least half of the window, so the stabilized median can hold it.
    # Shorter dwell belongs to the variation pathway and is not what L or L - B should find.
    nh = [r for r in near if r['history_available'] and r['window_presence'] >= 0.5]
    negh = [r for r in neg_rows if r['history_available']]
    return dict(
        annotations=len(clip_rows), near_stationary=len(near),
        near_categories=dict(cats),
        near_static_whole_clip_fraction=(cats['static_whole_clip'] / len(near)) if near else math.nan,
        near_resident=sum(r['window_presence'] >= 0.5 for r in near),
        near_resident_with_history=len(nh), negatives=len(neg_rows), negatives_with_history=len(negh),
        windows=n_windows,
        auc_L=auc([r['contrast_L'] for r in nh], [r['contrast_L'] for r in negh]),
        auc_D=auc([r['contrast_D'] for r in nh], [r['contrast_D'] for r in negh]),
        spots_per_window_at_half_recall_L=false_spots_at_half_recall(
            [r['contrast_L'] for r in nh], [r['contrast_L'] for r in negh], n_windows),
        spots_per_window_at_half_recall_D=false_spots_at_half_recall(
            [r['contrast_D'] for r in nh], [r['contrast_D'] for r in negh], n_windows),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', nargs='*', default=['val'])
    ap.add_argument('--clips', default=None, help='comma-separated clip names (overrides --split)')
    ap.add_argument('--targets', default='results/profile/targets.csv')
    ap.add_argument('--seqlen', type=int, default=40)
    ap.add_argument('--hist-len', dest='hist_len', type=int, default=80)
    ap.add_argument('--hist-step', dest='hist_step', type=int, default=4)
    ap.add_argument('--mode', choices=('both', 'past'), default='both')
    ap.add_argument('--min-hist', dest='min_hist', type=int, default=3)
    ap.add_argument('--motion-gate', dest='motion_gate', type=float, default=0.0,
                    help='px; windows moving less use recorded frames (0 = always warp)')
    ap.add_argument('--out', default='results/history_diag')
    ap.add_argument('--nproc', type=int, default=4)
    args = ap.parse_args()
    clips = args.clips.split(',') if args.clips else [c for s in args.split for c in splits.clips(s)]
    speed = read_motion(args.targets, clips)
    os.makedirs(args.out, exist_ok=True)
    tasks = [(c, args, {k: v for k, v in speed.items() if k[0] == c}) for c in clips]
    per_clip, all_pos, all_neg = {}, [], []
    with Pool(max(1, min(args.nproc, len(tasks)))) as pool:
        for clip, n, pos, neg in pool.imap_unordered(process, tasks):
            nwin = len(st.windows(n, args.seqlen))
            per_clip[clip] = summarize(pos, neg, nwin)
            all_pos += pos
            all_neg += neg
            s = per_clip[clip]
            print(f"{clip:12s} near={s['near_stationary']:5d} static_whole={s['near_static_whole_clip_fraction']:.2f} "
                  f"AUC L={s['auc_L']:.3f} D={s['auc_D']:.3f}", flush=True)
    for name, rows in (('targets.csv', all_pos), ('negatives.csv', all_neg)):
        if rows:
            with open(os.path.join(args.out, name), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    clips_with_near = [c for c, s in per_clip.items()
                       if s['near_resident_with_history'] and s['negatives_with_history']]
    macro = {k: float(np.nanmean([per_clip[c][k] for c in clips_with_near])) if clips_with_near else math.nan
             for k in ('auc_L', 'auc_D')}
    summary = dict(settings=dict(vars(args)), per_clip=per_clip,
                   macro_over_clips_with_near_stationary=dict(clips=sorted(clips_with_near), **macro),
                   note='AUCs compare near-stationary annotations with persistent bright spots, '
                        'both restricted to pixels where History is available. Pooled numbers are '
                        'not reported because one clip dominates the near-stationary count.')
    json.dump(summary, open(os.path.join(args.out, 'summary.json'), 'w'), indent=1, default=float)
    print('\nclip          near  resid+hist  static_whole  AUC_L  AUC_D  spots/win@50%: L     D')
    for c in sorted(per_clip):
        s = per_clip[c]
        print(f"{c:12s} {s['near_stationary']:5d} {s['near_resident_with_history']:11d} "
              f"{s['near_static_whole_clip_fraction']:12.2f} {s['auc_L']:6.3f} {s['auc_D']:6.3f} "
              f"{s['spots_per_window_at_half_recall_L']:16.2f} {s['spots_per_window_at_half_recall_D']:6.2f}")
    print(f"macro AUC over {len(clips_with_near)} clips with near-stationary targets: "
          f"L={macro['auc_L']:.3f} D={macro['auc_D']:.3f}")


if __name__ == '__main__':
    main()
