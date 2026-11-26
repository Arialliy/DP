"""Grouped K-fold cross-validation over the non-test clips (train + val of splits.json).

Why: the validation split has 8 clips, and one of them (DJI_0319_2) holds 80 % of the
near-stationary boxes, so a single split cannot support a claim about slow targets.

Rules (fixed before any multi-rate model is trained)
  1. Groups are drone recordings (DJI_XXXX). Recordings that the visual scene review found
     to show the same site are one group: (DJI_0043, DJI_0057) and (DJI_0113, DJI_0115).
  2. Moving-camera groups (any clip >= 0.3 px/frame) are dealt first, largest first, to the
     fold with the fewest moving groups (ties: fewest boxes).
  3. Static-camera groups are dealt by near-stationary box count (rel_speed < 0.2,
     results/profile/targets.csv), largest first, to the fold with the fewest such boxes
     (ties: fewest groups). Without targets.csv every group weighs the same.
  4. Ties in ordering are broken by the SHA-256 of the group name.
Writes configs/folds.json. splits.clips('fold2/val') is fold 2 and splits.clips('fold2/train')
the other non-test clips, for train.py --split, infer_mr.py --clips and score.py --split.

Usage: python code/folds.py [--k 5] [--targets results/profile/targets.csv]
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
import splits  # noqa: E402

MERGE = (('DJI_0043', 'DJI_0057'), ('DJI_0113', 'DJI_0115'))
MOVING = 0.3
NEAR = 0.2


def recording_id(clip):
    return '_'.join(clip.split('_')[:2])


def groups_of(clips, merge=MERGE):
    owner = {}
    for pair in merge:
        for r in pair:
            owner[r] = '+'.join(pair)
    g = defaultdict(list)
    for c in clips:
        r = recording_id(c)
        g[owner.get(r, r)].append(c)
    return {k: sorted(v) for k, v in g.items()}


def build_folds(clips, cam, near, boxes, k=5, merge=MERGE):
    """clips: list; cam: clip -> camera speed; near/boxes: clip -> counts. -> list of k clip lists."""
    groups = groups_of(clips, merge)
    h = lambda name: hashlib.sha256(name.encode()).hexdigest()
    moving = [g for g, cs in groups.items() if max(cam[c] for c in cs) >= MOVING]
    static = [g for g in groups if g not in moving]
    folds = [dict(groups=[], moving=0, near=0, boxes=0) for _ in range(k)]

    def add(f, g):
        f['groups'].append(g)
        f['near'] += sum(near.get(c, 0) for c in groups[g])
        f['boxes'] += sum(boxes.get(c, 0) for c in groups[g])

    for g in sorted(moving, key=lambda g: (-sum(boxes.get(c, 0) for c in groups[g]), h(g))):
        f = min(folds, key=lambda f: (f['moving'], f['boxes'], folds.index(f)))
        add(f, g)
        f['moving'] += 1
    for g in sorted(static, key=lambda g: (-sum(near.get(c, 0) for c in groups[g]), h(g))):
        f = min(folds, key=lambda f: (f['near'], len(f['groups']), folds.index(f)))
        add(f, g)
    return [sorted(c for g in f['groups'] for c in groups[g]) for f in folds], folds


def read_counts(path):
    near, boxes = defaultdict(int), defaultdict(int)
    if not os.path.exists(path):
        return None, None
    for r in csv.DictReader(open(path)):
        boxes[r['scene']] += 1
        v = r.get('rel_speed', '')
        if v not in ('', 'nan') and float(v) < NEAR:
            near[r['scene']] += 1
    return near, boxes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--targets', default='results/profile/targets.csv')
    args = ap.parse_args()
    s = json.load(open(splits.SPLIT_FILE))
    clips = sorted(s['train'] + s['val'])
    cam = s['camera_speed_px_per_frame']
    near, boxes = read_counts(args.targets)
    if near is None:
        print(f'{args.targets} not found: static groups are balanced by count only', flush=True)
        near, boxes = {}, {}
    fold_clips, stats = build_folds(clips, cam, near, boxes, args.k)
    out = dict(rule=__doc__.strip(), k=args.k, folds=fold_clips,
               stats=[dict(clips=len(fc), groups=f['groups'], moving_groups=f['moving'],
                           near_stationary_boxes=f['near'], boxes=f['boxes'])
                      for fc, f in zip(fold_clips, stats)],
               counts_from=args.targets if near else None)
    json.dump(out, open(splits.FOLD_FILE, 'w'), indent=1)
    for i, (fc, f) in enumerate(zip(fold_clips, stats)):
        print(f'fold{i}: {len(fc):2d} clips, {f["moving"]} moving groups, {f["near"]:5d} near-stationary / '
              f'{f["boxes"]:6d} boxes: {" ".join(fc)}')
    print(f'wrote {splits.FOLD_FILE}')


if __name__ == '__main__':
    main()
