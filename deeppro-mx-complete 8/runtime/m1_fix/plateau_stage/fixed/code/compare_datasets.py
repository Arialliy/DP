"""Are two copies of the RGB-T clips the same data? (e.g. the VT-Tiny-MOT copy used for the
earlier experiments and the RGB-T folder built from the RGBT-Tiny release)

For every clip of the official lists it compares
  frames   the sorted frame names of <root>/<split>/<clip>/01
  images   the bytes of each frame (SHA-256); with --pixels also the decoded grey values
  boxes    per frame, the multiset of (x, y, w, h, category), ignoring track ids
  tracks   whether the two copies' track ids map one to one over the matched boxes (the
           RGBT-Tiny release numbers ids within each category, vtdata.track_of)
and reports each clip as
  SAME         frames, images and boxes equal, and every box's track pairing confirmed
  UNCONFIRMED  frames, images and boxes equal, no conflict, but some boxes share their
               position and category with another box of the frame, and their pairing is
               not pinned down by the clip's other boxes ("track relation cannot be confirmed")
  DIFF         anything differs, or a pairing contradicts another
A box with a position of its own pins a track pair. Boxes with a shared position are
confirmed only if each of their tracks is paired elsewhere and those partners are exactly
the other copy's tracks there.
Exit status 0 if every clip is SAME, 2 if some are UNCONFIRMED and none differ, 1 otherwise.

Usage
  python code/compare_datasets.py --a /data/VT-Tiny-MOT --b /path/to/DP/datasets/RGB-T
  python code/compare_datasets.py --a ... --b ... --b-ann /path/to/RGBT-Tiny/annotations_coco --pixels
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import vtdata  # noqa: E402


def load_boxes(root, ann_dir, split):
    """{clip: {frame name: [(x, y, w, h, cat, track)]}} of one split, tracks as vtdata.track_of."""
    dirs = [ann_dir] if ann_dir else [os.path.join(root, d) for d in ('annotations', 'annotations_coco')]
    path = next((os.path.join(d, f'instances_01_{split}.json') for d in dirs
                 if os.path.exists(os.path.join(d, f'instances_01_{split}.json'))), None)
    if path is None:
        raise FileNotFoundError(f'instances_01_{split}.json not in {dirs}')
    ann = json.load(open(path))
    imgs = {im['id']: im['file_name'] for im in ann['images']}
    out = defaultdict(lambda: defaultdict(list))
    for a in ann['annotations']:
        if a.get('ignore', 0):
            continue
        clip, _, fname = imgs[a['image_id']].split('/')
        x, y, w, h = a['bbox']
        out[clip][fname].append((round(float(x), 3), round(float(y), 3), round(float(w), 3), round(float(h), 3),
                                 int(a['category_id']), vtdata.track_of(a)))
    return out


def frame_dir(root, clip):
    for s in ('train2017', 'test2017'):
        d = os.path.join(root, s, clip, '01')
        if os.path.isdir(d):
            return s, d
    raise FileNotFoundError(f'{clip} not in {root}/train2017 or test2017')


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def compare_clip(clip, A, B, pixels=False):
    (sa, da), (sb, db) = frame_dir(A['root'], clip), frame_dir(B['root'], clip)
    fa = sorted(f for f in os.listdir(da) if f.endswith('.jpg'))
    fb = sorted(f for f in os.listdir(db) if f.endswith('.jpg'))
    r = dict(split_a=sa, split_b=sb, frames_a=len(fa), frames_b=len(fb), frame_names_equal=fa == fb,
             images_differing=0, max_pixel_diff=0.0, frames_with_box_differences=0, boxes_a=0, boxes_b=0,
             track_conflicts=0, tracks_a=0, tracks_b=0, duplicate_position_boxes=0, unconfirmed_boxes=0)
    ba, bb = A['boxes'][sa].get(clip, {}), B['boxes'][sb].get(clip, {})
    a2b, b2a = defaultdict(set), defaultdict(set)
    shared = []                          # (tracks in A, tracks in B) of positions held by several boxes
    for f in sorted(set(fa) & set(fb)):
        pa, pb = os.path.join(da, f), os.path.join(db, f)
        if sha(pa) != sha(pb):
            r['images_differing'] += 1
            if pixels:
                ia = np.asarray(Image.open(pa).convert('L'), np.float32)
                ib = np.asarray(Image.open(pb).convert('L'), np.float32)
                d = float(np.abs(ia - ib).max()) if ia.shape == ib.shape else float('inf')
                r['max_pixel_diff'] = max(r['max_pixel_diff'], d)
        xa, xb = ba.get(f, []), bb.get(f, [])
        r['boxes_a'] += len(xa)
        r['boxes_b'] += len(xb)
        if Counter(b[:5] for b in xa) != Counter(b[:5] for b in xb):
            r['frames_with_box_differences'] += 1
            continue
        # match boxes by position and category; a position held by one box pins a track pair
        ga, gb = defaultdict(list), defaultdict(list)
        for b in xa:
            ga[b[:5]].append(b[5])
        for b in xb:
            gb[b[:5]].append(b[5])
        for key, ids in ga.items():                  # the multisets are equal, so gb[key] has len(ids) ids
            if len(ids) == 1:
                a2b[ids[0]].add(gb[key][0])
                b2a[gb[key][0]].add(ids[0])
            else:
                shared.append((ids, gb[key]))
    r['tracks_a'], r['tracks_b'] = len(a2b), len(b2a)
    conflicts = sum(len(v) > 1 for v in a2b.values()) + sum(len(v) > 1 for v in b2a.values())
    for ids_a, ids_b in shared:
        r['duplicate_position_boxes'] += len(ids_a)
        if all(len(a2b.get(i, ())) == 1 for i in ids_a):
            if Counter(next(iter(a2b[i])) for i in ids_a) != Counter(ids_b):
                conflicts += 1                       # the partners known elsewhere are not these tracks
        else:
            r['unconfirmed_boxes'] += len(ids_a)
    r['track_conflicts'] = conflicts
    equal = (r['frame_names_equal'] and r['images_differing'] == 0 and r['frames_with_box_differences'] == 0
             and conflicts == 0)
    r['status'] = 'DIFF' if not equal else ('UNCONFIRMED' if r['unconfirmed_boxes'] else 'SAME')
    r['same'] = r['status'] == 'SAME'
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--a', required=True, help='first copy (VT_ROOT layout)')
    ap.add_argument('--b', required=True, help='second copy')
    ap.add_argument('--a-ann', dest='a_ann', default=None, help='annotation folder of --a (default: inside it)')
    ap.add_argument('--b-ann', dest='b_ann', default=None, help='annotation folder of --b (default: inside it)')
    ap.add_argument('--lists', default=None, help='folder with train.txt and test.txt (default: --b)')
    ap.add_argument('--pixels', action='store_true', help='also compare decoded pixels of differing files')
    ap.add_argument('--out', default=None, help='optional json report')
    args = ap.parse_args(argv)
    lists = args.lists or args.b
    clips = [c for n in ('train', 'test') for c in open(os.path.join(lists, f'{n}.txt')).read().split()]
    A = dict(root=args.a, boxes={s: load_boxes(args.a, args.a_ann, s) for s in ('train2017', 'test2017')})
    B = dict(root=args.b, boxes={s: load_boxes(args.b, args.b_ann, s) for s in ('train2017', 'test2017')})
    res = {}
    for c in clips:
        r = compare_clip(c, A, B, args.pixels)
        res[c] = r
        print(f"{r['status']:11s} {c}: frames {r['frames_a']}/{r['frames_b']}, "
              f"{r['images_differing']} images differ, {r['frames_with_box_differences']} frames with other boxes, "
              f"boxes {r['boxes_a']}/{r['boxes_b']}, tracks {r['tracks_a']}/{r['tracks_b']}, "
              f"{r['track_conflicts']} track conflicts, {r['unconfirmed_boxes']} of "
              f"{r['duplicate_position_boxes']} boxes at shared positions unconfirmed", flush=True)
    diff = sorted(c for c, r in res.items() if r['status'] == 'DIFF')
    unconfirmed = sorted(c for c, r in res.items() if r['status'] == 'UNCONFIRMED')
    keys = ('frames_a', 'frames_b', 'boxes_a', 'boxes_b', 'images_differing', 'frames_with_box_differences',
            'duplicate_position_boxes', 'unconfirmed_boxes')
    tot = {k: sum(r[k] for r in res.values()) for k in keys}
    print(f"\n{len(res) - len(diff) - len(unconfirmed)}/{len(res)} clips SAME; totals {tot}" +
          (f"; track relation cannot be confirmed: {', '.join(unconfirmed)}" if unconfirmed else '') +
          (f"; differ: {', '.join(diff)}" if diff else ''))
    if args.out:
        json.dump(dict(a=args.a, b=args.b, totals=tot, clips=res), open(args.out, 'w'), indent=1)
    return 1 if diff else (2 if unconfirmed else 0)


if __name__ == '__main__':
    sys.exit(main())
