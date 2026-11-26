"""Write RGBT-Tiny Gaussian masks in the layout the official DeepPro code reads ('RGB-T').

The official loaders expect, under one data root:
    train2017/<clip>/01/*.jpg     thermal frames (from the RGBT-Tiny release)
    test2017/<clip>/01/*.jpg
    segmentations/<clip>/*.png    one mask per frame; images and masks are each sorted
                                  by name and paired by position
    train.txt, test.txt           the 45 + 15 clips of the paper
Each mask is gaussian_masks.store(gaussian_masks.heatmap(boxes, H, W, k), storage): the
Gaussian itself in 8 bits, not a binary image. The loaders binarise it (> 0) themselves.
segmentations/MASKS.json records the parameters, the paper sentence they implement, and
per clip the boxes drawn, the boxes skipped (ignored, empty) and the boxes too small to
leave a pixel.

Usage
  python code/export_masks.py --k 6 --storage round           # 60 official clips into $VT_ROOT
  python code/export_masks.py --k 6 --clips DJI_0049_2 --out /tmp/masks
k, storage and --ignored are not given by the paper; code/check_mask_protocol.py
measures which values reproduce the authors' released per-clip target counts.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import gaussian_masks as gm  # noqa: E402
import vtdata  # noqa: E402

PAPER = ('DeepPro (arXiv:2506.12766), Sec. 4.6: "For training, we generated rough Gaussian masks as labels '
         'according to the bounding boxes provided by its authors."')


def annotated_frames(scene, ignored=False):
    """-> list over frames of (boxes (n, 6) array, n_ignored, n_empty) for one vtdata.Scene."""
    per = vtdata._annotations(scene.split)[scene.name]
    out = []
    for f in scene.frames:
        rows, n_ign, n_empty = [], 0, 0
        for a in per.get(f, []):
            if a.get('ignore', 0) and not ignored:
                n_ign += 1
                continue
            x, y, w, h = a['bbox']
            if not (w > 0 and h > 0):
                n_empty += 1
                continue
            rows.append((x, y, w, h, a['category_id'], a['track_id']))
        out.append((np.array(rows, dtype=np.float64).reshape(-1, 6), n_ign, n_empty))
    return out


def official_clips():
    return vtdata.scene_list('train') + vtdata.scene_list('test')


def export_clip(task):
    clip, out_root, k, storage, ignored = task
    sc = vtdata.Scene(clip)
    d = os.path.join(out_root, clip)
    os.makedirs(d, exist_ok=True)
    stats = dict(frames=len(sc), boxes=0, ignored_skipped=0, empty_skipped=0, vanished=0)
    for fname, (boxes, n_ign, n_empty) in zip(sc.frames, annotated_frames(sc, ignored)):
        img = gm.store(gm.heatmap(boxes, sc.H, sc.W, k), storage)
        Image.fromarray(img, mode='L').save(os.path.join(d, os.path.splitext(fname)[0] + '.png'))
        stats['boxes'] += len(boxes)
        stats['ignored_skipped'] += n_ign
        stats['empty_skipped'] += n_empty
        stats['vanished'] += gm.vanished(boxes, sc.H, sc.W, k, storage)
    return clip, stats


def git_version():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=os.path.dirname(__file__),
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=vtdata.DATA_ROOT, help='RGBT-Tiny root holding train2017/ and test2017/')
    ap.add_argument('--out', default=None, help='mask folder (default <root>/segmentations)')
    ap.add_argument('--clips', default='official', help="'official' (45 + 15 clips of the paper) or a comma list")
    ap.add_argument('--k', type=float, required=True, help='sigma = box size / k (not given by the paper)')
    ap.add_argument('--storage', choices=('round', 'floor'), default='round')
    ap.add_argument('--ignored', action='store_true', help="also draw boxes flagged 'ignore'")
    ap.add_argument('--no-lists', dest='lists', action='store_false',
                    help='do not copy train.txt / test.txt into the root')
    ap.add_argument('--nproc', type=int, default=8)
    args = ap.parse_args()
    vtdata.DATA_ROOT = args.root
    out = args.out or os.path.join(args.root, 'segmentations')
    clips = official_clips() if args.clips == 'official' else args.clips.split(',')
    os.makedirs(out, exist_ok=True)
    if args.lists:
        for name in ('train', 'test'):
            dst = os.path.join(args.root, f'{name}.txt')
            if not os.path.exists(dst):
                shutil.copyfile(os.path.join(vtdata.OFFICIAL_LISTS, f'{name}.txt'), dst)
    tasks = [(c, out, args.k, args.storage, args.ignored) for c in clips]
    per_clip = {}
    with Pool(max(1, min(args.nproc, len(tasks)))) as pool:
        for clip, s in pool.imap_unordered(export_clip, tasks):
            per_clip[clip] = s
            print(f"{clip}: {s['frames']} masks, {s['boxes']} boxes, {s['vanished']} too small for 8 bits, "
                  f"{s['ignored_skipped']} ignored and {s['empty_skipped']} empty boxes skipped", flush=True)
    meta = dict(paper=PAPER, formula=gm.FORMULA, k=args.k, storage=args.storage, ignored_drawn=args.ignored,
                label_semi_axes_in_box_units=gm.semi_axis(args.storage) / args.k, code=git_version())
    path = os.path.join(out, 'MASKS.json')
    old = json.load(open(path)) if os.path.exists(path) else {}
    same = all(old.get(key) == meta[key] for key in ('k', 'storage', 'ignored_drawn'))
    if old and not same:
        print(f'note: {path} described other parameters; masks of clips not written now may not match', flush=True)
    meta['clips'] = dict(sorted({**(old.get('clips', {}) if same else {}), **per_clip}.items()))
    json.dump(meta, open(path, 'w'), indent=1)
    print(f'wrote {len(per_clip)} clips to {out}; parameters in {path}')


if __name__ == '__main__':
    main()
