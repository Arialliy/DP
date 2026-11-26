"""Check that the training labels equal the official-layout masks on disk.

train.py does not read <VT_ROOT>/segmentations: it renders each frame's label from the
boxes with vtdata.gaussian_mask(..., --label-rule), cropped or warped with the frame. With
--label-rule paper_k6_floor this is meant to be the mask export_masks.py wrote for the
official code. This script compares, frame by frame, the rendered label with the PNG as
the authors' loader reads it (uint8 / 255, every value > 0 set to 1) and reports the frames
and pixels that differ, PNGs that are missing and PNGs without a frame.

Usage
  export VT_ROOT=/path/to/RGB-T VT_ANN_DIR=/path/to/RGBT-Tiny/annotations_coco OFFICIAL_LISTS=/path/to/RGB-T
  python code/check_labels.py                       # the 60 clips of train.txt and test.txt
  python code/check_labels.py --clips val --rule paper_k6_floor
Exit status 1 if any clip differs.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import gaussian_masks as gm  # noqa: E402
import vtdata  # noqa: E402


def check_clip(task):
    clip, rule, seg_root = task
    sc = vtdata.Scene(clip)
    d = os.path.join(seg_root, clip)
    pngs = sorted(f for f in os.listdir(d) if f.endswith('.png')) if os.path.isdir(d) else []
    extra = sorted({os.path.splitext(f)[0] for f in pngs} - {os.path.splitext(f)[0] for f in sc.frames})
    out = dict(frames=len(sc), pngs=len(pngs), missing=0, extra=len(extra), extra_examples=extra[:5],
               differing_frames=0, differing_pixels=0, label_pixels=0, first_difference=None)
    for t, fname in enumerate(sc.frames):
        p = os.path.join(d, os.path.splitext(fname)[0] + '.png')
        if not os.path.exists(p):
            out['missing'] += 1
            continue
        png = gm.loader_label(np.array(Image.open(p)))
        lab = vtdata.gaussian_mask(sc.boxes[t], sc.H, sc.W, rule)
        if png.shape != lab.shape:
            raise ValueError(f'{p}: mask {png.shape}, frame {lab.shape}')
        diff = int((png != lab).sum())
        out['label_pixels'] += int(lab.sum())
        if diff:
            out['differing_frames'] += 1
            out['differing_pixels'] += diff
            if out['first_difference'] is None:
                out['first_difference'] = dict(frame=fname, pixels=diff)
    out['ok'] = out['missing'] == 0 and not out['extra'] and out['differing_frames'] == 0
    return clip, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--clips', default='official', help="'official' (train.txt + test.txt), a split name or a list")
    ap.add_argument('--rule', default='paper_k6_floor')
    ap.add_argument('--seg', default=None, help='mask folder (default <VT_ROOT>/segmentations)')
    ap.add_argument('--nproc', type=int, default=8)
    ap.add_argument('--out', default=None, help='optional json report')
    args = ap.parse_args(argv)
    if args.clips == 'official':
        clips = vtdata.scene_list('train') + vtdata.scene_list('test')
    else:
        import splits
        clips = splits.clips(args.clips)
    seg = args.seg or os.path.join(vtdata.DATA_ROOT, 'segmentations')
    res = {}
    with Pool(max(1, min(args.nproc, len(clips)))) as pool:
        for clip, r in pool.imap_unordered(check_clip, [(c, args.rule, seg) for c in clips]):
            res[clip] = r
            flag = 'OK  ' if r['ok'] else 'DIFF'
            print(f"{flag} {clip}: {r['frames']} frames, {r['missing']} missing, {r['differing_frames']} frames / "
                  f"{r['differing_pixels']} pixels differ of {r['label_pixels']} label pixels", flush=True)
    bad = sorted(c for c, r in res.items() if not r['ok'])
    print(f"\n{len(res) - len(bad)}/{len(res)} clips: labels of rule {args.rule} equal {seg}" +
          (f"; differ: {', '.join(bad)}" if bad else ''))
    if args.out:
        json.dump(dict(rule=args.rule, segmentations=seg, clips=res), open(args.out, 'w'), indent=1)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
