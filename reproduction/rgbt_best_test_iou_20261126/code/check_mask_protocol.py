"""Which Gaussian-mask parameters reproduce the authors' RGB-T target counts?

The paper does not give sigma (gaussian_masks.py). The authors' released evaluation log
does give, for each of the 15 test clips, the number of targets their evaluation found:
the 8-connected components of the binarised mask, summed over frames (ShootingRules.py).
Wider Gaussians merge neighbouring targets into one component, so the count falls as
sigma grows; drawing ignored boxes adds components. This script rebuilds the labels for
every combination of
    k        sigma = box size / k            (--ks)
    storage  'round' or 'floor' to 8 bits    (--storage)
    ignored  boxes flagged 'ignore' drawn    (no / yes)
counts the targets per clip exactly as ShootingRules.py does, and compares with the log.
The legacy vtdata.gaussian_mask('sigma_sixth') labels used for training so far are
counted too.

Usage
  python code/check_mask_protocol.py                       # counts embedded from the log
  python code/check_mask_protocol.py --log path/to/eval_ShootingRules_LocLen2_4.txt --ks 4,5,6,7,8
Writes results/mask_protocol/check.json.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import export_masks  # noqa: E402
import gaussian_masks as gm  # noqa: E402
import vtdata  # noqa: E402


def variants(ks, storages):
    return [(k, s, ign) for k in ks for s in storages for ign in (False, True)] + [('legacy', 'round', False)]


def key(v):
    k, s, ign = v
    return 'legacy_sigma_sixth' if k == 'legacy' else f"k{k:g}_{s}_{'with' if ign else 'without'}_ignored"


def count_clip(task):
    clip, ks, storages = task
    sc = vtdata.Scene(clip)
    frames = {ign: export_masks.annotated_frames(sc, ign) for ign in (False, True)}
    counts = {key(v): 0 for v in variants(ks, storages)}
    boxes = sum(len(b) for b, _, _ in frames[False])
    tiny = sum(int(((b[:, 2] < 3) | (b[:, 3] < 3)).sum()) for b, _, _ in frames[False] if len(b))
    for t in range(len(sc)):
        for ign in (False, True):
            b = frames[ign][t][0]
            for k in ks:
                g = gm.heatmap(b, sc.H, sc.W, k)
                for s in storages:
                    counts[key((k, s, ign))] += gm.count_targets(gm.loader_label(gm.store(g, s)))
        counts['legacy_sigma_sixth'] += gm.count_targets(
            vtdata.gaussian_mask(frames[False][t][0], sc.H, sc.W, 'sigma_sixth', quant=True))
    return clip, dict(frames=len(sc), boxes=boxes, boxes_under_3px=tiny, counts=counts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--log', default=None, help='authors\' eval_ShootingRules_LocLen2_4.txt (default: counts embedded)')
    ap.add_argument('--ks', default='2,3,4,5,6,7,8,10,12')
    ap.add_argument('--storage', default='round,floor')
    ap.add_argument('--nproc', type=int, default=8)
    ap.add_argument('--out', default='results/mask_protocol')
    args = ap.parse_args()
    ref = gm.parse_eval_log(open(args.log).read()) if args.log else dict(gm.AUTHOR_TEST_COUNTS)
    ks = [float(k) for k in args.ks.split(',')]
    storages = args.storage.split(',')
    clips = sorted(ref)
    res = {}
    with Pool(max(1, min(args.nproc, len(clips)))) as pool:
        for clip, r in pool.imap_unordered(count_clip, [(c, ks, storages) for c in clips]):
            res[clip] = r
            print(f"{clip}: {r['frames']} frames, {r['boxes']} boxes ({r['boxes_under_3px']} under 3 px), "
                  f"log {ref[clip]}", flush=True)
    rows = []
    for v in variants(ks, storages):
        name = key(v)
        diff = {c: res[c]['counts'][name] - ref[c] for c in clips}
        rows.append(dict(variant=name, total=sum(res[c]['counts'][name] for c in clips),
                         exact_clips=sum(d == 0 for d in diff.values()),
                         abs_diff=sum(abs(d) for d in diff.values()), diff=diff))
    rows.sort(key=lambda r: (-r['exact_clips'], r['abs_diff']))
    print(f"\nlog total {sum(ref.values())} over {len(clips)} clips; boxes drawn (no ignored) "
          f"{sum(res[c]['boxes'] for c in clips)}")
    print(f"{'variant':34s} {'total':>7s} {'exact':>6s} {'|diff|':>7s}  worst clips")
    for r in rows:
        worst = sorted(r['diff'].items(), key=lambda kv: -abs(kv[1]))[:3]
        print(f"{r['variant']:34s} {r['total']:7d} {r['exact_clips']:3d}/{len(clips):<2d} {r['abs_diff']:7d}  "
              + ' '.join(f'{c}:{d:+d}' for c, d in worst if d))
    exact = [r['variant'] for r in rows if r['exact_clips'] == len(clips)]
    print('\nreproduce every clip exactly: ' + (', '.join(exact) if exact else 'none'))
    os.makedirs(args.out, exist_ok=True)
    json.dump(dict(reference=ref, reference_source=args.log or 'gaussian_masks.AUTHOR_TEST_COUNTS',
                   per_clip=res, variants=rows, exact=exact),
              open(os.path.join(args.out, 'check.json'), 'w'), indent=1)
    print(f"wrote {os.path.join(args.out, 'check.json')}")


if __name__ == '__main__':
    main()
