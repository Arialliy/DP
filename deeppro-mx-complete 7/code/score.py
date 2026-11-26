"""Score cached logits of a run on a split.

Writes results/scores/<run>/<split>/{gt.csv, fp.npz, summary.json}.
  gt.csv : clip,t,track,cat,score   (score = logit of the matched detection, -inf if missed)
  fp.npz : per-clip arrays of false-alarm detection logits
Usage: python code/score.py --run official_rgbt --split test [--author [--label-rule paper_k6_floor]]
--label-rule sets the masks of the authors' protocol (--author); the primary box metrics use
no masks. The default paper_k6_floor matches the masks export_masks.py writes for the
official code; sigma_sixth reproduces author-protocol numbers made before DeepPro-MX.
The authors' protocol also reports IoU (prob > th, as test.py) and, by default, treats the
image edge as ShootingRules.py does (--author-edges, evaluate.shooting_rules).
"""
import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import cachekey  # noqa: E402
import evaluate as ev  # noqa: E402
import splits  # noqa: E402
import vtdata  # noqa: E402


def _one(a):
    run, clip, author, rule, edges = a
    lg = np.load(f'cache/preds/{run}/{clip}.npy')
    sc = vtdata.Scene(clip)
    assert lg.shape[0] == len(sc), (clip, lg.shape)
    r = ev.eval_clip(lg, sc)
    if author:
        r['author'] = ev.author_metrics(lg, sc, rule=rule, edges=edges)
    return clip, r


def summarize(gt_rows, fps, frames):
    s = np.array([g[-1] for g in gt_rows], float)
    out = dict(gt=len(s), frames=frames, fp_candidates=int(len(fps)),
               cpm=ev.cpm(s, fps, frames), ap=ev.ap(s, fps))
    for fa in ev.CPM_FA:
        pd, thr = ev.pd_at(s, fps, frames, fa)
        out[f'pd@{fa}'] = pd
        out[f'thr@{fa}'] = thr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--author', action='store_true')
    ap.add_argument('--label-rule', dest='label_rule', choices=('sigma_sixth', 'paper_k6_floor'),
                    default='paper_k6_floor', help="masks of the authors' protocol")
    ap.add_argument('--author-edges', dest='author_edges', choices=('official', 'clipped'), default='official',
                    help="ShootingRules at the image edge: as ShootingRules.py (default) or clipped")
    ap.add_argument('--nproc', type=int, default=8)
    args = ap.parse_args()
    clips = splits.clips(args.split)
    with Pool(args.nproc) as p:
        res = dict(p.map(_one, [(args.run, c, args.author, args.label_rule, args.author_edges) for c in clips]))
    out = f'results/scores/{args.run}/{args.split}'
    os.makedirs(out, exist_ok=True)
    rows, fps, frames, per_clip = [], [], 0, {}
    for c in clips:
        r = res[c]
        rows += [(c, *g) for g in r['gt']]
        fps.append(r['fp'])
        frames += r['frames']
        per_clip[c] = summarize([(c, *g) for g in r['gt']], r['fp'], r['frames'])
    with open(f'{out}/gt.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['clip', 't', 'track', 'cat', 'score'])
        w.writerows(rows)
    np.savez_compressed(f'{out}/fp.npz', **{c: res[c]['fp'].astype(np.float32) for c in clips})
    fps_of = {c: cachekey.stored(args.run, c) for c in clips}
    summ = dict(run=args.run, split=args.split, overall=summarize(rows, np.concatenate(fps), frames),
                per_clip=per_clip,
                predictions={c: (dict(fingerprint=cachekey.digest(fp), ckpt=fp.get('ckpt'),
                                      ckpt_sha256=fp.get('ckpt_sha256')) if fp else None)
                             for c, fp in fps_of.items()})
    ckpts = {fp.get('ckpt_sha256') for fp in fps_of.values() if fp}
    if len(ckpts) > 1:
        raise SystemExit(f'{args.run}: the clips were predicted by {len(ckpts)} different checkpoints; '
                         'run the inference again')
    if any(fp is None for fp in fps_of.values()):
        print(f'warning: {sum(fp is None for fp in fps_of.values())} clips of {args.run} have no prediction '
              'fingerprint (made before cachekey.py, or outside this code): their checkpoint is unknown', flush=True)
    if args.author:
        tot = {}
        for c in clips:
            for th, v in res[c]['author'].items():
                t = tot.setdefault(str(th), dict(false_px=0, tp=0, targets=0, pixels=0, intersection=0, union=0))
                for k in t:
                    t[k] += v[k]
        for t in tot.values():
            t['Pd'] = t['tp'] / max(t['targets'], 1)
            t['Fa'] = t['false_px'] / t['pixels']
            t['IoU'] = t['intersection'] / max(t['union'], 1)
        summ['author_protocol'] = tot
        # test.py's headline numbers for one checkpoint: IoU at threshold_eval 0.5, Pd and Fa
        # at the same threshold, and the AUC of the Pd / Fa points of the 27 thresholds
        at = tot[str(0.5)]
        summ['author_headline'] = dict(IoU=at['IoU'], Pd=at['Pd'], Fa=at['Fa'], threshold=0.5,
                                       AUC=ev.official_auc([tot[str(th)]['Fa'] for th in ev.TH_SEG],
                                                           [tot[str(th)]['Pd'] for th in ev.TH_SEG]),
                                       thresholds=list(ev.TH_SEG))
        summ['author_label_rule'] = args.label_rule
        summ['author_edges'] = args.author_edges
        summ['author_protocol_per_clip'] = {c: {str(k): v for k, v in res[c]['author'].items()} for c in clips}
    json.dump(summ, open(f'{out}/summary.json', 'w'), indent=1)
    o = summ['overall']
    print(args.run, args.split, 'CPM %.4f AP %.4f' % (o['cpm'], o['ap']),
          ' '.join('Pd@%g=%.4f' % (fa, o[f'pd@{fa}']) for fa in ev.CPM_FA))
    if args.author:
        for th, t in summ['author_protocol'].items():
            print('  author th=%-6s Pd=[%d/%d, %.5f] Fa=[%d, %.6e] IoU=%.6f'
                  % (th, t['tp'], t['targets'], t['Pd'], t['false_px'], t['Fa'], t['IoU']))
        h = summ['author_headline']
        print('  author headline (threshold 0.5): IoU=%.6f Pd=%.5f Fa=%.6e; AUC over %d thresholds %.5f'
              % (h['IoU'], h['Pd'], h['Fa'], len(h['thresholds']), h['AUC']))


if __name__ == '__main__':
    main()
