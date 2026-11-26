"""Mine persistent bright spots (hard negatives for the level pathway).

The level pathway of DeepPro-LV was trained with Soft-IoU on 128 x 128 crops, three in four
of them centred on a target. Soft-IoU has zero gradient on a crop without positive
pixels, so a ground hot spot is only ever penalised when a persistent target happens to
share its crop. This script lists, per inference window and in camera-compensated
reference coordinates, the local maxima of the stabilized median's local contrast that are
away from every annotation of the window. Training then centres a fraction of crops on
them (train.py --hard-neg).

Usage: python code/hotspots.py --split train
Writes cache/hotspots/<clip>.npz with arrays ref, x, y, contrast.
Unannotated real targets can appear in the list; inspect a few before trusting it.
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
import stabilize as st  # noqa: E402

CACHE = 'cache/hotspots'


def mine(video, boxes, chain, T=40, min_contrast=10.0, sep=8.0, per_window=20, motion_gate=0.0):
    n = len(video)
    out = {k: [] for k in ('ref', 'x', 'y', 'contrast')}
    full_chain = chain
    for a, b in st.windows(n, T):
        ref = st.reference_frame(a, b)
        chain, _ = st.gate_chain(full_chain, range(a, b), ref, motion_gate, video.shape[1:3])
        stack, valid = st.warp_frames(video, chain, range(a, b), ref)
        L, count = st.masked_median(stack, valid)
        L = np.where(count >= max(1, (b - a) // 2), L, np.nan)
        c = np.nan_to_num(st.local_contrast(L), nan=-np.inf)
        peak = (c == ndimage.maximum_filter(c, size=9, mode='nearest')) & (c >= min_contrast)
        ys, xs = np.nonzero(peak)
        ann = [st.map_boxes(boxes[t], st.ref_to(chain, t, ref)) for t in range(a, b)]
        ann = [q[:, :2] + q[:, 2:4] / 2 - 0.5 for q in ann if len(q)]
        if len(xs) and ann:
            ann = np.concatenate(ann)
            d = np.max(np.abs(np.stack([xs, ys], 1)[:, None, :] - ann[None]), axis=2).min(1)
            xs, ys = xs[d > sep], ys[d > sep]
        order = np.argsort(-c[ys, xs])[:per_window]
        out['ref'] += [ref] * len(order)
        out['x'] += xs[order].tolist()
        out['y'] += ys[order].tolist()
        out['contrast'] += c[ys[order], xs[order]].tolist()
    return {k: np.asarray(v, dtype=np.int64 if k == 'ref' else np.float32) for k, v in out.items()}


def load_or_mine(clip, video, boxes, chain, cache_dir=CACHE, **kw):
    if kw.get('motion_gate', 0.0) > 0:  # gated mining differs from the ungated cache
        cache_dir = os.path.join(cache_dir, f"gate{kw['motion_gate']:g}")
    path = os.path.join(cache_dir, f'{clip}.npz')
    if os.path.exists(path):
        d = np.load(path)
        return {k: d[k] for k in d.files}
    res = mine(video, boxes, chain, **kw)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = path + '.tmp.npz'
    np.savez(tmp, **res)
    os.replace(tmp, path)
    return res


def _one(clip):
    import vtdata
    sc = vtdata.Scene(clip)
    video = st.load_video_u8(clip, sc)
    chain = st.load_chain(clip, len(sc))
    res = load_or_mine(clip, video, sc.boxes, chain)
    return clip, len(res['x'])


def main():
    import splits
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', default='train')
    ap.add_argument('--nproc', type=int, default=4)
    args = ap.parse_args()
    clips = splits.clips(args.split)
    with Pool(max(1, min(args.nproc, len(clips)))) as pool:
        for clip, k in pool.imap_unordered(_one, clips):
            print(f'{clip}: {k} hot-spot candidates', flush=True)


if __name__ == '__main__':
    main()
