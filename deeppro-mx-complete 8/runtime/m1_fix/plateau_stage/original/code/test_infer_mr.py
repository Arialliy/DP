"""End-to-end check of multi-rate inference with a toy temporal-profile detector (no torch).

The toy detector scores |x - window median|, the evidence any temporal-profile network
starts from. A slow target (0.1 px/frame) is invisible to it at rate 1 and visible at
rate 4; a fast target (1 px/frame) is visible at rate 1. The fused maps keep both.

Run: python code/test_infer_mr.py
"""
import os
import sys
import tempfile
from argparse import Namespace

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import infer_mr as im  # noqa: E402
import multirate as mr  # noqa: E402
import stabilize as st  # noqa: E402

H, W, N = 96, 160, 420


def video_two_targets(seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(80, 2, size=(N, H, W)).astype(np.float32)
    xs = np.arange(W)
    for t in range(N):
        for row, cx, s in ((20, 30 + 0.1 * t, 6), (70, (5 + 1.0 * t) % (W - 10) + 5, 4)):
            prof = np.clip(np.minimum(xs + 1 - (cx - s / 2), (cx + s / 2) - xs), 0, 1)
            v[t, row:row + s] += 40 * prof[None]
    return v


def toy_scorer(video, chain):
    def toy(stack):
        return np.abs(stack - np.median(stack, axis=0, keepdims=True)) / 4.0 - 6.0

    def score(frames, coords):
        frames = [int(t) for t in frames]
        ref = frames[len(frames) // 2]
        if coords == 'warp':
            stack, pv = st.warp_frames(video, chain, frames, ref)
            return st.warp_back(np.where(pv, toy(stack), -20.0), chain, frames, ref, im.FILL)
        return toy(video[frames])
    return score


def args_for(**kw):
    a = dict(rates=[1, 4], seqlen=40, coords='raw', lowres=4, margin=0.1, abs_tol=0.5, min_cover=0.1,
             fusion={'max', 'cal'}, cal_m=10, cal_half=18)
    a.update(kw)
    return Namespace(**a)


def _run(tmp, video, chain, **kw):
    args = args_for(**kw)
    dirs = im.outputs(os.path.join(tmp, 'run'), args.rates, args.fusion)
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    meta = im.process_clip('FAKE', video, chain, toy_scorer(video, chain), args, dirs)
    return {k: np.load(f'{d}/FAKE.npy').astype(np.float32) for k, d in dirs.items()}, meta


def test_fusion_keeps_slow_and_fast_targets():
    video = video_two_targets()
    chain = im.identity_chain(N)
    with tempfile.TemporaryDirectory() as tmp:
        out, meta = _run(tmp, video, chain)
    assert set(out) == {'r1', 'r4', 'r1cal', 'r4cal', 'mrmax', 'mrcal'}
    assert all(o.shape == (N, H, W) and o.dtype == np.float32 for o in out.values())
    assert meta['rates']['4']['used'] and meta['rates']['4']['windows'] == 4 * len(st.windows(N // 4))
    # Frame 200 is the middle of the rate-1 window 180-219. The slow target (6 px, moving
    # 4 px in that window) covers its centre in every frame, so the median absorbs it; only
    # thin edges remain in the variation. At rate 4 it moves 16 px per window and stays whole.
    t = 200
    slow = (slice(21, 25), slice(int(round(30 + 0.1 * t)) - 1, int(round(30 + 0.1 * t)) + 2))
    cxf = (5 + 1.0 * t) % (W - 10) + 5
    fast = (slice(70, 74), slice(int(cxf) - 2, int(cxf) + 3))
    peak = lambda key, r: float(out[key][t][r].max())
    assert peak('r1', slow) < peak('r4', slow) - 4, (peak('r1', slow), peak('r4', slow))   # blind at rate 1
    assert peak('r1', fast) > 1.0                                                          # fast seen at rate 1
    assert peak('mrmax', slow) >= peak('r4', slow) - 1e-2 and peak('mrmax', fast) >= peak('r1', fast) - 1e-2
    assert np.isfinite(out['mrcal']).all() and np.isfinite(out['r1cal']).all()


def test_short_clip_marks_slow_rate_unused():
    video = video_two_targets()[:120]
    chain = im.identity_chain(120)
    with tempfile.TemporaryDirectory() as tmp:
        out, meta = _run(tmp, video, chain)
    assert not meta['rates']['4']['used'] and np.isneginf(out['r4']).all()
    assert np.allclose(out['mrmax'], out['r1'])


def test_auto_coordinates_compensate_a_panning_camera():
    rng = np.random.default_rng(1)
    world = (np.kron(rng.normal(size=(H // 8, (W + 200) // 8)), np.ones((8, 8))) * 30 + 90).astype(np.float32)
    n, vx = 200, 0.5
    video = np.stack([world[:, int(round(vx * t)):int(round(vx * t)) + W] for t in range(n)])
    video += rng.normal(0, 1, video.shape).astype(np.float32)
    shifts = np.diff(np.round(vx * np.arange(n)))
    chain = st.chain_from_pairs(np.stack([st.translation(-s, 0.0) for s in shifts]))
    with tempfile.TemporaryDirectory() as tmp:
        out, meta = _run(tmp, video, chain, rates=[1, 2], coords='auto')
    r1, r2 = meta['rates']['1'], meta['rates']['2']
    assert r1['warp'] == r1['windows'] and r2['warp'] + r2['skip'] == r2['windows'], (r1, r2)
    # compensation removes the moving edges: far fewer strong responses than raw scoring
    with tempfile.TemporaryDirectory() as tmp:
        raw, _ = _run(tmp, video, chain, rates=[1], coords='raw', fusion=set())
    assert (out['r1'] > 0).mean() < 0.2 * (raw['r1'] > 0).mean()


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
