"""Checks of the two-pass DeepPro-MX inference loop with stand-in networks (numpy only).

Run: python code/test_infer_mx.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import evaluate  # noqa: E402
import infer_mx  # noqa: E402
import multirate as mr  # noqa: E402

N, H, W, T = 130, 32, 64, 40


def _profile(frames, coords):
    """A moving 'target' at (5 + t % 50, 10) with logit 5; elsewhere -10."""
    out = np.full((len(frames), H, W), -10.0, np.float32)
    for i, t in enumerate(frames):
        out[i, 10, 5 + int(t) % 50] = 5.0
    return out


def _score(frames, coords, rows):
    """Fused map = number of exemplars of this window (so the assembly is visible)."""
    return {'fused': np.full((len(frames), H, W), float(len(rows)), np.float32)}


def _decide(frames):
    return 'raw', {}


def test_outside_bank_uses_other_windows_only():
    bank = infer_mx.make_bank('outside', 16, N, H, W)
    maps, meta, cands = infer_mx.run_clip(N, (H, W), T, _decide, _profile, _score, bank, evaluate.peaks, 0.0)
    plan = mr.plan_windows(N, 1, T)
    assert len(meta) == len(plan) and maps['fused'].shape == (N, H, W)
    assert len(cands) == N and (cands[:, 3] == 5.0).all()           # one peak per frame
    for w in meta:
        rows = np.array(w['exemplar_rows']).reshape(-1, 4)
        assert 0 < w['exemplars'] <= 16
        assert ((rows[:, 0] < w['first']) | (rows[:, 0] > w['last'])).all()
        for i in range(len(rows)):                                    # greedy separation
            for j in range(i):
                same = abs(rows[i, 0] - rows[j, 0]) <= 20 and np.hypot(*(rows[i, 1:3] - rows[j, 1:3])) <= 16
                assert not same
    # overlapping frames take the larger window value
    for t in range(N):
        holding = [w['exemplars'] for w in meta if w['first'] <= t <= w['last']]
        assert maps['fused'][t, 0, 0] == max(holding)


def test_causal_bank_and_none_bank():
    bank = infer_mx.make_bank('causal', 16, N, H, W)
    _, meta, _ = infer_mx.run_clip(N, (H, W), T, _decide, _profile, _score, bank, evaluate.peaks, 0.0)
    assert meta[0]['exemplars'] == 0 and meta[-1]['exemplars'] > 0
    for w in meta:
        assert all(r[0] < w['first'] for r in w['exemplar_rows'])

    def no_profile(frames, coords):
        raise AssertionError('pass 1 must not run without a detection bank')
    _, meta, cands = infer_mx.run_clip(N, (H, W), T, _decide, no_profile, _score,
                                       infer_mx.make_bank('none', 16, N, H, W), evaluate.peaks, 0.0,
                                       need_profile=False)
    assert len(cands) == 0 and all(w['exemplars'] == 0 for w in meta)


def test_min_score_and_clip_bank():
    _, meta, cands = infer_mx.run_clip(N, (H, W), T, _decide, _profile, _score,
                                       infer_mx.make_bank('clip', 16, N, H, W), evaluate.peaks, 6.0)
    assert len(cands) == 0 and all(w['exemplars'] == 0 for w in meta)   # nothing reaches logit 6
    _, meta, _ = infer_mx.run_clip(N, (H, W), T, _decide, _profile, _score,
                                   infer_mx.make_bank('clip', 4, N, H, W), evaluate.peaks, 0.0)
    assert all(w['exemplars'] == 4 for w in meta)


def test_coordinates_are_decided_once_and_shared_by_both_passes():
    calls, seen = [], []

    def decide(frames):
        calls.append(int(frames[0]))
        return ('warp' if int(frames[0]) % 2 == 0 else 'raw'), {'raw': 1.0, 'warp': 0.5}

    def profile(frames, coords):
        seen.append(('p', int(frames[0]), coords))
        return _profile(frames, coords)

    def score(frames, coords, rows):
        seen.append(('s', int(frames[0]), coords))
        return _score(frames, coords, rows)
    infer_mx.run_clip(N, (H, W), T, decide, profile, score, infer_mx.make_bank('outside', 16, N, H, W),
                      evaluate.peaks, 0.0)
    plan = mr.plan_windows(N, 1, T)
    assert sorted(calls) == sorted(int(f[0]) for _, f in plan)          # one decision per window
    for first in calls:
        assert {c for k, f, c in seen if f == first} == {'warp' if first % 2 == 0 else 'raw'}


def test_gt_and_random_banks():
    moving = np.array([[t, 5 + t % 50, 10, 1] for t in range(N)], float)
    g = infer_mx.make_bank('gt', 16, N, H, W, moving=moving)
    rows = g(np.zeros((0, 4)), 40, 79)
    assert 0 < len(rows) <= 16 and ((rows[:, 0] < 40) | (rows[:, 0] > 79)).all()
    r = infer_mx.make_bank('random', 16, N, H, W)(np.zeros((0, 4)), 40, 79)
    assert r.shape == (16, 4) and ((r[:, 0] < 40) | (r[:, 0] > 79)).all()
    try:
        infer_mx.make_bank('gt', 16, N, H, W)
        raise AssertionError('gt bank without a moving table')
    except ValueError:
        pass


def test_arms_are_float16_and_fused_float32():
    def score(frames, coords, rows):
        z = np.zeros((len(frames), H, W), np.float32)
        return {'fused': z + 1, 'variation': z + 2, 'level': z + 3, 'match': z + 0.5}
    maps, _, _ = infer_mx.run_clip(N, (H, W), T, _decide, _profile, score,
                                   infer_mx.make_bank('none', 16, N, H, W), evaluate.peaks, 0.0,
                                   names=('fused', 'variation', 'level', 'match'), need_profile=False)
    assert maps['fused'].dtype == np.float32 and maps['level'].dtype == np.float16
    assert float(maps['match'].max()) == 0.5 and np.isfinite(maps['fused']).all()


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
