"""Checks of multi-rate training samples (numpy only).

Run: python code/test_mr_data.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import lvh_data  # noqa: E402
import mr_data  # noqa: E402
import stabilize as st  # noqa: E402
import test_stabilize as ts  # noqa: E402


def _scene():
    video, boxes, pairs = ts.scene()          # 1.5 px/frame pan, a car parked for frames 60-119
    return video, boxes, st.chain_from_pairs(pairs)


def test_rate_tables_and_windows():
    lengths = {'a': 70, 'b': 160, 'c': 600}
    tab = mr_data.rate_tables(lengths, [1, 2, 4])
    assert set(tab) == {2, 4}
    names, p = tab[4]
    assert names == ['b', 'c'] and np.isclose(p.sum(), 1) and p[1] > p[0]
    assert 'a' not in tab[2][0]                                 # 70 < 79 frames
    rng = np.random.default_rng(0)
    for _ in range(50):
        f = mr_data.draw_window(rng, 4, 160)
        assert f[0] >= 0 and f[-1] <= 159 and np.all(np.diff(f) == 4) and len(f) == 40


def test_raw_sample_equals_direct_crop():
    video, boxes, chain = _scene()
    frames = np.arange(76, 76 + 79, 2)[:40]                    # car in view for frames 71-119
    P, r0, c0 = 32, 30, 85
    s = mr_data.build_window_sample(video, boxes, None, frames, 'raw', 40, P, r0, c0, True,
                                    mean=80.0, std=10.0)
    assert s['img'].shape == (1, 40, P, P) and s['valid'].all() and s['pvalid'].all()
    assert np.allclose(s['img'][0], (video[frames, r0:r0 + P, c0:c0 + P] - 80.0) / 10.0)
    checked = 0
    for k, t in enumerate(frames):                              # labels follow the box per frame
        if len(boxes[t]):
            b = boxes[t][0]
            cy, cx = int(b[1] + 1) - r0, int(b[0] + 1) - c0
            if 0 <= cx < P:
                assert s['lab'][k, cy, cx] == 1.0
                checked += 1
    assert checked >= 5


def test_identity_warp_equals_raw():
    video, boxes, _ = _scene()
    ident = st.chain_from_pairs(np.repeat(np.eye(3)[None], len(video) - 1, 0))
    frames = np.arange(60, 100)
    a = mr_data.build_window_sample(video, boxes, ident, frames, 'raw', 40, 32, 30, 40, True)
    b = mr_data.build_window_sample(video, boxes, ident, frames, 'warp', 40, 32, 30, 40, True)
    for key in ('img', 'lab', 'valid', 'pvalid'):
        assert np.allclose(a[key], b[key]), key


def test_warp_sample_holds_the_parked_car_still():
    video, boxes, chain = _scene()
    frames = np.arange(62, 62 + 2 * 39 + 1, 2)                 # rate 2 over the parked interval
    ref = frames[20]
    car = boxes[ref][0]
    r0, c0 = int(car[1]) - 10, int(car[0]) - 12
    s = mr_data.build_window_sample(video, boxes, chain, frames, 'warp', 40, 32, r0, c0, True,
                                    mean=80.0, std=10.0)
    cy, cx = int(car[1] + 1) - r0, int(car[0] + 1) - c0
    present = np.array([len(boxes[t]) > 0 for t in frames])    # the car is annotated in 71-119
    ok = s['pvalid'][0, :, cy, cx] & present
    assert ok.sum() >= 20, ok.sum()
    assert (s['lab'][ok, cy, cx] == 1.0).all()                  # same reference pixel every sample
    assert (s['img'][0, ok, cy, cx] > 2.5).all()                # (120 - 80) / 10 = 4 at the car


def test_short_window_padding_matches_trainset_convention():
    video, boxes, chain = _scene()
    s = mr_data.build_window_sample(video, boxes, chain, np.arange(0, 10), 'warp', 40, 32, 0, 0, True)
    assert s['valid'].sum() == 10 and s['valid'][-10:].all() and not s['pvalid'][0, :30].any()
    s = mr_data.build_window_sample(video, boxes, chain, np.arange(0, 10), 'raw', 40, 32, 0, 0, False)
    assert s['valid'][:10].all() and not s['valid'][10:].any()


def test_decide_follows_the_stationarity_rule():
    video, boxes, chain = _scene()                             # real pan: compensation wins
    frames = np.arange(0, 40)
    coords, stats = mr_data.decide(video, chain, frames, 'auto', 1)
    assert coords == 'warp', stats
    assert mr_data.decide(video, chain, frames, 'raw', 1) == ('raw', {})


def test_centre_and_crop():
    rng = np.random.default_rng(1)
    boxes = np.array([[50.0, 40.0, 4.0, 4.0, 1, 3]])
    hits = [mr_data.choose_center(boxes, rng) for _ in range(400)]
    frac = np.mean([h is not None for h in hits])
    assert 0.65 < frac < 0.85
    c = next(h for h in hits if h is not None)
    assert c == (52.0, 42.0)
    r0, c0 = lvh_data.crop_origin(c, 32, 96, 128, rng)
    assert 0 <= r0 <= 96 - 33 and 0 <= c0 <= 128 - 33


def test_draw_spec_rates_and_skip_fallback():
    lengths = {'short': 70, 'long': 600}
    samples = [(c, max(0, i - 40), i) for c, n in lengths.items() for i in range(4, n)]
    w = np.array([b - a for _, a, b in samples], float)
    tables = mr_data.rate_tables(lengths, [1, 2, 4])
    rng = np.random.default_rng(0)
    got = [mr_data.draw_spec(rng, [1, 2, 4], [0.5, 0.25, 0.25], tables, samples, w / w.sum(), lengths,
                             lambda c, f, k: 'raw') for _ in range(2000)]
    ks = np.array([g[0] for g in got])
    assert abs((ks == 1).mean() - 0.5) < 0.05 and abs((ks == 4).mean() - 0.25) < 0.05
    for k, c, f, coords in got:
        assert f[-1] < lengths[c] and (k == 1 or (len(f) == 40 and np.all(np.diff(f) == k)))
        assert not (k == 4 and c == 'short')                 # 70 frames hold no rate-4 window
    skip_slow = lambda c, f, k: 'skip' if k > 1 else 'warp'
    got = [mr_data.draw_spec(rng, [1, 4], [0.0, 1.0], tables, samples, w / w.sum(), lengths, skip_slow)
           for _ in range(50)]
    assert all(g[0] == 1 and g[3] == 'warp' for g in got)    # skipped rate-4 windows fall back to rate 1


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
