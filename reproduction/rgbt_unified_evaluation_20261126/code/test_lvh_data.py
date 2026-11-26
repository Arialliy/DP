"""Checks of hot-spot mining and stabilized sample construction (numpy only).

Run: python code/test_lvh_data.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import hotspots  # noqa: E402
import lvh_data  # noqa: E402
import stabilize as st  # noqa: E402
import test_stabilize as ts  # noqa: E402


def _scene():
    video, boxes, pairs = ts.scene()
    return video, boxes, st.chain_from_pairs(pairs)


def test_mining_finds_hot_spot_not_car():
    video, boxes, chain = _scene()
    hot = hotspots.mine(video, boxes, chain, min_contrast=20)
    assert len(hot['x']) > 0
    for ref, x, y in zip(hot['ref'], hot['x'], hot['y']):
        world_x = x + ts.PAN * ref
        assert abs(world_x - 201) <= 2 and abs(y - 41) <= 1, (ref, x, y, world_x)


def test_build_sample_aligns_labels_and_frames():
    video, boxes, chain = _scene()
    a, b, T, P = 80, 120, 40, 32
    ref = st.reference_frame(a, b)
    car = boxes[ref][0]
    r0, c0 = int(car[1]) - 10, int(car[0]) - 12
    s = lvh_data.build_sample(video, boxes, chain, a, b, T, P, True, r0, c0, 60, 2, 'both',
                              mean=80.0, std=10.0)
    assert s['img'].shape == (1, T, P, P) and s['lab'].shape == (T, P, P)
    assert s['valid'].all() and s['pvalid'].shape == (1, T, P, P)
    cy, cx = int(car[1] + 1) - r0, int(car[0] + 1) - c0
    assert s['lab'][:, cy, cx].min() == 1.0                     # car labelled in every frame
    assert (s['img'][0, :, cy, cx] > 2.5).all()                 # (120 - 80) / 10 = 4 at the car
    assert s['hvalid'][0, cy, cx] and abs(s['hist'][0, cy, cx]) < 1.0  # car absent from History


def test_short_window_padding():
    video, boxes, chain = _scene()
    s = lvh_data.build_sample(video, boxes, chain, 0, 10, 40, 32, True, 0, 0, 60, 2, 'both')
    assert s['valid'].sum() == 10 and s['valid'][-10:].all()
    assert not s['pvalid'][0, :30].any()


def test_hotspot_center_moves_to_sample_reference():
    video, boxes, chain = _scene()
    hot = hotspots.mine(video, boxes, chain, min_contrast=20)
    rng = np.random.default_rng(0)
    a, b = 80, 120
    ref = st.reference_frame(a, b)
    c, kind = lvh_data.choose_center(boxes[ref], hot, chain, a, b, ref, rng, hard_neg=1.0)
    assert kind == 'hotspot'
    assert abs(c[0] + ts.PAN * ref - 201) <= 2 and abs(c[1] - 41) <= 1, c


def test_history_label_occupancy_arrived_vs_resident():
    video, boxes, chain = _scene()
    a, b, T, P = 80, 120, 40, 32
    ref = st.reference_frame(a, b)
    car = boxes[ref][0]
    r0, c0 = int(car[1]) - 10, int(car[0]) - 12
    cy, cx = int(car[1] + 1) - r0, int(car[0] + 1) - c0
    s = lvh_data.build_sample(video, boxes, chain, a, b, T, P, True, r0, c0, 60, 2, 'both')
    assert s['hlab'].shape == (1, P, P)
    assert s['hlab'][0, cy, cx] < 0.5, s['hlab'][0, cy, cx]      # arrived: mostly absent from History
    # Resident: annotate the car's ground position in every frame where it is in view.
    resident = []
    for t in range(len(boxes)):
        x = int(round(230 - ts.PAN * t))
        resident.append(np.array([(x - 1, 59, 3, 3, 1, 7)], float) if 2 <= x < ts.W - 3
                        else np.zeros((0, 6)))
    s = lvh_data.build_sample(video, resident, chain, a, b, T, P, True, r0, c0, 60, 2, 'both')
    assert s['hlab'][0, cy, cx] > 0.9, s['hlab'][0, cy, cx]


def test_motion_gate_uses_recorded_frames_on_static_camera():
    rng = np.random.default_rng(2)
    video = rng.normal(80, 5, size=(120, 48, 64)).astype(np.float32)
    boxes = [np.zeros((0, 6)) for _ in range(120)]
    noisy = st.chain_from_pairs(np.stack([st.translation(*rng.normal(0, 0.02, 2)) for _ in range(119)]))
    a, b, P = 40, 80, 32
    s = lvh_data.build_sample(video, boxes, noisy, a, b, 40, P, True, 5, 7, 20, 2, 'both',
                              mean=80.0, std=5.0, motion_gate=1.0)
    assert bool(s['gated'])
    assert np.allclose(s['img'][0], (video[a:b, 5:5 + P, 7:7 + P] - 80.0) / 5.0, atol=1e-5)
    s = lvh_data.build_sample(video, boxes, noisy, a, b, 40, P, True, 5, 7, 20, 2, 'both',
                              mean=80.0, std=5.0, motion_gate=0.0)
    assert not bool(s['gated'])
    assert not np.allclose(s['img'][0], (video[a:b, 5:5 + P, 7:7 + P] - 80.0) / 5.0, atol=1e-3)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
