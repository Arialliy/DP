"""Synthetic checks of the camera-compensation path (numpy + OpenCV only).

Run: python code/test_stabilize.py
A camera pans 1.5 px/frame to the right over a textured scene that contains a fixed hot
spot and a car that parks (relative to the ground) in the middle of the clip.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import stabilize as st  # noqa: E402

H, W, N, PAN = 96, 128, 160, 1.5


def scene(seed=0):
    rng = np.random.default_rng(seed)
    world = rng.normal(80, 4, size=(H, 3 * W)).astype(np.float32)
    world[40:43, 200:203] += 60          # ground hot spot, present the whole clip
    car_xy = (230.0, 60.0)               # ground position where the car parks
    frames, boxes = [], []
    for t in range(N):
        ox = PAN * t
        img = np.empty((H, W), np.float32)
        xs = np.arange(W) + ox
        x0 = np.floor(xs).astype(int)
        f = (xs - x0)[None]
        img[:] = world[:, x0] * (1 - f) + world[:, x0 + 1] * f
        b = []
        if 60 <= t < 120:                 # car present (parked) only in the middle
            cx, cy = car_xy[0] - ox, car_xy[1]
            if 2 <= cx < W - 3:
                yi, xi = int(cy), int(round(cx))
                img[yi - 1:yi + 2, xi - 1:xi + 2] += 40
                b.append((xi - 1, yi - 1, 3, 3, 1, 7))
        frames.append(img)
        boxes.append(np.array(b, float).reshape(-1, 6))
    pairs = np.repeat(st.translation(-PAN, 0)[None], N - 1, 0)  # frame i -> i+1: x - PAN
    return np.stack(frames), boxes, pairs


def test_chain_and_mapping():
    _, _, pairs = scene()
    chain = st.chain_from_pairs(pairs)
    M = st.ref_to(chain, 50, 60)
    p = st.apply_h(M, [[30.0, 20.0]])[0]
    assert np.allclose(p, [30.0 - 10 * PAN, 20.0]), p


def test_window_is_stabilized():
    video, boxes, pairs = scene()
    chain = st.chain_from_pairs(pairs)
    a, b = 80, 120
    ref = (a + b) // 2
    stack, valid = st.warp_frames(video, chain, range(a, b), ref)
    # In reference coordinates the parked car stays on one pixel for the whole window.
    car = st.map_boxes(boxes[ref], np.eye(3))[0]
    cy, cx = int(car[1] + 1), int(car[0] + 1)
    prof = stack[:, cy, cx][valid[:, cy, cx]]
    assert prof.min() > 100, prof.min()
    # Raw fixed-pixel profile: the car passes through the pixel in only a few frames.
    raw = video[a:b, cy, cx]
    assert (raw > 100).sum() < 5, (raw > 100).sum()


def test_box_mapping_round_trip():
    video, boxes, pairs = scene()
    chain = st.chain_from_pairs(pairs)
    t, ref = 100, 80
    mapped = st.map_boxes(boxes[t], st.ref_to(chain, t, ref))
    back = st.map_boxes(mapped, st.ref_to(chain, ref, t))
    assert np.allclose(back, boxes[t])
    # The box mapped into reference coordinates equals the car's position at the reference.
    assert np.allclose(mapped[:, :2], boxes[ref][:, :2], atol=1e-6), (mapped, boxes[ref])


def test_history_separates_hot_spot_from_parked_car():
    video, boxes, pairs = scene()
    chain = st.chain_from_pairs(pairs)
    a, b = 80, 120
    ref = (a + b) // 2
    stack, valid = st.warp_frames(video, chain, range(a, b), ref)
    L, _ = st.masked_median(stack, valid)
    B, avail = st.history_field(video, chain, a, b, ref, length=60, step=2, mode='both')
    D = np.where(avail, L - B, np.nan)
    car = boxes[ref][0]
    cy, cx = int(car[1] + 1), int(car[0] + 1)
    spot_x = int(round(200 - PAN * ref)) + 1
    assert 0 <= spot_x < W
    cL, cD = st.local_contrast(L), st.local_contrast(D)
    assert cL[cy, cx] > 20 and cL[41, spot_x] > 20, (cL[cy, cx], cL[41, spot_x])
    assert cD[cy, cx] > 20, cD[cy, cx]                 # car kept: absent from History
    assert abs(cD[41, spot_x]) < 5, cD[41, spot_x]     # hot spot removed: present in History


def test_warp_back_inverts_warp():
    video, _, pairs = scene()
    chain = st.chain_from_pairs(pairs)
    frames = list(range(70, 110))
    ref = 90
    stack, valid = st.warp_frames(video, chain, frames, ref)
    back = st.warp_back(stack, chain, frames, ref, fill=-99.0)
    k = 5
    inside = back[k] > -99
    err = np.abs(back[k][inside] - video[frames[k]][inside])
    assert np.median(err) < 1.0, np.median(err)


def test_crop_matches_full_warp():
    video, _, pairs = scene()
    chain = st.chain_from_pairs(pairs)
    full, _ = st.warp_frames(video, chain, range(70, 80), 75)
    crop, _ = st.warp_frames(video, chain, range(70, 80), 75, r0=10, c0=20, size=(32, 40))
    assert np.allclose(full[:, 10:42, 20:60], crop, atol=1e-3)


def test_gate_chain_static_vs_panning():
    rng = np.random.default_rng(1)
    noisy = np.stack([st.translation(*rng.normal(0, 0.02, 2)) for _ in range(N - 1)])
    chain = st.chain_from_pairs(noisy)
    g, gated = st.gate_chain(chain, range(60, 100), 80, 1.0, (H, W))
    assert gated and np.allclose(g, np.eye(3))
    _, _, pairs = scene()
    pan = st.chain_from_pairs(pairs)
    g, gated = st.gate_chain(pan, range(60, 100), 80, 1.0, (H, W))
    assert not gated and g is pan
    g, gated = st.gate_chain(chain, range(60, 100), 80, 0.0, (H, W))
    assert not gated and g is chain


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
