"""Checks of the multi-rate core (numpy / OpenCV only).

Run: python code/test_multirate.py
"""
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
import multirate as mr  # noqa: E402
import stabilize as st  # noqa: E402

H, W = 96, 128


def texture(width, seed=0):
    """Smooth random texture with sharp blocks (edges for the stationarity test)."""
    rng = np.random.default_rng(seed)
    t = ndimage.gaussian_filter(rng.normal(size=(H, width)), 2.0) * 60 + 90
    for _ in range(12):
        y, x = rng.integers(0, H - 12), rng.integers(0, width - 12)
        t[y:y + 10, x:x + 10] += rng.choice([-40, 40])
    return t.astype(np.float32)


def pan_video(n, vx, seed=0, noise=2.0):
    """Camera panning vx px/frame to the right over a static world. -> video, chain."""
    world = texture(W + int(abs(vx) * n) + 4, seed)
    rng = np.random.default_rng(seed + 1)
    xs = np.arange(W, dtype=np.float64)
    out = np.empty((n, H, W), np.float32)
    for t in range(n):
        p = xs + vx * t
        x0 = np.floor(p).astype(int)
        fr = p - x0
        out[t] = world[:, x0] * (1 - fr) + world[:, x0 + 1] * fr
    out += rng.normal(0, noise, out.shape).astype(np.float32)
    pairs = np.repeat(st.translation(-vx, 0.0)[None], n - 1, 0)
    return out, st.chain_from_pairs(pairs)


def test_constants_match_evaluate():
    import evaluate as ev
    assert mr.FLOOR == ev.FLOOR and mr.NMS == ev.NMS


def test_detectable_band():
    lo, hi = mr.detectable_band(6, 1, 40)
    assert np.isclose(lo, 0.3) and hi == 6
    lo4, hi4 = mr.detectable_band(6, 4, 40)
    assert np.isclose(lo4, 0.075) and hi4 == 1.5


def test_plan_windows_cover_every_frame_once_per_rate():
    for n in (160, 333, 598):
        assert [tuple(f[[0, -1]]) for _, f in mr.plan_windows(n, 1)] == \
               [(a, b - 1) for a, b in st.windows(n, 40)]
        for k in (2, 4):
            plan = mr.plan_windows(n, k)
            seen = np.zeros(n, int)
            for phase, f in plan:
                assert len(f) == 40 and np.all(np.diff(f) == k) and f[0] % k == phase
                seen[np.unique(f)] += 1
            assert (seen >= 1).all(), (n, k)
            # within a phase, windows overlap as in DeepPro's rate-1 scheme (10 % margin,
            # plus the last window, which is aligned to the end)
            assert seen.max() <= 3
    assert mr.plan_windows(150, 4) == []
    assert mr.rate_starts(160, 4) == 4 and mr.rate_starts(156, 4) == 0


def test_rate_parsing():
    assert mr.parse_rates('4,1,2') == [1, 2, 4]
    assert np.allclose(mr.rate_probs([1, 2, 4]), [0.5, 0.25, 0.25])
    assert np.allclose(mr.rate_probs([1]), [1.0])
    assert np.allclose(mr.rate_probs([1, 2], '3,1'), [0.75, 0.25])


def test_scale_homography_matches_block_means():
    import cv2
    rng = np.random.default_rng(3)
    img = (ndimage.gaussian_filter(rng.normal(size=(H, W)), 8) * 400 + 80).astype(np.float32)
    two = np.stack([img, img])                         # frame 1 seen from frame 0's coordinates
    low = mr.lowres(img[None], 4)[0]
    inner = (slice(4, -4), slice(4, -4))
    errs = {}
    for name, M in (('shift4', st.translation(8.0, -4.0)),
                    ('zoom', np.array([[1.08, 0.0, -5.0], [0.0, 1.08, -3.0], [0.0, 0.0, 1.0]]))):
        full = st.warp_frames(two, st.chain_from_pairs(M[None]), [1], 0)[0][0]
        target = mr.lowres(full[None], 4)[0]
        for conv, Ml in (('centres', mr.scale_homography(M, 4)),
                         ('corners', np.linalg.inv(np.diag([4., 4., 1.])) @ M @ np.diag([4., 4., 1.]))):
            wl = cv2.warpPerspective(low, Ml, low.shape[::-1], flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            errs[(name, conv)] = float(np.abs(wl[inner] - target[inner]).mean())
    assert errs[('shift4', 'centres')] < 1e-4, errs           # whole low-res pixels: exact
    assert errs[('zoom', 'centres')] < errs[('zoom', 'corners')], errs  # block centres, not corners
    assert errs[('zoom', 'centres')] < 0.05 * img.std(), errs


def _decide(video, chain, frames, policy='auto'):
    low = mr.lowres(video[frames], 4)
    stats = mr.stationarity(low, chain, frames, 4)
    return mr.choose_coords(policy, stats, frames[1] - frames[0], ), stats


def test_stationarity_prefers_compensation_when_the_camera_drifts():
    video, chain = pan_video(200, 0.1)
    frames = list(range(10, 10 + 157, 4))[:40]        # rate 4: 15.6 px of drift
    choice, s = _decide(video, chain, frames)
    assert choice == 'warp', s
    assert s['warp'] < 0.5 * s['raw'], s


def test_stationarity_keeps_recorded_frames_when_compensation_is_wrong():
    video, _ = pan_video(200, 0.0)                     # static camera ...
    bad = st.chain_from_pairs(np.repeat(st.translation(-0.1, 0.0)[None], 199, 0))  # ... wrong chain
    frames = list(range(10, 10 + 157, 4))[:40]
    choice, s = _decide(video, bad, frames)
    assert choice == 'raw', s
    good = st.chain_from_pairs(np.repeat(np.eye(3)[None], 199, 0))
    choice, s = _decide(video, good, frames)
    assert choice == 'raw', s                          # nothing to gain: keep recorded frames


def test_fast_pan_is_skipped_at_slow_rates_only():
    video, chain = pan_video(170, 3.0)
    f4 = list(range(0, 157, 4))[:40]                   # 468 px sweep: nothing stays in view
    assert _decide(video, chain, f4)[0] == 'skip'
    assert _decide(video, chain, f4, 'raw')[0] == 'raw'
    f1 = list(range(0, 40))                            # 117 px: compensated at rate 1
    assert _decide(video, chain, f1)[0] == 'warp'


def test_assemble_scores_every_frame_at_every_rate():
    n = 330
    for k in (1, 2, 4):
        plan = mr.plan_windows(n, k)
        out, meta = mr.assemble(n, (2, 3), plan, lambda f: ('raw', {}),
                                lambda f, c: np.broadcast_to(np.asarray(f, np.float32)[:, None, None], (len(f), 2, 3)))
        assert np.array_equal(out[:, 0, 0], np.arange(n, dtype=np.float32)), k
        assert len(meta) == len(plan)
    out, meta = mr.assemble(n, (2, 3), mr.plan_windows(n, 2), lambda f: ('skip', {'cover': 0.0}),
                            lambda f, c: None)
    assert np.isneginf(out).all() and meta[0]['coords'] == 'skip'


def toy_profile_detector(stack):
    """|x - window median| per sample: what a temporal-profile detector can see at all."""
    return np.abs(stack - np.median(stack, axis=0, keepdims=True))


def slow_target_video(n, v, s=4, contrast=40.0, noise=2.0, seed=0):
    rng = np.random.default_rng(seed)
    video = rng.normal(80, noise, size=(n, H, W)).astype(np.float32)
    xs = np.arange(W)
    centres = []
    for t in range(n):
        cx = 20 + v * t
        prof = np.clip(np.minimum(xs + 1 - (cx - s / 2), (cx + s / 2) - xs), 0, 1)  # anti-aliased box
        video[t, 46:46 + s] += contrast * prof[None, :]
        centres.append(cx)
    return video, np.array(centres)


def test_blind_zone_and_its_removal_by_rate():
    # 0.1 px/frame, 4 px target: dwell 40 frames. Rate 1 (40-frame window) is blind;
    # rate 4 (40 samples over 157 frames, dwell 10 samples) sees it.
    video, cx = slow_target_video(400, 0.1)
    resp = {}
    for k in (1, 4):
        frames = np.arange(100, 100 + 39 * k + 1, k)
        mid = frames[20]
        r = toy_profile_detector(video[frames])[20]
        x = int(round(cx[mid]))
        resp[k] = r[46:50, x - 1:x + 2].max()
    assert resp[1] < 0.3 * 40 and resp[4] > 0.7 * 40, resp
    # A fast target (1 px/frame) is visible at rate 1, so fusing rates keeps it.
    video, cx = slow_target_video(400, 1.0)
    frames = np.arange(40, 80)                         # centre at x = 80 in the middle frame
    r = toy_profile_detector(video[frames])[20]
    x = int(round(cx[frames[20]]))
    assert r[46:50, x - 1:x + 2].max() > 0.7 * 40


def test_peak_scores_match_evaluate_peaks():
    import evaluate as ev
    rng = np.random.default_rng(5)
    lg = (ndimage.gaussian_filter(rng.normal(size=(H, W)), 2) * 40 - 5).astype(np.float32)
    a = mr.peak_scores(lg)
    b = np.sort(ev.peaks(lg)[:, 2])[::-1]
    assert len(a) == len(b) and np.allclose(a, b)
    assert len(mr.peak_scores(np.full((8, 8), -np.inf))) == 0


def test_cfar_reference_and_fusion():
    peaks = [np.arange(30, 0, -1, dtype=np.float32) for _ in range(5)]   # 30 peaks per frame
    ref = mr.cfar_reference(peaks, m=10, half=1)
    assert np.allclose(ref, 21.0), ref                                   # 10th largest
    sparse = [np.array([5.0, 4.0], np.float32)] * 5
    assert np.allclose(mr.cfar_reference(sparse, m=10, half=1), mr.FLOOR)
    lg = np.zeros((5, 2, 2), np.float32)
    z = mr.calibrate(lg, ref)
    assert np.allclose(z, -21.0)
    f = mr.fuse_max([z, lg - 1, np.full_like(lg, -np.inf)])
    assert np.allclose(f, -1.0)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
