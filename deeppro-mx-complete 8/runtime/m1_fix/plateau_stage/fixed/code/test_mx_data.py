"""Checks of the DeepPro-MX exemplar data path (numpy only).

Run: python code/test_mx_data.py
"""
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import mx_data as mxd  # noqa: E402


def _boxes(n=100):
    """Track 1 moves 1 px/frame along row 20; track 2 is parked at (60, 40); 5 x 3 boxes."""
    boxes, speed = [], {}
    for t in range(n):
        b = [(10 + t, 19, 5, 3, 1, 1), (58, 39, 5, 3, 1, 2)]
        boxes.append(np.array(b, float))
        speed[('c', t, 1)] = 1.0
        speed[('c', t, 2)] = 0.0
    return boxes, speed


def test_read_speeds_and_moving_table():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'targets.csv')
        with open(p, 'w') as f:
            f.write('scene,t,track,cat,cx,cy,w,h,image_speed,camera_speed,rel_speed,span\n'
                    'c,0,1,1,0,0,1,1,1,0,0.7,5\nc,1,1,1,0,0,1,1,1,0,nan,0\nd,0,3,1,0,0,1,1,1,0,2.0,5\n')
        sp = mxd.read_speeds(p, ['c'])
        assert sp[('c', 0, 1)] == 0.7 and math.isnan(sp[('c', 1, 1)]) and ('d', 0, 3) not in sp
    boxes, speed = _boxes()
    m = mxd.moving_table('c', boxes, speed)
    assert m.shape == (100, 4) and set(m[:, 3]) == {1.0}
    assert np.allclose(m[5, :3], [5, 10 + 5 + 2.5 - 0.5, 19 + 1.5 - 0.5])     # centre pixel of the box
    assert len(mxd.moving_table('c', boxes, {})) == 0                            # unknown speed: not moving


def test_greedy_select_and_pools():
    rows = np.array([[10, 0, 0, 0], [12, 3, 0, 0], [40, 3, 0, 0], [12, 30, 0, 0], [13, 31, 0, 0]], float)
    sel = mxd.greedy_select(rows, 10, r_xy=16, dt=20)
    assert list(sel) == [0, 2, 3]                       # row 1 is near row 0 in space and time
    assert list(mxd.greedy_select(rows, 2)) == [0, 2]
    t = np.arange(10)
    assert list(np.flatnonzero(mxd.pool_mask(t, 3, 5, 'outside'))) == [0, 1, 2, 6, 7, 8, 9]
    assert list(np.flatnonzero(mxd.pool_mask(t, 3, 5, 'causal'))) == [0, 1, 2]
    assert mxd.pool_mask(t, 3, 5, 'clip').all()


def test_draw_exemplars_stays_outside_the_window():
    boxes, speed = _boxes()
    moving = mxd.moving_table('c', boxes, speed)
    rng = np.random.default_rng(0)
    seen_none, sizes = 0, []
    for _ in range(300):
        sel = mxd.draw_exemplars(rng, moving, 30, 69, 16, p_none=0.3)
        if len(sel) == 0:
            seen_none += 1
            continue
        assert ((sel[:, 0] < 30) | (sel[:, 0] > 69)).all()
        assert len(mxd.greedy_select(sel, 99)) == len(sel)      # already separated
        sizes.append(len(sel))
    assert 50 < seen_none < 130                                  # p_none = 0.3
    assert max(sizes) > 1 and min(sizes) == 1                    # full and partial banks
    assert len(mxd.draw_exemplars(rng, moving, 0, 99, 16, p_none=0.0)) == 0   # nothing outside


def test_patches_centre_edge_padding_and_normalisation():
    video = np.arange(2 * 20 * 30, dtype=np.float32).reshape(2, 20, 30)
    p = mxd.patch2d(video[1], 12.4, 7.6, 5)
    assert p.shape == (5, 5) and p[2, 2] == video[1, 8, 12]
    q = mxd.patch2d(video[0], 0.0, 0.0, 5)                       # corner: edge padding
    assert q[0, 0] == video[0, 0, 0] and q[2, 2] == video[0, 0, 0] and q[4, 4] == video[0, 2, 2]
    ex, valid = mxd.extract_patches(video, np.array([[1, 12.4, 7.6, 0]]), 5, 10.0, 2.0, K=3)
    assert ex.shape == (3, 1, 5, 5) and list(valid) == [True, False, False]
    assert np.allclose(ex[0, 0], (p - 10.0) / 2.0) and not ex[1:].any()
    try:
        mxd.patch2d(video[0], 3, 3, 4)
        raise AssertionError('even size accepted')
    except ValueError:
        pass


def test_background_point_avoids_boxes():
    rng = np.random.default_rng(1)
    frame = rng.normal(80, 3, (64, 96)).astype(np.float32)
    frame[30:33, 40:45] += 80                                    # the strongest spot is a target
    boxes = np.array([[40, 30, 5, 3, 1, 1]], float)
    for _ in range(20):
        x, y = mxd.background_point(rng, frame, boxes)
        assert mxd.point_box_distance(x, y, boxes).min() > 8.0
        assert 0 <= x < 96 and 0 <= y < 64


def test_exemplar_sample_shapes_fakes_and_frames():
    boxes, speed = _boxes()
    moving = mxd.moving_table('c', boxes, speed)
    rng = np.random.default_rng(2)
    video = rng.normal(80, 3, (100, 64, 128)).astype(np.float32)
    frames = np.arange(40, 80)
    n_fake = n_valid = 0
    for _ in range(200):
        s = mxd.exemplar_sample(rng, video, boxes, moving, frames, 16, 39, 80.0, 3.0, p_none=0.2, p_fake=0.25)
        assert s['ex'].shape == (16, 1, 39, 39) and s['ex'].dtype == np.float32
        assert s['ex_valid'].dtype == bool and s['ex_t'].dtype == np.int64
        v = s['ex_valid']
        assert (s['ex_t'][v] >= 0).all() and (s['ex_t'][~v] == -1).all()
        assert ((s['ex_t'][v] < 40) | (s['ex_t'][v] > 79)).all()
        assert not s['ex_fake'][~v].any() and not s['ex'][~v].any()
        n_fake += s['ex_fake'].sum()
        n_valid += v.sum()
    assert 0.15 < n_fake / n_valid < 0.35


def test_detection_bank_modes():
    cands = np.array([[5, 10, 10, 3.0], [6, 11, 10, 2.9], [50, 10, 10, 2.0], [90, 40, 40, 1.0],
                      [45, 70, 20, 5.0]], float)
    out = mxd.select_bank(cands, 40, 79, 'outside', 16)
    assert set(out[:, 0]) == {5, 90} and out[0, 3] == 3.0       # in-window rows excluded, NMS on 5/6
    assert set(mxd.select_bank(cands, 40, 79, 'causal', 16)[:, 0]) == {5}
    assert set(mxd.select_bank(cands, 40, 79, 'clip', 16)[:, 0]) == {45, 5, 50, 90}
    assert len(mxd.select_bank(cands, 40, 79, 'none', 16)) == 0
    assert len(mxd.select_bank(np.zeros((0, 4)), 40, 79, 'outside', 16)) == 0
    rng = np.random.default_rng(3)
    r = mxd.random_bank(rng, 100, 64, 128, 40, 79, 'outside', 16)
    assert r.shape == (16, 4) and ((r[:, 0] < 40) | (r[:, 0] > 79)).all()
    assert (r[:, 1] < 128).all() and (r[:, 2] < 64).all()
    assert len(mxd.random_bank(rng, 100, 64, 128, 0, 99, 'outside', 16)) == 0
    boxes, speed = _boxes()
    g = mxd.gt_bank(rng, mxd.moving_table('c', boxes, speed), 40, 79, 'causal', 16)
    assert len(g) and (g[:, 0] < 40).all()


def test_paper_label_rule_is_the_exported_mask():
    import gaussian_masks as gm
    import vtdata
    rng = np.random.default_rng(4)
    for _ in range(50):
        b = np.array([(rng.uniform(-3, 60), rng.uniform(-3, 40), rng.uniform(1, 12), rng.uniform(1, 12), 1, i)
                      for i in range(int(rng.integers(0, 6)))], float).reshape(-1, 6)
        lab = vtdata.gaussian_mask(b, 48, 64, 'paper_k6_floor')
        assert lab.dtype == np.float32 and np.array_equal(lab, gm.label(b, 48, 64, 6.0, 'floor'))
        assert np.allclose(vtdata.gaussian_mask(b, 48, 64, 'paper_k6_floor', quant=False), gm.heatmap(b, 48, 64, 6.0))
    assert not vtdata.gaussian_mask(np.zeros((0, 6)), 8, 8, 'paper_k6_floor').any()


def test_peak_rows():
    def fake_peaks(lg):
        return np.array([[3.0, 4.0, 2.0], [7.0, 1.0, -1.0]])
    r = mxd.peak_rows(np.zeros((8, 8)), 12, fake_peaks, 0.0)
    assert r.shape == (1, 4) and list(r[0]) == [12, 3, 4, 2]


def test_self_bank_is_label_free_and_outside_the_window():
    """The no-motion control: the strongest blobs of frames outside the window, from the
    image only; deterministic at inference, random frames in training."""
    rng = np.random.default_rng(0)
    n, H, W = 100, 60, 80
    video = rng.normal(80, 1.0, (n, H, W)).astype(np.float32)
    video[:, 20:23, 30:33] += 25                       # a parked target
    video[:, 40:43, 60:63] += 40                       # a hot spot, brighter
    for t in range(n):                                 # a mover
        x = 5 + (t % 60)
        video[t, 10:13, x:x + 3] += 30
    peaks = mxd.SelfPeaks(video, 16)
    rows = mxd.self_bank(peaks, n, 40, 79, 16)
    assert len(rows) and np.all((rows[:, 0] < 40) | (rows[:, 0] > 79))
    assert np.array_equal(rows, mxd.self_bank(peaks, n, 40, 79, 16))           # inference: deterministic
    assert rows[0, 1:3].tolist() == [61.0, 41.0]                                # the strongest blob first
    xy = {(int(r[1]), int(r[2])) for r in rows}
    assert (31, 21) in xy                                                       # the parked target is in it
    assert np.all(np.diff(rows[:, 3]) <= 1e-9)
    tr = mxd.self_bank(peaks, n, 40, 79, 16, rng=np.random.default_rng(1))
    assert len(tr) and np.all((tr[:, 0] < 40) | (tr[:, 0] > 79))
    assert len(mxd.self_bank(peaks, n, 0, n - 1, 16)) == 0                     # no frame outside
    e = mxd.self_sample(np.random.default_rng(2), video, peaks, np.arange(40, 80), 16, 39, 85.0, 47.0, p_none=0.0)
    assert e['ex'].shape == (16, 1, 39, 39) and e['ex_valid'].any() and not e['ex_fake'].any()
    assert np.all((e['ex_t'][e['ex_valid']] < 40) | (e['ex_t'][e['ex_valid']] > 79))
    none = mxd.self_sample(np.random.default_rng(2), video, peaks, np.arange(40, 80), 16, 39, 85.0, 47.0, p_none=1.0)
    assert not none['ex_valid'].any()


def _peaks_from_contrast(contrast, t=0, k=100):
    """Isolate plateau geometry; end-to-end image tests below keep real local_contrast."""
    previous = mxd.st.local_contrast
    try:
        mxd.st.local_contrast = lambda frame: np.asarray(contrast, np.float32)
        return mxd.frame_peaks(np.zeros_like(contrast), t, k)
    finally:
        mxd.st.local_contrast = previous


def test_frame_peaks_single_pixel_is_one_blob():
    frame = np.zeros((47, 61), np.float32)
    frame[23, 30] = 90
    # The real 3x3 inner mean creates nine equal maxima, all from this one point.
    peaks = mxd.frame_peaks(frame, 7, 100)
    assert np.array_equal(peaks, [[7., 30., 23., 10.]])
    assert peaks.dtype == np.float64


def test_frame_peaks_deduplicates_before_top100():
    frame = np.zeros((141, 171), np.float32)
    strong = [(20 + 30 * x, 20 + 30 * y) for y in range(3) for x in range(4)]
    for x, y in strong:
        frame[y, x] = 90
    frame[110, 140] = 45
    peaks = mxd.frame_peaks(frame, 0, 100)
    # Twelve 3x3 plateaus used to consume all 100 slots and hide the 13th blob.
    assert peaks.shape == (13, 4)
    assert np.array_equal(peaks[:12, 1:3], strong)
    assert np.array_equal(peaks[:12, 3], np.full(12, 10.))
    assert np.array_equal(peaks[-1], [0., 140., 110., 5.])


def test_frame_peaks_nonconvex_plateau_uses_an_actual_peak():
    contrast = np.zeros((31, 31), np.float32)
    contrast[10, 10:15] = contrast[14, 10:15] = 10
    contrast[10:15, 10] = contrast[10:15, 14] = 10
    # The ring centroid (12,12) is not a peak; nearest actual members tie.
    assert contrast[12, 12] == 0
    expected = np.array([[0., 12., 10., 10.]])
    for _ in range(3):
        peaks = _peaks_from_contrast(contrast)
        assert np.array_equal(peaks, expected)
        x, y = peaks[0, 1:3].astype(int)
        assert contrast[y, x] == peaks[0, 3]


def test_frame_peaks_separate_equal_plateaus_and_centroid_ties():
    contrast = np.zeros((40, 50), np.float32)
    contrast[8:10, 8:10] = 7
    contrast[8:10, 30:32] = 7
    # Four equally close actual members per plateau: smallest (y,x) wins.
    assert np.array_equal(_peaks_from_contrast(contrast), [[0., 8., 8., 7.], [0., 30., 8., 7.]])


def test_frame_peaks_diagonal_members_are_one_eight_connected_plateau():
    contrast = np.zeros((30, 30), np.float32)
    contrast[10, 10] = contrast[11, 11] = 8
    assert np.array_equal(_peaks_from_contrast(contrast), [[0., 10., 10., 8.]])


def test_frame_peaks_isolated_scores_and_sorting_are_unchanged():
    contrast = np.zeros((41, 61), np.float32)
    contrast[6, 8], contrast[6, 40], contrast[30, 20] = 4., 11., -7.
    expected = np.array([[3., 40., 6., 11.], [3., 20., 30., 7.], [3., 8., 6., 4.]])
    assert np.array_equal(_peaks_from_contrast(contrast, t=3), expected)
    assert np.array_equal(_peaks_from_contrast(contrast, t=3, k=2), expected[:2])


def test_frame_peaks_equal_scores_sort_representatives_in_row_major_order():
    contrast = np.zeros((40, 45), np.float32)
    contrast[5, 10:31] = contrast[25, 10:31] = 10
    contrast[5:26, 10] = 10                    # C-shaped component is labelled first
    contrast[10, 20] = 10                     # but its representative comes first
    assert np.array_equal(_peaks_from_contrast(contrast), [[0., 20., 10., 10.], [0., 10., 15., 10.]])
    assert np.array_equal(_peaks_from_contrast(contrast, k=1), [[0., 20., 10., 10.]])


def test_frame_peaks_floor_is_strict_and_empty_shape_is_preserved():
    contrast = np.zeros((41, 61), np.float32)
    contrast[10, 10] = 3
    contrast[10, 30] = -3
    contrast[30, 10] = np.nextafter(np.float32(3), np.float32(np.inf))
    peaks = _peaks_from_contrast(contrast)
    assert peaks.shape == (1, 4)
    assert np.array_equal(peaks[0, 1:3], [10., 30.])
    assert peaks[0, 3] == contrast[30, 10]
    assert _peaks_from_contrast(contrast, k=0).shape == (0, 4)
    assert mxd.frame_peaks(np.zeros((25, 25), np.float32), 0, 100).shape == (0, 4)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
