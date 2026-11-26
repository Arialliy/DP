"""Checks of motion_image.py, camera motion without annotations (numpy + OpenCV).

Run: python code/test_motion_image.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motion_image as mi  # noqa: E402
import profile_motion as pm  # noqa: E402
import stabilize as st  # noqa: E402
import test_stabilize as ts  # noqa: E402


def _with_movers(video, speed=3.0, amp=60):
    """Bright 3 x 3 targets that drive across the frame (no annotation is given)."""
    v = video.copy()
    n, H, W = v.shape
    for t in range(n):
        for k, y in enumerate((15, 50, 85)):
            x = int(10 + (speed * t + 40 * k) % (W - 20))
            if 2 <= y < H - 2:
                v[t, y - 1:y + 2, x - 1:x + 2] += amp
    return v


def test_pan_is_recovered_without_boxes():
    video, boxes, pairs = ts.scene()
    video = _with_movers(video)
    dx = []
    for t in range(0, len(video) - 1, 7):
        H, n, inl = mi.image_homography(pm.to_u8(video[t]), pm.to_u8(video[t + 1]))
        dx.append(H[0, 2])
        assert n >= 12
    assert abs(np.median(dx) + ts.PAN) < 0.1, np.median(dx)


def test_blob_mask_covers_small_targets_only():
    video, _, _ = ts.scene()
    frame = _with_movers(video)[10]
    m = mi.blob_mask(frame)
    assert m.dtype == np.uint8 and set(np.unique(m)) <= {0, 255}
    assert (m == 0).mean() < 0.1                            # most of the frame keeps its corners
    for y in (15, 50, 85):                                  # every driving target is masked
        xs = np.flatnonzero(_with_movers(video)[10, y] - video[10, y] > 30)
        assert len(xs) and np.all(m[y, xs] == 0)


def test_static_camera_with_movers_stays_still():
    rng = np.random.default_rng(3)
    base = rng.normal(80, 4, (96, 160)).astype(np.float32)
    video = _with_movers(np.repeat(base[None], 30, 0) + rng.normal(0, 0.5, (30, 96, 160)).astype(np.float32))
    for t in range(0, 29, 5):
        H, _, _ = mi.image_homography(pm.to_u8(video[t]), pm.to_u8(video[t + 1]))
        pts = mi.grid_points(96, 160)
        assert np.linalg.norm(st.apply_h(H, pts) - pts, axis=1).max() < 0.2


def test_compare_measures_window_error():
    eye = np.repeat(np.eye(3)[None], 79, 0)
    pair, win = mi.compare(eye, eye, (100, 120))
    assert np.allclose(pair, 0) and np.allclose(win, 0)
    shifted = np.repeat(st.translation(0.1, 0)[None], 79, 0)
    pair, win = mi.compare(shifted, eye, (100, 120))
    assert np.allclose(pair, 0.1) and np.isclose(win.max(), 2.0, atol=1e-6)   # 20 frames from the reference


def test_main_writes_homographies_and_compares():
    import fake_dataset
    import vtdata
    video, boxes, pairs = ts.scene()
    saved = vtdata.DATA_ROOT, vtdata.ANN_DIR, vtdata.OFFICIAL_LISTS, os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        try:
            root = os.path.join(d, 'RGB-T')
            fake_dataset.write(root, {'pan_1': (video, boxes, 'train2017')})
            open(os.path.join(root, 'train.txt'), 'w').write('pan_1\n')
            open(os.path.join(root, 'test.txt'), 'w').write('\n')
            vtdata.DATA_ROOT, vtdata.ANN_DIR, vtdata.OFFICIAL_LISTS = root, None, root
            vtdata._ANN_CACHE.clear()
            os.chdir(d)
            os.makedirs('results/profile/homographies')
            ims = [pm.to_u8(vtdata.Scene('pan_1').image(i)) for i in range(len(video))]
            boxed = np.array([pm.pair_homography(ims[i], ims[i + 1], boxes[i])[0] for i in range(len(ims) - 1)])
            np.save('results/profile/homographies/pan_1.npy', boxed)        # box-masked, as profile_motion.py
            rows = mi.main(['--nproc', '1'])
            H = np.load('results/motion/homographies/pan_1.npy')
            assert H.shape == pairs.shape and abs(np.median(H[:, 0, 2]) + ts.PAN) < 0.1
            # this synthetic pan accumulates about 3 px of LK error for either estimator (vs the
            # true pan); the two estimators differ by about 1 px within a window
            assert rows[0]['window_diff_p95'] < 1.5 and os.path.exists('results/motion/clips.csv')
            # against the true pan, the annotation-free estimate is no worse than the box-masked one
            err_image = mi.compare(H, pairs, video.shape[1:])[1]
            err_boxed = mi.compare(boxed, pairs, video.shape[1:])[1]
            assert np.percentile(err_image, 95) <= np.percentile(err_boxed, 95) + 0.5
            chain = st.load_chain('pan_1', len(video), directory='results/motion/homographies')
            assert chain.shape == (len(video), 3, 3)
        finally:
            vtdata.DATA_ROOT, vtdata.ANN_DIR, vtdata.OFFICIAL_LISTS = saved[:3]
            vtdata._ANN_CACHE.clear()
            os.chdir(saved[3])


def test_default_folder_is_the_annotation_free_one():
    assert st.HOMOGRAPHY_DIR == os.environ.get('HOMOGRAPHY_DIR', 'results/motion/homographies')


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
