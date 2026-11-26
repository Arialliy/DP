"""Checks of the M0 motion-exemplar diagnostic on synthetic clips (numpy only).

Run: python code/test_exemplar_diag.py
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exemplar_diag as ed  # noqa: E402
import stabilize as st  # noqa: E402
import test_stabilize as ts  # noqa: E402

H, W, N = 96, 160, 120


def _args(**kw):
    a = dict(seqlen=40, coords='raw', patch=11, k=16, step=4, shift=2, seed=0, min_pos=20, delta=0.10, frac=0.6,
             key_clips=['syn'])
    a.update(kw)
    return argparse.Namespace(**a)


def _car(img, cx, cy, amp):
    """A 3 x 7 car: bright body with a cooler middle (a shape, not a blob)."""
    x0, y0 = int(round(cx)) - 3, int(round(cy)) - 1
    img[y0:y0 + 3, x0:x0 + 7] += amp
    img[y0 + 1, x0 + 2:x0 + 5] -= 0.6 * amp


HOT = [(20, 70), (60, 15), (100, 88), (140, 30), (130, 72), (12, 45), (95, 40), (55, 88), (150, 60), (80, 12)]


def scene(seed=0, hot_shape='round'):
    """Static camera. Three cars drive at 1 px/frame and four are parked for the whole clip.
    Ten unannotated ground hot spots span the cars' contrast (some dimmer, some brighter),
    so contrast alone cannot separate them. hot_shape 'round': Gaussian blobs, which do not
    look like cars; 'car': car-shaped spots (unlabelled parked cars), which exemplars
    cannot separate either."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    base = 80 + 0.05 * xx + 0.03 * yy
    hot = HOT
    for (x, y), amp in zip(hot, rng.permutation(np.linspace(8, 60, len(hot)))):
        if hot_shape == 'round':
            base += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 3.0 ** 2))
        else:
            _car(base, x, y, amp * 0.8)
    parked = [(35, 40), (75, 60), (115, 50), (150, 88)]
    parked_amp = [14, 22, 30, 38]                       # parked cars differ in contrast as well
    lanes = [8, 25, 80]
    video, boxes, speed = [], [], {}
    for t in range(N):
        img = base + rng.normal(0, 1.5, (H, W))
        b = []
        for i, y in enumerate(lanes):
            x = 5 + (t + 37 * i) % (W - 10)
            _car(img, x, y, 25)
            b.append((round(x) - 3, y - 1, 7, 3, 1, 100 + i))
            speed[('syn', t, 100 + i)] = 1.0
        for j, (x, y) in enumerate(parked):
            _car(img, x, y, parked_amp[j])
            b.append((x - 3, y - 1, 7, 3, 1, 200 + j))
            speed[('syn', t, 200 + j)] = 0.0
        video.append(img.astype(np.float32))
        boxes.append(np.array(b, float))
    return np.stack(video), boxes, speed, hot, parked


def test_exemplars_separate_parked_cars_from_hot_spots():
    video, boxes, speed, hot, parked = scene()
    args = _args()
    pos, neg, win = ed.process_clip('syn', video, boxes, speed, None, args)
    s = ed.summarize(pos, neg, win, args)
    assert s['positives_with_exemplars'] >= 4 * 30 and s['negatives_with_exemplars'] >= 4 * 8
    assert 0.2 < s['auc_contrast'] < 0.8 and s['auc_lr_base'] < 0.85, s   # contrast cannot separate
    assert s['auc_ncc_L'] > 0.9 and s['auc_ncc_med'] > 0.9, s
    assert s['auc_lr_full'] > 0.95 and s['gain'] > 0.15 and s['eligible'], s
    verdict = ed.decide({'syn': s}, args)
    assert verdict['go'] and verdict['key_clips'] == {'syn': True}
    # every negative is a hot spot or texture, never a car; positives are only parked cars
    for r in pos:
        assert r['track'] >= 200 and r['rel_speed'] < 0.2
    # exemplars come from frames outside each window
    for w in win:
        assert all(t < w['window_start'] or t >= w['window_end'] for t in w['exemplar_frames'])
        assert w['exemplars'] > 0


def test_car_shaped_hot_spots_give_no_go():
    """The control: if the clutter looks like the targets, exemplars add nothing."""
    video, boxes, speed, _, _ = scene(hot_shape='car')
    args = _args()
    pos, neg, win = ed.process_clip('syn', video, boxes, speed, None, args)
    s = ed.summarize(pos, neg, win, args)
    assert s['eligible'] and s['gain'] < 0.10, s
    assert not ed.decide({'syn': s}, args)['go']


def test_heldout_logistic_auc():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(200), np.zeros(200)]
    fold = np.arange(400) % 2
    informative = y * 2.0 + rng.normal(size=400)
    noise = rng.normal(size=400)
    a = ed.heldout_auc(np.c_[noise], y, fold)
    b = ed.heldout_auc(np.c_[noise, informative], y, fold)
    assert 0.4 < a < 0.6 and b > 0.85
    flipped = ed.heldout_auc(np.c_[-informative], y, fold)      # a reversed feature is still information
    assert flipped > 0.85
    assert np.isnan(ed.heldout_auc(np.c_[noise], np.r_[np.ones(200), np.zeros(200)], np.r_[np.zeros(200),
                                                                                           np.ones(200)]))


def test_no_moving_targets_means_no_exemplars_and_no_decision():
    video, boxes, speed, _, _ = scene()
    speed = {k: 0.0 for k in speed}                     # nothing moves
    args = _args()
    pos, neg, win = ed.process_clip('syn', video, boxes, speed, None, args)
    s = ed.summarize(pos, neg, win, args)
    assert s['positives_with_exemplars'] == 0 and s['windows_without_exemplars'] == len(win)
    assert not s['eligible'] and not ed.decide({'syn': s}, args)['go']


def test_detection_exemplars_are_used():
    video, boxes, speed, _, _ = scene()
    det = []                                            # 'detections' on the moving cars only
    for t, b in enumerate(boxes):
        for row in b[:3]:
            det.append((t, row[0] + 3, row[1] + 1, 5.0))
    args = _args()
    pos, neg, win = ed.process_clip('syn', video, boxes, speed, None, args, det=np.array(det))
    s = ed.summarize(pos, neg, win, args)
    assert s['auc_ncc_det'] > 0.9 and all(w['det_exemplars'] > 0 for w in win)


def test_warped_windows_follow_a_panning_camera():
    video, boxes, pairs = ts.scene()                    # 1.5 px/frame pan, car parked at frames 60-119
    chain = st.chain_from_pairs(pairs)
    speed = {('pan', t, 7): 0.0 for t in range(len(video))}
    args = _args(coords='warp', min_pos=1)
    pos, neg, win = ed.process_clip('pan', video, boxes, speed, chain, args)
    assert all(w['coords'] == 'warp' for w in win)
    assert pos and all(np.isfinite(r['contrast']) and r['contrast'] > 15 for r in pos)   # +40 car, sharp in L
    assert all(r['exemplars'] == 0 for r in pos)       # nothing moves, so no exemplars
    raw_pos, _, _ = ed.process_clip('pan', video, boxes, speed, None, _args(coords='raw', min_pos=1))
    assert np.nanmean([r['contrast'] for r in raw_pos]) < np.nanmean([r['contrast'] for r in pos])


def test_ncc_matrix():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(3, 5, 5))
    Q = np.stack([E[1] * 4 + 7, -E[2], np.ones((5, 5))])
    c = ed.ncc_matrix(Q, E)
    assert np.isclose(c[0, 1], 1) and np.isclose(c[1, 2], -1) and np.allclose(c[2], 0)
    assert ed.ncc_matrix(np.zeros((0, 5, 5)), E).shape == (0, 3)


def test_command_line_on_a_dataset_on_disk():
    """main() end to end: JPEG frames, COCO json with 'tracking_id', targets.csv."""
    import json
    import tempfile
    import fake_dataset
    import vtdata
    video, boxes, speed, _, _ = scene()
    old_root, old_ann, old_cwd = vtdata.DATA_ROOT, vtdata.ANN_DIR, os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        try:
            fake_dataset.write(d, {'syn': (video, boxes, 'train2017')}, track_key='tracking_id')
            fake_dataset.write_targets(os.path.join(d, 'targets.csv'), speed, {'syn': boxes})
            vtdata.DATA_ROOT, vtdata.ANN_DIR = d, None
            vtdata._ANN_CACHE.clear()
            os.chdir(d)
            summary = ed.main(['--clips', 'syn', '--targets', 'targets.csv', '--coords', 'raw', '--nproc', '1',
                               '--min-pos', '10', '--key-clips', 'syn', '--out', 'out'])
            s = summary['per_clip']['syn']
            assert s['auc_ncc_L'] > 0.9 and s['gain'] > 0.15 and summary['decision']['go']
            assert json.load(open('out/summary.json'))['decision']['go']
            assert os.path.exists('out/positives.csv') and os.path.exists('out/negatives.csv')
        finally:
            os.chdir(old_cwd)
            vtdata.DATA_ROOT, vtdata.ANN_DIR = old_root, old_ann
            vtdata._ANN_CACHE.clear()


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
