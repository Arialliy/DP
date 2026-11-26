"""Checks of M1, miss_recovery.py (numpy only).

Synthetic clips: cars drive and park; the 'detector' (cached logits) finds every car from
frame 60 on and none before (a weak temporal start). The clutter is round hot spots of the
cars' brightness. Single frames, taught by the detector's own later detections, must recover
the early misses and not admit more hot spots than the budget.
Run: python code/test_miss_recovery.py
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import miss_recovery as m1  # noqa: E402
import test_exemplar_diag as te  # noqa: E402

H, W, N = 96, 160, 150


def _args(**kw):
    a = dict(thresholds=[1e-5, 0.5], teacher_th=0.5, k=16, patch=11, shift=2, per_frame=60, r_pos=3.0, r_neg=8.0,
             budgets=[0.25, 1.0], early=60, seqlen=40)
    a.update(kw)
    return argparse.Namespace(**a)


def clip(seed, amp):
    """Cars (driving and parked) of contrast amp, round hot spots of the same brightness,
    and logits that detect every car from frame 60 on (none before) and no hot spot."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    base = 80 + 0.05 * xx
    for (x, y) in te.HOT:
        base += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 2.0 ** 2))
    video, boxes = [], []
    logits = np.full((N, H, W), -20.0, np.float32)
    for t in range(N):
        img = base + rng.normal(0, 1.5, (H, W))
        b = []
        for i, y in enumerate((8, 25, 80)):
            x = 5 + (t + 37 * i) % (W - 10)
            te._car(img, x, y, amp)
            b.append((round(x) - 3, y - 1, 7, 3, 1, 100 + i))
        for j, (x, y) in enumerate(((35, 40), (75, 60), (115, 50))):
            te._car(img, x, y, amp)
            b.append((x - 3, y - 1, 7, 3, 1, 200 + j))
        if t >= 60:
            for bx, by, bw, bh, *_ in b:
                logits[t, int(by + bh // 2), int(bx + bw // 2)] = 8.0
        video.append(img.astype(np.float32))
        boxes.append(np.array(b, float))
    return np.stack(video), boxes, logits


def run():
    args = _args()
    targets, cands = [], []
    for k, (name, amp) in enumerate((('DJI_8001_1', 20), ('DJI_8002_1', 35), ('DJI_8003_1', 50))):
        v, b, lg = clip(k, amp)
        t, c, w = m1.process_clip(name, v, b, lg, args)
        targets += t
        cands += c
    m1.loro_scores(targets, cands)
    frames = {c: N for c in ('DJI_8001_1', 'DJI_8002_1', 'DJI_8003_1')}
    return targets, cands, frames, w


def test_early_misses_are_recovered_and_hot_spots_are_not():
    targets, cands, frames, w = run()
    early = [r for r in targets if r['t'] < 60]
    assert early and not any(r['hit_1e-05'] for r in early)            # the detector misses them all
    assert all(r['hit_1e-05'] for r in targets if r['t'] >= 60)
    thr = m1.thresholds(cands, frames, 0.25, 'pooled')
    s = m1.summarize(targets, thr, 1e-5, 60)
    assert s['all']['misses'] == len(early)
    assert s['frames < 60']['recovered_fraction'] > 0.8, s['frames < 60']
    assert s['frames < 60']['recall_with_recovery'] > 0.8
    # the admitted clutter is the budget
    for c in frames:
        n_admit = sum(r['score'] >= thr[c] for r in cands if r['clip'] == c)
        assert n_admit <= 0.25 * sum(frames.values())


def test_teacher_is_outside_the_window_and_from_confident_detections():
    args = _args()
    v, b, lg = clip(0, 30)
    targets, cands, wins = m1.process_clip('DJI_8001_1', v, b, lg, args)
    assert wins[0]['teacher'] > 0                                      # window 0 is taught by later frames
    lg2 = np.full_like(lg, -20.0)                                      # a detector that finds nothing
    t2, _, w2 = m1.process_clip('DJI_8001_1', v, b, lg2, args)
    assert all(w['teacher'] == 0 for w in w2) and all(r['teacher'] == 0 for r in t2)
    m1.loro_scores(t2, [])
    assert all(r['score'] == -np.inf for r in t2)                      # nothing to learn from: nothing recovered


def test_box_hits_follow_the_mask():
    prob = np.zeros((30, 30), np.float32)
    boxes = np.array([(10, 10, 3, 3, 1, 1), (20, 20, 3, 3, 1, 2)], float)
    prob[11, 12] = 0.6                         # on the first box's mask
    prob[19, 19] = 0.7                         # next to the second box's mask (3 x 3 neighbourhood)
    h = m1.box_hits(prob, boxes, 30, 30, [1e-5, 0.5, 0.65])
    assert h.tolist() == [[True, True, False], [True, True, True]]


def test_command_line_on_a_dataset_on_disk():
    import json
    import tempfile
    import fake_dataset
    import vtdata
    saved = vtdata.DATA_ROOT, vtdata.ANN_DIR, vtdata.OFFICIAL_LISTS, os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        try:
            data = {name: clip(k, amp) for k, (name, amp) in enumerate((('DJI_8001_1', 20), ('DJI_8002_1', 35),
                                                                         ('DJI_8003_1', 50)))}
            fake_dataset.write(d, {k: (v, b, 'test2017') for k, (v, b, _) in data.items()})
            vtdata.DATA_ROOT, vtdata.ANN_DIR, vtdata.OFFICIAL_LISTS = d, None, d
            vtdata._ANN_CACHE.clear()
            os.chdir(d)
            os.makedirs('cache/preds/fake')
            for k, (_, _, lg) in data.items():
                np.save(f'cache/preds/fake/{k}.npy', lg.astype(np.float16))
            res = m1.main(['--run', 'fake', '--split', ','.join(data), '--nproc', '1', '--per-frame', '60'])
            s = res['pooled budget 0.25 / official th 1e-05']
            assert s['frames < 60']['recovered_fraction'] > 0.8
            summ = json.load(open('results/miss_recovery/fake/' + ','.join(data).replace('/', '_') + '/summary.json'))
            assert summ['frames'] == {k: N for k in data}
            assert os.path.exists('results/miss_recovery/fake/' + ','.join(data) + '/windows.csv')
        finally:
            vtdata.DATA_ROOT, vtdata.ANN_DIR, vtdata.OFFICIAL_LISTS = saved[:3]
            vtdata._ANN_CACHE.clear()
            os.chdir(saved[3])


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
