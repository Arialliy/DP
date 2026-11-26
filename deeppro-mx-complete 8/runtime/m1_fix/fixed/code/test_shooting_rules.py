"""evaluate.shooting_rules against a transcription of the authors' ShootingRules.forward
(numpy only; the loop below is ShootingRules.py with skimage's 8-connected labelling
replaced by scipy's, which gives the same components).

Run: python code/test_shooting_rules.py
"""
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as ev  # noqa: E402


def official(output, target, th):
    """ShootingRules.forward for one frame, with its numpy slicing; the comparison in float32,
    as the authors' float32 map with a threshold under NumPy 1."""
    output = output.astype(np.float32)
    th = np.float32(th)
    output[np.where(output < th)] = 0
    output[np.where(output >= th)] = 1
    lab, n = ndimage.label(target, structure=np.ones((3, 3)))
    true = 0
    box = np.ones(output.shape)
    for k in range(1, n + 1):
        flag = 0
        for i, j in np.argwhere(lab == k):
            box[i - 4:i + 4 + 1, j - 4:j + 4 + 1] = 0
            if output[i - 1:i + 1 + 1, j - 1:j + 1 + 1].sum() >= 1:
                flag = 1
        true += flag
    return int(np.count_nonzero(output * box)), true, n


def test_official_edges_match_shooting_rules():
    rng = np.random.default_rng(0)
    differ_clipped = 0
    for trial in range(400):
        H, W = 40, 50
        mask = np.zeros((H, W))
        prob = rng.random((H, W)) * (rng.random((H, W)) > 0.95)
        for _ in range(3):
            near_edge = trial % 2 == 0
            y = int(rng.integers(0, 5)) if near_edge else int(rng.integers(0, H - 2))
            x = int(rng.integers(0, W - 2)) if trial % 4 else int(rng.integers(0, 5))
            mask[y:y + 2, x:x + 2] = 1
            if rng.random() < 0.6:
                prob[min(y + int(rng.integers(0, 3)), H - 1), x] = 0.9
        for th in (0.5, 0.99):
            ref = official(prob, mask, th)
            assert ev.shooting_rules(prob, mask, th) == ref, (trial, th)
            differ_clipped += ev.shooting_rules(prob, mask, th, 'clipped') != ref
    assert differ_clipped > 0              # the edge quirk matters on these frames


def test_inner_targets_are_the_same_either_way():
    rng = np.random.default_rng(1)
    for _ in range(100):
        mask = np.zeros((30, 30))
        y, x = rng.integers(6, 22, 2)
        mask[y:y + 3, x:x + 3] = 1
        prob = rng.random((30, 30))
        prob[:5], prob[:, :5] = 0, 0                 # nothing near the top or left edge
        ref = official(prob, mask, 0.5)
        assert ev.shooting_rules(prob, mask, 0.5) == ev.shooting_rules(prob, mask, 0.5, 'clipped') == ref


def test_sweep_equals_every_threshold_of_test_py():
    """All 27 thresholds of test.py at once, with values exactly at thresholds, 0 and 1."""
    rng = np.random.default_rng(2)
    ths = ev.TH_SEG
    for trial in range(60):
        H, W = 30, 36
        mask = np.zeros((H, W))
        for _ in range(int(rng.integers(0, 5))):
            y, x = int(rng.integers(0, H - 2)), int(rng.integers(0, W - 2))
            mask[y:y + int(rng.integers(1, 4)), x:x + int(rng.integers(1, 4))] = 1
        prob = (rng.random((H, W)) ** 6).astype(np.float32)
        prob[rng.random((H, W)) < 0.05] = 0.5                     # exactly at test.py's threshold_eval
        prob[rng.random((H, W)) < 0.03] = 1.0
        prob[rng.random((H, W)) < 0.03] = 0.0
        prob[rng.random((H, W)) < 0.03] = np.float32(1e-6)
        f, t, n, inter, union = ev.shooting_rules_sweep(prob, mask, ths)
        for j, th in enumerate(ths):
            assert (f[j], t[j], n) == official(prob, mask, th), (trial, th)
            pred = prob > th                                      # test.py's IoU comparison
            assert inter[j] == np.count_nonzero(pred & (mask > 0)) and union[j] == np.count_nonzero(pred | (mask > 0))


def test_official_auc_is_sklearns():
    fa = np.array([1.0, 0.3, 0.1, 1e-3, 1e-5, 0.0])
    pd = np.array([1.0, 0.95, 0.9, 0.8, 0.5, 0.1])
    try:
        from sklearn.metrics import auc
    except ImportError:
        auc = None
    a = ev.official_auc(fa, pd)
    assert np.isclose(a, np.sum((fa[:-1] - fa[1:]) * (pd[:-1] + pd[1:]) / 2))
    if auc is not None:
        assert np.isclose(a, auc(fa, pd))


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
