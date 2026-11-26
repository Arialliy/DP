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
    """ShootingRules.forward for one frame, with its numpy slicing."""
    output = output.copy()
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


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
