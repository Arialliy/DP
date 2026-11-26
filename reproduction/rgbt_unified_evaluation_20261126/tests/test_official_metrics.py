"""CPU count-for-count checks against the author's unmodified ShootingRules.

Run in the compatible NumPy 1.23.5 runtime without GPU access:
    python -m unittest discover -s tests -p test_official_metrics.py -v
DEEPPRO_ROOT optionally overrides the repository path for bind mounts.
"""

import importlib.util
import os
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image
from sklearn.metrics import auc


RUN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(os.environ.get('DEEPPRO_ROOT', RUN_ROOT / 'official'))
sys.path.insert(0, str(RUN_ROOT))
import official_metrics as metrics


def load_original():
    path = REPO_ROOT / 'ShootingRules.py'
    if not path.is_file():
        raise RuntimeError('The original ShootingRules.py is required: ' + str(path))
    spec = importlib.util.spec_from_file_location('original_shooting_rules_oracle', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ShootingRules()


ORIGINAL = load_original()


def original_score(prob, mask):
    target = (mask > 0).astype(np.float64)[None]
    rows = np.asarray([ORIGINAL(prob[None], target, threshold)
                       for threshold in metrics.THRESHOLDS], dtype=np.int64)
    prediction = prob > .5
    foreground = mask > 0
    return {
        'false_pixels': rows[:, 0],
        'true_targets': rows[:, 1],
        'targets': rows[:, 2],
        'intersection': int(np.sum(prediction & foreground)),
        'union': int(np.sum(prediction | foreground)),
        'pixels': prob.size,
        'frames': 1,
    }


def edge_probabilities():
    values = np.asarray(metrics.THRESHOLDS, dtype=np.float32)
    return np.unique(np.clip(np.concatenate([
        values,
        np.nextafter(values, np.float32(-np.inf)),
        np.nextafter(values, np.float32(np.inf)),
    ]), 0, 1)).astype(np.float32)


class ExactOfficialMetrics(unittest.TestCase):
    def assert_counts_equal(self, prob, mask):
        prob_before, mask_before = prob.copy(), mask.copy()
        expected = original_score(prob, mask)
        actual = metrics.score_frame(prob, mask)
        for key in ('false_pixels', 'true_targets', 'targets'):
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
            self.assertEqual(actual[key].dtype, np.int64)
        for key in ('intersection', 'union', 'pixels', 'frames'):
            self.assertEqual(actual[key], expected[key], key)
        np.testing.assert_array_equal(prob, prob_before)
        np.testing.assert_array_equal(mask, mask_before)
        return actual

    def test_scalar_float32_boundaries(self):
        values = edge_probabilities()
        for threshold, boundary in zip(metrics.THRESHOLDS, metrics._BOUNDARIES):
            with self.subTest(threshold=threshold):
                np.testing.assert_array_equal(values >= threshold, values >= boundary)
        probability = np.resize(values, (17, 19))
        self.assert_counts_equal(probability, np.zeros(probability.shape, dtype=np.uint8))

    def test_threshold_zero_and_negative_slice_quirks(self):
        # At t=0 even zero probabilities become 1. A target at (0,0) has
        # an empty detection slice, but its exclusion slice wraps strangely.
        for size, excluded in ((7, 4), (8, 1), (9, 0), (17, 0)):
            mask = np.zeros((size, size), dtype=np.uint8)
            mask[0, 0] = 255
            actual = self.assert_counts_equal(np.zeros(mask.shape, dtype=np.float32), mask)
            self.assertEqual(actual['false_pixels'][0], size * size - excluded)
            self.assertEqual(actual['true_targets'][0], 0)
            self.assertEqual(actual['targets'][0], 1)
            np.testing.assert_array_equal(actual['false_pixels'][1:], 0)

    def test_small_shapes_all_borders(self):
        rng = np.random.RandomState(46)
        values = edge_probabilities()
        for height in range(1, 11):
            for width in range(1, 11):
                with self.subTest(shape=(height, width)):
                    mask = (rng.random_sample((height, width)) < .18).astype(np.uint8) * 128
                    mask[0, 0] = 1
                    mask[-1, -1] = 255
                    probability = rng.choice(values, size=mask.shape).astype(np.float32)
                    self.assert_counts_equal(probability, mask)

    def test_eight_connected_components_and_strict_iou(self):
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[1, 1], mask[2, 2], mask[5, 5], mask[7, 2] = 255, 128, 1, 255
        probability = np.full(mask.shape, .5, dtype=np.float32)
        actual = self.assert_counts_equal(probability, mask)
        np.testing.assert_array_equal(actual['targets'], 3)
        self.assertEqual(actual['intersection'], 0)
        self.assertEqual(actual['union'], 4)
        self.assertEqual(actual['true_targets'][16], 3)
        self.assertEqual(actual['true_targets'][17], 0)

    def test_random_full_size_mask_and_saturated_probabilities(self):
        rng = np.random.RandomState(123)
        mask = np.zeros((512, 640), dtype=np.uint8)
        mask[0, :7], mask[-1, -7:] = 255, 128
        mask[250:256, 320:328] = 1
        mask[3, 3], mask[4, 4], mask[30, 637] = 255, 255, 255
        for fill in (0., 1.):
            self.assert_counts_equal(np.full(mask.shape, fill, dtype=np.float32), mask)
        probability = rng.uniform(0, 1, mask.shape).astype(np.float32)
        probability[:7, :7] = np.resize(edge_probabilities(), (7, 7))
        self.assert_counts_equal(probability, mask)

    def test_real_rgbt_png_masks(self):
        rng = np.random.RandomState(73)
        dataset = Path(os.environ.get('VT_ROOT', RUN_ROOT.parent.parent / 'datasets' / 'RGB-T'))
        for clip in ('DJI_0049_2', 'DJI_0237_2', 'DJI_0303_2'):
            paths = sorted((dataset / 'segmentations' / clip).glob('*.png'))
            self.assertTrue(paths, 'Missing required real mask fixture: ' + clip)
            for path in (paths[0], paths[len(paths) // 2]):
                with self.subTest(mask=str(path)):
                    mask = np.asarray(Image.open(path), dtype=np.uint8)
                    if mask.ndim == 3:
                        mask = mask[..., 0]
                    # Cover both very small probabilities and an active
                    # foreground while retaining real connected geometry.
                    probability = (10 ** rng.uniform(-22, 0, mask.shape)).astype(np.float32)
                    probability[mask > 0] = rng.choice(edge_probabilities(), size=np.count_nonzero(mask))
                    self.assert_counts_equal(probability, mask)

    def test_global_aggregation_and_unaugmented_auc(self):
        rng = np.random.RandomState(97)
        scores, expected = [], []
        for shape in ((17, 19), (11, 9), (25, 12)):
            probability = rng.choice(edge_probabilities(), size=shape).astype(np.float32)
            mask = (rng.random_sample(shape) > .93).astype(np.uint8) * 255
            scores.append(metrics.score_frame(probability, mask))
            expected.append(original_score(probability, mask))
        total = metrics.aggregate_results(iter(scores))
        nested = metrics.aggregate_results([metrics.aggregate_clip(scores[:2]), scores[2]])
        for key in ('false_pixels', 'true_targets', 'targets'):
            reference = sum(row[key] for row in expected)
            np.testing.assert_array_equal(total[key], reference)
            np.testing.assert_array_equal(nested[key], reference)
        pixels = sum(row['pixels'] for row in expected)
        pd = sum(row['true_targets'] for row in expected) / sum(row['targets'] for row in expected)
        fa = sum(row['false_pixels'] for row in expected) / pixels
        np.testing.assert_array_equal(total['pd'], pd)
        np.testing.assert_array_equal(total['fa'], fa)
        self.assertEqual(total['auc'], float(auc(fa, pd)))
        self.assertEqual(total['iou'], sum(row['intersection'] for row in expected) / sum(row['union'] for row in expected))
        self.assertEqual(total['iou_at_0_5'], total['iou'])
        self.assertEqual(total['frames'], 3)
        self.assertEqual(total['pixels'], pixels)
        self.assertEqual(len(total['thresholds']), 27)
        np.testing.assert_array_equal(total['thresholds'], metrics.THRESHOLDS)

    def test_empty_targets_and_invalid_inputs(self):
        score = self.assert_counts_equal(np.zeros((9, 9), dtype=np.float32), np.zeros((9, 9), dtype=np.uint8))
        total = metrics.aggregate_results([score])
        self.assertTrue(np.isnan(total['pd']).all())
        self.assertTrue(np.isnan(total['auc']))
        self.assertTrue(np.isnan(total['iou']))
        with self.assertRaises(ValueError):
            metrics.aggregate_results([])
        with self.assertRaises(TypeError):
            metrics.score_frame(np.zeros((2, 2), dtype=np.float64), np.zeros((2, 2)))
        for value in (np.nan, np.inf, -0.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                metrics.score_frame(np.full((2, 2), value, dtype=np.float32), np.zeros((2, 2)))


if __name__ == '__main__':
    unittest.main()
