"""CPU-only exhaustive-oracle tests for the bounded-memory exact Fa diagnostic."""

import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exact_fa import DEFAULT_FA_BUDGETS, DEFAULT_PIXELS_TOTAL, ExactFaAccumulator


def exhaustive_points(background, targets, pixels):
    """All distinct change points, including every threshold just above a score."""
    scores = np.concatenate([background, targets[np.isfinite(targets)], np.array([0, 1], np.float32)])
    candidates = np.unique(np.concatenate([scores, np.nextafter(scores, np.float32(np.inf))]))
    candidates = candidates[(candidates >= 0) & (candidates <= 1)]
    return [dict(threshold=float(t), false_pixels=int(np.count_nonzero(background >= t)),
                 true_targets=int(np.count_nonzero(targets >= t))) for t in candidates]


class ExactFaTests(unittest.TestCase):
    def run_stream(self, background, targets, caps, *, chunk=19, buffer=13, pixels=None):
        background = np.asarray(background, np.float32)
        targets = np.asarray(targets, np.float32)
        pixels = int(pixels if pixels is not None else max(1, len(background)))
        collector = ExactFaAccumulator(pixels_total=pixels, fa_budgets=[cap / pixels for cap in caps],
                                       expected_targets=len(targets), buffer_size=buffer)
        if len(background):
            for start in range(0, len(background), chunk):
                part = background[start:start + chunk]
                count = len(part) + (pixels - len(background) if start == 0 else 0)
                collector.update_scores(part, targets if start == 0 else np.empty(0, np.float32),
                                        pixels=count)
        else:
            collector.update_scores(background, targets, pixels=pixels)
        result = collector.finalize()
        self.assertEqual([r['fp_budget'] for r in result['budgets']], caps)
        self.assertLessEqual(result['memory']['retained_background_scores'], max(caps) + 1)
        self.assertLessEqual(result['memory']['peak_buffered_scores'], buffer)
        np.testing.assert_array_equal(collector.target_maxima, targets)
        json.dumps(result, allow_nan=False)
        return result

    def check_exhaustive(self, background, targets, result):
        background, targets = np.asarray(background, np.float32), np.asarray(targets, np.float32)
        exhaustive = exhaustive_points(background, targets, result['pixels'])
        for row in result['budgets']:
            cap = row['fp_budget']
            feasible = [p for p in exhaustive if p['false_pixels'] <= cap]
            self.assertEqual(row['feasible'], bool(feasible))
            if feasible:
                expected = min(feasible, key=lambda p: p['threshold'])
                actual = row['best_point']
                self.assertEqual(actual['threshold'], expected['threshold'])
                self.assertEqual(actual['false_pixels'], expected['false_pixels'])
                self.assertEqual(actual['true_targets'], max(p['true_targets'] for p in feasible))
                self.assertEqual(row['lower_fa_point'], actual)
                above = [p['false_pixels'] for p in exhaustive if p['false_pixels'] > cap]
                self.assertEqual(row['upper_fa_point'] is not None, bool(above))
                if above:
                    self.assertEqual(row['upper_fa_point']['false_pixels'], min(above))
            else:
                self.assertIsNone(row['best_point'])
                self.assertIsNone(row['lower_fa_point'])
                self.assertEqual(row['upper_fa_point']['threshold'], 1.)
                self.assertIn('scores_equal_one', row['reason'])
            for point in (row['best_point'], row['lower_fa_point'], row['upper_fa_point']):
                if point is None:
                    continue
                threshold = np.float32(point['threshold'])
                self.assertEqual(point['false_pixels'], int(np.count_nonzero(background >= threshold)))
                self.assertEqual(point['true_targets'], int(np.count_nonzero(targets >= threshold)))
                self.assertEqual(point['fa'], point['false_pixels'] / result['pixels'])
                self.assertEqual(point['pd'], point['true_targets'] / len(targets) if len(targets) else None)

    def test_fixed_real_budgets_and_one_ulp_rounding(self):
        collector = ExactFaAccumulator()
        self.assertEqual(collector.fp_budgets, (87474, 886700, 1201645))
        self.assertEqual(collector.pixels_total, DEFAULT_PIXELS_TOTAL)
        self.assertEqual(collector.fa_budgets, DEFAULT_FA_BUDGETS)
        # 0.29 * 100 is 28.999999999999996 in binary floating point.
        rounded = ExactFaAccumulator(pixels_total=100, fa_budgets=[.29], expected_targets=None)
        self.assertEqual(rounded.fp_budgets, (29,))

    def test_randomized_exhaustive_and_chunk_consistency(self):
        for seed in range(12):
            rng = np.random.RandomState(seed)
            n = 47 + seed * 3
            values = np.concatenate([rng.uniform(0, 1, 19).astype(np.float32),
                                     np.array([0, .3, .5, .99, 1], np.float32)])
            background = rng.choice(values, n).astype(np.float32)
            targets = np.r_[rng.choice(values, 17), np.float32(-np.inf)].astype(np.float32)
            caps = [0, 1, 3, n // 3, n]
            reference = None
            for chunk, buffer in ((1, 1), (7, 11), (23, 17), (1000, 1000)):
                with self.subTest(seed=seed, chunk=chunk, buffer=buffer):
                    result = self.run_stream(background, targets, caps, chunk=chunk, buffer=buffer)
                    self.check_exhaustive(background, targets, result)
                    if reference is not None:
                        self.assertEqual(result['budgets'], reference)
                    reference = result['budgets']

    def test_ties_larger_than_retained_set_and_cutoff_changes(self):
        background = np.r_[np.full(5000, .4), np.full(6000, .8),
                            np.full(7000, .8), np.ones(2000)].astype(np.float32)
        targets = np.array([0, .4, .8, np.nextafter(np.float32(.8), np.float32(np.inf)), .9, 1, -np.inf], np.float32)
        reference = None
        for chunk, buffer in ((31, 97), (500, 4096), (30000, 20000)):
            result = self.run_stream(background, targets, [0, 2000, 2200, 8000], chunk=chunk, buffer=buffer)
            self.check_exhaustive(background, targets, result)
            self.assertEqual(result['budgets'][-1]['upper_fa_point']['false_pixels'], 15000)
            self.assertEqual(result['budgets'][-1]['best_point']['false_pixels'], 2000)
            self.assertEqual(result['memory']['discarded_boundary_ties'], 6999)
            if reference is not None:
                self.assertEqual(result['budgets'], reference)
            reference = result['budgets']
        # The retained boundary rises .2 -> .6 -> .9; lower-score tie counts
        # must not leak into the final .9 multiplicity.
        background = np.r_[np.full(100, .2), np.full(106, .6), np.full(105, .9), np.ones(3)].astype(np.float32)
        for chunk, buffer in ((1, 1), (7, 4), (1000, 19)):
            result = self.run_stream(background, targets, [3, 4], chunk=chunk, buffer=buffer)
            self.check_exhaustive(background, targets, result)
            self.assertEqual(result['budgets'][-1]['upper_fa_point']['false_pixels'], 108)

    def test_float32_one_saturation_and_threshold_domain(self):
        targets = np.array([0, .99, 1, -np.inf], np.float32)
        background = np.ones(100000, np.float32)
        result = self.run_stream(background, targets, [0, 2, 7], chunk=3333, buffer=19)
        self.check_exhaustive(background, targets, result)
        self.assertEqual(result['minimum_attainable_point']['false_pixels'], 100000)
        self.assertEqual(result['memory']['discarded_boundary_ties'], 99992)
        below_one = np.array([np.nextafter(np.float32(1), np.float32(0))], np.float32)
        result = self.run_stream(below_one, np.array([1], np.float32), [0], chunk=1, buffer=1)
        self.check_exhaustive(below_one, np.array([1], np.float32), result)
        self.assertEqual(result['budgets'][0]['best_point']['threshold'], 1.)
        self.assertEqual(result['budgets'][0]['best_point']['pd'], 1.)

    def test_empty_gt_zero_scores_and_no_background(self):
        for background, targets, caps, pixels in (
            (np.zeros(13, np.float32), np.empty(0, np.float32), [0, 3, 13], 13),
            (np.empty(0, np.float32), np.array([0, .5, -np.inf], np.float32), [0, 10], 10),
        ):
            result = self.run_stream(background, targets, caps, pixels=pixels)
            self.check_exhaustive(background, targets, result)
            self.assertEqual(result['pd_defined'], len(targets) > 0)

    def test_frame_input_matches_original_shooting_rules(self):
        path = ROOT.parents[1] / 'ShootingRules.py'
        spec = importlib.util.spec_from_file_location('original_exact_fa_oracle', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        oracle = module.ShootingRules()
        rng = np.random.RandomState(92)
        frames = []
        for shape in ((7, 8), (19, 17), (8, 9)):
            prob = rng.choice(np.array([0, .1, .5, .99, 1], np.float32), size=shape)
            mask = np.zeros(shape, np.uint8)
            mask[0, 0], mask[-2, -2] = 128, 255
            frames.append((prob, mask))
        pixels = sum(prob.size for prob, _ in frames)
        collector = ExactFaAccumulator(pixels_total=pixels, fa_budgets=[0, .1, .3, 1],
                                       expected_targets=None, buffer_size=13)
        for prob, mask in frames:
            collector.update(prob, mask)
        result = collector.finalize()
        for row in result['budgets']:
            for point in (row['best_point'], row['upper_fa_point']):
                if point is None:
                    continue
                sums = np.asarray([oracle(prob[None], (mask > 0).astype(float)[None], np.float32(point['threshold']))
                                   for prob, mask in frames]).sum(axis=0)
                self.assertEqual(tuple(sums), (point['false_pixels'], point['true_targets'], point['targets']))

    def test_incomplete_and_invalid_inputs(self):
        collector = ExactFaAccumulator(pixels_total=10, fa_budgets=[0], expected_targets=None)
        collector.update_scores(np.array([.5], np.float32), np.array([-np.inf], np.float32), pixels=1)
        with self.assertRaises(ValueError):
            collector.finalize()
        for values in (np.array([np.nan], np.float32), np.array([np.inf], np.float32), np.array([-.1], np.float32)):
            with self.assertRaises(ValueError):
                collector.update_scores(np.array([.5], np.float32), values, pixels=1)
        with self.assertRaises(TypeError):
            collector.update_scores(np.array([.5], np.float64), np.empty(0, np.float32), pixels=1)
        done = ExactFaAccumulator(pixels_total=1, fa_budgets=[0], expected_targets=0)
        done.update_scores(np.array([.5], np.float32), np.empty(0, np.float32), pixels=1)
        first = done.finalize()
        self.assertEqual(done.finalize(), first)
        with self.assertRaises(RuntimeError):
            done.update_scores(np.array([.5], np.float32), np.empty(0, np.float32), pixels=1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
