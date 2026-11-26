"""Independent synthetic tests of the frozen M2 protocol; no dataset/model access.

Run with the existing CPU scientific Python environment from the project root.
The tests import only the runtime implementation and fixed JSON plan.
"""
import copy
import csv
import hashlib
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
import m2_backcast as m2

PARAMS = json.loads((ROOT / 'results/m2/plan.json').read_text())['parameters']


def observation(t, x, y, logit=1.):
    return dict(t=t, x=float(x), y=float(y), logit=float(logit))


def track(track_id, observations):
    return dict(track_id=track_id, observations=observations)


def fitted(track_id, observations):
    item = track(track_id, observations)
    return m2.fit_track(item, PARAMS)


def brute_assignment(distance, valid):
    """Enumerate tiny graphs independently to verify cardinality before cost."""
    n, width = valid.shape
    best_cardinality, best_cost = -1, np.inf
    for columns in itertools.product(range(-1, width), repeat=n):
        chosen = [j for j in columns if j >= 0]
        if len(chosen) != len(set(chosen)):
            continue
        pairs = [(i, j) for i, j in enumerate(columns) if j >= 0]
        if any(not valid[i, j] for i, j in pairs):
            continue
        card, cost = len(pairs), sum(distance[i, j] for i, j in pairs)
        if card > best_cardinality or (card == best_cardinality and cost < best_cost):
            best_cardinality, best_cost = card, cost
    return best_cardinality, best_cost


class AssignmentTests(unittest.TestCase):
    def check_assignment(self, distance, valid):
        result = m2.maximum_valid_assignment(distance, valid)
        self.assertEqual(len({i for i, _ in result}), len(result))
        self.assertEqual(len({j for _, j in result}), len(result))
        self.assertTrue(all(valid[i, j] for i, j in result))
        count, cost = brute_assignment(distance, valid)
        self.assertEqual(len(result), count)
        self.assertAlmostEqual(sum(distance[i, j] for i, j in result), cost, places=9)

    def test_hard_gate_preserves_maximum_cardinality(self):
        # Greedy nearest, or Hungarian followed by a gate at too-cheap invalid
        # costs, loses a match: the required assignment is (0,1),(1,0).
        d = np.array([[1., 9.], [1., 0.]])
        valid = np.array([[True, True], [True, False]])
        self.check_assignment(d, valid)
        self.assertEqual(set(m2.maximum_valid_assignment(d, valid)), {(0, 1), (1, 0)})

    def test_assignment_agrees_with_independent_tiny_graph_enumeration(self):
        rng = np.random.default_rng(8123)
        for n, width in [(1, 4), (4, 1), (2, 3), (3, 2), (4, 4)]:
            for _ in range(8):
                distance = rng.uniform(0., 1000., (n, width))
                valid = rng.random((n, width)) < .55
                with self.subTest(n=n, width=width, valid=valid.tolist()):
                    self.check_assignment(distance, valid)

    def test_empty_and_all_invalid_graphs(self):
        for shape in [(0, 0), (0, 3), (3, 0), (3, 3)]:
            self.assertEqual(m2.maximum_valid_assignment(np.zeros(shape), np.zeros(shape, bool)), [])


class TeacherAssociationTests(unittest.TestCase):
    def test_teacher_t60_and_logit_zero_are_inclusive(self):
        logits = np.full((32, 32), -20., np.float32)
        logits[6, 7] = 0.
        logits[22, 25] = np.nextafter(np.float32(0), np.float32(-1))
        before, _ = m2.extract_teacher(logits, 59, PARAMS)
        self.assertEqual(before, [])
        at, _ = m2.extract_teacher(logits, 60, PARAMS)
        self.assertEqual([(r['t'], r['x'], r['y'], r['logit']) for r in at], [(60, 7., 6., 0.)])

    def test_teacher_top300_ties_are_y_then_x_deterministic(self):
        logits = np.full((210, 210), -20., np.float32)
        expected = [(float(x), float(y)) for y in range(5, 205, 10) for x in range(5, 205, 10)]
        for x, y in expected:
            logits[int(y), int(x)] = 0.
        rows, _ = m2.extract_teacher(logits, 60, PARAMS)
        self.assertEqual([(r['x'], r['y']) for r in rows], expected[:300])

    def test_association_is_one_to_one_and_maximum_cardinality(self):
        rows = [observation(60, 0, 0), observation(60, 10, 0),
                observation(61, 9, 0), observation(61, 11, 0)]
        tracks = m2.associate(rows, PARAMS)
        self.assertEqual(len(tracks), 2)
        self.assertEqual(sorted(len(t['observations']) for t in tracks), [2, 2])
        pairs = {(t['observations'][0]['x'], t['observations'][1]['x']) for t in tracks}
        self.assertEqual(pairs, {(0., 9.), (10., 11.)})

    def test_association_gap3_and_speed_boundary_pass_but_gap4_starts_new(self):
        rows = [observation(60, 10, 20), observation(63, 40, 20), observation(67, 40, 20)]
        tracks = m2.associate(rows, PARAMS)
        self.assertEqual(sorted([r['t'] for r in t['observations']] for t in tracks), [[60, 63], [67]])


class FitProjectionTests(unittest.TestCase):
    def test_stop_and_constant_velocity_math(self):
        stop = m2.fit_track(track(1, [observation(t, 20, 30) for t in range(60, 70)]), PARAMS)
        self.assertEqual(stop['n_fit'], 10)
        self.assertTrue(stop['stop_ok'])
        self.assertTrue(stop['cv_ok'])
        self.assertAlmostEqual(stop['stop_x'], 20.)
        self.assertAlmostEqual(stop['stop_y'], 30.)
        self.assertAlmostEqual(stop['stop_rms'], 0.)
        moving = m2.fit_track(track(2, [observation(t, 100 + 2*(t-60), 200-(t-60))
                                      for t in range(60, 70)]), PARAMS)
        self.assertFalse(moving['stop_ok'])
        self.assertTrue(moving['cv_ok'])
        self.assertAlmostEqual(moving['cv_vx'], 2.)
        self.assertAlmostEqual(moving['cv_vy'], -1.)
        self.assertAlmostEqual(moving['cv_x'], 100 + 2*(moving['cv_anchor_t']-60))
        self.assertAlmostEqual(moving['cv_y'], 200-(moving['cv_anchor_t']-60))

    def test_first40_frame_fit_ignores_observations_from_frame100(self):
        observations = [observation(t, 100 + 2*(t-60), 200-(t-60)) for t in range(60, 100)]
        early = m2.fit_track(track(1, observations), PARAMS)
        late = m2.fit_track(track(1, observations+[observation(100, 1e6, -1e6),
                                                  observation(110, -1e6, 1e6)]), PARAMS)
        self.assertEqual(early['n_fit'], 40)
        self.assertEqual(late['n_fit'], 40)
        for key in ['stop_x', 'stop_y', 'stop_rms', 'cv_rms', 'cv_anchor_t', 'cv_x', 'cv_y', 'cv_vx', 'cv_vy']:
            self.assertAlmostEqual(early[key], late[key], msg=key)

    def test_minimum_points_and_rms_inclusive_boundary(self):
        too_short = m2.fit_track(track(1, [observation(t, 20, 20) for t in range(60, 69)]), PARAMS)
        self.assertFalse(too_short['stop_ok'])
        self.assertFalse(too_short['cv_ok'])
        exact = m2.fit_track(track(2, [observation(t, 20+(-3 if t%2 else 3), 20)
                                     for t in range(60, 70)]), PARAMS)
        self.assertAlmostEqual(exact['stop_rms'], 3.)
        self.assertTrue(exact['stop_ok'])
        beyond = m2.fit_track(track(3, [observation(t, 20+(-3.001 if t%2 else 3.001), 20)
                                      for t in range(60, 70)]), PARAMS)
        self.assertFalse(beyond['stop_ok'])

    def test_projection_is_prediction_only_and_retains_out_of_bounds_records(self):
        item = fitted(2, [observation(t, 100+2*(t-60), 200-(t-60)) for t in range(60, 70)])
        projected = m2.project_tracks([item], 100, 400, 400, PARAMS)
        cv = [r for r in projected if r['model'] == 'raw_cv']
        self.assertEqual(sorted(r['t'] for r in cv), list(range(60)))
        by_time = {r['t']: r for r in cv}
        self.assertAlmostEqual(by_time[59]['x'], 98.)
        self.assertAlmostEqual(by_time[59]['y'], 201.)
        self.assertEqual(by_time[59]['status'], 'kept')
        self.assertEqual(by_time[0]['status'], 'out_of_bounds')
        self.assertAlmostEqual(by_time[0]['x'], -20.)
        self.assertEqual(by_time[0]['horizon'], 60)
        self.assertEqual(by_time[10]['x'], 0.)
        self.assertEqual(by_time[10]['y'], 250.)
        self.assertEqual(by_time[10]['status'], 'kept')

    def test_nms_separate_models_uses_count_then_score_then_track_id(self):
        items = [fitted(1, [observation(t, 40, 40, 9.) for t in range(60, 70)]),
                 fitted(2, [observation(t, 43, 40, 5.) for t in range(60, 71)]),
                 fitted(3, [observation(t, 43, 43, 4.) for t in range(60, 71)]),
                 fitted(4, [observation(t, 46.1, 40, 4.) for t in range(60, 71)])]
        projected = m2.project_tracks(items, 100, 100, 100, PARAMS)
        for model in ['raw_stop', 'raw_cv']:
            rows = {r['track_id']: r for r in projected if r['t']==0 and r['model']==model}
            self.assertEqual({k for k,r in rows.items() if r['status']=='kept'}, {2,4})
            self.assertEqual(rows[1]['status'], 'nms_suppressed')
            self.assertEqual(rows[3]['status'], 'nms_suppressed')
            self.assertEqual(rows[1]['suppressed_by'], 2)


class EvaluationMatchingTests(unittest.TestCase):
    def test_matching_uses_euclidean3_not_square3_and_counts_all_unmatched(self):
        points = [dict(x=3., y=0.), dict(x=3., y=3.), dict(x=6., y=0.)]
        gt = [dict(x=0., y=0.)]
        matches = m2.match_points(points, gt, radius=3.)
        self.assertEqual([(i,j) for i,j,_ in matches], [(0,0)])
        self.assertAlmostEqual(matches[0][2], 3.)
        self.assertEqual(len(points)-len(matches), 2)  # Includes both 3..8 px wrong points.

    def test_duplicate_candidates_and_no_gt_are_false_candidates(self):
        points = [dict(x=0.,y=0.), dict(x=1.,y=0.)]
        matches = m2.match_points(points, [dict(x=0., y=0.)], radius=3.)
        self.assertEqual(len(matches), 1)
        self.assertEqual(len(points)-len(matches), 1)
        empty = m2.match_points(points, [], radius=3.)
        self.assertEqual(empty, [])
        self.assertEqual(len(points)-len(empty), 2)


class IsolationIntegrationTests(unittest.TestCase):
    def test_sealed_generation_without_gt_and_gt_replacement_never_changes_proposals(self):
        fixture_dir = ROOT / 'runtime/test_fixtures'
        fixture_dir.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='m2_', dir=fixture_dir) as temporary:
            root = Path(temporary)
            clip = 'synthetic_only'
            plan = copy.deepcopy(json.loads((ROOT / 'results/m2/plan.json').read_text()))
            plan.update(clips=[clip], predictions='preds/official_rgbt', teacher_checkpoint='checkpoint.bin',
                        baseline_hit_source='targets.csv')
            plan.pop('expected_first60_baseline_misses', None)
            gt = []
            for t in range(1, 100):
                gt.append(dict(clip=clip, t=t, box_index=0, track=1, x=35 if t == 1 else 30,
                               y=30, **{'hit_0.5': False, 'hit_1e-05': False}))
            buffer = io.StringIO(newline='')
            writer = csv.DictWriter(buffer, fieldnames=list(gt[0]))
            writer.writeheader()
            writer.writerows(gt)
            gt_bytes = buffer.getvalue().encode()
            plan['baseline_hit_source_sha256'] = hashlib.sha256(gt_bytes).hexdigest()
            (root / 'checkpoint.bin').write_bytes(b'synthetic fixture only, not a model')
            logits = np.full((100, 64, 64), -20., np.float16)
            logits[:60, 5, 5] = 10  # Excluded early detections must not become teachers.
            logits[60:, 30, 30] = 0  # Exactly probability .5 must remain a teacher.
            (root / plan['predictions']).mkdir(parents=True)
            np.save(root / plan['predictions'] / (clip + '.npy'), logits)
            m2.write_json(root / 'preds/official_rgbt__meta' / (clip + '.json'),
                          {'fingerprint': {'clip': clip, 'ckpt_sha256': m2.sha256(root / 'checkpoint.bin'),
                                           'data': {'frames': 100}}})
            m2.write_json(root / 'plan.json', plan)
            self.assertFalse((root / 'targets.csv').exists())
            first = m2.generate(root, root / 'plan.json', root / 'out_a', 1)
            self.assertFalse(first['gt_loaded'])
            self.assertEqual(first['clips'][0]['teacher_points'], 40)
            self.assertEqual(first['clips'][0]['tracks'], 1)
            (root / 'targets.csv').write_bytes(gt_bytes)
            result = m2.evaluate(root, root / 'plan.json', root / 'out_a', root / 'targets.csv')
            for model in ('raw_stop', 'raw_cv'):
                total = result['totals'][model]
                self.assertEqual(total['kept'], 60)
                self.assertEqual(total['wrong_candidates'], 2)  # Empty-GT frame0 and 5px mismatch frame1.
                self.assertEqual(total['true_candidates'], 58)
                self.assertEqual(total['baseline_thresholds']['0.5']['original_misses'], 59)
                self.assertEqual(total['baseline_thresholds']['0.5']['recovered'], 58)
                self.assertAlmostEqual(total['precision'], 58/60)
            # Deliberately replace GT before another generation; proposal file SHA must be unchanged.
            (root / 'targets.csv').write_bytes(b'changed GT must have no effect on generation\n')
            second = m2.generate(root, root / 'plan.json', root / 'out_b', 1)
            self.assertEqual(first['clips'][0]['files'], second['clips'][0]['files'])
            with self.assertRaisesRegex(ValueError, 'Frozen GT'):
                m2.evaluate(root, root / 'plan.json', root / 'out_b', root / 'targets.csv')


if __name__ == '__main__':
    unittest.main(verbosity=2)
