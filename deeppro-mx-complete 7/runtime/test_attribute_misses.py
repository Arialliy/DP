"""Small independent witnesses for component attribution, including merged masks."""
import json
from pathlib import Path

import numpy as np

import attribute_misses as am


def test_component_hits_equal_author_sweep():
    rng = np.random.default_rng(12)
    for trial in range(12):
        boxes = np.array([[x, y, w, h, 1, i] for i, (x, y, w, h) in enumerate(
            [(0, 0, 2, 2), (5, 5, 5, 3), (9, 5, 5, 3), (24, 24, 5, 5), (1, 28, .2, .2)])], float)
        prob = rng.uniform(0, 1e-5 if trial % 2 else 1, (32, 32)).astype(np.float32)
        records, vanished = am.component_records(prob, boxes, [0.1, 0.1, 0.8, 0.6, 0.1])
        mask = am.gm.label(boxes, 32, 32, 6, 'floor')
        _, expected_hits, expected_targets, _, _ = am.ev.shooting_rules_sweep(prob, mask, am.THRESHOLDS)
        assert len(records) == expected_targets
        got = [sum(np.float32(r['best_probability']) >= np.float32(t) for r in records) for t in am.THRESHOLDS]
        assert got == expected_hits.tolist(), (got, expected_hits)
        assert vanished == 1


def test_cross_band_merge_and_unknown_are_not_arbitrarily_attributed():
    boxes = np.array([[5, 5, 5, 3, 1, 1], [8, 5, 5, 3, 1, 2]], float)
    prob = np.zeros((24, 24), np.float32)
    records, gone = am.component_records(prob, boxes, [0.1, 0.8])
    assert len(records) == 1 and records[0]['merged'] and records[0]['target_group'] == 'mixed'
    assert records[0]['box_indices'] == [0, 1] and gone == 0
    assert am.component_records(prob, boxes, [0.1, 0.05])[0][0]['target_group'] == 'near_stationary'
    assert am.component_records(prob, boxes, [0.1, float('nan')])[0][0]['target_group'] == 'unknown'


def test_official_top_left_edge_can_remain_missed_at_probability_one():
    boxes = np.array([[0, 0, 1, 1, 1, 1]], float)
    probability = np.ones((8, 8), np.float32)
    records, gone = am.component_records(probability, boxes, [0.0])
    assert len(records) == 1 and gone == 0
    assert records[0]['best_probability'] == -np.inf
    mask = am.gm.label(boxes, 8, 8, 6, 'floor')
    assert am.ev.shooting_rules(probability, mask, .5) == (64, 0, 1)


def test_aggregation_denominators_and_score_validation():
    rows = [dict(clip='c', threshold=t, camera_group='static', target_group=g, hit=h, merged=False)
            for t in am.THRESHOLDS for g, h in [('near_stationary', 0), ('near_stationary', 1), ('moving', 1)]]
    groups = am.aggregate(rows)
    r = next(r for r in groups if r['threshold'] == .5 and r['clip'] == 'all'
             and r['camera_group'] == 'all' and r['target_group'] == 'near_stationary')
    assert r['miss_rate'] == .5 and r['miss_share'] == 1 and r['exposure_share'] == 2 / 3
    table = {str(t): dict(targets=3, tp=2) for t in am.THRESHOLDS}
    summary = dict(author_protocol=table, author_protocol_per_clip={'c': table})
    assert am.validate_summary(groups, summary)['threshold_clip_totals_checked'] == 4
    summary['author_protocol'] = {str(t): dict(targets=3, tp=1) for t in am.THRESHOLDS}
    try:
        am.validate_summary(groups, summary)
    except AssertionError:
        pass
    else:
        raise AssertionError('Mismatched reference totals were accepted')


if __name__ == '__main__':
    tests = [test_component_hits_equal_author_sweep,
             test_cross_band_merge_and_unknown_are_not_arbitrarily_attributed,
             test_official_top_left_edge_can_remain_missed_at_probability_one,
             test_aggregation_denominators_and_score_validation]
    for test in tests:
        test()
        print('ok', test.__name__)
    output = Path(__file__).with_name('attribution_validation.json')
    output.write_text(json.dumps(dict(status='passed', tests=[t.__name__ for t in tests],
                                      script_sha256=am.sha256(am.__file__)), indent=2) + '\n')
