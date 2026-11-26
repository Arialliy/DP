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
        n_admit = sum(r['label'] == 0 and np.isfinite(r['score']) and r['score'] >= thr[c]
                      for r in cands if r['clip'] == c)
        assert n_admit <= 0.25 * sum(frames.values())
    admitted = m1.clutter_admissions(cands, frames, thr, 0.25, 'pooled')
    assert admitted['within_budget'] and admitted['per_frame'] <= 0.25
    assert s['all']['recovered'] <= s['all']['candidate_ceiling']


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
            assert summ['decision']['go'] and summ['decision']['diagnostic_only']
            assert summ['measured_clutter']['pooled budget 1']['per_frame'] <= 1
            assert os.path.exists('results/miss_recovery/fake/' + ','.join(data) + '/candidates.csv')
            assert os.path.exists('results/miss_recovery/fake/' + ','.join(data) + '/windows.csv')
        finally:
            vtdata.DATA_ROOT, vtdata.ANN_DIR, vtdata.OFFICIAL_LISTS = saved[:3]
            vtdata._ANN_CACHE.clear()
            os.chdir(saved[3])



def _target(box=0, teacher=1, candidate_ids=None):
    return dict(clip='DJI_8100_1', t=0, box_index=box, track=box, cat=1, rel_speed=1.0,
                camera='static', teacher=teacher, candidate_ids=[0] if candidate_ids is None else candidate_ids,
                score=-np.inf, **{'hit_1e-05': False})


def _candidate(candidate=0, score=1.0, label=0, teacher=1, matches=None, distances=None):
    return dict(clip='DJI_8100_1', t=0, candidate_id=candidate, score=score, label=label, teacher=teacher,
                target_indices=matches or [], target_distances=distances or [],
                **{name: float(score + j * 0.1) for j, name in enumerate(m1.FEATURES)})


def test_no_candidate_means_no_recovery_even_with_a_teacher():
    args = _args()
    video, boxes, logits = clip(0, 30)
    old = m1.mxd.frame_peaks
    try:
        m1.mxd.frame_peaks = lambda *a, **k: np.zeros((0, 4))
        targets, candidates, _ = m1.process_clip('DJI_8100_1', video, boxes, logits, args)
    finally:
        m1.mxd.frame_peaks = old
    assert any(r['teacher'] > 0 for r in targets) and not candidates
    assert all(r['candidate_count'] == 0 for r in targets)
    m1.loro_scores(targets, candidates)
    threshold = m1.thresholds(candidates, {'DJI_8100_1': N}, 1.0, 'pooled')
    result = m1.summarize(targets, threshold, 1e-5, 60)
    assert result['all']['misses'] > 0
    assert result['all']['recovered'] == result['all']['candidate_ceiling'] == 0


def test_no_teacher_and_nonfinite_scores_cannot_recover():
    target = _target(teacher=0)
    candidates = [_candidate(score=-np.inf, teacher=0)]
    threshold = m1.thresholds(candidates, {'DJI_8100_1': 10}, 1.0, 'pooled')
    assert threshold['DJI_8100_1'] == -np.inf
    result = m1.summarize([target], threshold, 1e-5, 60)['all']
    assert result['recovered'] == result['candidate_ceiling'] == 0
    target['teacher'] = 1             # failed LORO fit still yields no finite score
    assert m1.summarize([target], threshold, 1e-5, 60)['all']['recovered'] == 0


def test_tied_clutter_scores_never_exceed_the_budget():
    candidates = [_candidate(i) for i in range(3)]
    frames = {'DJI_8100_1': 1}
    for mode in ('pooled', 'clip'):
        threshold = m1.thresholds(candidates, frames, 1.0, mode)
        report = m1.clutter_admissions(candidates, frames, threshold, 1.0, mode)
        assert report['admitted'] == 0 and report['within_budget']
    candidates[0]['score'], candidates[1]['score'], candidates[2]['score'] = 3.0, 2.0, 1.0
    threshold = m1.thresholds(candidates, frames, 1.0, 'pooled')
    assert m1.clutter_admissions(candidates, frames, threshold, 1.0, 'pooled')['admitted'] == 1


def test_one_candidate_cannot_recover_two_adjacent_targets():
    targets = [_target(0), _target(1)]
    candidates = [_candidate(label=1, matches=[0, 1], distances=[1., 1.])]
    m1.assign_candidates(targets, candidates)
    result = m1.summarize(targets, {'DJI_8100_1': 0.0}, 1e-5, 60)['all']
    assert result['recovered'] == result['candidate_ceiling'] == 1
    assert targets[0]['matched_candidate'] == 0 and targets[1]['matched_candidate'] is None
    # The ceiling uses augmenting paths instead of a greedy underestimate.
    assert m1.candidate_ceiling([_target(0, candidate_ids=[0, 1]), _target(1, candidate_ids=[0])]) == 2


def test_loro_excludes_labels_and_features_of_the_whole_heldout_recording():
    import copy
    candidates = []
    for clip_name in ('DJI_8100_1', 'DJI_8100_2', 'DJI_8200_1', 'DJI_8300_1'):
        for i, value in enumerate((-1.0, 1.0)):
            r = _candidate(i, value, i)
            r['clip'] = clip_name
            candidates.append(r)
    changed = copy.deepcopy(candidates)
    for r in changed:
        if r['clip'].startswith('DJI_8100'):
            r['label'] = 1 - r['label']
        if r['clip'] == 'DJI_8100_2':
            for name in m1.FEATURES:
                r[name] += 1000
    m1.loro_scores([], candidates)
    m1.loro_scores([], changed)
    original_scores = [r['score'] for r in candidates if r['clip'] == 'DJI_8100_1']
    changed_scores = [r['score'] for r in changed if r['clip'] == 'DJI_8100_1']
    assert all(np.isfinite(original_scores)) and original_scores == changed_scores


def test_gate_uses_exact_predeclared_boundaries():
    import copy
    key = 'pooled budget 1 / official th 1e-05'
    result = {key: {'all': dict(misses=100, recovered_fraction=0.20),
                    'frames < 60': dict(misses=10, recovered_fraction=0.30)}}
    observed = {'pooled budget 1': dict(per_frame=1.0, within_budget=True)}
    assert m1.decision(result, observed)['go']
    below = copy.deepcopy(result)
    below[key]['frames < 60']['recovered_fraction'] = np.nextafter(.30, -np.inf)
    assert not m1.decision(below, observed)['go']
    below = copy.deepcopy(result)
    below[key]['all']['recovered_fraction'] = np.nextafter(.20, -np.inf)
    assert not m1.decision(below, observed)['go']
    assert not m1.decision(result, {'pooled budget 1': dict(per_frame=1.01, within_budget=True)})['go']
    assert not m1.decision(result, observed, early=40)['go']
    assert not m1.decision({}, observed)['go']


def test_box_hits_keep_official_top_left_edge_behavior():
    import evaluate
    import vtdata
    prob = np.zeros((8, 8), np.float32)
    prob[0, 0] = 1.0
    boxes = np.array([(0, 0, 1, 1, 1, 0)], float)
    mask = vtdata.gaussian_mask(boxes, 8, 8, 'paper_k6_floor')
    assert evaluate.shooting_rules(prob, mask, .5)[1:] == (0, 1)
    assert m1.box_hits(prob, boxes, 8, 8, [1e-5, .5]).tolist() == [[False, False]]


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
