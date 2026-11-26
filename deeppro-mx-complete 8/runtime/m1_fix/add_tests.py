from pathlib import Path
p=Path(__file__).parent/'fixed/code/test_miss_recovery.py'
s=p.read_text()
s=s.replace("n_admit = sum(r['score'] >= thr[c] for r in cands if r['clip'] == c)","n_admit = sum(r['label'] == 0 and np.isfinite(r['score']) and r['score'] >= thr[c]\n                      for r in cands if r['clip'] == c)")
s=s.replace("        assert n_admit <= 0.25 * sum(frames.values())", "        assert n_admit <= 0.25 * sum(frames.values())\n    admitted = m1.clutter_admissions(cands, frames, thr, 0.25, 'pooled')\n    assert admitted['within_budget'] and admitted['per_frame'] <= 0.25\n    assert s['all']['recovered'] <= s['all']['candidate_ceiling']")
s=s.replace("            assert summ['frames'] == {k: N for k in data}", "            assert summ['frames'] == {k: N for k in data}\n            assert summ['decision']['go'] and summ['decision']['diagnostic_only']\n            assert summ['measured_clutter']['pooled budget 1']['per_frame'] <= 1\n            assert os.path.exists('results/miss_recovery/fake/' + ','.join(data) + '/candidates.csv')")
tests=r'''
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


'''
s=s.replace("if __name__ == '__main__':",tests+"if __name__ == '__main__':")
p.write_text(s)

p=Path(__file__).parent/'fixed/code/miss_recovery.py'
s=p.read_text()
a=s.index('    single     every frame')
b=s.index('Reported per group',a)
s=s[:a]+'''    single     every frame on its own: actual local maxima of |local contrast|, top --per-frame
    frame      candidates only. Candidates within --r-pos px of a GT centre are positive;
               beyond --r-neg of every centre are clutter; others are excluded from fitting.
    features   calibrated contrast R, |R|, D, P and NCC / |NCC|, sampled at each actual
               candidate location. Ground-truth centres are never detection proposals.
    score      logistic regression fitted leave-one-recording-out on actual positive vs
               clutter candidates, balanced by clip and class. All candidates of the held-out
               clip are scored without using their labels in the fit. No teacher means no recovery.
    recovery   finite scored candidates are assigned to boxes one-to-one, in descending score
               order, nearest eligible centre first. The geometric candidate ceiling uses
               maximum matching, ignoring scores and budgets, among the misses of each group.
    budget     conservative thresholds admit at most --budgets labelled clutter candidates per
               frame (pooled or per clip). Ties are excluded when needed; actual rates are reported.
''' + s[b:]
s=s.replace("A test-free version fits on --fit-split clips scored by the\nsame run (e.g. a cross-validation run's predictions on its own validation folds).", "Thresholds also use the evaluated clips' labelled clutter. Neither fitting nor threshold\ncalibration is a test-free performance result.")
s=s.replace("Writes results/miss_recovery/<run>/<split>/{targets.csv, windows.csv, summary.json}.","Writes results/miss_recovery/<run>/<split>/{targets.csv, candidates.csv, windows.csv, summary.json}.\nThe summary includes the predeclared pooled 1/frame, baseline 1e-5 gate: early recovery\n>=30% and overall >=20%, with early fixed to frames 0..59. It remains diagnostic only.")
s=s.replace("        ceiling = candidate_ceiling([r for r in rows if not r[key]])", "        ceiling = candidate_ceiling([r for r in rows if not r[key]])\n        if int((~hit & rec).sum()) > ceiling:\n            raise AssertionError('Recovery exceeds the actual candidate matching ceiling')")
p.write_text(s)
