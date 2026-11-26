"""Summarize existing author-component hits without inference or fitting."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/temporal_diagnosis'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def windows(n, T=40):
    return [(max(0, min(n, (i + 1) * 36 + 4) - T), min(n, (i + 1) * 36 + 4))
            for i in range(math.ceil((n - 4) / 36))]


def count(rows):
    n = len(rows)
    hits = sum(r['hit'] for r in rows)
    return dict(components=n, hits=hits, misses=n - hits,
                miss_rate=(n - hits) / n if n else None)


def main():
    clips = json.loads((ROOT / 'configs/splits.json').read_text())['official_test']
    frames = {c: json.loads((ROOT / f'cache/preds/official_rgbt__meta/{c}.json').read_text())[
        'fingerprint']['data']['frames'] for c in clips}
    positions = {}
    for c, n in frames.items():
        for t in range(n):
            positions[c, t] = [t - a for a, b in windows(n) if a <= t < b]
            assert positions[c, t]
    raw = list(csv.DictReader((OUT / 'source_components.csv').open()))
    rows = [dict(r, frame=int(r['frame']), component=int(r['component']),
                 threshold=float(r['threshold']), hit=int(r['hit'])) for r in raw]
    assert len({(r['clip'], r['frame'], r['component'], r['threshold']) for r in rows}) == len(rows)
    assert all(0 <= r['frame'] < frames[r['clip']] for r in rows)
    original = json.loads((OUT / 'source_attribution_summary.json').read_text())
    all_stats, per_clip, coverage_stats, phase_stats = [], [], [], []
    for th in (1e-5, 0.5):
        rr = [r for r in rows if r['threshold'] == th]
        total = count(rr)
        check = next(r for r in original['component_groups'] if r['threshold'] == th and
                     r['clip'] == r['camera_group'] == r['target_group'] == 'all')
        assert (total['components'], total['hits']) == (check['targets'], check['hits'])
        for early in (True, False):
            chosen = [r for r in rr if (r['frame'] < 60) == early]
            s = count(chosen)
            nf = sum(min(n, 60) if early else max(n - 60, 0) for n in frames.values())
            all_stats.append(dict(threshold=th, period='first60' if early else 'later', frames=nf,
                                  frame_share=nf / sum(frames.values()),
                                  component_share=s['components'] / total['components'],
                                  miss_share=s['misses'] / total['misses'], **s))
            for overlap in range(1, max(map(len, positions.values())) + 1):
                subset = [r for r in chosen if len(positions[r['clip'], r['frame']]) == overlap]
                cov_frames = sum(1 for (c, t), pp in positions.items()
                                 if (t < 60) == early and len(pp) == overlap)
                coverage_stats.append(dict(threshold=th, period='first60' if early else 'later',
                                           covering_windows=overlap, frames=cov_frames, **count(subset)))
            partition = [r for r in coverage_stats if r['threshold'] == th and
                         r['period'] == ('first60' if early else 'later')]
            assert sum(r['frames'] for r in partition) == nf
            assert sum(r['components'] for r in partition) == len(chosen)
        for c in clips:
            for cut in (40, 60):
                for early in (True, False):
                    s = count([r for r in rr if r['clip'] == c and (r['frame'] < cut) == early])
                    per_clip.append(dict(threshold=th, clip=c, cutoff=cut,
                                         period='early' if early else 'later', **s))
        for lo, hi in ((0, 4), (4, 10), (10, 20), (20, 30), (30, 36), (36, 40)):
            chosen = [r for r in rr if len(positions[r['clip'], r['frame']]) == 1 and
                      lo <= positions[r['clip'], r['frame']][0] < hi]
            phase_stats.append(dict(threshold=th, relative_position_start=lo,
                                    relative_position_stop=hi, scope='single_coverage_frames_only',
                                    **count(chosen)))
    sources = ['results/temporal_diagnosis/source_components.csv',
               'results/temporal_diagnosis/source_attribution_summary.json',
               'configs/splits.json', 'code/infer.py', 'code/models.py']
    transitions = {}
    for c in ('DJI_0255_1', 'DJI_0307_2'):
        early_rows = [r for r in rows if r['clip'] == c and r['threshold'] == 1e-5 and r['frame'] < 80]
        transitions[c] = dict(first_annotated_frame=min(r['frame'] for r in early_rows),
                              missed_frames_before80=[r['frame'] for r in early_rows if not r['hit']],
                              hit_frames_before40=[r['frame'] for r in early_rows if r['hit'] and r['frame'] < 40],
                              windows=windows(frames[c]),
                              interpretation='Descriptive boundary coincidence, not a causal intervention.')
    result = dict(status='verified', unit='8-connected mask component per frame, not a distinct physical target',
                  source_run='official_rgbt', total_frames=sum(frames.values()),
                  first60_frames=sum(min(n, 60) for n in frames.values()), clips=frames,
                  total_windows=sum(len(windows(n)) for n in frames.values()),
                  total_frame_window_instances=sum(len(p) for p in positions.values()),
                  early60=all_stats, per_clip=per_clip, window_coverage=coverage_stats,
                  inspected_transitions=transitions,
                  unique_window_relative_position=phase_stats,
                  method='Replay existing fixed-threshold component hit records. No new predictions, fitting, '
                         'speed relabeling or model execution. Half-open early periods: frame<40 or frame<60.',
                  limitations=[
                      'The same function is applied to each window, but this does not rule out a window-scheme effect.',
                      'models.py:66-77 uses separate learned T×T temporal matrices and output-position biases; '
                      'there is no constraint enforcing temporal shift equivariance.',
                      'infer.py:38-46 shifts the final window; infer.py:63 max-fuses overlapping windows. '
                      'Frames differ in context, relative position and number of scoring opportunities.',
                      'The single-coverage phase table excludes overlaps to make position unique. '
                      'It is descriptive and confounded with clip position, clip identity and target composition.',
                      'No per-window logits were retained; for overlaps the winning window cannot be identified '
                      'from the fused predictions. An offset/context intervention is required to test causation.',
                      'Shared-track, isolated-target and category claims were not independently recomputed here.',
                      'Early windows with many repeated frames/targets are not independent statistical samples.'
                  ], source_sha256={p: sha(ROOT / p) for p in sources},
                  analysis_sha256=sha(Path(__file__)))
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, table in [('early60.csv', all_stats), ('per_clip.csv', per_clip),
                        ('window_coverage.csv', coverage_stats), ('unique_window_phase.csv', phase_stats)]:
        with (OUT / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    print(json.dumps(all_stats, indent=2))
    print(json.dumps([r for r in per_clip if r['clip'] in ('DJI_0255_1', 'DJI_0307_2') and
                      r['cutoff'] == 40], indent=2))


if __name__ == '__main__':
    main()
