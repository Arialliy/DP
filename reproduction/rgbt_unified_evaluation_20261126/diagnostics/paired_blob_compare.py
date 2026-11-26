"""CPU-only comparison of prediction areas on the same, jointly detected GTs.

Reads completed breakdown JSONs, never loads a model or changes inputs. The
default comparison is author-released vs final and interim retrained Plus at
0.99. Output is paired_blob_comparison.json beside this helper.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RETRAINED = ('deeppro_plus_416648f66cbee93f', 'deeppro_plus_c1c818212eb0c1d8')


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256') if hasattr(hashlib, 'file_digest') else None
        if digest is None:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(4*1024*1024), b''):
                digest.update(block)
    return digest.hexdigest()


def stats(values):
    values = sorted(values)
    if not values:
        return dict(count=0, mean=None, median=None, quantiles=None)
    quantiles = {}
    for percentile in (0, 5, 25, 50, 75, 95, 100):
        rank = (len(values)-1)*percentile/100
        lo, hi = math.floor(rank), math.ceil(rank)
        quantiles[str(percentile)] = values[lo] + (values[hi]-values[lo])*(rank-lo)
    return dict(count=len(values), mean=statistics.fmean(values), median=statistics.median(values),
                quantiles=quantiles)


def target_index(result):
    index = {}
    clips = set()
    for clip in result['per_clip']:
        name = clip['clip']
        if name in clips:
            raise ValueError('Duplicate clip: ' + name)
        clips.add(name)
        records = clip['per_target']
        if len(records) != clip['targets'] or sum(bool(x['hit']) for x in records) != clip['true_targets']:
            raise ValueError('Per-clip target/hit counts disagree with target records: ' + name)
        for target in records:
            key = (name, str(target['frame_id']), int(target['target_id']))
            if key in index:
                raise ValueError('Duplicate target key: ' + str(key))
            if target['target_id'] < 1 or target['gt_area'] < 1:
                raise ValueError('Invalid positive target ID/area')
            components = target['hit_prediction_components']
            ids = [x['component_id'] for x in components]
            if len(ids) != len(set(ids)) or any(x['area'] < 1 for x in components):
                raise ValueError('Nonunique component IDs or nonpositive component area')
            if target['hit_component_area_sum'] != sum(x['area'] for x in components):
                raise ValueError('Recorded target association area does not equal unique component areas')
            if bool(target['hit']) != bool(components):
                raise ValueError('Hit flag and associated components disagree')
            index[key] = target
    if len(index) != result['targets'] or sum(bool(x['hit']) for x in index.values()) != result['true_targets']:
        raise ValueError('Global target/hit counts disagree with target records')
    return index


def paired_summary(keys, reference, retrained):
    paired = [key for key in keys if reference[key]['hit'] and retrained[key]['hit']]
    old_area = [reference[key]['hit_component_area_sum'] for key in paired]
    new_area = [retrained[key]['hit_component_area_sum'] for key in paired]
    differences = [new-old for old, new in zip(old_area, new_area)]
    ratios = [new/old for old, new in zip(old_area, new_area)]
    directions = Counter('larger' if difference > 0 else 'smaller' if difference < 0 else 'equal'
                         for difference in differences)
    reference_components, retrained_components = set(), set()
    for key in paired:
        reference_components.update((key[0], key[1], x['component_id']) for x in reference[key]['hit_prediction_components'])
        retrained_components.update((key[0], key[1], x['component_id']) for x in retrained[key]['hit_prediction_components'])
    counts = dict(total_targets=len(keys),
                  reference_hit_targets=sum(bool(reference[key]['hit']) for key in keys),
                  retrained_hit_targets=sum(bool(retrained[key]['hit']) for key in keys),
                  paired_targets=len(paired),
                  reference_only_hits=sum(reference[key]['hit'] and not retrained[key]['hit'] for key in keys),
                  retrained_only_hits=sum(retrained[key]['hit'] and not reference[key]['hit'] for key in keys),
                  both_missed=sum(not reference[key]['hit'] and not retrained[key]['hit'] for key in keys))
    assert counts['paired_targets']+counts['reference_only_hits']+counts['retrained_only_hits']+counts['both_missed'] == len(keys)
    return dict(**counts, paired_frames=len({key[:2] for key in paired}),
                reference_area_pixels=stats(old_area), retrained_area_pixels=stats(new_area),
                retrained_minus_reference_area_pixels=stats(differences),
                retrained_to_reference_area_ratio=stats(ratios),
                retrained_area_direction={name: dict(count=directions[name], fraction=directions[name]/len(paired) if paired else None)
                                          for name in ('larger', 'equal', 'smaller')},
                unique_associated_reference_components=len(reference_components),
                unique_associated_retrained_components=len(retrained_components))


def load_input(job_id, variant, threshold, directory):
    if Path(job_id).name != job_id:
        raise ValueError('Expected plain job_id')
    path = directory / (job_id+'.json')
    metadata = dict(job_id=job_id, variant=variant, path=str(path), status='pending')
    if not path.is_file():
        return metadata | dict(reason='Breakdown file not present'), None
    data = read_json(path)
    metadata.update(checkpoint_sha256=data.get('checkpoint_sha256'), checkpoint_status=data.get('checkpoint_status'),
                    breakdown_status=data.get('status'), model=data.get('model'))
    result = data.get('results', {}).get(variant)
    if not data.get('complete') or not result or not result.get('complete'):
        return metadata | dict(reason='Complete breakdown not available'), None
    if data.get('status') not in ('complete', 'mismatch') or result.get('status') not in ('complete', 'mismatch'):
        raise ValueError('Completed breakdown has unexpected status')
    if data.get('job_id') != job_id or result.get('variant') != variant:
        raise ValueError('Job/variant identity mismatch')
    manifest = read_json(ROOT / 'jobs' / (job_id+'.json'))
    if data['checkpoint_sha256'] != manifest['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA256 does not match frozen job manifest')
    if data.get('split') != 'official_test' or not math.isclose(data.get('threshold', -1), threshold, rel_tol=0, abs_tol=1e-12):
        raise ValueError('Split or detection threshold mismatch')
    for field in ('frames', 'pixels'):
        if result[field] != data[field] or result[field] != data['expected_'+field]:
            raise ValueError('Incomplete '+field+' denominator')
    if result['targets'] != data['expected_targets']:
        raise ValueError('Incomplete target denominator')
    metadata.update(status='complete', breakdown_sha256=sha256(path),
                    frames=result['frames'], pixels=result['pixels'], targets=result['targets'],
                    reference_comparison=result.get('reference_comparison'))
    return metadata, result


def compare(reference, retrained):
    for field in ('frames', 'pixels', 'targets'):
        if reference[field] != retrained[field]:
            raise ValueError('Comparison denominator differs: '+field)
    ref_index, new_index = target_index(reference), target_index(retrained)
    if set(ref_index) != set(new_index):
        raise ValueError('GT target keys differ between the two complete inputs')
    for key in ref_index:
        if ref_index[key]['gt_area'] != new_index[key]['gt_area']:
            raise ValueError('GT area differs for '+str(key))
    ref_clips = {x['clip']: x for x in reference['per_clip']}
    new_clips = {x['clip']: x for x in retrained['per_clip']}
    for clip in ref_clips:
        for field in ('frames', 'pixels', 'targets'):
            if ref_clips[clip][field] != new_clips[clip][field]:
                raise ValueError('Per-clip denominator differs: '+clip+'/'+field)
    keys = sorted(ref_index)
    summary = paired_summary(keys, ref_index, new_index)
    summary.update(status='complete', gt_keys_and_areas_identical=True,
                   per_clip=[dict(clip=clip, **paired_summary([key for key in keys if key[0] == clip], ref_index, new_index))
                             for clip in sorted(ref_clips)])
    return summary


def self_test():
    def target(area):
        return dict(hit=area > 0, hit_component_area_sum=area,
                    hit_prediction_components=[dict(component_id=1, area=area)] if area else [])
    keys = [('clip', str(i), 1) for i in range(5)]
    reference = dict(zip(keys, [target(x) for x in (10, 30, 100, 0, 1)]))
    retrained = dict(zip(keys, [target(x) for x in (20, 10, 0, 120, 1)]))
    result = paired_summary(keys, reference, retrained)
    assert result['paired_targets'] == 3 and result['reference_only_hits'] == result['retrained_only_hits'] == 1
    assert result['reference_area_pixels']['mean'] == 41/3
    assert result['retrained_area_pixels']['mean'] == 31/3
    assert result['retrained_to_reference_area_ratio']['median'] == 1
    assert all(x['count'] == 1 for x in result['retrained_area_direction'].values())
    empty = paired_summary(keys[2:4], reference, retrained)
    assert empty['paired_targets'] == 0 and empty['retrained_to_reference_area_ratio']['mean'] is None
    print('CPU fixture tests passed: shared-hit selection, unmatched exclusion, directions, ratios, empty cohort.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default='author_released_reference')
    parser.add_argument('--retrained', nargs='+', default=DEFAULT_RETRAINED)
    parser.add_argument('--variant', default='native')
    parser.add_argument('--threshold', type=float, default=.99)
    parser.add_argument('--breakdown-dir', type=Path, default=ROOT / 'diagnostics/breakdown')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('paired_blob_comparison.json'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    ref_meta, reference = load_input(args.reference, args.variant, args.threshold, args.breakdown_dir)
    comparisons = []
    for job_id in args.retrained:
        metadata, retrained = load_input(job_id, args.variant, args.threshold, args.breakdown_dir)
        row = dict(retrained=metadata, status='pending', reason='Both complete input breakdowns are required')
        if reference is not None and retrained is not None:
            row = dict(retrained=metadata, **compare(reference, retrained))
        row['warnings'] = [item['job_id']+': recomputed breakdown differs from stored ROC reference counts; '
                           'paired geometry uses the real recomputed predictions and does not establish reference reproduction.'
                           for item in (ref_meta, metadata) if item.get('breakdown_status') == 'mismatch']
        comparisons.append(row)
    complete = all(row['status'] == 'complete' for row in comparisons)
    warnings = list(dict.fromkeys(message for row in comparisons for message in row['warnings']))
    report = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
                  status='complete' if complete else 'pending', complete=complete,
                  evidence_status='reference_mismatch' if warnings else 'no_reference_mismatch_reported', warnings=warnings,
                  split='official_test', threshold=args.threshold, variant=args.variant,
                  reference=ref_meta, comparisons=comparisons, helper_sha256=sha256(Path(__file__)),
                  protocol=dict(
                      join=['clip', 'frame_id', 'target_id'], cohort='GT targets detected by BOTH models at the same threshold',
                      area='Sum of whole pixel areas of unique prediction components associated with each GT under original +/-1 hit rule',
                      ratio='retrained area / reference area, calculated per jointly detected GT before summarizing',
                      direction='Positive area difference means retrained is larger; integer equality is exact',
                      quantiles='Linear interpolation between sorted observations at percentiles 0,5,25,50,75,95,100',
                      limitations=[
                          'The same prediction component can associate with multiple GTs; its whole area is then repeated across GT-level pairs.',
                          'One GT can associate with multiple prediction components. Summed area is not mask overlap, false-positive area, or an independent object estimate.',
                          'Frames and targets are correlated; no independence, significance test, or causal interpretation is asserted.',
                          'Matching removes differing-hit-set composition from this comparison, but conditions on joint detection and does not describe missed targets.',
                          'Released-vs-final and released-vs-interim may use different jointly detected target cohorts; their means/medians alone do not establish a final-vs-interim change.',
                          'No conclusion about the cause of distant clutter or training-mask effects follows from these spatial summaries.']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name+'.%d.tmp' % os.getpid())
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(args.output)
    print(str(args.output)+': '+report['status'])
    for message in warnings:
        print('WARNING: '+message)
    for row in comparisons:
        if row['status'] == 'complete':
            print(row['retrained']['job_id'], 'paired=', row['paired_targets'],
                  'median areas=', row['reference_area_pixels']['median'], row['retrained_area_pixels']['median'],
                  'median ratio=', row['retrained_to_reference_area_ratio']['median'])


if __name__ == '__main__':
    main()
