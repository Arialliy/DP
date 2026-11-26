"""Describe real false-positive geometry without changing frozen evaluation code.

Run inside the evaluation container: breakdown.py JOB_ID --threshold .99.
CPU-only validation: breakdown.py --self-test --dataset /data/RGB-T.
No predictions are saved. Only one clip is consumed at a time. Distance bins
describe geometry relative to the PNG foreground; they do not identify causes
or exclude any influence of training masks. Prediction-component associations
are many-to-many and are not independent object detections or causal attributions.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
from scipy import ndimage

from common import ROOT, config, read_json, save_json, sha256, verify_source
from official_metrics import THRESHOLDS, _protected_pixels, _target_maxima, aggregate_results, score_frame


BIN_NAMES = ('d_0_4', 'd_4_8', 'd_8_16', 'd_16_32', 'd_gt_32', 'empty_gt')
SCALARS = ('frames', 'pixels', 'targets', 'true_targets', 'false_pixels',
           'foreground_fp_pixels', 'near_fp_pixels', 'far_fp_pixels',
           'empty_gt_fp_pixels', 'foreground_pixels', 'prediction_pixels',
           'protected_prediction_pixels')
CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)


def threshold_index(value):
    indices = np.flatnonzero(np.isclose(THRESHOLDS, value, rtol=0, atol=1e-15))
    # Near zero, a tolerance could alias multiple official thresholds.
    exact = np.flatnonzero(THRESHOLDS == value)
    indices = exact if len(exact) else indices
    if len(indices) != 1:
        raise ValueError('Choose one of the original 27 thresholds: ' + str(THRESHOLDS.tolist()))
    return int(indices[0])


def area_summary(values):
    values = np.asarray(values, dtype=np.int64)
    if not values.size:
        return dict(count=0, sum=0, min=None, median=None, mean=None, p95=None, max=None)
    return dict(count=int(values.size), sum=int(values.sum()), min=int(values.min()),
                median=float(np.median(values)), mean=float(values.mean()),
                p95=float(np.percentile(values, 95)), max=int(values.max()))


def component_associations(foreground, prediction, frame_id):
    """Use exactly the author's +/-1 Python slices, including empty borders."""
    gt_labels, gt_count = ndimage.label(foreground, structure=CONNECTIVITY)
    pred_labels, pred_count = ndimage.label(prediction, structure=CONNECTIVITY)
    areas = np.bincount(pred_labels.ravel(), minlength=pred_count + 1)
    targets, linked = [], defaultdict(list)
    for target_id, bounds in enumerate(ndimage.find_objects(gt_labels), 1):
        coordinates = np.argwhere(gt_labels[bounds] == target_id)
        coordinates += np.array([bounds[0].start, bounds[1].start])
        ids = set()
        for row, column in coordinates:
            ids.update(int(x) for x in np.unique(pred_labels[row-1:row+2, column-1:column+2]) if x)
        ids = sorted(ids)
        for component_id in ids:
            linked[component_id].append(target_id)
        targets.append(dict(frame_id=frame_id, target_id=target_id,
                            gt_area=int(len(coordinates)), hit=bool(ids),
                            hit_component_area_sum=int(sum(areas[x] for x in ids)),
                            hit_prediction_components=[dict(component_id=x, area=int(areas[x])) for x in ids]))
    for target in targets:
        for component in target['hit_prediction_components']:
            component['associated_gt_count'] = len(linked[component['component_id']])
    return targets, dict(
        prediction_components=int(pred_count), hit_prediction_components=len(linked),
        unmatched_prediction_components=int(pred_count - len(linked)),
        target_component_associations=sum(len(x) for x in linked.values()),
        targets_with_multiple_components=sum(len(x['hit_prediction_components']) > 1 for x in targets),
        components_hitting_multiple_targets=sum(len(x) > 1 for x in linked.values()),
        hit_component_areas=[int(areas[x]) for x in linked])


def frame_breakdown(probability, mask, index, frame_id='', *, exact_fa=None):
    """Return a descriptive record plus unmodified official 27-point counts."""
    raw = score_frame(probability, mask)  # validates FP32, shape and probability range
    foreground = np.asarray(mask) > 0
    prediction = probability >= THRESHOLDS[index]
    rows, columns = np.nonzero(foreground)
    protected = _protected_pixels(foreground.shape, rows, columns)
    if exact_fa is not None:
        labels, count = ndimage.label(foreground, structure=CONNECTIVITY)
        maxima = _target_maxima(probability, labels, count, rows, columns)
        exact_fa.update_scores(probability[~protected], maxima, pixels=int(probability.size), frames=1)
    false_mask = prediction & ~protected
    bins = dict.fromkeys(BIN_NAMES, 0)
    foreground_fp = 0
    if rows.size:
        distances = ndimage.distance_transform_cdt(~foreground, metric='chessboard')[false_mask]
        counts = np.bincount(np.searchsorted([4, 8, 16, 32], distances, side='left'), minlength=5)
        bins.update(zip(BIN_NAMES[:5], (int(x) for x in counts)))
        foreground_fp = int(np.count_nonzero(distances == 0))
    else:
        bins['empty_gt'] = int(np.count_nonzero(false_mask))
    targets, associations = component_associations(foreground, prediction, frame_id)
    record = dict(frames=1, pixels=int(probability.size),
                  targets=int(raw['targets'][index]), true_targets=int(raw['true_targets'][index]),
                  false_pixels=int(raw['false_pixels'][index]), fp_distance_bins=bins,
                  foreground_fp_pixels=foreground_fp,
                  near_fp_pixels=sum(bins[x] for x in BIN_NAMES[:4]) - foreground_fp,
                  far_fp_pixels=bins['d_gt_32'], empty_gt_fp_pixels=bins['empty_gt'],
                  foreground_pixels=int(rows.size), prediction_pixels=int(prediction.sum()),
                  protected_prediction_pixels=int(np.count_nonzero(prediction & protected)),
                  per_target=targets, associations=associations)
    assert sum(bins.values()) == record['false_pixels'] == int(false_mask.sum())
    assert sum(x['hit'] for x in targets) == record['true_targets']
    assert len(targets) == record['targets']
    assert record['prediction_pixels'] == record['false_pixels'] + record['protected_prediction_pixels']
    return record, raw


def combine_records(records, *, keep_targets=False):
    totals = {key: sum(x[key] for x in records) for key in SCALARS}
    totals['fp_distance_bins'] = {key: sum(x['fp_distance_bins'][key] for x in records) for key in BIN_NAMES}
    totals['pd'] = totals['true_targets'] / totals['targets'] if totals['targets'] else None
    totals['fa'] = totals['false_pixels'] / totals['pixels'] if totals['pixels'] else None
    targets = [item for record in records for item in record['per_target']]
    associations = {key: sum(x['associations'][key] for x in records) for key in (
        'prediction_components', 'hit_prediction_components', 'unmatched_prediction_components',
        'target_component_associations', 'targets_with_multiple_components', 'components_hitting_multiple_targets')}
    areas = [area for record in records for area in record['associations']['hit_component_areas']]
    totals['object_component_summary'] = dict(
        **associations, hit_targets=totals['true_targets'], missed_targets=totals['targets'] - totals['true_targets'],
        gt_area=area_summary([x['gt_area'] for x in targets]),
        per_hit_target_associated_component_area_sum=area_summary([x['hit_component_area_sum'] for x in targets if x['hit']]),
        unique_hit_prediction_component_area=area_summary(areas))
    assert sum(totals['fp_distance_bins'].values()) == totals['false_pixels']
    assert sum(totals[k] for k in ('foreground_fp_pixels', 'near_fp_pixels', 'far_fp_pixels', 'empty_gt_fp_pixels')) == totals['false_pixels']
    if keep_targets:
        totals['per_target'] = targets
    return totals


def official_record(raw):
    return dict(
        auc=float(raw['auc']) if np.isfinite(raw['auc']) else None,
        iou_at_0_5=float(raw['iou']) if np.isfinite(raw['iou']) else None,
        intersection=int(raw['intersection']), union=int(raw['union']),
        thresholds=[dict(threshold=float(threshold), true_targets=int(raw['true_targets'][i]),
                         targets=int(raw['targets'][i]), false_pixels=int(raw['false_pixels'][i]),
                         pd=float(raw['pd'][i]) if np.isfinite(raw['pd'][i]) else None,
                         fa=float(raw['fa'][i])) for i, threshold in enumerate(THRESHOLDS)])


def compare_reference(record, path, checkpoint_sha256, complete):
    reference = read_json(path)
    result = dict(path=str(path), status='missing' if reference is None else 'partial')
    if reference is None or not complete:
        return result
    differences = []
    if reference.get('checkpoint_sha256') != checkpoint_sha256:
        differences.append(dict(field='checkpoint_sha256', expected=reference.get('checkpoint_sha256'), actual=checkpoint_sha256))
    reference_points = {float(x['threshold']): x for x in reference.get('thresholds', [])}
    for actual in record['thresholds']:
        expected = reference_points.get(actual['threshold'])
        if expected is None:
            differences.append(dict(threshold=actual['threshold'], field='threshold', expected='present', actual='missing in reference'))
            continue
        for key in ('true_targets', 'targets', 'false_pixels'):
            if expected.get(key) != actual[key]:
                differences.append(dict(threshold=actual['threshold'], field=key, expected=expected.get(key), actual=actual[key]))
    for key in ('frames', 'pixels', 'intersection', 'union'):
        if key in reference and reference[key] != record[key]:
            differences.append(dict(field=key, expected=reference[key], actual=record[key]))
    result.update(status='mismatch' if differences else 'matched', count_differences=differences,
                  checked_thresholds=len(record['thresholds']), reference_metrics_sha256=sha256(path),
                  auc_reference=reference.get('auc'), auc_actual=record['auc'],
                  iou_reference=reference.get('iou_at_0_5'), iou_actual=record['iou_at_0_5'],
                  note='Integer counts compared exactly. Reference IoU may have been rounded; both values retained.')
    return result


def read_mask(path):
    with Image.open(path) as image:
        mask = np.asarray(image, dtype=np.uint8)
    return mask[..., 0] if mask.ndim == 3 else mask


def self_test(dataset):
    """Synthetic predictions are exclusively unit fixtures, never model results."""
    rng = np.random.default_rng(4661)
    count = border_count = 0
    fixtures = []
    for case in range(20):
        mask = np.zeros((67, 73), np.uint8)
        if case == 0:
            mask[0, 0] = 1
        elif case == 1:
            mask[33, 36] = 1
        elif case == 2:
            pass  # empty-GT
        elif case == 3:
            mask[20, 20] = mask[20, 24] = 1  # shared prediction component
        elif case == 18:
            mask[20, 10:31] = 1  # one GT associated with two separate predictions
        elif case == 19:
            mask[20, 20] = mask[21, 21] = 1  # diagonal adjacency is 8-connected
        else:
            mask[rng.integers(0, 67, 12), rng.integers(0, 73, 12)] = 1
        probability = rng.random(mask.shape, dtype=np.float32)
        if case in (0, 1, 2, 3):
            probability.fill(1)
        elif case == 18:
            probability.fill(0)
            probability[20, 10] = probability[20, 30] = 1
        elif case == 19:
            probability = mask.astype(np.float32)
        for value in (0., .5, .99, 1.):
            index = threshold_index(value)
            record, _ = frame_breakdown(probability, mask, index, str(case))
            rows, columns = np.nonzero(mask)
            brute = np.zeros(mask.shape, dtype=bool)
            for row, column in zip(rows, columns):
                brute[row-4:row+5, column-4:column+5] = True
            assert np.array_equal(brute, _protected_pixels(mask.shape, rows, columns))
            # Independent direct distance computation on small fixtures.
            fp = (probability >= THRESHOLDS[index]) & ~brute
            bins = dict.fromkeys(BIN_NAMES, 0)
            for row, column in np.argwhere(fp):
                if len(rows):
                    distance = int(np.min(np.maximum(abs(rows-row), abs(columns-column))))
                    bins[BIN_NAMES[int(np.searchsorted([4, 8, 16, 32], distance, side='left'))]] += 1
                else:
                    bins['empty_gt'] += 1
            assert bins == record['fp_distance_bins']
            border_count += record['fp_distance_bins']['d_0_4'] > 0
            if case == 0:
                assert record['foreground_fp_pixels'] == 1 and record['true_targets'] == 0
            if case == 3:
                assert record['associations']['components_hitting_multiple_targets'] == 1
            if case == 18 and value > 0:
                assert record['associations']['targets_with_multiple_components'] == 1
            if case == 19:
                assert record['targets'] == 1
            fixtures.append(record)
            count += 1
    combined = combine_records(fixtures, keep_targets=True)
    assert combined['true_targets'] == sum(x['hit'] for x in combined['per_target'])
    real_frames = 0
    for clip in dict.fromkeys(x.split('/')[0] for x in (dataset / 'test.txt').read_text().split()):
        paths = sorted((dataset / 'segmentations' / clip).glob('*.png'))
        for frame in sorted({0, len(paths)//2, len(paths)-1}):
            mask = read_mask(paths[frame])
            probability = np.zeros(mask.shape, dtype=np.float32)
            probability[mask > 0] = np.float32(.995)
            probability[::31, ::29] = 1
            record, _ = frame_breakdown(probability, mask, threshold_index(.99), paths[frame].stem)
            assert record['false_pixels'] == sum(record['fp_distance_bins'].values())
            real_frames += 1
    from exact_fa import ExactFaAccumulator, extract_frame_scores
    integrated = ExactFaAccumulator(pixels_total=12*17*23, fa_budgets=(.02, .2, .7), expected_targets=None, buffer_size=89)
    independent = ExactFaAccumulator(pixels_total=12*17*23, fa_budgets=(.02, .2, .7), expected_targets=None, buffer_size=89)
    all_fp, all_maxima = [], []
    for frame in range(12):
        mask = np.zeros((17, 23), np.uint8)
        if frame % 3:
            mask[8, 11] = 1
        if frame % 2:
            mask[0, 0] = 1
        probability = rng.choice(np.array([0, .5, .99, 1], dtype=np.float32), size=mask.shape)
        frame_breakdown(probability, mask, threshold_index(.99), str(frame), exact_fa=integrated)
        independent.update(probability, mask)
        extracted = extract_frame_scores(probability, mask)
        all_fp.append(extracted['fp_scores'])
        all_maxima.append(extracted['target_maxima'])
    exact_result = integrated.finalize()
    assert exact_result == independent.finalize()
    fp, maxima = np.concatenate(all_fp), np.concatenate(all_maxima)
    for budget in exact_result['budgets']:
        for name in ('best_point', 'lower_fa_point', 'upper_fa_point'):
            point = budget[name]
            if point is not None:
                threshold = np.float32(point['threshold'])
                assert point['false_pixels'] == int(np.count_nonzero(fp >= threshold))
                assert point['true_targets'] == int(np.count_nonzero(maxima >= threshold))
    result = dict(status='passed', synthetic_cases=count, original_border_cases_observed=border_count,
                  real_png_frames=real_frames, real_predictions=False,
                  exact_fa_integration_frames=12, exact_fa_points_equal_full_score_counts=True,
                  note='All probabilities here are CPU unit fixtures; no model inference and no benchmark numbers.')
    print(result, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job_id', nargs='?')
    parser.add_argument('--threshold', type=float, default=.99)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dataset', type=Path, default=Path(os.environ.get('VT_ROOT', '/data/RGB-T')))
    parser.add_argument('--artifacts', type=Path, default=Path('/artifacts'))
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'diagnostics/breakdown')
    parser.add_argument('--variants', nargs='+', help='Optional subset; every requested variant remains explicit')
    parser.add_argument('--max-clips', type=int, help='Partial diagnostic only; complete will be false')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(args.dataset)
        return
    if not args.job_id or Path(args.job_id).name != args.job_id:
        parser.error('A plain job_id is required')
    if args.max_clips is not None and args.max_clips < 1:
        parser.error('--max-clips must be positive')
    index = threshold_index(args.threshold)
    import torch
    from exact_fa import ExactFaAccumulator
    from inference import load_model, predict_clip, prediction_protocol
    torch.set_num_threads(4)
    cfg = config()
    job = read_json(ROOT / 'jobs' / (args.job_id + '.json'))
    if not job:
        raise FileNotFoundError('Frozen job manifest not found: ' + args.job_id)
    variants = args.variants or job['variants']
    if len(set(variants)) != len(variants) or not set(variants) <= set(job['variants']):
        raise ValueError('Variants must be a unique subset of the job variants')
    output = args.output_dir / (args.job_id + '.json')
    progress_path = args.output_dir / (args.job_id + '.status.json')
    started = time.monotonic()
    model_cfg = next((x for x in cfg['models'] if x['id'] == job['model_id']), {})
    report = {key: job.get(key) for key in ('job_id', 'model_id', 'model', 'arch', 'checkpoint_sha256',
                                         'training_epochs_completed', 'final_checkpoint')}
    report.update(schema_version=1, run=job.get('run', model_cfg.get('run', args.job_id)),
                  checkpoint_status=job.get('checkpoint_status', 'final' if job.get('final_checkpoint') else 'interim'),
                  split='official_test', threshold=float(THRESHOLDS[index]), status='running', complete=False,
                  device=args.device, dtype='float32', frames=0, pixels=0, results={},
                  expected_frames=cfg['expected_frames'], expected_pixels=cfg['expected_pixels'],
                  expected_targets=cfg['expected_targets'], started_at=datetime.now(timezone.utc).isoformat(),
                  breakdown_source_sha256=sha256(Path(__file__)),
                  exact_fa_source_sha256=sha256(ROOT / 'exact_fa.py'),
                  source_manifest_sha256=sha256(ROOT / 'source_manifest.json'),
                  geometry_protocol=dict(
                      distance='Chebyshev distance in source pixels to nearest PNG>0 pixel',
                      bins=dict(zip(BIN_NAMES, ('[0,4]', '(4,8]', '(8,16]', '(16,32]', '>32', 'frame has no PNG foreground'))),
                      protection='official_metrics._protected_pixels, original +/-4 slices including border anomalies',
                      matching='8-connected GT and prediction components; original +/-1 slices per GT pixel',
                      counts='foreground_fp_pixels: distance0; near_fp_pixels: distance1..32; far_fp_pixels: >32; empty_gt separate',
                      component_areas='Whole prediction-component areas. Per-GT union sums unique associated components; the same component can associate to multiple GTs.',
                      interpretation='Descriptive geometry only; no causal attribution and no exclusion of training-mask influence.'),
                  predictions_saved=False)

    def update(**values):
        save_json(progress_path, dict(job_id=args.job_id, checkpoint_sha256=job['checkpoint_sha256'],
                                     elapsed_seconds=time.monotonic()-started,
                                     updated_at=datetime.now(timezone.utc).isoformat(), **values))

    def persist():
        report.update(elapsed_seconds=time.monotonic()-started, updated_at=datetime.now(timezone.utc).isoformat())
        save_json(output, report)

    try:
        verify_source()
        checkpoint = ROOT / job['checkpoint']
        if sha256(checkpoint) != job['checkpoint_sha256']:
            raise ValueError('Checkpoint SHA256 differs from frozen job')
        train_cfg = read_json(ROOT / job['train_config'])
        report['train_config_sha256'] = sha256(ROOT / job['train_config'])
        protocol = prediction_protocol(job['arch'], train_cfg)
        if set(protocol['variants']) != set(job['variants']):
            raise ValueError('Job variants differ from frozen inference protocol')
        report['inference_protocol'] = protocol
        clips = list(dict.fromkeys(x.split('/')[0] for x in (args.dataset / 'test.txt').read_text().split()))
        if clips != cfg['test_clips']:
            raise ValueError('Official test split changed')
        report['test_split_sha256'] = sha256(args.dataset / 'test.txt')
        clips = clips if args.max_clips is None else clips[:args.max_clips]
        update(stage='loading', clips_total=len(clips), clips_done=0)
        model, needs = load_model(job['arch'], checkpoint, args.device)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        report['best_epoch'] = int(payload.get('best_epoch', int(payload['epoch']) + 1))
        del payload
        frame_records = {variant: [] for variant in variants}
        exact_collectors = {variant: ExactFaAccumulator(pixels_total=cfg['expected_pixels'], expected_targets=cfg['expected_targets'])
                            for variant in variants}
        clip_raw = {variant: [] for variant in variants}
        clip_output = {variant: [] for variant in variants}
        persist()
        for clip_index, clip in enumerate(clips):
            images = sorted((args.dataset / 'test2017' / clip / '01').iterdir())
            masks = sorted((args.dataset / 'segmentations' / clip).iterdir())
            if not images or len(images) != len(masks) or any(a.stem != b.stem for a, b in zip(images, masks)):
                raise ValueError('Image/PNG alignment mismatch: ' + clip)
            video = np.asarray(np.load(args.artifacts / 'cache/video' / (clip + '.npy'), mmap_mode='r'), dtype=np.float32)
            if len(video) != len(images):
                raise ValueError('Cached frame count mismatch: ' + clip)
            for frame in sorted({0, len(video)//2, len(video)-1}):
                with Image.open(images[frame]) as image:
                    original = np.asarray(image, dtype=np.float32)
                original = original[..., 0] if original.ndim == 3 else original
                if not np.array_equal(original, video[frame]):
                    raise ValueError('Cached image mismatch: ' + clip)
            seen = set()
            update(stage='inference', clip=clip, clips_done=clip_index, clips_total=len(clips))
            for variant, probabilities in predict_clip(model, needs, video, clip, train_cfg, args.device,
                                                       progress=lambda values: update(stage='inference', **values)):
                if variant not in variants:
                    del probabilities
                    continue
                if variant in seen or probabilities.shape != video.shape or probabilities.dtype != np.float32:
                    raise ValueError('Invalid inference output: ' + variant)
                seen.add(variant)
                records, raw = [], []
                for frame, path in enumerate(masks):
                    record, counts = frame_breakdown(probabilities[frame], read_mask(path), index, path.stem,
                                                     exact_fa=exact_collectors[variant])
                    records.append(record)
                    raw.append(counts)
                    if (frame + 1) % 100 == 0:
                        update(stage='scoring', clip=clip, variant=variant, scoring_frame=frame + 1,
                               scoring_total=len(masks), clips_done=clip_index, clips_total=len(clips))
                counts = aggregate_results(raw)
                clip_raw[variant].append(counts)
                combined = combine_records(records, keep_targets=True)
                combined.update(clip=clip, **official_record(counts))
                clip_output[variant].append(combined)
                # Small counts/associations only; never retain probability maps.
                frame_records[variant].extend(records)
                del probabilities, records, raw
            if seen != set(variants):
                raise ValueError('Requested inference variants were not all produced')
            report['frames'] += len(video)
            report['pixels'] += int(video.size)
            del video
            gc.collect()
            for variant in variants:
                combined = combine_records(frame_records[variant])
                combined.update(variant=variant, status='running', complete=False,
                                **official_record(aggregate_results(clip_raw[variant])), per_clip=clip_output[variant])
                report['results'][variant] = combined
            persist()
            print(f'{job["model"]} breakdown {clip}: {clip_index+1}/{len(clips)} clips', flush=True)
        complete = clips == cfg['test_clips']
        if complete:
            assert report['frames'] == cfg['expected_frames'] and report['pixels'] == cfg['expected_pixels']
        for variant, result in report['results'].items():
            if complete:
                assert result['targets'] == cfg['expected_targets']
                result['exact_matched_fa'] = exact_collectors[variant].finalize()
            else:
                result['exact_matched_fa'] = dict(status='partial', complete=False,
                                         note='Full-test denominators required; exact Fa statistics not finalized.')
            result['reference_comparison'] = compare_reference(result, ROOT / 'results' / args.job_id / variant / 'metrics.json',
                                                               job['checkpoint_sha256'], complete)
            result.update(complete=complete, status=('mismatch' if result['reference_comparison']['status'] == 'mismatch'
                                                    else 'complete') if complete else 'partial')
        report.update(complete=complete, clips=clips,
                      status='mismatch' if any(x['status'] == 'mismatch' for x in report['results'].values()) else ('complete' if complete else 'partial'))
        persist()
        update(stage=report['status'], complete=complete, frames=report['frames'], output=str(output))
        print(f'{output}: {report["status"]}', flush=True)
    except BaseException as exc:
        report.update(status='failed', complete=False, error=repr(exc), traceback=traceback.format_exc())
        persist()
        update(stage='failed', error=repr(exc), output=str(output))
        raise


if __name__ == '__main__':
    main()
