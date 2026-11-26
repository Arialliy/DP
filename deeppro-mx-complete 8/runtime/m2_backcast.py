#!/usr/bin/env python3
"""M2 frozen prediction-only backcasting; generation and GT evaluation are separate commands.

generate reads only the fixed plan, prediction arrays and their provenance. It does not
import project data loaders. evaluate verifies the sealed files before reading GT.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

MODELS = ('raw_stop', 'raw_cv')
POINT_FIELDS = ('clip', 'model', 't', 'track_id', 'x', 'y', 'status', 'suppressed_by',
                'horizon', 'n_fit', 'teacher_logit', 'fit_first', 'fit_last')
TEACHER_FIELDS = ('clip', 'track_id', 't', 'x', 'y', 'logit', 'in_fit_window')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def maximum_valid_assignment(distance, valid):
    """Maximum valid cardinality, then minimum distance; invalid pairs are never returned."""
    distance, valid = np.asarray(distance, float), np.asarray(valid, bool)
    if distance.ndim != 2 or distance.shape != valid.shape:
        raise ValueError('distance/valid must be equally shaped matrices')
    if not distance.size or not valid.any():
        return []
    if not np.isfinite(distance[valid]).all() or (distance[valid] < 0).any():
        raise ValueError('valid distances must be finite and nonnegative')
    # One invalid assignment costs more than the sum of every possible valid cost.
    penalty = (min(distance.shape) + 1) * (float(distance[valid].max()) + 1.0)
    cost = np.where(valid, distance, penalty)
    rr, cc = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(rr, cc) if valid[i, j]]


def extract_teacher(frame_logit, t, params):
    if t < params['teacher_first']:
        return [], dict(local_peaks=0, retained=0, capped=False)
    logit = np.asarray(frame_logit, np.float32)
    mx = ndimage.maximum_filter(logit, size=params['teacher_peak_nms_size'], mode='nearest')
    # Existing floor=-12, followed by inclusive logit>=0: exact p=.5 is retained.
    mask = (logit >= mx) & (logit > -12.0) & (logit >= params['teacher_logit_min_inclusive']) & np.isfinite(logit)
    lab, count = ndimage.label(mask, structure=np.ones((3, 3)))
    if not count:
        return [], dict(local_peaks=0, retained=0, capped=False)
    yy, xx = np.indices(logit.shape)
    index = np.arange(1, count + 1)
    x = ndimage.mean(xx, lab, index)
    y = ndimage.mean(yy, lab, index)
    score = ndimage.maximum(logit, lab, index)
    order = np.lexsort((x, y, -score))[:params['teacher_peak_topk']]
    rows = [dict(t=int(t), x=float(x[i]), y=float(y[i]), logit=float(score[i])) for i in order]
    return rows, dict(local_peaks=int(count), retained=len(rows), capped=count > len(rows))


def associate(rows, params):
    """No GT or motion-model preference: link to the last observed detection in raw coordinates."""
    by_frame = defaultdict(list)
    for row in rows:
        if row['t'] >= params['teacher_first']:
            by_frame[int(row['t'])].append(dict(row))
    tracks = []
    for t in sorted(by_frame):
        detections = sorted(by_frame[t], key=lambda p: (-p['logit'], p['y'], p['x']))
        active = [track for track in tracks
                  if 0 < t - track['observations'][-1]['t'] <= params['association_max_gap']]
        matches = []
        if active and detections:
            a = np.array([[tr['observations'][-1]['x'], tr['observations'][-1]['y']] for tr in active])
            b = np.array([[p['x'], p['y']] for p in detections])
            dt = np.array([t - tr['observations'][-1]['t'] for tr in active])
            distance = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
            valid = distance <= params['association_max_px_per_frame'] * dt[:, None]
            matches = maximum_valid_assignment(distance, valid)
        assigned = set()
        for i, j in matches:
            active[i]['observations'].append(detections[j])
            assigned.add(j)
        for j, point in enumerate(detections):
            if j not in assigned:
                tracks.append(dict(track_id=len(tracks), observations=[point]))
    return tracks


def fit_track(track, params):
    observations = track['observations']
    first, last = observations[0]['t'], observations[-1]['t']
    fit = [p for p in observations if first <= p['t'] <= first + params['fit_first_window_frames'] - 1]
    ts = np.array([p['t'] for p in fit], float)
    xy = np.array([[p['x'], p['y']] for p in fit], float)
    out = dict(track_id=track['track_id'], first=int(first), last=int(last), n_observations=len(observations),
               n_fit=len(fit), fit_first=int(ts[0]), fit_last=int(ts[-1]),
               fit_span=int(ts[-1] - ts[0]), teacher_logit=float(np.median([p['logit'] for p in fit])),
               stop_ok=False, cv_ok=False, stop_rms=None, cv_rms=None,
               stop_x=None, stop_y=None, cv_anchor_t=None, cv_x=None, cv_y=None, cv_vx=None, cv_vy=None)
    if len(fit) < params['minimum_fit_points'] or out['fit_span'] < params['minimum_fit_span']:
        out['fit_status'] = 'insufficient_observations'
        return out
    stop = np.median(xy, axis=0)
    stop_rms = float(np.sqrt(np.mean(np.sum((xy - stop) ** 2, axis=1))))
    anchor = float(ts.mean())
    design = np.c_[ts - anchor, np.ones(len(ts))]
    # Centered one-regressor OLS in closed form avoids an unnecessary SVD's
    # tiny boundary shifts for exactly linear trajectories (e.g. x=0 at t=10).
    mean_xy = xy.mean(axis=0)
    centered_time = ts - anchor
    velocity = (centered_time[:, None] * (xy - mean_xy)).sum(axis=0) / np.dot(centered_time, centered_time)
    beta = np.vstack([velocity, mean_xy])
    if np.all(xy == xy[0]):  # Exact constant trajectory; avoid roundoff changing a 3px NMS boundary.
        beta = np.vstack([np.zeros(2), xy[0]])
    cv_rms = float(np.sqrt(np.mean(np.sum((xy - design @ beta) ** 2, axis=1))))
    out.update(fit_status='fitted', stop_x=float(stop[0]), stop_y=float(stop[1]), stop_rms=stop_rms,
               cv_anchor_t=anchor, cv_vx=float(beta[0, 0]), cv_vy=float(beta[0, 1]),
               cv_x=float(beta[1, 0]), cv_y=float(beta[1, 1]), cv_rms=cv_rms,
               stop_ok=stop_rms <= params['maximum_fit_rms_px'], cv_ok=cv_rms <= params['maximum_fit_rms_px'])
    return out


def project_tracks(tracks_with_fit, n_frames, height, width, params):
    points = []
    for fit in tracks_with_fit:
        for model, accepted in (('raw_stop', fit['stop_ok']), ('raw_cv', fit['cv_ok'])):
            if not accepted:
                continue
            for t in range(params['early_first'], min(n_frames - 1, params['early_last']) + 1):
                if model == 'raw_stop':
                    x, y = fit['stop_x'], fit['stop_y']
                else:
                    dt = t - fit['cv_anchor_t']
                    x, y = fit['cv_x'] + dt * fit['cv_vx'], fit['cv_y'] + dt * fit['cv_vy']
                finite = math.isfinite(x) and math.isfinite(y)
                status = ('nonfinite' if not finite else 'kept' if 0 <= x <= width - 1 and 0 <= y <= height - 1
                          else 'out_of_bounds')
                points.append(dict(model=model, t=t, track_id=fit['track_id'], x=float(x) if finite else None,
                                   y=float(y) if finite else None, status=status, suppressed_by='',
                                   horizon=fit['first'] - t, n_fit=fit['n_fit'], teacher_logit=fit['teacher_logit'],
                                   fit_first=fit['fit_first'], fit_last=fit['fit_last']))
    groups = defaultdict(list)
    for point in points:
        if point['status'] == 'kept':
            groups[point['model'], point['t']].append(point)
    radius = params['candidate_nms_radius_px']
    for rows in groups.values():
        kept = []
        for point in sorted(rows, key=lambda p: (-p['n_fit'], -p['teacher_logit'], p['track_id'])):
            suppressor = next((p for p in kept if math.hypot(point['x'] - p['x'], point['y'] - p['y']) <= radius), None)
            if suppressor is None:
                kept.append(point)
            else:
                point.update(status='nms_suppressed', suppressed_by=suppressor['track_id'])
    return sorted(points, key=lambda p: (p['model'], p['t'], p['track_id']))


def match_points(points, gt, radius=3.0):
    if not points or not gt:
        return []
    a = np.array([[p['x'], p['y']] for p in points], float)
    b = np.array([[p['x'], p['y']] for p in gt], float)
    distance = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return [(i, j, float(distance[i, j])) for i, j in maximum_valid_assignment(distance, distance <= radius)]


def _generate_clip(task):
    root, out, clip, plan, checkpoint_sha = task
    root, out = Path(root), Path(out)
    params = plan['parameters']
    prediction = root / plan['predictions'] / (clip + '.npy')
    metadata = prediction.parent.parent / (prediction.parent.name + '__meta') / (clip + '.json')
    fingerprint = read_json(metadata)['fingerprint']
    if fingerprint['clip'] != clip or fingerprint['ckpt_sha256'] != checkpoint_sha:
        raise ValueError(clip + ': prediction checkpoint provenance mismatch')
    before = prediction.stat()
    logits = np.load(prediction, mmap_mode='r')
    if logits.ndim != 3:
        raise ValueError(clip + ': expected N,H,W prediction array')
    n, h, w = map(int, logits.shape)
    if fingerprint['data']['frames'] != n:
        raise ValueError(clip + ': metadata frame count mismatch')
    rows, peak_stats = [], []
    for t in range(params['teacher_first'], n):
        points, stats = extract_teacher(logits[t], t, params)
        rows.extend(points)
        peak_stats.append(dict(t=t, **stats))
    tracks = associate(rows, params)
    fits = [fit_track(track, params) for track in tracks]
    projected = project_tracks(fits, n, h, w, params)
    teacher = []
    for track, fit in zip(tracks, fits):
        for point in track['observations']:
            teacher.append(dict(clip=clip, track_id=track['track_id'], **point,
                                in_fit_window=point['t'] <= fit['first'] + params['fit_first_window_frames'] - 1))
    paths = {kind: out / clip / (kind + '.csv') for kind in ('tracks', 'teacher_points', 'points', 'teacher_frames')}
    write_csv(paths['tracks'], [dict(clip=clip, **fit) for fit in fits],
              ['clip', *list(fits[0])] if fits else ['clip', 'track_id', 'first', 'last', 'n_fit', 'stop_ok', 'cv_ok'])
    write_csv(paths['teacher_points'], teacher, TEACHER_FIELDS)
    write_csv(paths['points'], [dict(clip=clip, **point) for point in projected], POINT_FIELDS)
    write_csv(paths['teacher_frames'], peak_stats, ('t', 'local_peaks', 'retained', 'capped'))
    prediction_sha = sha256(prediction)
    after = prediction.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(clip + ': predictions changed during generation')
    return dict(clip=clip, shape=[n, h, w], early_frames=min(n, params['early_last'] + 1),
                teacher_frames=max(0, n - params['teacher_first']), teacher_points=len(teacher),
                teacher_frames_capped=sum(r['capped'] for r in peak_stats), tracks=len(tracks),
                sufficient_fit_tracks=sum(f['fit_status'] == 'fitted' for f in fits),
                accepted_tracks={model: sum(f['stop_ok' if model == 'raw_stop' else 'cv_ok'] for f in fits) for model in MODELS},
                projected_counts={model: dict(Counter(p['status'] for p in projected if p['model'] == model)) for model in MODELS},
                sources={'prediction': str(prediction.relative_to(root)), 'prediction_sha256': prediction_sha,
                         'metadata': str(metadata.relative_to(root)), 'metadata_sha256': sha256(metadata)},
                files={str(path.relative_to(out)): sha256(path) for path in paths.values()})


def generate(root, plan_path, out, nproc=4):
    root, plan_path, out = Path(root).resolve(), Path(plan_path).resolve(), Path(out).resolve()
    plan = read_json(plan_path)
    if plan['models'] != list(MODELS) or plan['coords'] != 'raw' or plan['homography_enabled']:
        raise ValueError('This frozen diagnostic implements only raw_stop and raw_cv')
    generation = out / 'generation'
    if generation.exists() and any(generation.iterdir()):
        raise ValueError('Generation output already exists; immutable generation is not overwritten')
    generation.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = sha256(root / plan['teacher_checkpoint'])
    tasks = [(str(root), str(generation), clip, plan, checkpoint_sha) for clip in plan['clips']]
    records = []
    with Pool(min(max(1, nproc), len(tasks))) as pool:
        for record in pool.imap_unordered(_generate_clip, tasks):
            records.append(record)
            print('generated', record['clip'], 'teacher_points', record['teacher_points'],
                  'tracks', record['tracks'], 'accepted', record['accepted_tracks'], flush=True)
    records.sort(key=lambda r: plan['clips'].index(r['clip']))
    manifest = dict(status='generation_complete', gt_loaded=False, plan_path=str(plan_path.relative_to(root)),
                    plan_sha256=sha256(plan_path), generator_sha256=sha256(__file__),
                    checkpoint_sha256=checkpoint_sha, parameters=plan['parameters'], clips=records)
    write_json(generation / 'manifest.json', manifest)
    print('sealed', generation / 'manifest.json', 'GT not loaded', flush=True)
    return manifest


def parse_bool(value):
    if value not in ('True', 'False', True, False):
        raise ValueError('invalid boolean: ' + str(value))
    return value in ('True', True)


def csv_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def distance_band(value, limits):
    low = 1
    for high in limits:
        if value <= high:
            return f'{low}-{high}'
        low = high + 1
    return f'>{limits[-1]}'


def evaluate(root, plan_path, out, targets_path):
    root, plan_path, out, targets_path = map(lambda p: Path(p).resolve(), (root, plan_path, out, targets_path))
    plan = read_json(plan_path)
    generation = out / 'generation'
    manifest = read_json(generation / 'manifest.json')
    if manifest['status'] != 'generation_complete' or manifest['gt_loaded'] is not False:
        raise ValueError('Evaluation requires a sealed GT-free generation')
    if manifest['plan_sha256'] != sha256(plan_path) or manifest['generator_sha256'] != sha256(__file__):
        raise ValueError('Plan/code changed after generation')
    for clip in manifest['clips']:
        for path, digest in clip['files'].items():
            if sha256(generation / path) != digest:
                raise ValueError('Sealed generation file changed: ' + path)
    # First GT read occurs here, after every proposal output has been verified.
    if plan.get('baseline_hit_source_sha256') and sha256(targets_path) != plan['baseline_hit_source_sha256']:
        raise ValueError('Frozen GT evaluation source changed')
    targets = csv_rows(targets_path)
    gt_frames = defaultdict(list)
    for row in targets:
        if row['clip'] not in plan['clips']:
            continue
        row.update(t=int(row['t']), box_index=int(row['box_index']), track=int(row['track']),
                   x=float(row['x']), y=float(row['y']))
        for th in (0.5, 1e-5):
            row[f'hit_{th:g}'] = parse_bool(row[f'hit_{th:g}'])
        gt_frames[row['clip'], row['t']].append(row)
    early = [r for rows in gt_frames.values() for r in rows if r['t'] <= plan['parameters']['early_last']]
    expected = plan.get('expected_first60_baseline_misses', {})
    for th, misses in expected.items():
        actual = sum(not row[f'hit_{float(th):g}'] for row in early)
        if actual != misses:
            raise ValueError(f'Baseline miss count {th}: {actual}, expected {misses}')
    matched_early = {model: set() for model in MODELS}
    any_teacher = set()
    qualified_teacher = {model: set() for model in MODELS}
    teacher_first = {}
    eval_points, evaluated_tracks, per_clip, distance_rows, coverage_rows = [], [], [], [], []
    radius = plan['parameters']['evaluation_radius_px']
    for record in manifest['clips']:
        clip = record['clip']
        fits = {int(r['track_id']): r for r in csv_rows(generation / clip / 'tracks.csv')}
        teachers = defaultdict(list)
        for row in csv_rows(generation / clip / 'teacher_points.csv'):
            row.update(t=int(row['t']), track_id=int(row['track_id']), x=float(row['x']), y=float(row['y']),
                       in_fit_window=parse_bool(row['in_fit_window']))
            teachers[row['t']].append(row)
        fit_truth = defaultdict(Counter)
        for t, points in sorted(teachers.items()):
            gt = gt_frames[clip, t]
            for i, j, _ in match_points(points, gt, radius):
                target_key = (clip, gt[j]['track'])
                any_teacher.add(target_key)
                teacher_first[target_key] = min(t, teacher_first.get(target_key, t))
                if points[i]['in_fit_window']:
                    fit_truth[points[i]['track_id']][gt[j]['track']] += 1
        for track_id, fit in fits.items():
            truth = fit_truth[track_id]
            dominant, maximum = (min(truth.items(), key=lambda kv: (-kv[1], kv[0])) if truth else (None, 0))
            evaluated_tracks.append(dict(clip=clip, track_id=track_id, n_fit=int(fit['n_fit']),
                                         matched_fit_observations=sum(truth.values()), dominant_gt_track=dominant,
                                         dominant_count=maximum, posthoc_purity=maximum / int(fit['n_fit']),
                                         stop_accepted=parse_bool(fit['stop_ok']), cv_accepted=parse_bool(fit['cv_ok']),
                                         gt_track_counts=json.dumps(dict(truth), sort_keys=True)))
            for model, flag in (('raw_stop', 'stop_ok'), ('raw_cv', 'cv_ok')):
                if parse_bool(fit[flag]):
                    for gt_id, count in truth.items():
                        if count >= plan['parameters']['minimum_fit_points']:
                            qualified_teacher[model].add((clip, gt_id))
        points_by_frame = defaultdict(list)
        all_points = csv_rows(generation / clip / 'points.csv')
        for point in all_points:
            if point['status'] == 'kept':
                point.update(t=int(point['t']), track_id=int(point['track_id']), x=float(point['x']),
                             y=float(point['y']), horizon=int(point['horizon']))
                points_by_frame[point['model'], point['t']].append(point)
        for model in MODELS:
            evaluated = []
            for t in range(record['early_frames']):
                points, gt = points_by_frame[model, t], gt_frames[clip, t]
                mapping = {i: (j, d) for i, j, d in match_points(points, gt, radius)}
                for i, point in enumerate(points):
                    match = mapping.get(i)
                    target = gt[match[0]] if match else None
                    if target is not None:
                        matched_early[model].add((clip, t, target['box_index']))
                    row = dict(clip=clip, model=model, t=t, track_id=point['track_id'], x=point['x'], y=point['y'],
                               horizon=point['horizon'], horizon_band=distance_band(point['horizon'], plan['parameters']['distance_bin_upper_frames']),
                               wrong=target is None, gt_box_index=target['box_index'] if target else None,
                               gt_track=target['track'] if target else None, distance=match[1] if match else None,
                               recovered_at_0_5=target is not None and not target['hit_0.5'],
                               recovered_at_1e_5=target is not None and not target['hit_1e-05'])
                    evaluated.append(row)
            eval_points.extend(evaluated)
            original = [p for p in all_points if p['model'] == model]
            status_counts = Counter(p['status'] for p in original)
            clip_early = [r for r in early if r['clip'] == clip]
            wrong = sum(p['wrong'] for p in evaluated)
            base = dict(clip=clip, model=model, frames=record['early_frames'], teacher_points=record['teacher_points'],
                        tracks=record['tracks'], accepted_tracks=record['accepted_tracks'][model], raw_points=len(original),
                        in_image_points=status_counts['kept'] + status_counts['nms_suppressed'],
                        nms_suppressed=status_counts['nms_suppressed'], out_of_bounds=status_counts['out_of_bounds'],
                        nonfinite=status_counts['nonfinite'], kept=len(evaluated), true_candidates=len(evaluated) - wrong,
                        wrong_candidates=wrong, wrong_per_frame=wrong / record['early_frames'],
                        precision=(len(evaluated) - wrong) / len(evaluated) if evaluated else None)
            for th in (0.5, 1e-5):
                missed = [r for r in clip_early if not r[f'hit_{th:g}']]
                recovered = sum((clip, r['t'], r['box_index']) in matched_early[model] for r in missed)
                per_clip.append(dict(**base, baseline_threshold=th, early_targets=len(clip_early), original_misses=len(missed),
                                     recovered=recovered, recovery_fraction=recovered / len(missed) if missed else None,
                                     teacher_any_missed=sum((clip, r['track']) in any_teacher for r in missed),
                                     teacher_qualified_missed=sum((clip, r['track']) in qualified_teacher[model] for r in missed)))
            for band in ('1-30', '31-60', '61-120', '121-240', '>240'):
                chosen = [p for p in evaluated if p['horizon_band'] == band]
                bad = sum(p['wrong'] for p in chosen)
                distance_rows.append(dict(clip=clip, model=model, horizon_band=band, points=len(chosen),
                                          wrong=bad, true_candidates=len(chosen) - bad,
                                          precision=(len(chosen) - bad) / len(chosen) if chosen else None,
                                          recovered_at_0_5=sum(p['recovered_at_0_5'] for p in chosen),
                                          recovered_at_1e_5=sum(p['recovered_at_1e_5'] for p in chosen)))
        print('evaluated', clip, 'early_GT', len([r for r in early if r['clip'] == clip]), flush=True)
    for row in early:
        key = row['clip'], row['track']
        first = teacher_first.get(key)
        coverage_rows.append(dict(clip=row['clip'], t=row['t'], box_index=row['box_index'], gt_track=row['track'],
                                  baseline_hit_0_5=row['hit_0.5'], baseline_hit_1e_5=row['hit_1e-05'],
                                  any_later_teacher=key in any_teacher, first_matched_teacher=first,
                                  distance_to_first_teacher=first - row['t'] if first is not None else None,
                                  stop_qualified_teacher=key in qualified_teacher['raw_stop'],
                                  cv_qualified_teacher=key in qualified_teacher['raw_cv'],
                                  stop_matched=(row['clip'], row['t'], row['box_index']) in matched_early['raw_stop'],
                                  cv_matched=(row['clip'], row['t'], row['box_index']) in matched_early['raw_cv']))
    totals = {}
    for model in MODELS:
        main = [r for r in per_clip if r['model'] == model and r['baseline_threshold'] == .5]
        count_keys = ('frames', 'teacher_points', 'tracks', 'accepted_tracks', 'raw_points', 'in_image_points',
                      'nms_suppressed', 'out_of_bounds', 'nonfinite', 'kept', 'true_candidates', 'wrong_candidates')
        total = {k: sum(r[k] for r in main) for k in count_keys}
        total['precision'] = total['true_candidates'] / total['kept'] if total['kept'] else None
        total['wrong_per_frame'] = total['wrong_candidates'] / total['frames']
        total['baseline_thresholds'] = {}
        for th in (.5, 1e-5):
            rr = [r for r in per_clip if r['model'] == model and r['baseline_threshold'] == th]
            summed = {k: sum(r[k] for r in rr) for k in ('early_targets', 'original_misses', 'recovered',
                                                       'teacher_any_missed', 'teacher_qualified_missed')}
            summed['recovery_fraction'] = summed['recovered'] / summed['original_misses'] if summed['original_misses'] else None
            summed['teacher_any_coverage'] = summed['teacher_any_missed'] / summed['original_misses'] if summed['original_misses'] else None
            summed['teacher_qualified_coverage'] = summed['teacher_qualified_missed'] / summed['original_misses'] if summed['original_misses'] else None
            total['baseline_thresholds'][f'{th:g}'] = summed
        totals[model] = total
    result = dict(status='diagnostic_complete', training_authorized=False, M2_GO_threshold=None,
                  plan_sha256=sha256(plan_path), generation_manifest_sha256=sha256(generation / 'manifest.json'),
                  targets_source=str(targets_path.relative_to(root)), targets_sha256=sha256(targets_path),
                  checkpoint_sha256=manifest['checkpoint_sha256'], parameters=plan['parameters'],
                  coordinate_system='raw', unit='GT boxes; Euclidean3px, not official merged-component Pd',
                  primary_baseline_threshold=.5, secondary_baseline_threshold=1e-5, totals=totals,
                  limitations=plan['limitations'])
    write_csv(out / 'per_clip.csv', per_clip)
    write_csv(out / 'distance_bands.csv', distance_rows)
    write_csv(out / 'evaluated_points.csv', eval_points,
              list(eval_points[0]) if eval_points else ['clip', 'model', 't', 'wrong'])
    write_csv(out / 'teacher_track_evaluation.csv', evaluated_tracks)
    write_csv(out / 'early_target_coverage.csv', coverage_rows)
    write_json(out / 'summary.json', result)
    lines = ['M2 免训练回推诊断：固定raw_stop与raw_cv均报告，不按GT择优；无预设GO门槛。',
             'Teacher只来自官方缓存预测t>=60、p>=0.5；所有拟合只用轨迹首次观测起40帧内的点。',
             '生成文件先封存SHA，之后独立评价进程读取GT。下面是box口径、欧氏3px；不是官方component Pd/Fa/IoU。',
             '所有未匹配图内/NMS保留点计错点，包括无标注帧；没有3–8px忽略带。', '']
    for model, total in totals.items():
        metric = total['baseline_thresholds']['0.5']
        secondary = total['baseline_thresholds']['1e-05']
        lines.append(f"{model}: 原始点{total['raw_points']}，图内{total['in_image_points']}，NMS后{total['kept']}；"
                     f"错点{total['wrong_candidates']}（{total['wrong_per_frame']:.6f}/帧），precision={total['precision']}。")
        lines.append(f"  baseline0.5漏检恢复{metric['recovered']}/{metric['original_misses']}；"
                     f"1e-5漏检恢复{secondary['recovered']}/{secondary['original_misses']}。")
        lines.append(f"  0.5漏检中后段任一teacher覆盖{metric['teacher_any_missed']}/{metric['original_misses']}；"
                     f"合格轨迹中同GT至少10个fit观测覆盖{metric['teacher_qualified_missed']}/{metric['original_misses']}。")
    lines.extend(['', 'M1失败不能直接证明更长时间的回推无效；本诊断的覆盖、错点和距离分组分别报告。',
                  'GT轨迹ID仅用于事后覆盖/纯度统计，从未用于关联、拟合、NMS或候选筛选。',
                  '详细回推点、原始轨迹、NMS抑制原因、距离分组和每clip结果均保留。', ''])
    (out / 'notes.txt').write_text('\n'.join(lines))
    print(json.dumps(totals, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('generate', 'evaluate'))
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--plan', default='results/m2/plan.json')
    parser.add_argument('--out', default='results/m2')
    parser.add_argument('--targets', default='results/m2/inputs/m1_targets_baseline.csv')
    parser.add_argument('--nproc', type=int, default=4)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == 'generate':
        generate(root, root / args.plan, root / args.out, args.nproc)
    else:
        evaluate(root, root / args.plan, root / args.out, root / args.targets)


if __name__ == '__main__':
    main()
