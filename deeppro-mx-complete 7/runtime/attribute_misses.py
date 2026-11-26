"""Attribute fixed-threshold author-protocol misses to measured motion strata.

This is an observational diagnostic, not a causal test of the MXC mechanism.
Counts are target-frame mask components (or, separately, annotated box-frames),
not distinct physical targets. A component joining different speed bands stays
mixed; a missing/ambiguous speed stays unknown. No track is chosen arbitrarily.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

CODE = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE))
import evaluate as ev
import gaussian_masks as gm
import splits
import vtdata

THRESHOLDS = (1e-5, 0.5)
TARGET_GROUPS = ('near_stationary', 'intermediate', 'moving', 'mixed', 'unknown')


def speed_group(value):
    if not math.isfinite(value) or value < 0:
        return 'unknown'
    return 'near_stationary' if value < 0.2 else 'intermediate' if value < 0.5 else 'moving'


def component_group(speeds):
    groups = {speed_group(v) for v in speeds}
    if not groups or 'unknown' in groups:
        return 'unknown'
    return next(iter(groups)) if len(groups) == 1 else 'mixed'


def component_records(prob, boxes, speeds):
    """One record per 8-connected mask component; match evaluate.shooting_rules.

Individual k=6, floor-stored Gaussian supports identify every box contributing
to each component of their union. ``speeds`` is aligned with boxes.
    """
    prob = np.asarray(prob, np.float32)
    H, W = prob.shape
    mask = gm.label(boxes, H, W, 6.0, 'floor') > 0
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    valid = mask.copy()
    valid[0, :] = False
    valid[:, 0] = False
    near = ndimage.maximum_filter(prob, size=3, mode='constant', cval=-np.inf)
    best = np.full(n, -np.inf, np.float32)
    np.maximum.at(best, labels[valid] - 1, near[valid])
    members = [[] for _ in range(n)]
    vanished = 0
    for i, box in enumerate(boxes):
        support = gm.label([box], H, W, 6.0, 'floor') > 0
        ids = np.unique(labels[support])
        if len(ids) == 0:
            vanished += 1
        if np.any(ids == 0):
            raise AssertionError('Individual support lies outside the combined mask')
        for component in ids:
            members[int(component) - 1].append(i)
    records = []
    for component, indices in enumerate(members, 1):
        records.append(dict(component=component, box_indices=indices,
                            target_group=component_group([speeds[i] for i in indices]),
                            best_probability=float(best[component - 1]),
                            merged=len(indices) > 1))
    return records, vanished


def read_profile(speeds_path, clips_path):
    speeds, duplicate_keys = {}, set()
    with open(speeds_path, newline='') as f:
        for r in csv.DictReader(f):
            key = (r['scene'], int(r['t']), int(r['track']))
            if key in speeds:
                duplicate_keys.add(key)
            speeds[key] = float(r['rel_speed']) if r['rel_speed'] else math.nan
    for key in duplicate_keys:
        speeds[key] = math.nan
    with open(clips_path, newline='') as f:
        camera = {r['scene']: float(r['cam_speed_mean']) for r in csv.DictReader(f)}
    return speeds, camera, duplicate_keys


def process_clip(task):
    clip, run, frame_speeds, camera_speed = task
    scene = vtdata.Scene(clip)
    logits = np.load(f'cache/preds/{run}/{clip}.npy', mmap_mode='r')
    if logits.shape != (len(scene), scene.H, scene.W):
        raise ValueError(f'{clip}: wrong prediction shape {logits.shape}')
    camera_group = 'moving' if camera_speed >= 0.3 else 'static'
    rows = []
    vanished = 0
    for t, boxes in enumerate(scene.boxes):
        probability = 1.0 / (1.0 + np.exp(-logits[t].astype(np.float32)))
        if not np.isfinite(probability).all():
            raise ValueError(f'{clip}/{t}: non-finite probabilities')
        speeds = [frame_speeds.get((t, int(b[5])), math.nan) for b in boxes]
        records, gone = component_records(probability, boxes, speeds)
        vanished += gone
        for r in records:
            ids = r.pop('box_indices')
            base = dict(clip=clip, frame=t, camera_group=camera_group, camera_speed=camera_speed,
                        tracks=';'.join(str(int(boxes[i, 5])) for i in ids),
                        speeds=';'.join(format(speeds[i], '.9g') for i in ids), box_count=len(ids), **r)
            for threshold in THRESHOLDS:
                rows.append(dict(base, threshold=threshold,
                                 hit=int(np.float32(r['best_probability']) >= np.float32(threshold))))
    return clip, rows, vanished


def aggregate(rows):
    counts = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        for clip in ('all', r['clip']):
            for camera in ('all', r['camera_group']):
                for target in ('all', r['target_group']):
                    c = counts[(r['threshold'], clip, camera, target)]
                    c[0] += 1
                    c[1] += int(r['hit'])
                    c[2] += int(r.get('merged', False))
    out = []
    for (threshold, clip, camera, target), (n, hits, merged) in sorted(counts.items()):
        total, total_hits, _ = counts[(threshold, clip, camera, 'all')]
        misses, all_misses = n - hits, total - total_hits
        out.append(dict(threshold=threshold, clip=clip, camera_group=camera, target_group=target,
                        targets=n, hits=hits, misses=misses, recall=hits / n,
                        miss_rate=misses / n, exposure_share=n / total,
                        miss_share=misses / all_misses if all_misses else None,
                        merged_components=merged))
    return out


def box_records(path, speeds, camera):
    rows = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            clip, t, track = r['clip'], int(r['t']), int(r['track'])
            logit = np.float32(r['score'])
            with np.errstate(over='ignore'):
                probability = np.float32(1.0) / (np.float32(1.0) + np.exp(-logit))
            for threshold in THRESHOLDS:
                rows.append(dict(clip=clip, threshold=threshold,
                                 camera_group='moving' if camera[clip] >= 0.3 else 'static',
                                 target_group=speed_group(speeds.get((clip, t, track), math.nan)),
                                 hit=int(probability >= np.float32(threshold))))
    return rows


def validate_summary(rows, summary):
    """Fail instead of publishing a different total from score.py's author table."""
    grouped = {(r['threshold'], r['clip']): r for r in rows
               if r['camera_group'] == 'all' and r['target_group'] == 'all'}
    checked = 0
    for threshold in THRESHOLDS:
        for clip, table in [('all', summary['author_protocol'])] + list(
                summary['author_protocol_per_clip'].items()):
            expected = table[str(threshold)]
            observed = grouped[(threshold, clip)]
            if observed['targets'] != expected['targets'] or observed['hits'] != expected['tp']:
                raise AssertionError(f'Author score mismatch: {threshold=} {clip=} '
                                     f'observed={observed} expected={expected}')
            checked += 1
    return dict(status='passed', threshold_clip_totals_checked=checked)


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='official_rgbt')
    parser.add_argument('--split', default='test')
    parser.add_argument('--speeds', default='results/profile/targets.csv')
    parser.add_argument('--camera', default='results/profile/clips.csv')
    parser.add_argument('--nproc', type=int, default=4)
    parser.add_argument('--out', default='results/miss_attribution')
    args = parser.parse_args()
    clips = splits.clips(args.split)
    speeds, camera, duplicate_keys = read_profile(args.speeds, args.camera)
    for clip in clips:
        if clip not in camera or not math.isfinite(camera[clip]):
            raise ValueError(f'Missing finite fresh camera speed for {clip}')
    tasks = [(c, args.run, {(t, track): v for (clip, t, track), v in speeds.items() if clip == c},
              camera[c]) for c in clips]
    details, vanished = [], {}
    with Pool(max(1, min(args.nproc, len(tasks)))) as pool:
        for clip, rows, gone in pool.imap_unordered(process_clip, tasks):
            details.extend(rows)
            vanished[clip] = gone
            print(f'{clip}: {len(rows) // len(THRESHOLDS)} component-frames; {gone} vanished box-frames', flush=True)
    details.sort(key=lambda r: (r['clip'], r['frame'], r['component'], r['threshold']))
    groups = aggregate(details)
    score_dir = Path('results/scores') / args.run / args.split
    score_path = score_dir / 'summary.json'
    score = json.loads(score_path.read_text())
    if score.get('author_label_rule') != 'paper_k6_floor' or score.get('author_edges') != 'official':
        raise ValueError('Attribution requires paper_k6_floor masks and official edge semantics')
    validation = validate_summary(groups, score)
    config_camera = json.loads(Path(splits.SPLIT_FILE).read_text())['camera_speed_px_per_frame']
    differences = [dict(clip=c, fresh_speed=camera[c], config_speed=config_camera.get(c),
                        fresh_group='moving' if camera[c] >= 0.3 else 'static',
                        config_group='moving' if config_camera.get(c, 0) >= 0.3 else 'static')
                   for c in clips if (camera[c] >= 0.3) != (config_camera.get(c, 0) >= 0.3)]
    box_groups = aggregate(box_records(score_dir / 'gt.csv', speeds, camera))
    report = dict(run=args.run, split=args.split, thresholds=list(THRESHOLDS),
                  validation=validation, component_groups=groups, box_groups=box_groups,
                  camera_group_disagreements=differences, camera_speed_source=args.camera,
                  camera_groups={c: dict(speed=camera[c], group='moving' if camera[c] >= 0.3 else 'static')
                                 for c in clips}, vanished_box_frames=vanished,
                  duplicated_speed_keys=len(duplicate_keys),
                  evidence=dict(score_summary_sha256=sha256(score_path), speeds_sha256=sha256(args.speeds),
                                camera_sha256=sha256(args.camera), script_sha256=sha256(__file__),
                                checkpoint_sha256=sorted({v['ckpt_sha256'] for v in score['predictions'].values() if v})),
                  method=dict(mask='k=6, floor storage; 8-connected components',
                              hit='Maximum probability within 1 pixel of any eligible target pixel; '
                                  'first row/column target pixels ineligible, matching official ShootingRules',
                              speed_bands='near_stationary [0,0.2), intermediate [0.2,0.5), moving >=0.5 px/frame',
                              camera_split='Fresh profile mean >=0.3 px/frame is moving',
                              merged='Same-band merged components retain that band; cross-band are mixed; '
                                     'any unknown or duplicate speed key gives unknown',
                              shares='Exposure and miss shares within each threshold/clip/camera stratum',
                              box_metric='Separate NMS+greedy box matching from gt.csv; not ShootingRules'),
                  limitations=['Counts are target-frame components or box-frames, not unique physical targets.',
                               'Relative speeds depend on box-masked homography estimates; residual error may change a speed band.',
                               'Reconstructed Gaussian masks and cached float16 logits can differ from the released official log.',
                               'This stratification establishes association, not that stationarity caused a miss or MXC will fix it.'])
    out = Path(args.out) / args.run / args.split
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'components.csv', details)
    write_csv(out / 'component_groups.csv', groups)
    write_csv(out / 'box_groups.csv', box_groups)
    (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for row in groups:
        if row['clip'] == 'all' and row['camera_group'] == 'all':
            print(f"p>={row['threshold']:g} {row['target_group']:>15}: "
                  f"miss {row['misses']}/{row['targets']} ({row['miss_rate']:.1%}); "
                  f"miss share {row['miss_share']:.1%}" if row['miss_share'] is not None else str(row))
    print(f'Wrote {out}; author counts verified against {score_path}', flush=True)


if __name__ == '__main__':
    main()
