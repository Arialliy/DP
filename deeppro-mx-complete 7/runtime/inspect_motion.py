"""Read-only geometry/speed diagnostics; optional CPU image routing, never a model.

The initial investigation ran these calculations as stdin snippets. This file
preserves their methods for reproduction. Run from the v7 directory, with NumPy;
--images additionally needs the existing CPU Docker runtime's OpenCV/SciPy/Pillow.
Only --out writes an optional new report, and only inside this project.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOCUS = {'DJI_0277_2': ([324, 326], [333, 334, 336, 344]),
         'DJI_0309_2': ([504], [511, 513, 519]),
         'DJI_0237_2': ([324], [339, 340, 341, 342])}


def norm(matrix):
    return matrix / matrix[2, 2]


def grid(n):
    return np.stack(np.meshgrid(np.linspace(0, 639, n), np.linspace(0, 511, n)), -1).reshape(-1, 2)


def project(matrix, points):
    q = np.c_[points, np.ones(len(points))] @ matrix.T
    return q[:, :2] / q[:, 2, None], q[:, 2]


def chain(pairs):
    out = [np.eye(3)]
    for matrix in pairs:
        out.append(norm(matrix @ out[-1]))
    return np.array(out)


def direct(pairs, ref, target):
    matrix = np.eye(3)
    for pair in pairs[min(ref, target):max(ref, target)]:
        matrix = pair @ matrix
    return norm(np.linalg.inv(matrix) if target < ref else matrix)


def windows(n):
    return [(max(0, min(n, (i + 1) * 36 + 4) - 40), min(n, (i + 1) * 36 + 4))
            for i in range(math.ceil((n - 4) / 36))]


def inputs(clip):
    return [np.load(ROOT / f'results/{folder}/homographies/{clip}.npy')
            for folder in ('motion', 'profile')]


def geometry(clip):
    pairs = inputs(clip)
    chains = [chain(p) for p in pairs]
    points = grid(9)
    errors = [[], []]
    differences = []
    worst = None
    for a, b in windows(len(chains[0])):
        ref = a + (b - a) // 2
        window_max = 0.0
        for t in range(a, b):
            local = [direct(p, ref, t) for p in pairs]
            cumulative = [norm(c[t] @ np.linalg.inv(c[ref])) for c in chains]
            coords = [project(m, points)[0] for m in local]
            for k in range(2):
                errors[k].append(float(np.linalg.norm(coords[k] - project(cumulative[k], points)[0], axis=1).max()))
            d = np.linalg.norm(coords[0] - coords[1], axis=1)
            j = int(d.argmax())
            value = float(d[j])
            window_max = max(window_max, value)
            if worst is None or value > worst['difference_px']:
                worst = dict(a=a, b=b, ref=ref, frame=t, point=points[j].tolist(), difference_px=value,
                             image_xy=coords[0][j].tolist(), masked_xy=coords[1][j].tolist())
        differences.append(window_max)
    return dict(clip=clip, chain_vs_local_max_px=[max(e) for e in errors],
                direct_window_p95_px=float(np.percentile(differences, 95)),
                direct_window_max_px=max(differences), worst=worst)


def common_fov(clip, a):
    pairs = inputs(clip)
    points = grid(33)
    ref, b = a + 20, a + 40
    whole = common = 0.0
    min_cover = 1.0
    horizon = [0, 0]
    for t in range(a, b):
        values = [project(direct(p, ref, t), points) for p in pairs]
        valid = np.ones(len(points), bool)
        for k, (xy, denominator) in enumerate(values):
            valid &= (xy[:, 0] >= 0) & (xy[:, 0] <= 639) & (xy[:, 1] >= 0) & (xy[:, 1] <= 511)
            horizon[k] += int(denominator.min() < 0 < denominator.max())
        difference = np.linalg.norm(values[0][0] - values[1][0], axis=1)
        whole = max(whole, float(difference.max()))
        if valid.any():
            common = max(common, float(difference[valid].max()))
        min_cover = min(min_cover, float(valid.mean()))
    return dict(clip=clip, a=a, b=b, whole_grid33_max_px=whole, common_fov_grid33_max_px=common,
                min_common_grid_fraction=min_cover, horizon_cross_frames=horizon)


def speed_sensitivity(clips, detail_rows=None):
    rows = defaultdict(list)
    with (ROOT / 'results/profile/targets.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['scene'] in clips:
                rows[row['scene']].append(row)
    output = []
    for clip in clips:
        pairs = inputs(clip)
        tracks = defaultdict(dict)
        for row in rows[clip]:
            frame, track = int(row['t']), int(row['track'])
            if frame in tracks[track]:
                raise ValueError('Duplicate speed key; original investigation had none in the test split')
            tracks[track][frame] = np.array([float(row['cx']), float(row['cy'])])
        values, saved_errors = [], []
        for row in rows[clip]:
            frame, track = int(row['t']), int(row['track'])
            coords = tracks[track]
            endpoints = [t for t in range(max(0, frame - 5), min(len(pairs[0]) + 1, frame + 6)) if t in coords]
            a, b = min(endpoints), max(endpoints)
            if a == b:
                continue
            speeds = []
            for p in pairs:
                q = direct(p, a, b) @ np.r_[coords[a], 1]
                speeds.append(float(np.linalg.norm(coords[b] - q[:2] / q[2]) / (b - a)))
            values.append(speeds)
            saved_errors.append(abs(speeds[1] - float(row['rel_speed'])))
            if detail_rows is not None:
                names = ('near_stationary', 'intermediate', 'moving')
                detail_rows.append(dict(scene=clip, t=frame, track=track,
                                        masked_rel_speed=float(row['rel_speed']), image_rel_speed=speeds[0],
                                        masked_band=names[int(np.searchsorted([0.2, 0.5], speeds[1], side='right'))],
                                        image_band=names[int(np.searchsorted([0.2, 0.5], speeds[0], side='right'))]))
        values = np.array(values)
        delta = np.abs(values[:, 0] - values[:, 1])
        bands = np.searchsorted([0.2, 0.5], values, side='right')
        output.append(dict(clip=clip, box_frames=len(values), band_changes=int((bands[:, 0] != bands[:, 1]).sum()),
                           stored_recompute_max=float(max(saved_errors)), delta_median=float(np.median(delta)),
                           delta_p95=float(np.percentile(delta, 95)), delta_max=float(delta.max())))
    return output


def image_checks():
    # CPU image operations only. No model imports or inference/training calls.
    sys.path.insert(0, str(ROOT / 'code'))
    import cv2
    import multirate as mr
    import motion_image as mi
    import profile_motion as pm
    import stabilize as st
    import vtdata
    cv2.setNumThreads(1)
    output = dict(routing=[], fits=[])
    for clip, (starts, frame_pairs) in FOCUS.items():
        scene = vtdata.Scene(clip)
        pairs = inputs(clip)
        chains = [st.chain_from_pairs(p) for p in pairs]
        for a in starts:
            frames = list(range(a, a + 40))
            low = mr.lowres(np.stack([scene.image(t) for t in frames]), 4)
            stats = [mr.stationarity(low, c, frames, 4) for c in chains]
            output['routing'].append(dict(clip=clip, a=a, b=a + 40, stats=stats,
                                           auto=[mr.choose_coords('auto', s, 1) for s in stats]))
        for t in frame_pairs:
            first, second = pm.to_u8(scene.image(t)), pm.to_u8(scene.image(t + 1))
            image, n, inlier = mi.image_homography(first, second)
            masked, m, masked_inlier = pm.pair_homography(first, second, scene.boxes[t])
            output['fits'].append(dict(clip=clip, pair_start=t, corners=[n, m],
                                       inlier_fraction=[inlier, masked_inlier],
                                       reproduced_matrix_max_difference=[float(np.abs(image - pairs[0][t]).max()),
                                                                         float(np.abs(masked - pairs[1][t]).max())]))
    return output


def clean(value):
    if isinstance(value, dict):
        return {key: clean(v) for key, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', action='store_true')
    parser.add_argument('--out', help='Optional NEW report path, inside this project only')
    parser.add_argument('--speed-csv', help='Optional NEW CSV of image/masked speeds for test box-frames')
    parser.add_argument('--speeds-only', action='store_true', help='Skip geometry and image checks')
    args = parser.parse_args()
    os.chdir(ROOT)
    clips = json.loads((ROOT / 'configs/splits.json').read_text())['official_test']
    details = [] if args.speed_csv else None
    report = dict(estimator_order=['image_only', 'box_masked'],
                  speed_sensitivity=speed_sensitivity(clips, details),
                  image_checks_executed=args.images and not args.speeds_only,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if not args.speeds_only:
        report['geometry'] = [geometry(c) for c in clips + ['DJI_0309_2']]
        report['common_fov'] = [common_fov(c, a) for c, (starts, _) in FOCUS.items() for a in starts]
    if args.images and not args.speeds_only:
        report['image_checks'] = image_checks()
    if args.speed_csv:
        output = Path(args.speed_csv).resolve()
        output.relative_to(ROOT)
        with output.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(details[0]))
            writer.writeheader()
            writer.writerows(details)
    text = json.dumps(clean(report), indent=2, allow_nan=False) + '\n'
    if args.out:
        output = Path(args.out).resolve()
        output.relative_to(ROOT)
        with output.open('x') as stream:
            stream.write(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
