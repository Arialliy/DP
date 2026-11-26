"""Read completed v8 matrices only; audit coverage without rerunning image estimation."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/motion_review'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chain(pairs):
    out = [np.eye(3)]
    for h in pairs:
        m = h @ out[-1]
        out.append(m / m[2, 2])
    return np.asarray(out)


def windows(n):
    return [(max(0, min(n, (i + 1) * 36 + 4) - 40), min(n, (i + 1) * 36 + 4))
            for i in range(math.ceil((n - 4) / 36))]


def project(matrices, pts):
    q = matrices @ pts
    return q[:, :2] / q[:, 2:3]


def inside(q, h, w):
    return (q[:, 0] >= 0) & (q[:, 0] <= w - 1) & (q[:, 1] >= 0) & (q[:, 1] <= h - 1)


def grid(h, w, n=None):
    if n:
        xx, yy = np.meshgrid(np.linspace(0, w - 1, n), np.linspace(0, h - 1, n))
    else:
        yy, xx = np.mgrid[:h, :w]
    return np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])


def main():
    motion_path = ROOT / 'results/motion/clips.csv'
    profile_path = ROOT / 'results/profile/clips.csv'
    split_path = ROOT / 'configs/splits.json'
    rows = list(csv.DictReader(motion_path.open()))
    shapes = {r['scene']: (int(r['H']), int(r['W'])) for r in csv.DictReader(profile_path.open())}
    test = set(json.loads(split_path.read_text())['test'])
    audits, hashes = [], {}
    for row in rows:
        clip = row['scene']
        h, w = shapes[clip]
        p_image = ROOT / 'results/motion/homographies' / (clip + '.npy')
        p_masked = ROOT / 'results/profile/homographies' / (clip + '.npy')
        image_pairs, masked_pairs = np.load(p_image), np.load(p_masked)
        image, masked = chain(image_pairs), chain(masked_pairs)
        assert len(image) == int(row['frames']) == len(masked)
        assert np.isfinite(image).all() and np.isfinite(masked).all()
        hashes[clip] = {'image': sha(p_image), 'masked': sha(p_masked)}
        full_grid = grid(h, w, 33)
        small_grid = grid(h // 4, w // 4)
        block_centres = np.array([[4., 0, 1.5], [0, 4., 1.5], [0, 0, 1.]])
        inv_block_centres = np.linalg.inv(block_centres)
        for k, (a, b) in enumerate(windows(len(image))):
            ref = a + (b - a) // 2
            mi = image[a:b] @ np.linalg.inv(image[ref])
            mb = masked[a:b] @ np.linalg.inv(masked[ref])
            mi /= mi[:, 2:3, 2:3]
            mb /= mb[:, 2:3, 2:3]
            qi, qb = project(mi, full_grid), project(mb, full_grid)
            common = inside(qi, h, w) & inside(qb, h, w)
            distances = np.linalg.norm(qi - qb, axis=1)
            low = inv_block_centres @ mi @ block_centres
            low /= low[:, 2:3, 2:3]
            valid = inside(project(low, small_grid), h // 4, w // 4)
            cover = float((valid.mean(0) >= .75).mean())
            valid_distances = distances[common]
            audits.append(dict(clip=clip, test=clip in test, window=k, first=a, last=b - 1,
                               lowres_cover=cover, auto_low_cover_raw=cover < .1,
                               common_view_fraction_min=float(common.mean(1).min()),
                               common_view_fraction_mean=float(common.mean()),
                               frame_comparisons_without_common_view=int((~common.any(1)).sum()),
                               common_view_diff_max=float(valid_distances.max()) if len(valid_distances) else None))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'window_geometry.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(audits[0]))
        writer.writeheader()
        writer.writerows(audits)
    summary = dict(
        scope='All 60 completed v8 homography outputs; matrix geometry only, no images or model execution',
        sources={str(p.relative_to(ROOT)): sha(p) for p in [motion_path, profile_path, split_path]},
        homography_hashes=hashes,
        clips=len(rows), frame_pairs=sum(int(r['frames']) - 1 for r in rows),
        pair_methods={m: sum(int(r['pairs_' + m]) for r in rows) for m in ['homography', 'affine', 'shift', 'identity']},
        windows=len(audits), test_windows=sum(r['test'] for r in audits),
        low_cover_windows=[r for r in audits if r['auto_low_cover_raw']],
        no_common_view_comparisons=sum(r['frame_comparisons_without_common_view'] for r in audits),
        minimum_cover=min(r['lowres_cover'] for r in audits),
        minimum_test_cover=min(r['lowres_cover'] for r in audits if r['test']),
        top_common_view_disagreements=sorted(audits, key=lambda r: r['common_view_diff_max'] or 0, reverse=True)[:15],
        notes=[
            'Coverage exactly follows the 4x low-resolution block-centre geometry and >=75% visibility rule; it does not measure photometric residual.',
            'For auto at rate 1, cover<0.1 forces raw; otherwise raw-vs-warp requires measured stationarity residuals.',
            'Common-view disagreement compares two estimated maps; neither is ground truth.',
            'The source compare() maps an empty common view to zero difference; zero-common counts above audit whether this hides any window/frame comparisons.',
            'Per-pair inlier residuals are not persisted by motion_image.py; <=1px applies only to accepted homographies, not affine/shift/identity fallbacks.'
        ])
    (OUT / 'geometry_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ['sources', 'homography_hashes', 'top_common_view_disagreements']}, indent=2))


if __name__ == '__main__':
    main()
