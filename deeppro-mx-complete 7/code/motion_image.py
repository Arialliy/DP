"""Camera motion from the images alone, with no annotation, for every network input.

profile_motion.py fits each frame-pair homography to corner tracks with the annotated boxes
masked out. It therefore uses the boxes, the test clips' included. Since every compensated
window (lvh_*, mr_*, mx_*, mxc_*, M0 / M0c, checkpoint selection and inference) used those
homographies, the test pipeline needed test annotations. This script estimates the same
homographies from the frames only:
  corners  Shi-Tomasi corners of frame t, except within BLOB_PAD px of small bright or dark
           blobs that the image itself shows (|local contrast| >= max(BLOB_MIN, the
           BLOB_PCT percentile of the frame), stabilize.local_contrast); small targets are
           such blobs, so moving targets give few tracks;
  tracks   pyramidal LK into frame t + 1;
  model    RANSAC homography (profile_motion.masked_homography, the same parameters).
Writes, for every clip of train.txt and test.txt,
  results/motion/homographies/<clip>.npy   (N-1, 3, 3), H[i] maps frame i -> frame i + 1
  results/motion/clips.csv                 corners, inliers, camera speed and, when
                                           results/profile/homographies exists, the
                                           difference to the box-masked homographies
  results/motion/motion.json               method and parameters
The difference columns are reported as a check, not used: pair_diff is the mean (and 95th
percentile) displacement difference of a 9 x 9 grid between the two homographies of one
frame pair, window_diff the largest grid difference between a window's reference and any of
its frames (the error a compensated window would get), over the DeepPro windows.

stabilize.HOMOGRAPHY_DIR defaults to results/motion/homographies.

Usage: python code/motion_image.py [--clips DJI_0049_2,...] [--nproc 8]
"""
import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import profile_motion as pm  # noqa: E402
import stabilize as st  # noqa: E402
import vtdata  # noqa: E402

OUT = 'results/motion'
BLOB_MIN = 8.0      # grey levels
BLOB_PCT = 99.5
BLOB_PAD = 4        # px, as the +4 px around boxes in profile_motion.py


def blob_mask(frame):
    """uint8 mask (255 = corners allowed): 0 within BLOB_PAD px of small high-contrast blobs."""
    c = np.abs(np.nan_to_num(st.local_contrast(np.asarray(frame, np.float32)), nan=0.0))
    thr = max(BLOB_MIN, float(np.percentile(c, BLOB_PCT)))
    blobs = (c >= thr).astype(np.uint8)
    k = 2 * BLOB_PAD + 1
    blobs = cv2.dilate(blobs, np.ones((k, k), np.uint8))
    return np.where(blobs > 0, 0, 255).astype(np.uint8)


def image_homography(a, b):
    """Homography frame a -> frame b from the two frames only. -> (H, corners, inliers)."""
    return pm.masked_homography(a, b, blob_mask(a))


def grid_points(H, W, n=9):
    return np.stack(np.meshgrid(np.linspace(0, W - 1, n), np.linspace(0, H - 1, n)), -1).reshape(-1, 2)


def compare(Hs, Hg, shape, T=40):
    """Pair and window displacement differences (px) between two homography sequences."""
    pts = grid_points(*shape)
    pair = np.array([np.linalg.norm(st.apply_h(a, pts) - st.apply_h(b, pts), axis=1).mean()
                     for a, b in zip(Hs, Hg)]) if len(Hs) else np.zeros(0)
    ca, cb = st.chain_from_pairs(Hs), st.chain_from_pairs(Hg)
    win = []
    for a, b in st.windows(len(ca), T):
        ref = st.reference_frame(a, b)
        win.append(max(float(np.linalg.norm(st.apply_h(st.ref_to(ca, ref, t), pts)
                                            - st.apply_h(st.ref_to(cb, ref, t), pts), axis=1).max())
                       for t in range(a, b)))
    return pair, np.array(win)


def process(task):
    clip, out = task
    sc = vtdata.Scene(clip)
    ims = [pm.to_u8(sc.image(i)) for i in range(len(sc))]
    Hs, stats = [], []
    for i in range(len(ims) - 1):
        H, n, inl = image_homography(ims[i], ims[i + 1])
        Hs.append(H)
        stats.append((n, inl))
    Hs = np.array(Hs) if Hs else np.zeros((0, 3, 3))
    os.makedirs(f'{out}/homographies', exist_ok=True)
    tmp = f'{out}/homographies/{clip}.tmp.npy'
    np.save(tmp, Hs)
    os.replace(tmp, f'{out}/homographies/{clip}.npy')
    pts = grid_points(sc.H, sc.W)
    cam = [float(np.linalg.norm(st.apply_h(H, pts) - pts, axis=1).mean()) for H in Hs]
    row = dict(scene=clip, frames=len(sc), corners_mean=float(np.mean([s[0] for s in stats])) if stats else 0.0,
               inlier_mean=float(np.mean([s[1] for s in stats])) if stats else 0.0,
               cam_speed_mean=float(np.mean(cam)) if cam else 0.0)
    gt = os.path.join(pm.OUT, 'homographies', f'{clip}.npy')
    if os.path.exists(gt):
        Hg = np.load(gt)
        if len(Hg) == len(Hs):
            pair, win = compare(Hs, Hg, (sc.H, sc.W))
            row.update(pair_diff_mean=float(pair.mean()) if len(pair) else 0.0,
                       pair_diff_p95=float(np.percentile(pair, 95)) if len(pair) else 0.0,
                       window_diff_p95=float(np.percentile(win, 95)) if len(win) else 0.0,
                       window_diff_max=float(win.max()) if len(win) else 0.0)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--clips', default=None, help='comma-separated clips (default: train.txt and test.txt)')
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--nproc', type=int, default=int(os.environ.get('NPROC', 8)))
    args = ap.parse_args(argv)
    clips = args.clips.split(',') if args.clips else vtdata.scene_list('train') + vtdata.scene_list('test')
    os.makedirs(args.out, exist_ok=True)
    with Pool(max(1, min(args.nproc, len(clips)))) as p:
        rows = p.map(process, [(c, args.out) for c in clips])
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(f'{args.out}/clips.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    json.dump(dict(method='Shi-Tomasi + LK + RANSAC, corners suppressed around image blobs; no annotation',
                   blob_min=BLOB_MIN, blob_pct=BLOB_PCT, blob_pad=BLOB_PAD, clips=clips),
              open(f'{args.out}/motion.json', 'w'), indent=1)
    cmp = [r for r in rows if 'window_diff_max' in r]
    print(f'done: {len(rows)} clips in {args.out}/homographies')
    if cmp:
        worst = max(cmp, key=lambda r: r['window_diff_p95'])
        print(f"vs box-masked homographies: window difference p95 median "
              f"{np.median([r['window_diff_p95'] for r in cmp]):.3f} px, worst {worst['scene']} "
              f"{worst['window_diff_p95']:.3f} px (p95), {worst['window_diff_max']:.3f} px (max)")
    return rows


if __name__ == '__main__':
    main()
