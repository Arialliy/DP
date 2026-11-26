"""Per-clip camera motion and per-target motion statistics for the DeepPro RGB-T clips.

For every consecutive frame pair we fit a homography to background point tracks
(Shi-Tomasi corners + pyramidal LK, target boxes masked out, RANSAC). For every annotated
target at frame t we then measure, over a window of K frames:
  image_speed : |c_{t+K} - c_t| / K            (what a fixed-pixel temporal profile sees)
  camera_speed: |H_{t->t+K}(c_t) - c_t| / K    (background motion at the target position)
  rel_speed   : |c_{t+K} - H_{t->t+K}(c_t)| / K (target motion relative to its background)
Outputs results/profile/{homographies/<clip>.npy, targets.csv, clips.csv}.

targets.csv has one row per annotated box. Tracks are vtdata's clip-unique ids (for the
RGBT-Tiny release, category * vtdata.TRACK_BASE + tracking_id, since its tracking_id is
numbered within each category). A box that shares its frame's track id with another box is
ambiguous: it gets nan speeds, takes no part in any speed, and is counted in clips.csv
(duplicate_keys).
"""
import csv
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import vtdata  # noqa: E402

K = 5
OUT = 'results/profile'


def to_u8(im):
    return np.clip(im, 0, 255).astype(np.uint8)


def pair_homography(a, b, boxes):
    mask = np.full(a.shape, 255, np.uint8)
    for x, y, w, h, *_ in boxes:
        x0, y0 = int(max(x - 4, 0)), int(max(y - 4, 0))
        mask[y0:int(y + h + 4), x0:int(x + w + 4)] = 0
    p0 = cv2.goodFeaturesToTrack(a, maxCorners=800, qualityLevel=0.005, minDistance=8, mask=mask, blockSize=7)
    if p0 is None or len(p0) < 12:
        return np.eye(3), 0, 0.0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, winSize=(21, 21), maxLevel=3)
    ok = st.ravel() == 1
    if ok.sum() < 12:
        return np.eye(3), int(ok.sum()), 0.0
    H, inl = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 1.5)
    if H is None:
        return np.eye(3), int(ok.sum()), 0.0
    return H, int(ok.sum()), float(inl.mean())


def apply(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def target_rows(scene, boxes, Hs):
    """One row per box: (scene, t, track, cat, cx, cy, w, h, image_speed, camera_speed,
    rel_speed, span). -> (rows, number of tracks, number of ambiguous boxes)."""
    tracks, dup = {}, set()
    for t, b in enumerate(boxes):
        for x, y, w, h, c, tid in b:
            d = tracks.setdefault(int(tid), {})
            if t in d:
                dup.add((int(tid), t))
            d[t] = (x + w / 2, y + h / 2, w, h, int(c))
    for tid, t in dup:                      # an ambiguous position must not enter any speed
        tracks[tid].pop(t, None)
    rows = []
    for t, b in enumerate(boxes):
        for x, y, w, h, c, tid in b:
            tid, c = int(tid), int(c)
            cx, cy = x + w / 2, y + h / 2
            if (tid, t) in dup:
                rows.append((scene, t, tid, c, cx, cy, w, h, np.nan, np.nan, np.nan, 0))
                continue
            d = tracks[tid]
            # symmetric window, clipped to frames where the track exists
            t0, t1 = t, t
            for dt in range(1, K + 1):
                if t - dt in d:
                    t0 = t - dt
                if t + dt in d:
                    t1 = t + dt
            span = t1 - t0
            if span == 0:
                rows.append((scene, t, tid, c, cx, cy, w, h, np.nan, np.nan, np.nan, 0))
                continue
            Hc = np.eye(3)
            for k in range(t0, t1):
                Hc = Hs[k] @ Hc
            p0 = np.array([[d[t0][0], d[t0][1]]])
            p1 = np.array([[d[t1][0], d[t1][1]]])
            pw = apply(Hc, p0)
            rows.append((scene, t, tid, c, cx, cy, w, h,
                         float(np.linalg.norm(p1 - p0) / span), float(np.linalg.norm(pw - p0) / span),
                         float(np.linalg.norm(p1 - pw) / span), span))
    n_dup = sum(1 for t, b in enumerate(boxes) for row in b if (int(row[5]), t) in dup)
    return rows, len(tracks), n_dup


def process(scene):
    sc = vtdata.Scene(scene)
    ims = [to_u8(sc.image(i)) for i in range(len(sc))]
    Hs, stats = [], []
    for i in range(len(sc) - 1):
        H, n, inl = pair_homography(ims[i], ims[i + 1], sc.boxes[i])
        Hs.append(H)
        stats.append((n, inl))
    Hs = np.array(Hs) if Hs else np.zeros((0, 3, 3))
    os.makedirs(f'{OUT}/homographies', exist_ok=True)
    np.save(f'{OUT}/homographies/{scene}.npy', Hs)
    # camera speed at image centre / corners (mean over a grid)
    grid = np.stack(np.meshgrid(np.linspace(0, sc.W, 9), np.linspace(0, sc.H, 9)), -1).reshape(-1, 2)
    cam = [float(np.linalg.norm(apply(H, grid) - grid, axis=1).mean()) for H in Hs]
    rows, tracks, dup = target_rows(scene, sc.boxes, Hs)
    clip = dict(scene=scene, split=sc.split, frames=len(sc), H=sc.H, W=sc.W,
                boxes=int(sum(len(b) for b in sc.boxes)), tracks=tracks, duplicate_keys=dup,
                cam_speed_mean=float(np.mean(cam)) if cam else 0.0,
                cam_speed_p90=float(np.percentile(cam, 90)) if cam else 0.0,
                lk_inlier_mean=float(np.mean([s[1] for s in stats])) if stats else 0.0)
    return clip, rows


def main():
    scenes = vtdata.scene_list('train') + vtdata.scene_list('test')
    os.makedirs(OUT, exist_ok=True)
    with Pool(int(os.environ.get('NPROC', 24))) as p:
        res = p.map(process, scenes)
    with open(f'{OUT}/clips.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(res[0][0].keys()) + ['list'])
        w.writeheader()
        test = set(vtdata.scene_list('test'))
        for clip, _ in res:
            clip['list'] = 'test' if clip['scene'] in test else 'train'
            w.writerow(clip)
    with open(f'{OUT}/targets.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scene', 't', 'track', 'cat', 'cx', 'cy', 'w', 'h', 'image_speed', 'camera_speed', 'rel_speed',
                    'span'])
        for _, rows in res:
            w.writerows(rows)
    boxes, n_rows = sum(c['boxes'] for c, _ in res), sum(len(r) for _, r in res)
    dup = {c['scene']: c['duplicate_keys'] for c, _ in res if c['duplicate_keys']}
    if n_rows != boxes:
        raise RuntimeError(f'{n_rows} rows for {boxes} boxes')
    # read by mx_data.require_speeds: a targets.csv without it predates category-aware track ids
    with open(f'{OUT}/targets.json', 'w') as f:
        json.dump(dict(track_ids=vtdata.TRACK_SCHEME, boxes=boxes, rows=n_rows, duplicate_keys=dup,
                       clips={c['scene']: c['boxes'] for c, _ in res}), f, indent=1)
    print(f'done: {len(res)} clips, {boxes} boxes, {n_rows} rows in {OUT}/targets.csv')
    if dup:
        print(f'warning: {sum(dup.values())} boxes share a (frame, track) key and have nan speeds: {dup}')


if __name__ == '__main__':
    main()
