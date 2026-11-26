"""Numpy side of DeepPro-MX, motion exemplars (no torch: runs in DataLoader workers and tests).

DeepPro-MX lets targets the temporal profile can see teach the detector what the targets
it cannot see look like. An exemplar is an image patch of a moving target of the same clip:
  training   GT boxes with camera-compensated rel_speed >= MOVING (profile_motion.py),
             from frames outside the sample window                  (exemplar_sample)
  inference  peaks of the network's own variation logits             (select_bank)
             from frames outside the window ('outside', the training rule), only before it
             ('causal'), or anywhere in the clip ('clip');
             controls: GT moving boxes ('gt', gt_bank), random points ('random',
             random_bank), no exemplars ('none').
Tables are float arrays with one row per item: (t, x, y, extra), where t is the frame,
(x, y) the pixel centre (column, row; a COCO box (bx, by, w, h) has its centre pixel at
bx + w/2 - 0.5, by + h/2 - 0.5 as in evaluate.match_frame), and extra the track id
(GT tables) or the logit (detection tables).
"""
import csv
import math

import numpy as np

import stabilize as st

MOVING = 0.5   # px/frame; a GT box is an exemplar when its rel_speed is at least this
NEAR = 0.2     # px/frame; near-stationary, as in diagnose_history and the README
R_XY = 16.0    # two exemplars within R_XY px ...
DT = 20        # ... and DT frames of each other count as the same target (greedy_select)


# ---------------------------------------------------------------- GT exemplar tables
def read_speeds(path, clips=None):
    """targets.csv of profile_motion.py -> {(scene, t, track): rel_speed or nan}."""
    keep = None if clips is None else set(clips)
    out = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            if keep is not None and row['scene'] not in keep:
                continue
            v = row['rel_speed']
            out[(row['scene'], int(row['t']), int(float(row['track'])))] = (
                float(v) if v not in ('', 'nan') else math.nan)
    return out


def speed_coverage(clip, boxes, speed):
    """Fraction of a clip's boxes with an entry in `speed`. A targets.csv written by
    profile_motion.py from the same annotations covers every box (single-frame tracks have
    nan entries); a low value means its track ids come from other annotations."""
    n = hit = 0
    for t, b in enumerate(boxes):
        for row in np.asarray(b).reshape(len(b), -1) if len(b) else ():
            n += 1
            hit += (clip, t, int(row[5])) in speed
    return hit / n if n else 1.0


def require_speeds(boxes_by_clip, speed, path, min_cover=1.0):
    """Raise unless `path` (targets.csv) was written by the current profile_motion.py from these
    annotations: its targets.json must name vtdata.TRACK_SCHEME, and every box needs a row
    (profile_motion.py writes one per box, nan where the speed is undefined).

    The scheme check matters: a file keyed by the raw, per-category tracking_id still matches
    every category-0 (ship) box, so a mostly-ship clip would pass any coverage threshold below 1.
    Coverage is compared unrounded.
    """
    import json
    import os
    import vtdata
    meta = os.path.splitext(path)[0] + '.json'
    scheme = json.load(open(meta)).get('track_ids') if os.path.exists(meta) else None
    if scheme != vtdata.TRACK_SCHEME:
        raise ValueError(f'{path} was not written with the current track ids ({meta}: {scheme!r}, expected '
                         f'{vtdata.TRACK_SCHEME!r}): re-run code/profile_motion.py.')
    cover = {c: speed_coverage(c, b, speed) for c, b in boxes_by_clip.items()}
    low = {c: v for c, v in cover.items() if v < min_cover}
    if low:
        shown = {c: f'{100 * v:.4f}%' for c, v in low.items()}
        raise ValueError(f'{path} has no row for some boxes of {shown} (boxes covered). It was written from '
                         f'other annotations: re-run code/profile_motion.py on the current ones.')


def box_centres(boxes):
    """(n, >=4) COCO boxes -> (n, 2) centre pixels (x, y)."""
    b = np.asarray(boxes, np.float64)
    if b.size == 0:
        return np.zeros((0, 2))
    b = b.reshape(len(b), -1)
    return np.stack([b[:, 0] + b[:, 2] / 2.0 - 0.5, b[:, 1] + b[:, 3] / 2.0 - 0.5], 1)


def point_box_distance(x, y, boxes):
    """Distance (px) from pixel (x, y) to each box's pixel extent [bx, bx + w - 1] (0 inside)."""
    b = np.asarray(boxes, np.float64)
    if b.size == 0:
        return np.zeros(0)
    b = b.reshape(len(b), -1)
    dx = np.maximum(np.maximum(b[:, 0] - x, 0), x - (b[:, 0] + b[:, 2] - 1))
    dy = np.maximum(np.maximum(b[:, 1] - y, 0), y - (b[:, 1] + b[:, 3] - 1))
    return np.hypot(dx, dy)


def moving_table(clip, boxes, speed, thr=MOVING):
    """GT boxes of a clip moving at >= thr px/frame -> (m, 4) rows (t, x, y, track)."""
    rows = []
    for t, b in enumerate(boxes):
        if len(b) == 0:
            continue
        for (x, y), row in zip(box_centres(b), np.asarray(b).reshape(len(b), -1)):
            v = speed.get((clip, t, int(row[5])), math.nan)
            if np.isfinite(v) and v >= thr:
                rows.append((t, x, y, row[5]))
    return np.asarray(rows, np.float64).reshape(-1, 4)


# ---------------------------------------------------------------- selection
def pool_mask(t, lo, hi, mode):
    """Rows a window covering frames lo..hi (inclusive) may draw exemplars from."""
    t = np.asarray(t)
    if mode == 'outside':
        return (t < lo) | (t > hi)
    if mode == 'causal':
        return t < lo
    if mode == 'clip':
        return np.ones(t.shape, bool)
    raise ValueError(mode)


def greedy_select(rows, K, r_xy=R_XY, dt=DT):
    """Indices of up to K rows taken in the given order, skipping any row within r_xy px
    and dt frames of one already taken (one exemplar per target appearance)."""
    rows = np.asarray(rows, np.float64)
    chosen = []
    for i in range(len(rows)):
        if chosen:
            c = rows[chosen]
            same = (np.abs(c[:, 0] - rows[i, 0]) <= dt) & (np.hypot(c[:, 1] - rows[i, 1], c[:, 2] - rows[i, 2]) <= r_xy)
            if same.any():
                continue
        chosen.append(i)
        if len(chosen) == K:
            break
    return np.asarray(chosen, int)


def draw_exemplars(rng, moving, lo, hi, K, p_none=0.3, p_all=0.5, r_xy=R_XY, dt=DT):
    """Training exemplars of a sample spanning frames lo..hi: GT moving rows outside the
    span, in random order, greedily separated. With probability p_none there are none (the
    network must also work without them); otherwise all selected rows are kept with
    probability p_all, else a random number 1..n of them."""
    moving = np.asarray(moving, np.float64).reshape(-1, 4)
    if len(moving) == 0 or rng.random() < p_none:
        return moving[:0]
    cand = moving[pool_mask(moving[:, 0], lo, hi, 'outside')]
    if len(cand) == 0:
        return cand
    cand = cand[rng.permutation(len(cand))]
    sel = cand[greedy_select(cand, K, r_xy, dt)]
    if len(sel) > 1 and rng.random() >= p_all:
        sel = sel[:int(rng.integers(1, len(sel) + 1))]
    return sel


def background_point(rng, frame, boxes, sep=8.0, region=64, tries=8):
    """A clutter point of one frame at least sep px from every box: the strongest |local
    contrast| inside a random region (a plausible false detection), else a uniform point."""
    frame = np.asarray(frame, np.float32)
    H, W = frame.shape
    for _ in range(tries):
        r0 = int(rng.integers(0, max(1, H - region + 1)))
        c0 = int(rng.integers(0, max(1, W - region + 1)))
        lc = np.abs(np.nan_to_num(st.local_contrast(frame[r0:r0 + region, c0:c0 + region]), nan=0.0))
        order = np.argsort(-lc, axis=None)[:50]
        for y, x in zip(*np.unravel_index(order, lc.shape)):
            X, Y = float(c0 + x), float(r0 + y)
            if len(boxes) == 0 or point_box_distance(X, Y, boxes).min() > sep:
                return X, Y
    for _ in range(100):
        X, Y = float(rng.integers(0, W)), float(rng.integers(0, H))
        if len(boxes) == 0 or point_box_distance(X, Y, boxes).min() > sep:
            return X, Y
    return float(rng.integers(0, W)), float(rng.integers(0, H))


def select_bank(cands, lo, hi, mode, K, r_xy=R_XY, dt=DT, max_scan=20000):
    """Detection exemplars for a window covering lo..hi: candidates (t, x, y, logit) of the
    pool, highest logit first, greedily separated; mode 'none' gives none."""
    cands = np.asarray(cands, np.float64).reshape(-1, 4)
    if mode == 'none' or len(cands) == 0:
        return cands[:0]
    c = cands[pool_mask(cands[:, 0], lo, hi, mode)]
    c = c[np.argsort(-c[:, 3], kind='stable')][:max_scan]
    return c[greedy_select(c, K, r_xy, dt)]


def gt_bank(rng, moving, lo, hi, mode, K, r_xy=R_XY, dt=DT):
    """Oracle exemplars: GT moving rows of the pool, random order, greedily separated."""
    moving = np.asarray(moving, np.float64).reshape(-1, 4)
    if mode == 'none' or len(moving) == 0:
        return moving[:0]
    c = moving[pool_mask(moving[:, 0], lo, hi, mode)]
    c = c[rng.permutation(len(c))]
    return c[greedy_select(c, K, r_xy, dt)]


def random_bank(rng, n, H, W, lo, hi, mode, K):
    """Control exemplars: K uniform points in uniform frames of the pool (t, x, y, 0)."""
    frames = np.flatnonzero(pool_mask(np.arange(n), lo, hi, mode))
    if mode == 'none' or len(frames) == 0 or K <= 0:
        return np.zeros((0, 4))
    t = rng.choice(frames, size=K)
    return np.stack([t, rng.integers(0, W, K), rng.integers(0, H, K), np.zeros(K)], 1).astype(np.float64)


def peak_rows(logit, t, peaks_fn, min_score):
    """Candidate exemplars of one frame: peaks_fn(logit) -> (n, 3) x, y, score (evaluate.peaks);
    -> (m, 4) rows (t, x, y, score) with score >= min_score."""
    d = np.asarray(peaks_fn(np.asarray(logit, np.float32)), np.float64).reshape(-1, 3)
    d = d[d[:, 2] >= min_score]
    return np.c_[np.full(len(d), float(t)), d]


# ---------------------------------------------------------------- patches
def patch2d(img, x, y, S):
    """S x S patch (S odd) of a 2-D array centred on the pixel nearest (x, y), edge-padded."""
    if S % 2 != 1:
        raise ValueError(f'patch size must be odd, got {S}')
    img = np.asarray(img)
    H, W = img.shape
    m = S // 2
    xi = min(max(int(math.floor(x + 0.5)), 0), W - 1)
    yi = min(max(int(math.floor(y + 0.5)), 0), H - 1)
    y0, y1, x0, x1 = yi - m, yi + m + 1, xi - m, xi + m + 1
    sub = np.asarray(img[max(y0, 0):min(y1, H), max(x0, 0):min(x1, W)], np.float32)
    pad = ((max(0, -y0), max(0, y1 - H)), (max(0, -x0), max(0, x1 - W)))
    if any(p for pp in pad for p in pp):
        sub = np.pad(sub, pad, mode='edge')
    return sub


def extract_patches(video, rows, S, mean, std, K=None):
    """(K, 1, S, S) float32 normalised patches of video[t] around rows (t, x, y, ...) and a
    (K,) validity mask. Slots beyond len(rows) are zero and invalid."""
    rows = np.asarray(rows, np.float64).reshape(-1, np.shape(rows)[-1] if np.size(rows) else 4)
    K = len(rows) if K is None else int(K)
    if len(rows) > K:
        raise ValueError(f'{len(rows)} exemplars for {K} slots')
    out = np.zeros((K, 1, S, S), np.float32)
    valid = np.zeros(K, bool)
    for i, r in enumerate(rows):
        out[i, 0] = (patch2d(video[int(r[0])], r[1], r[2], S) - mean) / std
        valid[i] = True
    return out, valid


def exemplar_sample(rng, video, boxes, moving, frames, K, S, mean, std, p_none=0.3, p_fake=0.2,
                    p_all=0.5, jitter=1.0):
    """Exemplar inputs of one training sample whose window holds `frames` (clip frame indices).

    -> dict(ex (K,1,S,S) float32, ex_valid (K,) bool, ex_fake (K,) bool, ex_t (K,) int64)
    Exemplar centres are moved by up to `jitter` px, since at inference they are peaks of
    the variation logits rather than box centres. Each drawn exemplar is replaced, with
    probability p_fake, by a clutter point of the same frame (background_point), imitating
    false detections in the inference bank. ex_fake and ex_t are for logging and tests; the
    network sees ex and ex_valid only.
    """
    frames = np.asarray(frames)
    sel = draw_exemplars(rng, moving, int(frames.min()), int(frames.max()), K, p_none, p_all)
    rows = sel[:, :3].copy()
    if len(rows) and jitter > 0:
        rows[:, 1:3] += rng.uniform(-jitter, jitter, (len(rows), 2))
    fake = np.zeros(len(rows), bool)
    if len(rows) and p_fake > 0:
        fake = rng.random(len(rows)) < p_fake
        for i in np.flatnonzero(fake):
            t = int(rows[i, 0])
            rows[i, 1], rows[i, 2] = background_point(rng, video[t], boxes[t])
    ex, valid = extract_patches(video, rows, S, mean, std, K)
    ex_fake = np.zeros(K, bool)
    ex_fake[:len(fake)] = fake
    ex_t = np.full(K, -1, np.int64)
    ex_t[:len(rows)] = rows[:, 0].astype(np.int64)
    return dict(ex=ex, ex_valid=valid, ex_fake=ex_fake, ex_t=ex_t)
