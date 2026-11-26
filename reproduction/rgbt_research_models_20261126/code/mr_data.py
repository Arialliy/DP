"""Numpy side of multi-rate training samples (kept free of torch so it can be unit-tested).

A sample is T frames taken every k frames, s, s + k, ..., s + (T - 1) k, in recorded or
camera-compensated coordinates (multirate.choose_coords), cropped to P x P:
  img    (1,T,P,P) float32  normalised grey, 0 where invalid or padded
  lab    (T,P,P)   float32  sigma_sixth masks of the sampled frames' boxes
  valid  (T,)      bool     sample exists (False on the zero padding of short windows)
  pvalid (1,T,P,P) bool     pixel maps inside its source frame
Compensated samples use the middle sampled frame as reference (as at inference).
"""
import numpy as np

import multirate as mr
import stabilize as st
import vtdata


def rate_tables(lengths, rates, T=40):
    """For each rate k > 1: clips with room for a rate-k window, and the probability of
    each clip, proportional to its number of window starts (uniform over windows)."""
    out = {}
    for k in rates:
        if k == 1:
            continue
        names = [c for c, n in lengths.items() if mr.rate_starts(n, k, T) > 0]
        if names:
            w = np.array([mr.rate_starts(lengths[c], k, T) for c in names], float)
            out[k] = (names, w / w.sum())
    return out


def draw_window(rng, k, n, T=40):
    """Frames of a random rate-k window inside n frames."""
    s = int(rng.integers(0, mr.rate_starts(n, k, T)))
    return s + k * np.arange(T)


def draw_spec(rng, rates, rate_p, tables, samples, p, lengths, decide, T=40):
    """What one training sample holds: -> (k, clip, frames, coords).

    rates/rate_p: training rates and their probabilities; tables: rate_tables(...);
    samples/p: TrainSet's rate-1 windows (clip, a, b) and their probabilities;
    decide(clip, frames, k) -> 'raw' | 'warp' | 'skip'. A skipped window is redrawn at
    rate 1, where the coordinate rule never skips.
    """
    k = int(rng.choice(rates, p=rate_p))
    while True:
        if k == 1 or k not in tables:
            c, a, b = samples[rng.choice(len(samples), p=p)]
            k, frames = 1, np.arange(a, b)
        else:
            names, q = tables[k]
            c = names[rng.choice(len(names), p=q)]
            frames = draw_window(rng, k, lengths[c], T)
        coords = decide(c, frames, k)
        if coords != 'skip':
            return k, c, frames, coords
        if k == 1:
            raise RuntimeError('the coordinate rule skipped a rate-1 window')
        k = 1


def decide(video, chain, frames, policy, k, f=4, margin=0.1, abs_tol=0.5, min_cover=0.1):
    """Coordinate frame of one window; same rule as multi-rate inference."""
    if policy == 'raw':
        return 'raw', {}
    low = mr.lowres(np.stack([np.asarray(video[t]) for t in frames]), f)
    stats = mr.stationarity(low, chain, frames, f)
    return mr.choose_coords(policy, stats, k, margin, abs_tol, min_cover), stats


def choose_center(boxes_ref, rng, p_target=0.75):
    """A target of the reference frame (with probability p_target), else None (random crop)."""
    if len(boxes_ref) and rng.random() < p_target:
        x, y, w, h = boxes_ref[rng.integers(len(boxes_ref))][:4]
        return x + w / 2.0, y + h / 2.0
    return None


def build_window_sample(video, boxes, chain, frames, coords, T, P, r0, c0, front,
                        mean=vtdata.RGBT_MEAN, std=vtdata.RGBT_STD, rule='sigma_sixth'):
    frames = [int(t) for t in frames]
    t = len(frames)
    lo = T - t if front else 0
    ref = frames[t // 2]
    if coords == 'raw':
        crop = np.stack([np.asarray(video[tt, r0:r0 + P, c0:c0 + P], np.float32) for tt in frames])
        fv = np.ones(crop.shape, bool)
    elif coords == 'warp':
        crop, fv = st.warp_frames(video, chain, frames, ref, r0, c0, (P, P))
    else:
        raise ValueError(coords)
    img = np.zeros((T, P, P), np.float32)
    pv = np.zeros((T, P, P), bool)
    lab = np.zeros((T, P, P), np.float32)
    img[lo:lo + t] = np.where(fv, (crop - mean) / std, 0.0)
    pv[lo:lo + t] = fv
    for i, tt in enumerate(frames):
        if coords == 'raw':
            bx = np.array(boxes[tt], dtype=np.float64).reshape(-1, 6).copy()
        else:
            bx = st.map_boxes(boxes[tt], st.ref_to(chain, tt, ref))
        if len(bx):
            bx[:, 0] -= c0
            bx[:, 1] -= r0
        lab[lo + i] = vtdata.gaussian_mask(bx, P, P, rule) * fv[i]
    valid = np.zeros(T, bool)
    valid[lo:lo + t] = True
    return dict(img=img[None], lab=lab, valid=valid, pvalid=pv[None])
