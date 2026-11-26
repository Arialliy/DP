"""Numpy side of LVH training samples (kept free of torch so it can be unit-tested).

One sample is a T-frame window warped into the coordinates of its middle frame, cropped to
P x P, with:
  img    (1,T,P,P) float32  normalised grey, 0 where invalid
  lab    (T,P,P)   float32  sigma_sixth masks of every annotation, moved to the reference
  valid  (T,)      bool     frame exists (False for the zero padding of short windows)
  pvalid (1,T,P,P) bool     pixel maps inside its source frame
  hist   (1,P,P)   float32  normalised History median, 0 where unavailable
  hvalid (1,P,P)   bool     History available
"""
import numpy as np

import stabilize as st
import vtdata


def crop_origin(center, P, H, W, rng):
    """Top-left of a P x P crop around center (x, y) with DeepPro's +-P/6 jitter, or random."""
    if center is None:
        return int(rng.integers(0, H - P)), int(rng.integers(0, W - P))
    sh = int(P / 2 / 3)
    r0 = int(center[1] + (rng.random() - 0.5) * 2 * sh - P / 2)
    c0 = int(center[0] + (rng.random() - 0.5) * 2 * sh - P / 2)
    return min(max(r0, 0), H - P - 1), min(max(c0, 0), W - P - 1)


def hotspot_center(hot, chain, a, b, ref, rng, max_gap=40):
    """A mined hot spot near the window, moved to this window's reference coordinates."""
    if hot is None or not len(hot['x']):
        return None
    refs = hot['ref']
    inside = np.flatnonzero((refs >= a) & (refs < b))
    if not len(inside):
        gap = np.abs(refs - ref)
        if gap.min() > max_gap:
            return None
        inside = np.flatnonzero(gap == gap.min())
    i = int(inside[rng.integers(len(inside))])
    p = st.apply_h(st.ref_to(chain, int(refs[i]), ref), [[float(hot['x'][i]), float(hot['y'][i])]])[0]
    return float(p[0]), float(p[1])


def choose_center(boxes_ref, hot, chain, a, b, ref, rng, hard_neg=0.0, p_target=0.75):
    """-> ((x, y) or None, kind) with kind in {'hotspot', 'target', 'random'}."""
    if hard_neg > 0 and rng.random() < hard_neg:
        c = hotspot_center(hot, chain, a, b, ref, rng)
        if c is not None:
            return c, 'hotspot'
    if len(boxes_ref) and rng.random() < p_target:
        x, y, w, h = boxes_ref[rng.integers(len(boxes_ref))][:4]
        return (x + w / 2.0, y + h / 2.0), 'target'
    return None, 'random'


def window_chain(video, chain, a, b, hist_len, hist_step, hist_mode, motion_gate):
    """Chain for one window after motion gating (see stabilize.gate_chain). -> (chain, gated)"""
    ref = st.reference_frame(a, b)
    frames = list(range(a, b)) + st.history_frames(len(video), a, b, hist_len, hist_step, hist_mode)
    return st.gate_chain(chain, frames, ref, motion_gate, video.shape[1:3])


def build_sample(video, boxes, chain, a, b, T, P, front, r0, c0, hist_len=80, hist_step=4,
                 hist_mode='both', mean=vtdata.RGBT_MEAN, std=vtdata.RGBT_STD, rule='sigma_sixth',
                 motion_gate=0.0):
    t = b - a
    lo = T - t if front else 0
    ref = st.reference_frame(a, b)
    chain, gated = window_chain(video, chain, a, b, hist_len, hist_step, hist_mode, motion_gate)
    frames, fvalid = st.warp_frames(video, chain, range(a, b), ref, r0, c0, (P, P))
    img = np.zeros((T, P, P), np.float32)
    pv = np.zeros((T, P, P), bool)
    lab = np.zeros((T, P, P), np.float32)
    img[lo:lo + t] = np.where(fvalid, (frames - mean) / std, 0.0)
    pv[lo:lo + t] = fvalid
    for k, tt in enumerate(range(a, b)):
        bx = st.map_boxes(boxes[tt], st.ref_to(chain, tt, ref))
        if len(bx):
            bx[:, 0] -= c0
            bx[:, 1] -= r0
        lab[lo + k] = vtdata.gaussian_mask(bx, P, P, rule) * fvalid[k]
    hist, avail, hframes, hvalid_px = st.history_field(video, chain, a, b, ref, hist_len, hist_step,
                                                       hist_mode, r0, c0, (P, P), return_frames=True)
    hlab = history_label_occupancy(boxes, chain, hframes, hvalid_px, ref, r0, c0, P, rule)
    valid = np.zeros(T, bool)
    valid[lo:lo + t] = True
    return dict(img=img[None], lab=lab, valid=valid, pvalid=pv[None],
                hist=np.where(avail, (hist - mean) / std, 0.0).astype(np.float32)[None],
                hvalid=avail[None], hlab=hlab[None], gated=np.array(gated))


def history_label_occupancy(boxes, chain, frames, valid_px, ref, r0, c0, P, rule='sigma_sixth'):
    """Fraction of the valid History samples in which an annotation covers each pixel.

    A persistent target with high occupancy is also inside B ("resident"), so L - B carries
    no evidence for it; the arrival head of lvh_dec ignores such pixels instead of being
    taught to call them background.
    """
    if not len(frames):
        return np.zeros((P, P), np.float32)
    hit = np.zeros((P, P), np.float32)
    for k, tt in enumerate(frames):
        bx = st.map_boxes(boxes[tt], st.ref_to(chain, tt, ref))
        if len(bx):
            bx[:, 0] -= c0
            bx[:, 1] -= r0
            hit += vtdata.gaussian_mask(bx, P, P, rule) * valid_px[k]
    count = valid_px.sum(0)
    return np.where(count > 0, hit / np.maximum(count, 1), 0.0).astype(np.float32)
