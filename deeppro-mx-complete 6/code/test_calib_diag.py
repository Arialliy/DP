"""Checks of M0c, the motion-calibration diagnostic (numpy only).

Synthetic clips of different 'scenes': in each, moving and parked cars have one contrast
and car-shaped ground hot spots (which shape cannot separate) another, and the scenes
disagree on which is brighter. Absolute contrast then does not transfer across clips,
contrast in exemplar units does.
Run: python code/test_calib_diag.py
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib_diag as cd  # noqa: E402
import exemplar_diag as ed  # noqa: E402
import mx_data as mxd  # noqa: E402
import test_exemplar_diag as te  # noqa: E402

H, W, N = te.H, te.W, te.N
SCENES = {                       # clip: (moving car amp, parked car amps, hot spot amps)
    'DJI_9001_1': (14, (12, 14, 15, 16), (34, 38, 42, 46, 50, 36, 40, 44, 48, 52)),
    'DJI_9002_1': (46, (42, 45, 47, 50), (10, 12, 14, 16, 18, 11, 13, 15, 17, 19)),
    'DJI_9003_1': (26, (24, 26, 27, 29), (9, 10, 11, 12, 13, 56, 60, 64, 68, 72)),
}


def _args(**kw):
    a = dict(seqlen=40, coords='raw', patch=11, k=16, step=4, shift=2, seed=0, band=50, min_pos=20,
             min_gap=0.02, closure=0.30, closure_det=0.15, harm=0.03, harm_frac=0.2)
    a.update(kw)
    return argparse.Namespace(**a)


def scene(clip, seed=0, moving=None):
    """te.scene with this clip's amplitudes; moving overrides the moving cars' amplitude."""
    amp_m, amp_p, amp_h = SCENES[clip]
    amp_m = amp_m if moving is None else moving
    rng = np.random.default_rng(seed + int(clip[4:8]))
    yy, xx = np.mgrid[0:H, 0:W]
    base = 80 + 0.05 * xx + 0.03 * yy
    for (x, y), amp in zip(te.HOT, amp_h):
        te._car(base, x, y, amp)
    parked = [(35, 40), (75, 60), (115, 50), (150, 88)]
    lanes = [8, 25, 80]
    video, boxes, speed = [], [], {}
    for t in range(N):
        img = base + rng.normal(0, 1.5, (H, W))
        b = []
        for i, y in enumerate(lanes):
            x = 5 + (t + 37 * i) % (W - 10)
            te._car(img, x, y, amp_m)
            b.append((round(x) - 3, y - 1, 7, 3, 1, 100 + i))
            speed[(clip, t, 100 + i)] = 1.0
        for j, (x, y) in enumerate(parked):
            te._car(img, x, y, amp_p[j])
            b.append((x - 3, y - 1, 7, 3, 1, 200 + j))
            speed[(clip, t, 200 + j)] = 0.0
        video.append(img.astype(np.float32))
        boxes.append(np.array(b, float))
    return np.stack(video), boxes, speed


def run(moving=None, **kw):
    args = _args(**kw)
    pos, neg = [], []
    for c in SCENES:
        v, b, s = scene(c, moving=moving)
        p, q, _ = ed.process_clip(c, v, b, s, None, args, band=args.band, calib=True)
        pos += p
        neg += q
    per = cd.analyse(pos, neg, args)
    return per, cd.decide(per, args, False), pos, neg


def test_calibration_transfers_where_absolute_contrast_does_not():
    per, verdict, pos, neg = run()
    for c, s in per.items():
        assert s['eligible'] and s['calibration'], (c, s)
        assert s['auc_oracle'] > 0.95 and s['auc_base'] < 0.8, (c, s)       # the calibration gap
        assert s['auc_cal'] > 0.95 and s['auc_full'] > 0.95 and s['closure'] > 0.8, (c, s)
        assert s['auc_shape'] < 0.85, (c, s)                                 # shape cannot do it
    assert verdict['go'] and verdict['median_closure'] > 0.8, verdict
    # the features are those of mx_data.calibrated_features, from exemplars outside the window
    assert all(np.isfinite(r['cal_R']) for r in pos if r['exemplars'] > 0)
    assert all(abs(r['cal_R']) < 0.4 for r in pos if r['exemplars'] > 0)


def test_mismatched_exemplars_give_no_go():
    """The control: moving targets with another contrast than the parked ones calibrate wrongly."""
    per, verdict, _, _ = run(moving=90)
    assert not verdict['go'], verdict
    assert verdict['median_closure'] < 0.30, verdict


def test_band_negatives_lie_in_the_exemplar_band():
    v, b, s = scene('DJI_9003_1')
    v = v.copy()
    clutter = [(10, 40), (22, 52), (50, 38), (58, 66), (90, 64), (100, 38), (128, 62), (142, 42), (20, 64)]
    for img in v:                                        # unlabelled clutter with the targets' contrast
        for x, y in clutter:
            te._car(img, x, y, SCENES['DJI_9003_1'][0])
    args = _args()
    old = ed.NEG_PER_WINDOW
    try:
        ed.NEG_PER_WINDOW = 5                            # the 5 strongest peaks are 'top', the rest may be 'band'
        _, neg, win = ed.process_clip('DJI_9003_1', v, b, s, None, args, band=args.band, calib=True)
    finally:
        ed.NEG_PER_WINDOW = old
    band = [r for r in neg if r['kind'] == 'band']
    assert len(band) >= 3 * len(win) and sum(w['band_negatives'] for w in win) == len(band)
    for w in win:
        lo, hi = np.percentile(np.abs(w['exemplar_contrast']), [10, 90])
        rows = [r for r in band if r['window_start'] == w['window_start']]
        assert all(lo <= r['peak_abs_contrast'] <= hi for r in rows), (lo, hi, rows)
    assert {r['kind'] for r in neg} == {'top', 'band'}
    assert sum(r['kind'] == 'top' for r in neg) == 5 * len(win)


def test_exemplar_contrast_is_the_full_frame_value():
    rng = np.random.default_rng(0)
    img = rng.normal(80, 2, (60, 60)).astype(np.float32)
    img[30:33, 20:23] += 25
    img[10:13, 40:43] -= 9
    import stabilize as st
    full = st.local_contrast(img)
    rows = np.array([(0, 21.4, 31.2, 0), (0, 41, 11, 0)], float)
    c = ed.exemplar_contrasts(img[None], rows, 2)
    assert np.isclose(c[0], np.nanmax(full[29:34, 19:24])) and np.isclose(c[1], np.nanmin(full[9:14, 39:44]))


def test_calibrated_features_are_gain_invariant():
    rng = np.random.default_rng(1)
    c, ce = rng.normal(0, 20, 50), rng.normal(15, 5, 12)
    f = mxd.calibrated_features(c, ce)
    for g in (0.5, 1.7, 3.0):
        # with eps proportional to the gain the features are exactly invariant
        assert np.allclose(mxd.calibrated_features(g * c, g * ce, eps=g), f)
    R, absR, D, P = f.T
    assert np.allclose(absR, np.abs(R)) and np.all(D <= absR + 1e-9 + np.ptp(np.log(np.abs(ce) + 1)))
    assert np.allclose(mxd.calibrated_features(c, []), 0)
    assert np.isnan(mxd.calibrated_features([np.nan], ce)).all()
    one = mxd.calibrated_features([14.0], [14.0, -14.0, 14.0])
    assert np.allclose(one, [[0, 0, 0, 1 / 3]])


def test_loro_leaves_out_the_whole_recording():
    rng = np.random.default_rng(2)
    clips = np.array(['DJI_0001_1'] * 40 + ['DJI_0001_2'] * 40 + ['DJI_0002_1'] * 40)
    groups = np.array([cd.recording(c) for c in clips])
    y = np.tile(np.r_[np.ones(20), np.zeros(20)], 3)
    X = np.c_[rng.normal(size=120)]
    X[:80, 0] += 3 * y[:80]            # recording DJI_0001: informative feature
    X[80:, 0] -= 3 * y[80:]            # DJI_0002: the sign is flipped
    s = cd.loro_scores(X, y, clips, groups, 'DJI_0002_1')
    from diagnose_history import auc
    assert auc(s[y[80:] == 1], s[y[80:] == 0]) < 0.1          # fitted on DJI_0001 only
    s1 = cd.loro_scores(X, y, clips, groups, 'DJI_0001_1')    # fitted on DJI_0002 only, not on _2
    assert auc(s1[y[:40] == 1], s1[y[:40] == 0]) < 0.1
    w = cd.clip_weights(clips, y)
    assert np.isclose(w.sum(), 1) and np.allclose(w, 1 / 120)


def test_decision_rule():
    args = _args()

    def clip(gap, closure, harm=False, eligible=True):
        base = 0.8
        full = base + closure * gap
        return dict(eligible=eligible, calibration=eligible and gap >= 0.02, closure=closure if gap >= 0.02 else np.nan,
                    closure_det=np.nan, harm=harm, positives=100, gap=gap, auc_base=base, auc_full=full)
    ok = {'a': clip(0.1, 0.5), 'b': clip(0.2, 0.4), 'c': clip(0.0, 0.0)}
    assert cd.decide(ok, args, False)['go']
    assert not cd.decide({'a': clip(0.1, 0.5), 'c': clip(0.0, 0.0)}, args, False)['go']        # (a)
    assert not cd.decide({'a': clip(0.1, 0.2), 'b': clip(0.2, 0.25)}, args, False)['go']       # (b)
    many = dict(ok, d=clip(0.0, 0, harm=True), e=clip(0.0, 0, harm=True))
    assert not cd.decide(many, args, False)['go']                                              # (c)
    v = cd.decide(ok, args, True)
    assert not v['go'] and v['checks']['d_median_closure_det'] is False                        # (d)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
