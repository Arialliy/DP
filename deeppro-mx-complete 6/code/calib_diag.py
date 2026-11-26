"""M0c: training-free test of the motion-calibration hypothesis behind DeepPro-MXC (mx.py).

Why M0 (exemplar_diag.py) is replaced
    M0 asked whether exemplar similarity adds held-out AUC to [contrast, contrast^2] within
    each clip, and needed a gain of 0.10. Two faults: (1) its baseline was fitted on the
    same clip (window-parity folds), so it already knew the clip's contrast band: on
    DJI_0319_2 it reached 0.975 and left at most 0.025 to gain, below the 0.10 asked;
    (2) a detector is trained on other clips and never sees the test clip's band. Fitted on
    the other clips, the same baseline reaches 0.837 on DJI_0319_2. The 0.14 between the two
    is the calibration gap: the information is in the clip but only relative to the scene.

Hypothesis
    The moving targets of a clip (the exemplars) carry the scene's target contrast, so
    contrast measured in exemplar units (mx_data.calibrated_features: R, |R|, D, P) transfers
    across scenes where absolute contrast does not, and closes part of the gap.

Test
    process_clip of exemplar_diag.py, with the same positives (near-stationary resident
    annotations), the strongest contrast peaks as negatives ('top') and up to --band
    'band' negatives per window: peaks whose |contrast| lies inside the 10-90 % range of the
    window's GT exemplars, the clutter that calibration alone cannot reject. Rows of
    windows without GT exemplars are left out (as in M0).
    Leave-one-recording-out (LORO): for each clip, logistic regressions are fitted on the
    clips of all other recordings (every clip and class weighs the same) and scored on it:
        base       [contrast, contrast^2]                     absolute contrast (LV's cue)
        shape      base + [ncc_L, absncc_L, ncc_med]          M0's exemplar similarity
        cal        [R, |R|, D, P]                             calibrated contrast only
        full       cal + [ncc_L, absncc_L, ncc_med]           amplitude-free (DeepPro-MXC)
        all        base + cal + shape
        full_det   the 'full' features with --det-run exemplars (inference source)
    oracle = [contrast, contrast^2] fitted within the clip (window-parity folds): what a
    detector that knew the scene's band would reach.
    gap = oracle - base;  closure = (full - base) / gap.

Decision (pre-registered; written before M0c was run on the real clips)
    Only clips not in --seen decide (default: the validation clips, whose M0 numbers shaped
    this design); the result on every clip is reported as decision_all.
    eligible clips      >= --min-pos positives and negatives with GT exemplars
    calibration clips   eligible and gap >= --min-gap (0.02); the others have no gap to close
    GO if all hold:
      (a) at least 2 calibration clips;
      (b) median closure over the calibration clips >= --closure (0.30);
      (c) at most --harm-frac (20 %) of the eligible clips have full < base - --harm (0.03);
      (d) with --det-run: median closure of full_det >= --closure-det (0.15).
    Otherwise NO-GO, and DeepPro-MXC is not trained. The positive-weighted mean closure
    and the AUCs on band negatives alone are reported, but do not decide.

Usage (repository root, VT_ROOT set, profile_motion.py already run; no test clip is used)
    python code/calib_diag.py                                  # --split official_train (45 clips)
    python code/calib_diag.py --det-run official_rgbt          # also the inference exemplars
Writes results/calib_diag/{positives.csv, negatives.csv, summary.json}.
Exit status 0 on GO, 3 on NO-GO (1 on errors).
"""
import argparse
import csv
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import exemplar_diag as m0  # noqa: E402
import mx_data as mxd  # noqa: E402
from diagnose_history import auc  # noqa: E402

SHAPE = ('ncc_L', 'absncc_L', 'ncc_med')
SETS = {
    'base': ('contrast', 'contrast2'),
    'shape': ('contrast', 'contrast2') + SHAPE,
    'cal': m0.CAL,
    'full': m0.CAL + SHAPE,
    'all': ('contrast', 'contrast2') + m0.CAL + SHAPE,
    'full_det': m0.CAL_DET + ('ncc_det',),
}


def recording(clip):
    """Recording id (DJI_XXXX) of a clip; clips of one recording are held out together."""
    import vtdata
    return vtdata.recording_id(clip)


def table(rows, keys):
    """(n, len(keys)) float array of the rows; contrast2 is contrast squared."""
    def get(r, k):
        return float(r['contrast']) ** 2 if k == 'contrast2' else float(r.get(k, math.nan))
    return np.array([[get(r, k) for k in keys] for r in rows], float).reshape(len(rows), len(keys))


def fill_missing(X, ref):
    """NaNs of every column (ncc_med where the patch left the field) by ref's column median."""
    X = X.copy()
    for j in range(X.shape[1]):
        col = ref[:, j]
        med = np.nanmedian(col) if np.isfinite(col).any() else 0.0
        X[~np.isfinite(X[:, j]), j] = med
    return X


def weighted_logistic(Z, y, w, l2=1e-2, iters=50):
    """Newton steps of an L2 logistic regression with sample weights w (summing to 1).
    -> weights (d + 1,), the last the intercept."""
    X = np.c_[Z, np.ones(len(Z))]
    reg = np.full(X.shape[1], l2)
    reg[-1] = 0.0
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))
        g = X.T @ (w * (p - y)) + reg * beta
        hess = (X * (w * p * (1 - p))[:, None]).T @ X + np.diag(reg + 1e-9)
        step = np.linalg.solve(hess, g)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def clip_weights(clips, y):
    """Every (clip, class) pair weighs the same in total; the weights sum to 1."""
    w = np.zeros(len(y))
    keys = list(zip(clips, y))
    count = {}
    for k in keys:
        count[k] = count.get(k, 0) + 1
    for i, k in enumerate(keys):
        w[i] = 1.0 / count[k]
    return w / w.sum()


def loro_scores(X, y, clips, groups, target):
    """Logistic scores of the rows of clip `target`, fitted on the rows of every other
    recording (groups). NaN scores if the training rows lack a class."""
    tr, te = groups != groups[clips == target][0], clips == target
    if len(np.unique(y[tr])) < 2 or not te.any():
        return np.full(int(te.sum()), np.nan)
    Xtr = fill_missing(X[tr], X[tr])
    Xte = fill_missing(X[te], X[tr])
    w = clip_weights(clips[tr], y[tr])
    mu = (w[:, None] * Xtr).sum(0)
    sd = np.sqrt((w[:, None] * (Xtr - mu) ** 2).sum(0)) + 1e-9
    beta = weighted_logistic((Xtr - mu) / sd, y[tr], w)
    return np.c_[(Xte - mu) / sd, np.ones(int(te.sum()))] @ beta


def oracle_auc(rows, y):
    """[contrast, contrast^2] fitted within the clip, window-parity folds (M0's baseline)."""
    starts = sorted({r['window_start'] for r in rows})
    fold = np.array([starts.index(r['window_start']) % 2 for r in rows])
    return m0.heldout_auc(table(rows, SETS['base']), y, fold)


def analyse(pos, neg, args):
    """Per-clip LORO AUCs, gaps and closures -> {clip: dict}."""
    p = [r for r in pos if r['exemplars'] > 0 and np.isfinite(r['contrast'])]
    q = [r for r in neg if r['exemplars'] > 0 and np.isfinite(r['contrast'])]
    rows = p + q
    y = np.array([1.0] * len(p) + [0.0] * len(q))
    clips = np.array([r['clip'] for r in rows])
    groups = np.array([recording(c) for c in clips])
    has_det = any(r['det_exemplars'] > 0 for r in rows)
    sets = {k: v for k, v in SETS.items() if k != 'full_det' or has_det}
    X = {k: table(rows, v) for k, v in sets.items()}
    out = {}
    for c in sorted(set(clips)):
        te = clips == c
        yc = y[te]
        s = dict(recording=recording(c), positives=int(yc.sum()), negatives=int((1 - yc).sum()),
                 band_negatives=int(sum(r.get('kind') == 'band' for r, k in zip(rows, te) if k)))
        for k in sets:
            sc = loro_scores(X[k], y, clips, groups, c)
            s[f'auc_{k}'] = auc(sc[yc == 1], sc[yc == 0])
            band = np.array([r.get('kind') == 'band' for r, k in zip(rows, te) if k], bool)
            s[f'auc_{k}_band'] = auc(sc[yc == 1], sc[band]) if band.any() else math.nan
        crow = [r for r, k in zip(rows, te) if k]
        s['auc_oracle'] = oracle_auc(crow, yc) if 0 < yc.sum() < len(yc) else math.nan
        s['gap'] = s['auc_oracle'] - s['auc_base']
        s['eligible'] = bool(s['positives'] >= args.min_pos and s['negatives'] >= args.min_pos
                             and all(np.isfinite(s[f'auc_{k}']) for k in ('base', 'full'))
                             and np.isfinite(s['gap']))
        s['calibration'] = bool(s['eligible'] and s['gap'] >= args.min_gap)
        s['closure'] = (s['auc_full'] - s['auc_base']) / s['gap'] if s['calibration'] else math.nan
        s['closure_det'] = ((s['auc_full_det'] - s['auc_base']) / s['gap']
                            if s['calibration'] and 'full_det' in sets else math.nan)
        s['harm'] = bool(s['eligible'] and s['auc_full'] < s['auc_base'] - args.harm)
        out[c] = s
    return out


def decide(per_clip, args, det):
    """GO / NO-GO by the pre-registered rule of the module docstring."""
    eligible = sorted(c for c, s in per_clip.items() if s['eligible'])
    cal = sorted(c for c in eligible if per_clip[c]['calibration'])
    closures = [per_clip[c]['closure'] for c in cal]
    med = float(np.median(closures)) if closures else math.nan
    wts = [per_clip[c]['positives'] for c in cal]
    wmean = float(np.average(closures, weights=wts)) if closures else math.nan
    harmed = sorted(c for c in eligible if per_clip[c]['harm'])
    harm_frac = len(harmed) / len(eligible) if eligible else math.nan
    checks = {'a_calibration_clips': len(cal) >= 2,
              'b_median_closure': bool(np.isfinite(med) and med >= args.closure),
              'c_harm': bool(eligible and harm_frac <= args.harm_frac)}
    med_det = math.nan
    if det:
        cd = [per_clip[c]['closure_det'] for c in cal if np.isfinite(per_clip[c]['closure_det'])]
        med_det = float(np.median(cd)) if cd else math.nan
        checks['d_median_closure_det'] = bool(np.isfinite(med_det) and med_det >= args.closure_det)
    return dict(go=all(checks.values()), checks=checks, eligible_clips=eligible, calibration_clips=cal,
                median_closure=med, positive_weighted_closure=wmean, median_closure_det=med_det,
                harmed_clips=harmed, harm_fraction=harm_frac, inference_exemplars_checked=bool(det),
                rule=(f'(a) >= 2 calibration clips (eligible: >= {args.min_pos} positives and negatives; '
                      f'gap >= {args.min_gap}); (b) median closure >= {args.closure}; (c) <= '
                      f'{args.harm_frac:.0%} of eligible clips with full < base - {args.harm}'
                      + (f'; (d) median closure of full_det >= {args.closure_det}' if det else '')))


def _one(task):
    clip, args, speed = task
    import stabilize as st
    import vtdata
    sc = vtdata.Scene(clip)
    mxd.require_speeds({clip: sc.boxes}, speed, args.targets)
    video = st.load_video_u8(clip, sc)
    chain = None if args.coords == 'raw' else st.load_chain(clip, len(sc))
    det = m0.detection_candidates(args.det_run, clip, len(sc), args.det_min_score) if args.det_run else None
    return clip, m0.process_clip(clip, video, sc.boxes, speed, chain, args, det, band=args.band, calib=True)


def write_rows(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', nargs='*', default=['official_train'])
    ap.add_argument('--seen', nargs='*', default=['val'],
                    help='clips whose earlier M0 results shaped this design: reported, left out of the decision')
    ap.add_argument('--clips', default=None, help='comma-separated clips (overrides --split)')
    ap.add_argument('--targets', default='results/profile/targets.csv')
    ap.add_argument('--seqlen', type=int, default=40)
    ap.add_argument('--coords', choices=('raw', 'warp', 'auto'), default='auto')
    ap.add_argument('--patch', type=int, default=11, help='odd patch size for NCC')
    ap.add_argument('--k', type=int, default=16, help='exemplars per window')
    ap.add_argument('--step', type=int, default=4, help='frame step of ncc_med')
    ap.add_argument('--shift', type=int, default=2, help='every score is maximised over +-shift px')
    ap.add_argument('--band', type=int, default=50, help='band negatives per window')
    ap.add_argument('--det-run', dest='det_run', default=None, help='cached run whose peaks are exemplars')
    ap.add_argument('--det-min-score', dest='det_min_score', type=float, default=0.0)
    ap.add_argument('--min-pos', dest='min_pos', type=int, default=20)
    ap.add_argument('--min-gap', dest='min_gap', type=float, default=0.02)
    ap.add_argument('--closure', type=float, default=0.30)
    ap.add_argument('--closure-det', dest='closure_det', type=float, default=0.15)
    ap.add_argument('--harm', type=float, default=0.03)
    ap.add_argument('--harm-frac', dest='harm_frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results/calib_diag')
    ap.add_argument('--nproc', type=int, default=4)
    args = ap.parse_args(argv)
    if args.patch % 2 != 1:
        raise SystemExit('--patch must be odd')
    import splits
    clips = args.clips.split(',') if args.clips else [c for s in args.split for c in splits.clips(s)]
    used = sorted(set(splits.clips('test')) & set(clips))
    if used:
        raise SystemExit(f'M0c must not see test clips: {used}')
    speed = mxd.read_speeds(args.targets, clips)
    os.makedirs(args.out, exist_ok=True)
    tasks = [(c, args, {key: v for key, v in speed.items() if key[0] == c}) for c in clips]
    pos, neg, win = [], [], []
    with Pool(max(1, min(args.nproc, len(tasks)))) as pool:
        for clip, (p, q, w) in pool.imap_unordered(_one, tasks):
            pos += p
            neg += q
            win += w
            print(f'{clip:12s} positives={len(p):5d} negatives={len(q):5d} windows={len(w)}', flush=True)
    order = {c: i for i, c in enumerate(clips)}
    for rows in (pos, neg, win):
        rows.sort(key=lambda r: (order[r['clip']], r['window_start']))
    write_rows(os.path.join(args.out, 'positives.csv'), pos)
    write_rows(os.path.join(args.out, 'negatives.csv'), neg)
    per_clip = analyse(pos, neg, args)
    seen = {c for name in args.seen for c in splits.clips(name)}
    for c, sc in per_clip.items():
        sc['seen'] = c in seen
    verdict = decide({c: sc for c, sc in per_clip.items() if not sc['seen']}, args, bool(args.det_run))
    verdict['seen_clips_left_out'] = sorted(c for c in per_clip if c in seen)
    verdict_all = decide(per_clip, args, bool(args.det_run))
    json.dump(dict(settings=vars(args), per_clip=per_clip, decision=verdict, decision_all=verdict_all,
                   windows=win), open(os.path.join(args.out, 'summary.json'), 'w'), indent=1, default=float)
    cols = ['base', 'shape', 'cal', 'full', 'all'] + (['full_det'] if args.det_run else [])
    print('\nclip            pos   neg ' + ''.join(f'{k:>9s}' for k in cols) + '   oracle    gap  closure')
    for c in sorted(per_clip):
        s = per_clip[c]
        print(f"{c:12s} {s['positives']:6d} {s['negatives']:5d} " + ''.join(f"{s[f'auc_{k}']:9.3f}" for k in cols)
              + f"{s['auc_oracle']:9.3f} {s['gap']:+6.3f} {s['closure']:8.2f}"
              + ('' if s['eligible'] else '  (not eligible)')
              + ('  (no gap)' if s['eligible'] and not s['calibration'] else '') + ('  (seen)' if s['seen'] else ''))
    for name, v in (('all clips (reported)', verdict_all), ('unseen clips (decides)', verdict)):
        print(f"\n{name}: {'GO' if v['go'] else 'NO-GO'} {v['checks']}; median closure "
              f"{v['median_closure']:.3f} over {len(v['calibration_clips'])} calibration clips; "
              f"harmed {v['harmed_clips']}")
    print(f"rule: {verdict['rule']}")
    return dict(per_clip=per_clip, decision=verdict, decision_all=verdict_all)


if __name__ == '__main__':
    # Exit status 0 on GO and 3 on NO-GO, so that a pipeline (run_mx.sh) stops before training.
    sys.exit(0 if main()['decision']['go'] else 3)
