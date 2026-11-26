"""Two-pass inference of DeepPro-MX (motion exemplars, code/mx.py).

Pass 1  The network's variation pathway (DeepProMX.profile) scores every DeepPro window
        (rate 1; coordinates by --coords with the rule of infer_mr.py). The peaks of the
        clip's variation logits at or above --ex-min-score are the exemplar candidates
        (t, x, y, logit).
Pass 2  Every window is scored with an exemplar bank chosen by --bank:
          outside  candidates from frames outside the window (default; the training rule)
          causal   candidates from frames before the window only
          clip     candidates from the whole clip
          gt       oracle: moving GT boxes outside the window (rel_speed >= --ex-moving in
                   --targets); the upper bound of the exemplar source
          random   control: random points outside the window
          none     control: no exemplars (learned global prototypes only)
        The highest-scoring candidates are taken first, at most one per target appearance
        (mx_data.greedy_select), up to --ex-k.
Outputs (like infer.py): cache/preds/<run>/<clip>.npy float16 logits, the maximum over
overlapping windows; with --save-arms also <run>__variation, <run>__level and
<run>__match; and cache/preds/<run>__meta/<clip>.json with per-window coordinates and
exemplars. score.py scores every one of these runs.

Usage
  python code/infer_mx.py --run mx_s46 --ckpt runs/mx_s46/best_model.pth --clips val --device cuda
  python code/infer_mx.py --run mxc_s46 --arch mxc_lv --ckpt runs/mxc_s46/best_model.pth --clips test --device cuda
  python code/infer_mx.py --run mx_s46_none --ckpt runs/mx_s46/best_model.pth --clips val --bank none --device cuda
--arch defaults to the training config.json next to --ckpt (else mx_lv).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import multirate as mr  # noqa: E402
import mx_data as mxd  # noqa: E402
import stabilize as st  # noqa: E402
import vtdata  # noqa: E402

FILL = -30.0  # logit outside a compensated window's field (below evaluate.FLOOR), as infer_mr.py
BANKS = ('outside', 'causal', 'clip', 'gt', 'random', 'none')


def assemble_maps(n, shape, plan, decide, score, names, dtypes=None):
    """Like multirate.assemble for several outputs: score(frames, coords) -> {name: (len,H,W)}.
    -> ({name: (n,H,W)} maxima over the windows holding each frame, per-window records)."""
    dtypes = dtypes or {}
    out = {k: np.full((n,) + tuple(shape), -np.inf, dtypes.get(k, np.float32)) for k in names}
    meta = []
    for _, frames in plan:
        coords, stats = decide(frames)
        meta.append(dict(first=int(frames[0]), last=int(frames[-1]), coords=coords,
                         **{key: float(v) for key, v in stats.items()}))
        if coords == 'skip':
            continue
        maps = score(frames, coords)
        for k in names:
            out[k][frames] = np.maximum(out[k][frames], np.asarray(maps[k]).astype(out[k].dtype))
    return out, meta


def memo(decide):
    """Decide each window once, so both passes use the same coordinates."""
    cache = {}

    def f(frames):
        key = (int(frames[0]), int(frames[-1]), len(frames))
        if key not in cache:
            cache[key] = decide(frames)
        return cache[key]
    return f


def exemplar_candidates(n, shape, plan, decide, profile_fn, peaks_fn, min_score):
    """Pass 1: peaks (t, x, y, logit) of the variation logits, the maximum over the windows
    holding each frame, at or above min_score. checkpoint_selection.py uses the same."""
    lv, _ = assemble_maps(n, shape, plan, decide, lambda fr, co: {'lv': profile_fn(fr, co)}, ('lv',))
    rows = [mxd.peak_rows(lv['lv'][t], t, peaks_fn, min_score) for t in range(n)]
    return np.concatenate(rows) if rows else np.zeros((0, 4))


def run_clip(n, shape, T, decide, profile_fn, score_fn, bank_fn, peaks_fn, min_score, names=('fused',),
             need_profile=True):
    """Both passes over one clip.

    decide(frames) -> (coords, stats); profile_fn(frames, coords) -> (len,H,W) variation
    logits in frame coordinates; score_fn(frames, coords, rows) -> {name: (len,H,W)};
    bank_fn(cands, lo, hi) -> exemplar rows (t, x, y, extra) for a window covering lo..hi.
    -> (maps {name: (n,H,W)}, window records with their exemplars, candidates (m,4)).
    """
    plan = mr.plan_windows(n, 1, T)
    decide = memo(decide)
    cands = np.zeros((0, 4))
    if need_profile:
        cands = exemplar_candidates(n, shape, plan, decide, profile_fn, peaks_fn, min_score)
    banks = {}

    def score(frames, coords):
        lo, hi = int(frames[0]), int(frames[-1])
        rows = bank_fn(cands, lo, hi)
        banks[(lo, hi)] = rows
        return score_fn(frames, coords, rows)

    maps, meta = assemble_maps(n, shape, plan, decide, score, names,
                               {k: (np.float32 if k == 'fused' else np.float16) for k in names})
    for w in meta:
        rows = banks.get((w['first'], w['last']), np.zeros((0, 4)))
        w['exemplars'] = int(len(rows))
        w['exemplar_rows'] = [[round(float(v), 3) for v in r] for r in rows]
    return maps, meta, cands


def make_bank(mode, K, n, H, W, moving=None, seed=0):
    """bank_fn(cands, lo, hi) for a --bank mode."""
    rng = np.random.default_rng(seed)
    if mode in ('outside', 'causal', 'clip'):
        return lambda cands, lo, hi: mxd.select_bank(cands, lo, hi, mode, K)
    if mode == 'gt':
        if moving is None:
            raise ValueError("bank 'gt' needs the moving GT table")
        return lambda cands, lo, hi: mxd.gt_bank(rng, moving, lo, hi, 'outside', K)
    if mode == 'random':
        return lambda cands, lo, hi: mxd.random_bank(rng, n, H, W, lo, hi, 'outside', K)
    if mode == 'none':
        return lambda cands, lo, hi: np.zeros((0, 4))
    raise ValueError(mode)


def make_scorers(model, device, video, chain, K, arms):
    """profile_fn and score_fn of run_clip for a DeepProMX network (torch)."""
    import torch
    n, H, W = video.shape
    S = model.patch
    mean, std = vtdata.RGBT_MEAN, vtdata.RGBT_STD

    def tens(v, dt):
        return torch.from_numpy(np.ascontiguousarray(v)).to(device=device, dtype=dt)

    def window(frames, coords):
        frames = [int(t) for t in frames]
        ref = frames[len(frames) // 2]
        if coords == 'warp':
            stack, pv = st.warp_frames(video, chain, frames, ref)
            x = np.where(pv, (stack - mean) / std, 0.0).astype(np.float32)
        else:
            x = ((video[frames] - mean) / std).astype(np.float32)
            pv = np.ones(x.shape, bool)
        return frames, ref, tens(x[None, None], torch.float32), tens(pv[None, None], torch.bool)

    def back(maps, frames, coords, ref):
        maps = np.ascontiguousarray(maps, np.float32)
        return st.warp_back(maps, chain, frames, ref, FILL) if coords == 'warp' else maps

    @torch.no_grad()
    def profile_fn(frames, coords):
        frames, ref, x, pv = window(frames, coords)
        return back(model.profile(x, pvalid=pv)[0].float().cpu().numpy(), frames, coords, ref)

    @torch.no_grad()
    def score_fn(frames, coords, rows):
        frames, ref, x, pv = window(frames, coords)
        ex, exv = mxd.extract_patches(video, rows[:, :3] if len(rows) else np.zeros((0, 3)), S, mean, std, K)
        _, logit = model(x, pvalid=pv, ex=tens(ex[None], torch.float32), ex_valid=tens(exv[None], torch.bool))
        out = {'fused': back(logit[0].float().cpu().numpy(), frames, coords, ref)}
        if arms:
            last, T = model.last, len(frames)
            out['variation'] = back(last['lv'][0].float().cpu().numpy(), frames, coords, ref)
            out['level'] = back(np.broadcast_to(last['ll'][0, 0].float().cpu().numpy(), (T, H, W)), frames, coords, ref)
            out['match'] = back(np.broadcast_to(last['mx'][0].float().cpu().numpy(), (T, H, W)), frames, coords, ref)
        return out
    return profile_fn, score_fn


def save_f16(path, array):
    tmp = path + '.tmp.npy'
    np.save(tmp, np.asarray(array).astype(np.float16))
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--arch', default=None, help='default: the training config, else mx_lv')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--clips', default='val', help='split name, foldK/val, or comma-separated clips')
    ap.add_argument('--bank', choices=BANKS, default='outside')
    ap.add_argument('--coords', choices=('raw', 'warp', 'auto'), default='auto')
    ap.add_argument('--ex-k', dest='ex_k', type=int, default=None, help='default: the training config, else 16')
    ap.add_argument('--ex-min-score', dest='ex_min_score', type=float, default=0.0,
                    help='minimum variation logit of an exemplar candidate')
    ap.add_argument('--targets', default='results/profile/targets.csv', help="for --bank gt")
    ap.add_argument('--ex-moving', dest='ex_moving', type=float, default=mxd.MOVING, help="for --bank gt")
    ap.add_argument('--margin', type=float, default=0.1)
    ap.add_argument('--abs-tol', dest='abs_tol', type=float, default=0.5)
    ap.add_argument('--min-cover', dest='min_cover', type=float, default=0.1)
    ap.add_argument('--lowres', type=int, default=4)
    ap.add_argument('--seqlen', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0, help="for --bank gt and random")
    ap.add_argument('--save-arms', dest='save_arms', action='store_true')
    ap.add_argument('--threads', type=int, default=int(os.environ.get('OMP_NUM_THREADS', 16)))
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args(argv)

    import torch
    import evaluate
    import infer_mr
    import models
    import splits
    torch.set_num_threads(args.threads)
    cfg_path = os.path.join(os.path.dirname(args.ckpt), 'config.json')
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    if args.arch is None:
        args.arch = cfg.get('arch', 'mx_lv')
    elif cfg.get('arch') not in (None, args.arch):
        raise SystemExit(f"--arch {args.arch}, but {cfg_path} says {cfg['arch']}")
    model, needs = models.build(args.arch, seqlen=args.seqlen)
    if 'ex' not in needs:
        raise SystemExit(f'{args.arch} takes no exemplars; use infer.py or infer_mr.py')
    sd = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and ('model_state_dict' in sd or 'model' in sd):
        sd = sd.get('model_state_dict', sd.get('model'))
    model.load_state_dict(sd)
    device = torch.device(args.device)
    model.to(device)
    if device.type == 'cuda':
        torch.nn.utils.convert_conv3d_weight_memory_format(model, torch.channels_last_3d)
    model.eval()
    if args.ex_k is None:
        args.ex_k = int(cfg.get('ex_k', 16))
    names = ('fused', 'variation', 'level', 'match') if args.save_arms else ('fused',)
    base = f'cache/preds/{args.run}'
    dirs = {'fused': base, 'meta': f'{base}__meta'}
    dirs.update({k: f'{base}__{k}' for k in names if k != 'fused'})
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    clips = splits.clips(args.clips)
    speed = mxd.read_speeds(args.targets, clips) if args.bank == 'gt' else {}
    print(f'{args.run}: {args.arch} bank={args.bank} coords={args.coords} K={args.ex_k} '
          f'min_score={args.ex_min_score}', flush=True)
    summary = {}
    for c in clips:
        meta_path = f"{dirs['meta']}/{c}.json"
        if os.path.exists(meta_path) and all(os.path.exists(f'{dirs[k]}/{c}.npy') for k in names):
            summary[c] = json.load(open(meta_path))['summary']
            continue
        t0 = time.time()
        sc = vtdata.Scene(c)
        video = sc.video()
        n, H, W = video.shape
        chain = st.load_chain(c, n) if args.coords != 'raw' else infer_mr.identity_chain(n)
        low = mr.lowres(video, args.lowres) if args.coords != 'raw' else None
        decide = infer_mr.make_decider(args.coords, low, chain, args.lowres, args.margin, args.abs_tol,
                                       args.min_cover)
        moving = None
        if args.bank == 'gt':
            mxd.require_speeds({c: sc.boxes}, speed, args.targets)
            moving = mxd.moving_table(c, sc.boxes, speed, args.ex_moving)
        bank_fn = make_bank(args.bank, args.ex_k, n, H, W, moving, args.seed + sum(map(ord, c)))
        profile_fn, score_fn = make_scorers(model, device, video, chain, args.ex_k, args.save_arms)
        maps, meta, cands = run_clip(n, (H, W), args.seqlen, decide, profile_fn, score_fn, bank_fn,
                                     evaluate.peaks, args.ex_min_score, names,
                                     need_profile=args.bank in ('outside', 'causal', 'clip'))
        for k in names:
            save_f16(f'{dirs[k]}/{c}.npy', maps[k])
        summ = dict(frames=n, windows=len(meta), warped=sum(w['coords'] == 'warp' for w in meta),
                    candidates=int(len(cands)), mean_exemplars=float(np.mean([w['exemplars'] for w in meta])),
                    windows_without_exemplars=sum(w['exemplars'] == 0 for w in meta))
        json.dump(dict(clip=c, settings=vars(args), summary=summ, windows=meta), open(meta_path, 'w'), indent=1)
        summary[c] = summ
        print(f"{args.run} {c} frames={n} windows={summ['windows']} warped={summ['warped']} "
              f"candidates={summ['candidates']} exemplars/window={summ['mean_exemplars']:.1f} "
              f"{time.time() - t0:.1f}s", flush=True)
    json.dump(summary, open(f"{dirs['meta']}/summary.json", 'w'), indent=1)


if __name__ == '__main__':
    main()
