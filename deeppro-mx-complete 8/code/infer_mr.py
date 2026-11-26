"""Multi-rate inference (MRTP): one trained network at several temporal rates.

For each clip and each rate k in --rates, the network scores phase-interleaved windows of
T samples taken every k frames (multirate.plan_windows), each in the coordinate frame
chosen by --coords (raw | warp | auto, multirate.choose_coords). Outputs, each a run
that code/score.py can score:
  cache/preds/<run>__r<k>/<clip>.npy      rate k
  cache/preds/<run>__r<k>cal/<clip>.npy   rate k after CFAR calibration      (--fusion cal)
  cache/preds/<run>__mrmax/<clip>.npy     maximum over rates                 (--fusion max)
  cache/preds/<run>__mrcal/<clip>.npy     maximum over calibrated rates      (--fusion cal)
  cache/preds/<run>__meta/<clip>.json     per-window coordinates, residuals, skips, CFAR levels
A rate that needs more frames than a clip has (k T > n) is not used on that clip: its
__r<k> map is -inf there and the fused maps use the other rates.

Architectures
  lv_*, deeppro_plus   trained on recorded frames: --coords raw only
  mr_lv                any coordinates (train.py --rates ... --coords auto)
  lvh_*                any coordinates; --no-history zeroes the History input

Usage
  # T2, no training: the existing level-focal weights at rates 1, 2, 4 on recorded frames
  python code/infer_mr.py --run focal_s46_mr --arch lv_full \
      --ckpt runs/gpu_lvfull_focal_s46/final.pth --clips val --rates 1,2,4 --coords raw --device cuda
  # T3: lvh_dec at rate 1 without compensation, then without History
  python code/infer_mr.py --run dec_s46_raw --arch lvh_dec --ckpt runs/gpu_lvhdec_s46/final.pth \
      --clips val --rates 1 --coords raw --fusion none --device cuda
  python code/infer_mr.py --run dec_s46_nohist --arch lvh_dec --ckpt runs/gpu_lvhdec_s46/final.pth \
      --clips val --rates 1 --coords warp --no-history --fusion none --device cuda
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import multirate as mr  # noqa: E402
import stabilize as st  # noqa: E402
import vtdata  # noqa: E402

FILL = -30.0  # logit for frame pixels outside a compensated window (below evaluate.FLOOR)


def identity_chain(n):
    return np.broadcast_to(np.eye(3), (n, 3, 3)).copy()


def make_decider(policy, video_low, chain, f=4, margin=0.1, abs_tol=0.5, min_cover=0.1):
    def decide(frames):
        if policy == 'raw':
            return 'raw', {}
        k = int(frames[1] - frames[0]) if len(frames) > 1 else 1
        stats = mr.stationarity(video_low[np.asarray(frames)], chain, frames, f)
        return mr.choose_coords(policy, stats, k, margin, abs_tol, min_cover), stats
    return decide


def run_rates(n, shape, rates, T, decide, score, calibrate=True, cal_m=10, cal_half=18):
    """Yield (k, logits (n,H,W) float32 or None, window records, CFAR levels or None)."""
    for k in rates:
        plan = mr.plan_windows(n, k, T)
        if not plan:
            yield k, None, [], None
            continue
        out, meta = mr.assemble(n, shape, plan, decide, score)
        ref = mr.cfar_reference([mr.peak_scores(out[t]) for t in range(n)], cal_m, cal_half) if calibrate else None
        yield k, out, meta, ref


def save_f16(path, array):
    """(n,H,W) -> float16 .npy, written frame by frame."""
    tmp = path + '.tmp.npy'
    out = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float16, shape=array.shape)
    for t in range(array.shape[0]):
        out[t] = array[t]
    out.flush()
    del out
    os.replace(tmp, path)


def fuse_files(paths, out_path, refs=None):
    """Frame-wise maximum of saved maps, each minus its CFAR level when refs is given."""
    arrs = [np.load(p, mmap_mode='r') for p in paths]
    tmp = out_path + '.tmp.npy'
    out = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float16, shape=arrs[0].shape)
    for t in range(arrs[0].shape[0]):
        m = None
        for i, a in enumerate(arrs):
            v = np.asarray(a[t], np.float32)
            if refs is not None:
                v = v - refs[i][t]
            m = v if m is None else np.maximum(m, v)
        out[t] = m
    out.flush()
    del out
    os.replace(tmp, out_path)


def summarize_windows(meta):
    c = Counter(w['coords'] for w in meta)
    res = {key: [w[key] for w in meta if key in w and np.isfinite(w[key])] for key in ('raw', 'warp', 'cover')}
    return dict(windows=len(meta), raw=c['raw'], warp=c['warp'], skip=c['skip'],
                **{f'median_{key}': (float(np.median(v)) if v else None) for key, v in res.items()})


def outputs(base, rates, fusion):
    d = {f'r{k}': f'{base}__r{k}' for k in rates}
    if 'cal' in fusion:
        d.update({f'r{k}cal': f'{base}__r{k}cal' for k in rates})
    if len(rates) > 1:
        if 'max' in fusion:
            d['mrmax'] = f'{base}__mrmax'
        if 'cal' in fusion:
            d['mrcal'] = f'{base}__mrcal'
    return d


def process_clip(clip, video, chain, score, args, dirs):
    n, H, W = video.shape
    low = mr.lowres(video, args.lowres) if args.coords != 'raw' else None
    decide = make_decider(args.coords, low, chain, args.lowres, args.margin, args.abs_tol, args.min_cover)
    meta = dict(clip=clip, frames=n, coords=args.coords, rates={})
    refs = {}
    for k, out, wmeta, ref in run_rates(n, (H, W), args.rates, args.seqlen, decide, score,
                                        'cal' in args.fusion, args.cal_m, args.cal_half):
        path = f"{dirs[f'r{k}']}/{clip}.npy"
        if out is None:
            save_f16(path, np.full((n, H, W), -np.inf, np.float32))
            refs[k] = np.full(n, mr.FLOOR, np.float32)
            meta['rates'][str(k)] = dict(used=False, reason=f'needs {k * args.seqlen} frames, clip has {n}')
            continue
        save_f16(path, out)
        meta['rates'][str(k)] = dict(used=True, **summarize_windows(wmeta), windows_detail=wmeta)
        if ref is not None:
            refs[k] = ref
            meta['rates'][str(k)]['cfar_level'] = dict(zip(('p10', 'median', 'p90'),
                                                          map(float, np.percentile(ref, [10, 50, 90]))))
        del out
    paths = [f"{dirs[f'r{k}']}/{clip}.npy" for k in args.rates]
    if 'cal' in args.fusion:
        for k, p in zip(args.rates, paths):
            fuse_files([p], f"{dirs[f'r{k}cal']}/{clip}.npy", [refs[k]])
    if 'mrmax' in dirs:
        fuse_files(paths, f"{dirs['mrmax']}/{clip}.npy")
    if 'mrcal' in dirs:
        fuse_files(paths, f"{dirs['mrcal']}/{clip}.npy", [refs[k] for k in args.rates])
    return meta


def make_scorer(model, needs, device, video, chain, hist_cfg, no_history):
    """score(frames, coords) -> (len(frames), H, W) logits in frame coordinates."""
    import torch
    n, H, W = video.shape
    ident = identity_chain(n)

    def tens(v, dt):
        return torch.from_numpy(np.ascontiguousarray(v)).to(device=device, dtype=dt)

    @torch.no_grad()
    def score(frames, coords):
        frames = [int(t) for t in frames]
        ref = frames[len(frames) // 2]
        if coords == 'warp':
            stack, pv = st.warp_frames(video, chain, frames, ref)
            x = np.where(pv, (stack - vtdata.RGBT_MEAN) / vtdata.RGBT_STD, 0.0).astype(np.float32)
        else:
            x = ((video[frames] - vtdata.RGBT_MEAN) / vtdata.RGBT_STD).astype(np.float32)
            pv = np.ones(x.shape, bool)
        kw = {}
        if 'pvalid' in needs:
            kw['pvalid'] = tens(pv[None, None], torch.bool)
        if 'hist' in needs:
            if no_history:
                B, A = np.zeros((H, W), np.float32), np.zeros((H, W), bool)
            else:
                k = frames[1] - frames[0] if len(frames) > 1 else 1
                B, A = st.history_field(video, chain if coords == 'warp' else ident, frames[0], frames[-1] + 1,
                                        ref, hist_cfg['hist_len'] * k, hist_cfg['hist_step'] * k,
                                        hist_cfg['hist_mode'])
                B = np.where(A, (B - vtdata.RGBT_MEAN) / vtdata.RGBT_STD, 0.0).astype(np.float32)
            kw['hist'] = tens(B[None, None], torch.float32)
            kw['hvalid'] = tens(A[None, None], torch.bool)
        _, logit = model(tens(x[None, None], torch.float32), **kw)
        out = logit[0].float().cpu().numpy()
        if coords == 'warp':
            out = st.warp_back(out, chain, frames, ref, FILL)
        return out
    return score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--arch', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--clips', default='val', help='split name, foldK/val, or comma-separated clips')
    ap.add_argument('--rates', default='1,2,4')
    ap.add_argument('--coords', choices=('raw', 'warp', 'auto'), default=None,
                    help='default: auto for mr_*, warp for lvh_*, raw otherwise')
    ap.add_argument('--fusion', default='max,cal', help="comma list of max, cal, or 'none'")
    ap.add_argument('--cal-m', dest='cal_m', type=int, default=10, help='CFAR level: m-th candidate per frame')
    ap.add_argument('--cal-half', dest='cal_half', type=int, default=18, help='CFAR pooling, +-frames')
    ap.add_argument('--margin', type=float, default=0.1)
    ap.add_argument('--abs-tol', dest='abs_tol', type=float, default=0.5)
    ap.add_argument('--min-cover', dest='min_cover', type=float, default=0.1)
    ap.add_argument('--lowres', type=int, default=4)
    ap.add_argument('--no-history', dest='no_history', action='store_true')
    ap.add_argument('--seqlen', type=int, default=40)
    ap.add_argument('--threads', type=int, default=int(os.environ.get('OMP_NUM_THREADS', 16)))
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    args.rates = mr.parse_rates(args.rates)
    args.fusion = set() if args.fusion == 'none' else {f for f in args.fusion.split(',') if f}
    if not args.fusion <= {'max', 'cal'}:
        raise SystemExit(f'unknown fusion {args.fusion}')

    import torch
    import models
    import splits
    torch.set_num_threads(args.threads)
    model, needs = models.build(args.arch, seqlen=args.seqlen)
    if args.coords is None:
        args.coords = 'auto' if args.arch.startswith('mr_') else ('warp' if 'hist' in needs else 'raw')
    if args.coords != 'raw' and 'pvalid' not in needs:
        raise SystemExit(f'{args.arch} was trained on recorded frames; use --coords raw')
    if args.no_history and 'hist' not in needs:
        raise SystemExit('--no-history applies to lvh_* architectures only')
    sd = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and ('model_state_dict' in sd or 'model' in sd):
        sd = sd.get('model_state_dict', sd.get('model'))
    model.load_state_dict(sd)
    device = torch.device(args.device)
    model.to(device)
    if device.type == 'cuda':
        torch.nn.utils.convert_conv3d_weight_memory_format(model, torch.channels_last_3d)
    model.eval()
    cfg_path = os.path.join(os.path.dirname(args.ckpt), 'config.json')
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    hist_cfg = {k: cfg.get(k, d) for k, d in (('hist_len', 80), ('hist_step', 4), ('hist_mode', 'both'))}

    base = f'cache/preds/{args.run}'
    dirs = outputs(base, args.rates, args.fusion)
    dirs['meta'] = f'{base}__meta'
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    clips = splits.clips(args.clips)
    import cachekey
    settings = cachekey.settings_of(args)
    code = cachekey.code_sha256()
    print(f'{args.run}: {args.arch} rates={args.rates} coords={args.coords} fusion={sorted(args.fusion)} '
          f'no_history={args.no_history} -> {", ".join(sorted(os.path.basename(v) for v in dirs.values()))}',
          flush=True)
    summary = {}
    for c in clips:
        meta_path = f"{dirs['meta']}/{c}.json"
        sc = vtdata.Scene(c)
        fp = cachekey.fingerprint(args.ckpt, settings, c, len(sc),
                                  [os.path.join(st.HOMOGRAPHY_DIR, f'{c}.npy')] if args.coords != 'raw' else [], code)
        done = cachekey.current(meta_path, fp, [f'{d}/{c}.npy' for key, d in dirs.items() if key != 'meta'])
        if done:
            summary[c] = {k: {kk: vv for kk, vv in v.items() if kk != 'windows_detail'}
                          for k, v in json.load(open(meta_path))['rates'].items()}
            continue
        cachekey.invalidate(meta_path)
        t0 = time.time()
        video = sc.video()
        needs_chain = args.coords != 'raw'
        chain = st.load_chain(c, len(sc)) if needs_chain else identity_chain(len(sc))
        score = make_scorer(model, needs, device, video, chain, hist_cfg, args.no_history)
        meta = process_clip(c, video, chain, score, args, dirs)
        meta['settings'] = {k: (sorted(v) if isinstance(v, set) else v) for k, v in vars(args).items()}
        meta['fingerprint'] = fp
        cachekey.write_json(meta_path, meta)
        summary[c] = {k: {kk: vv for kk, vv in v.items() if kk != 'windows_detail'} for k, v in meta['rates'].items()}
        line = ' '.join(f"r{k}:{v['raw']}raw/{v['warp']}warp/{v['skip']}skip" if v['used'] else f'r{k}:unused'
                        for k, v in summary[c].items())
        print(f'{args.run} {c} frames={len(sc)} {line} {time.time() - t0:.1f}s', flush=True)
    json.dump(summary, open(f"{dirs['meta']}/summary.json", 'w'), indent=1)


if __name__ == '__main__':
    main()
