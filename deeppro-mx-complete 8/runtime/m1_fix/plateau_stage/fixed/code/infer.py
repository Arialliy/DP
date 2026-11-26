"""Clip-level inference with the DeepPro windowing scheme.

Windows of T=40 frames with stride T - round(0.1 T) = 36 (authors' TestDataLoader); the
last window ends at the last frame; overlapping frames take the max (authors' test.py takes the max
probability; sigmoid is monotonic so the max logit is equivalent).  Logits are saved as float16 arrays cache/preds/<run>/<clip>.npy (T,H,W).

LVH architectures (needs 'pvalid') run on camera-compensated windows: every window is
warped to its middle frame, scored there, and the per-frame logits are warped back to each
frame before the max over overlapping windows. --save-arms also writes the variation,
level and gated-level logits to cache/preds/<run>__<arm>/ for the calibrated fusion.

Usage:
  python code/infer.py --run official_rgbt --clips test
  python code/infer.py --run <run_name> --ckpt runs/<run>/best.pth --clips val
  python code/infer.py --run lvh_s46 --arch lvh_full --ckpt runs/lvh_s46/final.pth --clips val --save-arms
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import models  # noqa: E402
import splits  # noqa: E402
import stabilize  # noqa: E402
import vtdata  # noqa: E402

FILL = -30.0  # logit for frame pixels outside a window's reference field (below evaluate.FLOOR)

OFFICIAL_RGBT = '/official/log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'


def windows(n, T=40):
    cat = int(T * 0.1)
    num = math.ceil((n - cat) / (T - cat))
    out = []
    for i in range(num):
        last = min(n, (i + 1) * (T - cat) + cat)
        out.append((max(0, last - T), last))
    return out


@torch.no_grad()
def predict_clip(model, video, T=40, extra=None):
    """video: (N,H,W) float32 raw grey -> (N,H,W) float32 logits (max over overlapping windows)."""
    n = video.shape[0]
    device = next(model.parameters()).device
    x = (video - vtdata.RGBT_MEAN) / vtdata.RGBT_STD
    prob = np.full(video.shape, -np.inf, dtype=np.float32)
    for a, b in windows(n, T):
        inp = torch.from_numpy(x[a:b][None, None]).to(device=device, dtype=torch.float32)
        kw = {}
        if extra is not None:
            kw = {k: torch.from_numpy(v[a:b][None]).to(device=device, dtype=torch.float32)
                  for k, v in extra.items()}
        _, logit = model(inp, **kw) if kw else model(inp)
        p = logit[0].float().cpu().numpy()
        prob[a:b] = np.maximum(prob[a:b], p)
    return prob


@torch.no_grad()
def predict_clip_stabilized(model, video, chain, T=40, hist_len=80, hist_step=4, hist_mode='both',
                            arms=False, motion_gate=0.0):
    """video: (N,H,W) raw grey, chain: stabilize.load_chain -> fused logits (N,H,W) and,
    with arms=True, a dict of per-arm logits on frame pixels.
    motion_gate > 0 uses the recorded frames for windows whose camera motion stays below
    that many pixels (stabilize.gate_chain)."""
    n = video.shape[0]
    device = next(model.parameters()).device
    fused = np.full(video.shape, -np.inf, dtype=np.float32)
    names = ('variation', 'level', 'gated_level')
    if getattr(model, 'decoupled', False):
        names += ('appearance', 'arrival')
    arm = {k: np.full(video.shape, -np.inf, dtype=np.float32) for k in names} if arms else None
    full_chain = chain
    for a, b in windows(n, T):
        ref = stabilize.reference_frame(a, b)
        frames_all = list(range(a, b)) + stabilize.history_frames(n, a, b, hist_len, hist_step, hist_mode)
        chain, _ = stabilize.gate_chain(full_chain, frames_all, ref, motion_gate, video.shape[1:3])
        frames, fv = stabilize.warp_frames(video, chain, range(a, b), ref)
        x = np.where(fv, (frames - vtdata.RGBT_MEAN) / vtdata.RGBT_STD, 0.0).astype(np.float32)
        hist, avail = stabilize.history_field(video, chain, a, b, ref, hist_len, hist_step, hist_mode)
        hist = np.where(avail, (hist - vtdata.RGBT_MEAN) / vtdata.RGBT_STD, 0.0).astype(np.float32)
        tens = lambda v, dt: torch.from_numpy(np.ascontiguousarray(v)).to(device=device, dtype=dt)
        _, logit = model(tens(x[None, None], torch.float32), pvalid=tens(fv[None, None], torch.bool),
                         hist=tens(hist[None, None], torch.float32),
                         hvalid=tens(avail[None, None], torch.bool))
        frames_ab = range(a, b)
        back = stabilize.warp_back(logit[0].float().cpu().numpy(), chain, frames_ab, ref, FILL)
        fused[a:b] = np.maximum(fused[a:b], back)
        if arms:
            last = model.last
            per = {'variation': last['lv'][0],
                   'level': last['ll'][0].expand(b - a, -1, -1),
                   'gated_level': last['lq'][0]}
            if 'la' in last:
                per['appearance'] = last['la'][0].expand(b - a, -1, -1)
                per['arrival'] = last['lr'][0].expand(b - a, -1, -1)
            for k in names:
                v = stabilize.warp_back(per[k].float().cpu().numpy(), chain, frames_ab, ref, FILL)
                arm[k][a:b] = np.maximum(arm[k][a:b], v)
    return fused, arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--arch', default='deeppro_plus')
    ap.add_argument('--ckpt', default=OFFICIAL_RGBT)
    ap.add_argument('--clips', default='test')
    ap.add_argument('--threads', type=int, default=int(os.environ.get('OMP_NUM_THREADS', 16)))
    ap.add_argument('--seqlen', type=int, default=40)
    ap.add_argument('--device', default='cpu', help='PyTorch device, e.g. cpu or cuda:0')
    ap.add_argument('--hist-len', dest='hist_len', type=int, default=None,
                    help='History settings default to the training run config.json')
    ap.add_argument('--hist-step', dest='hist_step', type=int, default=None)
    ap.add_argument('--hist-mode', dest='hist_mode', choices=('both', 'past'), default=None)
    ap.add_argument('--motion-gate', dest='motion_gate', type=float, default=None,
                    help='px; windows moving less use recorded frames (default: training config)')
    ap.add_argument('--save-arms', dest='save_arms', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    model, needs = models.build(args.arch, seqlen=args.seqlen)
    sd = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = sd.get('model_state_dict', sd.get('model', sd)) if isinstance(sd, dict) and ('model_state_dict' in sd or 'model' in sd) else sd
    model.load_state_dict(sd)
    device = torch.device(args.device)
    model.to(device)
    if device.type == 'cuda':
        torch.nn.utils.convert_conv3d_weight_memory_format(model, torch.channels_last_3d)
    model.eval()
    out = f'cache/preds/{args.run}'
    os.makedirs(out, exist_ok=True)
    clips = splits.clips(args.clips)
    stabilized = 'pvalid' in needs
    if stabilized:
        cfg_path = os.path.join(os.path.dirname(args.ckpt), 'config.json')
        cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
        for k, default in (('hist_len', 80), ('hist_step', 4), ('hist_mode', 'both'), ('motion_gate', 0.0)):
            if getattr(args, k) is None:
                setattr(args, k, cfg.get(k, default))
        print(f'History: len={args.hist_len} step={args.hist_step} mode={args.hist_mode} '
              f'motion_gate={args.motion_gate}px', flush=True)
    arm_names = ('variation', 'level', 'gated_level')
    if getattr(model, 'decoupled', False):
        arm_names += ('appearance', 'arrival')
    arm_dirs = {k: f'{out}__{k}' for k in arm_names}
    if stabilized and args.save_arms:
        for d in arm_dirs.values():
            os.makedirs(d, exist_ok=True)

    def save(path, array):
        np.save(path + '.tmp.npy', array.astype(np.float16))
        os.replace(path + '.tmp.npy', path)

    import cachekey
    meta_dir = f'{out}__meta'
    os.makedirs(meta_dir, exist_ok=True)
    settings = cachekey.settings_of(args)
    code = cachekey.code_sha256()
    for c in clips:
        f = f'{out}/{c}.npy'
        sc = vtdata.Scene(c)
        meta_path = f'{meta_dir}/{c}.json'
        fp = cachekey.fingerprint(args.ckpt, settings, c, len(sc),
                                  [os.path.join(stabilize.HOMOGRAPHY_DIR, f'{c}.npy')] if stabilized else [], code)
        outputs = [f] + ([f'{d}/{c}.npy' for d in arm_dirs.values()] if stabilized and args.save_arms else [])
        if cachekey.current(meta_path, fp, outputs):
            print(f'{args.run} {c}: cached predictions of this checkpoint and settings', flush=True)
            continue
        cachekey.invalidate(meta_path)
        t0 = time.time()
        vid = sc.video()
        if stabilized:
            chain = stabilize.load_chain(c, len(sc))
            p, arms = predict_clip_stabilized(model, vid, chain, args.seqlen, args.hist_len,
                                              args.hist_step, args.hist_mode, args.save_arms,
                                              args.motion_gate)
            if arms is not None:
                for k, d in arm_dirs.items():
                    save(f'{d}/{c}.npy', arms[k])
        else:
            extra = models.extra_inputs(needs, c, vid)
            p = predict_clip(model, vid, args.seqlen, extra)
        save(f, p)
        cachekey.write_json(meta_path, dict(clip=c, settings=vars(args), fingerprint=fp))
        print(f'{args.run} {c} frames={len(sc)} {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
