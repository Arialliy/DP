"""Training with the DeepPro-Plus recipe on the recording-independent split.

Recipe follows the authors' train.py for 'RGB-T': 40-frame windows ending at every frame
i >= 4 (shorter windows zero-padded at the front or back), clip sampling proportional to
window length, 128x128 crops centred (+-21 px jitter) on a random target of the middle
frame with probability 0.75, Soft-IoU loss, Adam(lr 1e-3, wd 1e-4), lr x0.7 every 10
epochs, 32 epochs, batch 4, sample_rate 0.05.
Differences (documented): labels are reconstructed Gaussian masks (sigma = box/6); the
final checkpoint is used (the authors selected 'best_model' by IoU on the test set).

LVH architectures (lvh_*, see code/lvh.py) train on camera-compensated windows with a
History field (StabilizedTrainSet); they need results/profile/homographies from
code/profile_motion.py. Their level pathway adds a focal BCE term to Soft-IoU (Soft-IoU has
no gradient on crops without persistent targets) and can centre a fraction of crops on
mined hot spots (--hard-neg, code/hotspots.py).

Multi-rate training (MRTP, see code/multirate.py): --rates 1,2,4 draws windows of T
samples taken every k frames, and --coords raw|warp|auto puts each window in recorded or
camera-compensated coordinates by the same stationarity rule as code/infer_mr.py. mr_lv
defaults to --rates 1,2,4 --coords auto with the level-focal loss of gpu_lvfull_focal_s46.

Motion exemplars (DeepPro-MX, see code/mx.py): mx_* architectures train like mr_lv at rate 1
(--coords auto, level focal, hard persistence target) and every sample also carries up to
--ex-k patches of moving GT targets (rel_speed >= --ex-moving in --ex-targets) of the same
clip, from frames outside the sample window; --ex-p-none and --ex-p-fake set how often a
sample has none and how often an exemplar is replaced by a clutter point. --mx-aux weights
the focal loss of the match logit. --ex-p-none 1 trains the no-exemplar control.

Labels: --label-rule paper_k6_floor (default; the masks export_masks.py writes for the
official code, docs/MASKS.md) or sigma_sixth (the earlier reconstruction: pass it to
reproduce runs made before DeepPro-MX, whose config.json has no label_rule).

Usage: python code/train.py --run NAME --arch deeppro_plus [--seed 0] [--epochs 32]
       python code/train.py --run lvh_s46 --arch lvh_full --seed 46 --device cuda
       python code/train.py --run lvhdec_s46 --arch lvh_dec --seed 46 --device cuda
       python code/train.py --run mrlv_s46 --arch mr_lv --seed 46 --device cuda
       python code/train.py --run mx_s46 --arch mx_lv --seed 46 --device cuda --label-rule paper_k6_floor
"""
import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import models  # noqa: E402
import splits  # noqa: E402
import vtdata  # noqa: E402
import lvh  # noqa: E402
import hotspots  # noqa: E402
import lvh_data  # noqa: E402
import stabilize  # noqa: E402
import multirate  # noqa: E402
import mr_data  # noqa: E402
import mx_data  # noqa: E402
import mx  # noqa: E402

CACHE = 'cache/video'


def clip_array(clip):
    """uint8 (N,H,W) memmap cache of a clip's thermal frames."""
    os.makedirs(CACHE, exist_ok=True)
    f = f'{CACHE}/{clip}.npy'
    if not os.path.exists(f):
        v = vtdata.Scene(clip).video()
        np.save(f + '.tmp.npy', np.clip(v, 0, 255).astype(np.uint8))
        os.replace(f + '.tmp.npy', f)
    return np.load(f, mmap_mode='r')


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, clips, seqlen=40, patch=128, sample_rate=0.05, needs=(), seed=0, rule='sigma_sixth'):
        self.seqlen, self.patch, self.needs, self.rule = seqlen, patch, needs, rule
        self.scenes = {c: vtdata.Scene(c) for c in clips}
        self.samples, w = [], []
        for c, sc in self.scenes.items():
            for i in range(int(seqlen * 0.1), len(sc)):
                a = max(0, i - seqlen)
                self.samples.append((c, a, i))
                w.append(i - a)
        self.p = np.array(w, float) / np.sum(w)
        self.n = int(len(self.samples) * sample_rate)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        c, a, b = self.samples[self.rng.choice(len(self.samples), p=self.p)]
        return self._raw_sample(c, a, b, idx)

    def _raw_sample(self, c, a, b, idx):
        sc = self.scenes[c]
        vid = clip_array(c)[a:b]
        t = b - a
        boxes = sc.boxes[a:b]
        T, P = self.seqlen, self.patch
        # pad short windows (authors: zeros after normalisation, end if idx odd else start)
        front = (idx % 2 == 0)
        mid = (T - math.ceil(t / 2)) if front else int(t / 2)
        lo = T - t if front else 0
        H, W = sc.H, sc.W
        mid_boxes = boxes[mid - lo] if 0 <= mid - lo < t else np.zeros((0, 6))
        if len(mid_boxes) and self.rng.random() < 0.75:
            x, y, w, h = mid_boxes[self.rng.integers(len(mid_boxes)), :4]
            sh = int(P / 2 / 3)
            r0 = int(y + h / 2 + (self.rng.random() - 0.5) * 2 * sh - P / 2)
            c0 = int(x + w / 2 + (self.rng.random() - 0.5) * 2 * sh - P / 2)
            r0, c0 = min(max(r0, 0), H - P - 1), min(max(c0, 0), W - P - 1)
        else:
            r0, c0 = int(self.rng.integers(0, H - P)), int(self.rng.integers(0, W - P))
        img = np.zeros((T, P, P), np.float32)
        lab = np.zeros((T, P, P), np.float32)
        crop = vid[:, r0:r0 + P, c0:c0 + P].astype(np.float32)
        img[lo:lo + t] = (crop - vtdata.RGBT_MEAN) / vtdata.RGBT_STD
        for k in range(t):
            bx = boxes[k].copy()
            if len(bx):
                bx[:, 0] -= c0
                bx[:, 1] -= r0
            lab[lo + k] = vtdata.gaussian_mask(bx, P, P, self.rule)
        out = {'img': torch.from_numpy(img[None]), 'lab': torch.from_numpy(lab)}
        if 'valid' in self.needs:
            valid = np.zeros(T, bool)
            valid[lo:lo + t] = True
            out['valid'] = torch.from_numpy(valid)
        return out


class StabilizedTrainSet(TrainSet):
    """Windows warped to their middle frame, with the History field (LVH architectures)."""

    def __init__(self, clips, hist_len=80, hist_step=4, hist_mode='both', hard_neg=0.0, motion_gate=0.0, **kw):
        super().__init__(clips, **kw)
        self.hist_len, self.hist_step, self.hist_mode, self.hard_neg = hist_len, hist_step, hist_mode, hard_neg
        self.motion_gate = motion_gate
        self.chains = {c: stabilize.load_chain(c, len(sc)) for c, sc in self.scenes.items()}
        self.hot = {}
        if hard_neg > 0:
            for c, sc in self.scenes.items():
                self.hot[c] = hotspots.load_or_mine(c, clip_array(c), sc.boxes, self.chains[c], T=self.seqlen,
                                                    motion_gate=motion_gate)

    def __getitem__(self, idx):
        c, a, b = self.samples[self.rng.choice(len(self.samples), p=self.p)]
        sc, chain = self.scenes[c], self.chains[c]
        vid = clip_array(c)
        ref = stabilize.reference_frame(a, b)
        center, _ = lvh_data.choose_center(sc.boxes[ref], self.hot.get(c), chain, a, b, ref,
                                           self.rng, self.hard_neg)
        r0, c0 = lvh_data.crop_origin(center, self.patch, sc.H, sc.W, self.rng)
        s = lvh_data.build_sample(vid, sc.boxes, chain, a, b, self.seqlen, self.patch,
                                  front=(idx % 2 == 0), r0=r0, c0=c0, hist_len=self.hist_len,
                                  hist_step=self.hist_step, hist_mode=self.hist_mode,
                                  rule=self.rule, motion_gate=self.motion_gate)
        return {k: torch.from_numpy(v) for k, v in s.items()}


class MultiRateTrainSet(TrainSet):
    """Windows of T samples every k frames, in recorded or compensated coordinates (MRTP).

    Rate k is drawn with probability rate_p. A rate-k window (k > 1) is uniform over the
    windows that fit in the clips; rate 1 keeps TrainSet's windows, including the padded
    ones at clip starts, and with recorded coordinates TrainSet's exact sample. A window
    the coordinate rule skips (camera swept past the view) is redrawn at rate 1.
    """

    def __init__(self, clips, rates=(1,), rate_p=None, coords='raw', margin=0.1, abs_tol=0.5,
                 min_cover=0.1, lowres=4, **kw):
        super().__init__(clips, **kw)
        self.rates = list(rates)
        self.rate_p = np.asarray(multirate.rate_probs(self.rates) if rate_p is None else rate_p, float)
        self.coords, self.margin, self.abs_tol = coords, margin, abs_tol
        self.min_cover, self.lowres = min_cover, lowres
        self.chains = ({c: stabilize.load_chain(c, len(sc)) for c, sc in self.scenes.items()}
                       if coords != 'raw' else {})
        self.lengths = {c: len(sc) for c, sc in self.scenes.items()}
        self.tables = mr_data.rate_tables(self.lengths, self.rates, self.seqlen)

    def _decide(self, c, frames, k):
        return mr_data.decide(clip_array(c), self.chains.get(c), frames, self.coords, k,
                              self.lowres, self.margin, self.abs_tol, self.min_cover)[0]

    def __getitem__(self, idx):
        return self._sample(*self._draw(), idx)

    def _draw(self):
        return mr_data.draw_spec(self.rng, self.rates, self.rate_p, self.tables, self.samples,
                                 self.p, self.lengths, self._decide, self.seqlen)

    def _sample(self, k, c, frames, coords, idx):
        T, P = self.seqlen, self.patch
        if coords == 'raw' and k == 1:
            out = self._raw_sample(c, int(frames[0]), int(frames[-1]) + 1, idx)
            if 'pvalid' in self.needs:
                out['pvalid'] = out['valid'][None, :, None, None].expand(1, T, P, P).clone()
        else:
            sc = self.scenes[c]
            ref = int(frames[len(frames) // 2])
            center = mr_data.choose_center(sc.boxes[ref], self.rng)
            r0, c0 = lvh_data.crop_origin(center, P, sc.H, sc.W, self.rng)
            s = mr_data.build_window_sample(clip_array(c), sc.boxes, self.chains.get(c), frames, coords, T, P,
                                            front=(idx % 2 == 0), r0=r0, c0=c0, rule=self.rule)
            out = {key: torch.from_numpy(s[key]) for key in ('img', 'lab', 'valid', 'pvalid')
                   if key in ('img', 'lab') or key in self.needs}
        out['rate'] = torch.tensor(k)
        out['warped'] = torch.tensor(float(coords == 'warp'))
        return out


class ExemplarTrainSet(MultiRateTrainSet):
    """MultiRateTrainSet samples plus motion exemplars of the same clip (DeepPro-MX).

    Exemplars are GT boxes moving at >= moving px/frame (profile_motion.py rel_speed) from
    frames outside the sample window, as patches of the recorded frames (mx_data.py).
    Extra keys: ex (K,1,S,S), ex_valid (K,); ex_fake and ex_t only for logging.
    """

    def __init__(self, clips, targets, ex_k=16, ex_patch=39, moving=mx_data.MOVING, p_none=0.3, p_fake=0.2,
                 **kw):
        super().__init__(clips, **kw)
        speed = mx_data.read_speeds(targets, clips)
        mx_data.require_speeds({c: sc.boxes for c, sc in self.scenes.items()}, speed, targets)
        self.moving = {c: mx_data.moving_table(c, sc.boxes, speed, moving) for c, sc in self.scenes.items()}
        self.ex_k, self.ex_patch, self.p_none, self.p_fake = ex_k, ex_patch, p_none, p_fake

    def __getitem__(self, idx):
        k, c, frames, coords = self._draw()
        out = self._sample(k, c, frames, coords, idx)
        e = mx_data.exemplar_sample(self.rng, clip_array(c), self.scenes[c].boxes, self.moving[c], frames,
                                    self.ex_k, self.ex_patch, vtdata.RGBT_MEAN, vtdata.RGBT_STD,
                                    self.p_none, self.p_fake)
        out.update({key: torch.from_numpy(v) for key, v in e.items()})
        return out


def soft_iou_loss(logit, target):
    p = torch.sigmoid(logit)
    inter = (p * target).sum((1, 2, 3))
    return 1 - (inter / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) - inter)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--arch', default='deeppro_plus')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=32)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--sample_rate', type=float, default=0.05)
    ap.add_argument('--split', default='train')
    ap.add_argument('--threads', type=int, default=int(os.environ.get('OMP_NUM_THREADS', 16)))
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--device', default='cpu', help='PyTorch device, e.g. cpu or cuda:0')
    ap.add_argument('--aux', type=float, default=0.5, help='weight of the level-pathway loss (LV models)')
    ap.add_argument('--level-focal', dest='level_focal', type=float, default=None,
                    help='weight of focal BCE added to the level loss (default 1 for lvh_*, 0 otherwise)')
    ap.add_argument('--main-focal', dest='main_focal', type=float, default=0.0,
                    help='weight of focal BCE added to the fused per-frame loss')
    ap.add_argument('--persist', choices=('hard', 'ramp'), default=None,
                    help='level-pathway target (default ramp for lvh_*, hard for lv_*)')
    ap.add_argument('--hist-len', dest='hist_len', type=int, default=80)
    ap.add_argument('--hist-step', dest='hist_step', type=int, default=4)
    ap.add_argument('--hist-mode', dest='hist_mode', choices=('both', 'past'), default='both')
    ap.add_argument('--hard-neg', dest='hard_neg', type=float, default=None,
                    help='fraction of crops centred on mined hot spots '
                         '(default 0.15 for lvh_* except 0 for lvh_dec, 0 otherwise)')
    ap.add_argument('--motion-gate', dest='motion_gate', type=float, default=None,
                    help='px; windows whose camera motion stays below it use recorded frames '
                         '(default 1 for lvh_dec, 0 = always warp otherwise)')
    ap.add_argument('--rates', default=None,
                    help='temporal rates of the training windows, e.g. 1,2,4 (default 1,2,4 for mr_*, 1 otherwise)')
    ap.add_argument('--rate-probs', dest='rate_probs', default=None,
                    help='sampling probabilities of --rates (default 1/2 for rate 1, the rest equal)')
    ap.add_argument('--coords', choices=('raw', 'warp', 'auto'), default=None,
                    help='coordinates of multi-rate windows (default auto for mr_*, raw otherwise)')
    ap.add_argument('--stat-margin', dest='stat_margin', type=float, default=0.1)
    ap.add_argument('--abs-tol', dest='abs_tol', type=float, default=0.5)
    ap.add_argument('--min-cover', dest='min_cover', type=float, default=0.1)
    ap.add_argument('--lowres', type=int, default=4)
    ap.add_argument('--arrival-target', dest='arrival_target', choices=('history', 'shared'),
                    default='history', help="lvh_dec: arrival head ignores resident targets ('history') "
                                            "or uses the appearance head's target ('shared', ablation)")
    ap.add_argument('--label-rule', dest='label_rule', choices=('sigma_sixth', 'paper_k6_floor'),
                    default='paper_k6_floor',
                    help='training masks; sigma_sixth reproduces runs made before DeepPro-MX')
    ap.add_argument('--ex-targets', dest='ex_targets', default='results/profile/targets.csv',
                    help='mx_*: per-box rel_speed (profile_motion.py) that defines moving exemplars')
    ap.add_argument('--ex-k', dest='ex_k', type=int, default=16, help='mx_*: exemplar slots per sample')
    ap.add_argument('--ex-moving', dest='ex_moving', type=float, default=mx_data.MOVING,
                    help='mx_*: px/frame; GT boxes at least this fast are exemplars')
    ap.add_argument('--ex-p-none', dest='ex_p_none', type=float, default=0.3,
                    help='mx_*: probability that a sample has no exemplars (1 = no-exemplar control)')
    ap.add_argument('--ex-p-fake', dest='ex_p_fake', type=float, default=0.2,
                    help='mx_*: probability that an exemplar is replaced by a clutter point')
    ap.add_argument('--mx-aux', dest='mx_aux', type=float, default=0.5,
                    help='mx_*: weight of the focal loss of the match logit')
    args = ap.parse_args()
    is_lvh = args.arch.startswith('lvh')
    is_dec = args.arch == 'lvh_dec'
    is_mr = args.arch.startswith('mr_')
    is_mx = args.arch.startswith('mx_')
    args.rates = multirate.parse_rates(args.rates or ('1,2,4' if is_mr else '1'))
    rate_p = multirate.rate_probs(args.rates, args.rate_probs)
    args.rate_probs = [round(float(v), 6) for v in rate_p]
    if args.coords is None:
        args.coords = 'auto' if (is_mr or is_mx) else 'raw'
    if args.motion_gate is None:
        args.motion_gate = 1.0 if is_dec else 0.0
    if args.level_focal is None:
        args.level_focal = 1.0 if (is_lvh or is_mr or is_mx) else 0.0
    if args.persist is None:
        args.persist = 'ramp' if is_lvh else 'hard'
    if args.hard_neg is None:
        args.hard_neg = 0.0 if is_dec else (0.15 if is_lvh else 0.0)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    use_cuda = device.type == 'cuda'
    out = f'runs/{args.run}'
    os.makedirs(out, exist_ok=True)
    model, needs = models.build(args.arch)
    model.to(device)
    if use_cuda:
        torch.nn.utils.convert_conv3d_weight_memory_format(model, torch.channels_last_3d)
    clips = splits.clips(args.split)
    for c in clips:
        clip_array(c)
    multi_rate = args.rates != [1] or args.coords != 'raw' or is_mr
    if multi_rate and 'hist' in needs:
        raise SystemExit('--rates/--coords are for lv_* and mr_* architectures, not lvh_*')
    if args.coords != 'raw' and 'pvalid' not in needs:
        raise SystemExit(f'{args.arch} has no pixel-validity input; use --coords raw or an mr_* architecture')
    if 'ex' in needs and args.rates != [1]:
        raise SystemExit('mx_* architectures train at rate 1 (infer_mx.py scores rate-1 windows)')
    if 'hist' in needs:
        ds = StabilizedTrainSet(clips, hist_len=args.hist_len, hist_step=args.hist_step,
                                hist_mode=args.hist_mode, hard_neg=args.hard_neg,
                                motion_gate=args.motion_gate,
                                sample_rate=args.sample_rate, needs=needs, seed=args.seed, rule=args.label_rule)
    else:
        if args.hard_neg > 0:
            raise SystemExit('--hard-neg needs a stabilized (lvh_*) architecture')
        mr_kw = dict(rates=args.rates, rate_p=rate_p, coords=args.coords, margin=args.stat_margin,
                     abs_tol=args.abs_tol, min_cover=args.min_cover, lowres=args.lowres,
                     sample_rate=args.sample_rate, needs=needs, seed=args.seed, rule=args.label_rule)
        if 'ex' in needs:
            ds = ExemplarTrainSet(clips, args.ex_targets, ex_k=args.ex_k, ex_patch=model.patch,
                                  moving=args.ex_moving, p_none=args.ex_p_none, p_fake=args.ex_p_fake, **mr_kw)
        elif multi_rate:
            ds = MultiRateTrainSet(clips, **mr_kw)
        else:
            ds = TrainSet(clips, sample_rate=args.sample_rate, needs=needs, seed=args.seed, rule=args.label_rule)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, drop_last=True,
                                     num_workers=args.workers, persistent_workers=args.workers > 0,
                                     pin_memory=use_cuda,
                                     worker_init_fn=lambda w: setattr(ds, 'rng', np.random.default_rng(
                                         args.seed * 1000 + w + 17 * torch.initial_seed() % 9973)))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    loss_fn = soft_iou_loss
    start = 0
    if os.path.exists(f'{out}/last.pth'):
        ck = torch.load(f'{out}/last.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        start = ck['epoch'] + 1
    print(f'{args.run}: {args.arch}, labels {args.label_rule}, coords {args.coords}, rates {args.rates}', flush=True)
    json.dump(dict(vars(args), n_params=models.n_params(model), train_clips=clips, needs=list(needs)),
              open(f'{out}/config.json', 'w'), indent=1)
    log = open(f'{out}/train_log.txt', 'a')
    for ep in range(start, args.epochs):
        lr = max(args.lr * (0.7 ** (ep // 10)), 1e-5)
        for g in opt.param_groups:
            g['lr'] = lr
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        mix, n_warp, n_samp = Counter(), 0.0, 0
        ex_samples = ex_slots = ex_fake = 0
        for batch in dl:
            if 'rate' in batch:
                mix.update(int(r) for r in batch['rate'])
                n_warp += float(batch['warped'].sum())
                n_samp += len(batch['rate'])
            if 'ex_valid' in batch:
                ex_samples += int(batch['ex_valid'].any(1).sum())
                ex_slots += int(batch['ex_valid'].sum())
                ex_fake += int(batch['ex_fake'].sum())
            batch = {k: v.to(device, non_blocking=use_cuda) for k, v in batch.items()}
            opt.zero_grad()
            kw = {k: v for k, v in batch.items()
                  if k not in ('img', 'lab', 'hlab', 'gated', 'rate', 'warped', 'ex_fake', 'ex_t')}
            _, logit = model(batch['img'].float(), **kw)
            lab = batch['lab'].float()
            loss = loss_fn(logit, lab)
            if args.main_focal > 0:
                loss = loss + args.main_focal * lvh.focal_bce(logit, lab)
            if args.aux > 0 and getattr(model, 'decoupled', False):
                # Evidence-specific targets: the appearance head learns every persistent
                # target, the arrival head only those History does not already contain.
                pm = lvh.persistent_target(lab, kw['pvalid'], args.persist)[:, None]  # (b,1,H,W)
                la, arrival_logit = model.last_la, model.last_lr
                aux = loss_fn(la, pm)
                if args.level_focal > 0:
                    aux = aux + args.level_focal * lvh.focal_bce(la, pm)
                if args.arrival_target == 'history':
                    tgt, w = lvh.arrival_target(pm[:, 0], batch['hlab'].float(), kw['hvalid'])
                    tgt, w = tgt[:, None], w[:, None]
                else:
                    tgt, w = pm, torch.ones_like(pm)
                aux = aux + lvh.masked_soft_iou_loss(arrival_logit, tgt, w)
                if args.level_focal > 0:
                    aux = aux + args.level_focal * lvh.masked_focal_bce(arrival_logit, tgt, w)
                loss = loss + args.aux * aux
            elif args.aux > 0 and getattr(model, 'level', None) is not None:
                validity = kw['pvalid'] if 'pvalid' in kw else kw['valid']
                pm = lvh.persistent_target(lab, validity, args.persist)  # (b,H,W)
                ll = model.last_ll  # (b,1,H,W) level map, or (b,T,H,W) for the per-frame control
                target = pm[:, None].expand_as(ll)
                aux = loss_fn(ll, target)
                if args.level_focal > 0:
                    aux = aux + args.level_focal * lvh.focal_bce(ll, target)
                loss = loss + args.aux * aux
            if args.mx_aux > 0 and getattr(model, 'exemplar', False):
                loss = loss + args.mx_aux * mx.match_loss(model, lab, kw['pvalid'], args.persist)
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1
        msg = f'epoch {ep + 1}/{args.epochs} lr {lr:.2e} loss {tot / max(n, 1):.4f} iters {n} {time.time() - t0:.0f}s'
        if n_samp:
            msg += f' rates {dict(sorted(mix.items()))} warped {n_warp / n_samp:.2f}'
        if ex_slots or 'ex' in needs:
            msg += (f' exemplars: samples {ex_samples / max(n * args.batch, 1):.2f} '
                    f'per-sample {ex_slots / max(ex_samples, 1):.1f} fake {ex_fake / max(ex_slots, 1):.2f}')
        print(msg, flush=True)
        log.write(msg + '\n')
        log.flush()
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': ep}, f'{out}/last.pth.tmp')
        os.replace(f'{out}/last.pth.tmp', f'{out}/last.pth')
        if (ep + 1) % 8 == 0 and ep + 1 < args.epochs:
            torch.save(model.state_dict(), f'{out}/epoch_{ep + 1}.pth')
    torch.save(model.state_dict(), f'{out}/final.pth')


if __name__ == '__main__':
    main()
