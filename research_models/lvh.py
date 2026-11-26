"""DeepPro-LVH: Level-Variation-History decomposition in camera-compensated coordinates.

DeepPro-LV separates a window by *dwell time*: transient content goes to the variation V,
content that covers a pixel for more than half the window goes to the level L. A parked
target and a ground hot spot have the same dwell time inside one window, so the level
pathway cannot tell them apart (both are bright in L with E = mean|V| ~ 0). They differ in
*history*: the hot spot is also present outside the window, the target arrived or will
leave. LVH adds that third time scale.

  inputs (all in the coordinates of the window's middle frame, see stabilize.py)
    x      (b,1,T,H,W) stabilized, normalised window
    pvalid (b,1,T,H,W) pixel validity after warping
    hist   (b,1,H,W)   History B: median of frames outside the window (normalised)
    hvalid (b,1,H,W)   History available
  decomposition (masked over pvalid)
    L = median_t x,  V = x - L,  E = mean_t |V|,  cover = valid fraction
    D = (L - B) * hvalid     new persistent content (arrived or will leave)
  pathways
    variation: DeepPro-Plus backbone on V                         -> l_v(t)
    level:     2-D encoder-decoder on [L, E, cover, D, hvalid]    -> l_L
    gate:      3x3 spatial conv on [V, |V|, D, hvalid] per frame   -> g(t)
    fusion:    noisy-OR  p(t) = 1 - (1 - s(l_v(t))) (1 - g(t) s(l_L))

Variants (ablations)
  lvh_full       stabilized + History + spatial gate (proposed)
  lvh_stab       stabilized, no History, spatial gate   (isolates camera compensation)
  lvh_pointgate  stabilized + History, DeepPro-LV's pointwise gate on V
  lvh_nogate     stabilized + History, no gate
  lvh_dec        evidence-decoupled: separate appearance and arrival heads (see DeepProLVHDec)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lv import Backbone, Gate, LevelNet, noisy_or_logit
from .models import register

NEEDS = ('valid', 'pvalid', 'hist', 'hvalid')


def decompose_masked(x, pvalid):
    """x, pvalid: (b,1,T,H,W). -> L (b,1,1,H,W), V (b,1,T,H,W), count (b,1,1,H,W)."""
    xn = torch.where(pvalid, x, torch.full_like(x, float('nan')))
    L = torch.nanmedian(xn, dim=2, keepdim=True).values
    count = pvalid.sum(2, keepdim=True)
    L = torch.where(count > 0, L, torch.zeros_like(L))
    V = torch.where(pvalid, x - L, torch.zeros_like(x))
    return L, V, count


class SpatialGate(nn.Module):
    """Per-frame gate with 5x5 spatial context; starts mostly open like lv.Gate."""

    def __init__(self, cin):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, 8, (1, 3, 3), padding=(0, 1, 1)), nn.ReLU(inplace=True),
            nn.Conv3d(8, 8, (1, 3, 3), padding=(0, 1, 1)), nn.ReLU(inplace=True),
            nn.Conv3d(8, 1, 1))
        nn.init.constant_(self.net[-1].bias, 2.0)

    def forward(self, z):
        return torch.sigmoid(self.net(z)).squeeze(1)


class DeepProLVH(nn.Module):
    def __init__(self, seqlen=40, history=True, gate='spatial'):
        super().__init__()
        self.history = history
        self.var = Backbone(seqlen)
        self.level = LevelNet(5 if history else 3)
        if gate == 'spatial':
            self.gate = SpatialGate(4 if history else 2)
        elif gate == 'point':
            self.gate = Gate()
        else:
            self.gate = None
        self.gate_kind = gate
        self.last = {}

    def forward(self, x, valid=None, pvalid=None, hist=None, hvalid=None):
        b, _, T, H, W = x.shape
        if pvalid is None:
            pvalid = torch.ones_like(x, dtype=torch.bool)
        pvalid = pvalid.bool()
        if valid is not None:
            pvalid = pvalid & valid.bool()[:, None, :, None, None]
        L, V, count = decompose_masked(x, pvalid)
        f, lv = self.var(V)
        n = count[:, :, 0].clamp(min=1).to(x.dtype)                 # (b,1,H,W)
        E = V.abs().sum(2) / n
        cover = count[:, :, 0].to(x.dtype) / T
        L2 = L[:, :, 0]
        feats = [L2, E, cover]
        if self.history:
            if hist is None or hvalid is None:
                raise ValueError('this variant needs hist and hvalid')
            A = hvalid.to(x.dtype)
            D = (L2 - hist.to(x.dtype)) * A
            feats += [D, A]
        ll = self.level(torch.cat(feats, 1))[:, None]                # (b,1,H,W)
        lq = F.logsigmoid(ll)
        if self.gate is None:
            lq = lq.expand(b, T, H, W)
        else:
            if self.gate_kind == 'spatial':
                z = [V, V.abs()]
                if self.history:
                    z += [D[:, :, None].expand_as(V), A[:, :, None].expand_as(V)]
                g = self.gate(torch.cat(z, 1))
            else:
                g = self.gate(V)
            lq = lq + torch.log(g.clamp(min=1e-6))
        self.last_ll = ll
        self.last = {'lv': lv.detach(), 'll': ll.detach(), 'lq': lq.detach()}
        return f, noisy_or_logit(lv, lq)


@register('lvh_full', needs=NEEDS)
def _full(seqlen=40):
    return DeepProLVH(seqlen, history=True, gate='spatial')


@register('lvh_stab', needs=NEEDS)
def _stab(seqlen=40):
    return DeepProLVH(seqlen, history=False, gate='spatial')


@register('lvh_pointgate', needs=NEEDS)
def _point(seqlen=40):
    return DeepProLVH(seqlen, history=True, gate='point')


@register('lvh_nogate', needs=NEEDS)
def _nogate(seqlen=40):
    return DeepProLVH(seqlen, history=True, gate=None)


# ---------------------------------------------------------------- evidence-decoupled LVH
class DeepProLVHDec(nn.Module):
    """Evidence-decoupled LVH (lvh_dec).

    lvh_full feeds [L, E, cover, D, A] to one level network trained with one target, so a
    persistent target with D ~ 0 (present since before the window) looks like a ground hot
    spot and is suppressed: on DJI_0319_2, 81 -> 3 near-stationary hits at 1 FA/frame.
    Here the two kinds of persistent evidence get separate heads, joined by noisy-OR so
    that the absence of one cannot veto the other:
      appearance head  [L, E, cover] -> l_A   persistent and target-like
      arrival head     [D, A, cover] -> l_R   persistent and new relative to History
      level evidence   q = 1 - (1 - s(l_A)) (1 - s(l_R)),  gated per frame like lvh_full
    Each head has its own target in training (train.py, --arrival-target history): the
    arrival head ignores persistent targets that History already contains.
    """
    decoupled = True

    def __init__(self, seqlen=40):
        super().__init__()
        self.var = Backbone(seqlen)
        self.appear = LevelNet(3)
        self.arrive = LevelNet(3)
        self.gate = SpatialGate(4)
        self.last = {}

    def forward(self, x, valid=None, pvalid=None, hist=None, hvalid=None):
        b, _, T, H, W = x.shape
        if hist is None or hvalid is None:
            raise ValueError('lvh_dec needs hist and hvalid')
        if pvalid is None:
            pvalid = torch.ones_like(x, dtype=torch.bool)
        pvalid = pvalid.bool()
        if valid is not None:
            pvalid = pvalid & valid.bool()[:, None, :, None, None]
        L, V, count = decompose_masked(x, pvalid)
        f, lv = self.var(V)
        n = count[:, :, 0].clamp(min=1).to(x.dtype)
        E = V.abs().sum(2) / n
        cover = count[:, :, 0].to(x.dtype) / T
        L2 = L[:, :, 0]
        A = hvalid.to(x.dtype)
        D = (L2 - hist.to(x.dtype)) * A
        la = self.appear(torch.cat([L2, E, cover], 1))[:, None]      # (b,1,H,W)
        lr = self.arrive(torch.cat([D, A, cover], 1))[:, None]       # (b,1,H,W)
        ll = noisy_or_logit(la, F.logsigmoid(lr))                     # logit of q
        z = [V, V.abs(), D[:, :, None].expand_as(V), A[:, :, None].expand_as(V)]
        g = self.gate(torch.cat(z, 1))
        lq = F.logsigmoid(ll) + torch.log(g.clamp(min=1e-6))
        self.last_la, self.last_lr, self.last_ll = la, lr, ll
        self.last = {'lv': lv.detach(), 'll': ll.detach(), 'lq': lq.detach(),
                     'la': la.detach(), 'lr': lr.detach()}
        return f, noisy_or_logit(lv, lq)


@register('lvh_dec', needs=NEEDS)
def _dec(seqlen=40):
    return DeepProLVHDec(seqlen)


# ---------------------------------------------------------------- losses
def soft_iou_loss(logit, target):
    """Same as train.soft_iou_loss. Zero gradient on samples without positive pixels."""
    p = torch.sigmoid(logit)
    dims = tuple(range(1, logit.dim()))
    inter = (p * target).sum(dims)
    return 1 - (inter / (p.sum(dims) + target.sum(dims) - inter)).mean()


def focal_bce(logit, target, gamma=2.0):
    """Focal binary cross-entropy for soft targets, mean over pixels.

    Unlike Soft-IoU it penalises positive scores on crops with no target, which is what
    pushes the level pathway down on ground hot spots.
    """
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
    mod = (target - torch.sigmoid(logit)).abs().pow(gamma)
    return (mod * ce).mean()


def persistent_target(lab, valid_px, mode='ramp', frac=0.75, lo=0.3, hi=0.7):
    """Level-pathway target from per-frame labels.

    lab (b,T,H,W); valid_px (b,1,T,H,W) or (b,T,H,W) pixel validity, or (b,T) frame validity.
    occupancy = fraction of valid frames in which the pixel is labelled.
      'hard': occupancy >= frac (DeepPro-LV's auxiliary target)
      'ramp': 0 below lo, 1 above hi, linear between. The median holds content present in
              more than half of the frames, so the ramp is centred on 0.5 instead of
              asking for 75 percent; a slowly drifting target still gets credit.
    """
    v = valid_px.to(lab.dtype)
    if v.dim() == 5:
        v = v[:, 0]
    elif v.dim() == 2:
        v = v[:, :, None, None]
    occ = (lab * v).sum(1) / v.sum(1).clamp(min=1)
    if mode == 'hard':
        return (occ >= frac).to(lab.dtype)
    if mode == 'ramp':
        return ((occ - lo) / (hi - lo)).clamp(0, 1)
    raise ValueError(mode)


def arrival_target(pm, hlab, hvalid, resident=0.5):
    """Target and weight for the arrival head of lvh_dec.

    pm (b,H,W) persistence target; hlab (b,1,H,W) fraction of History samples in which an
    annotation covers the pixel; hvalid (b,1,H,W) History available.
      persistent and new (hlab < resident, History available) -> positive, weight 1
      persistent and resident, or no History                  -> ignored, weight 0
      everything else                                         -> negative, weight 1
    A resident target leaves no trace in L - B, so asking the arrival head to call it
    either target or background would teach it something the input cannot show.
    """
    new = ((hlab[:, 0] < resident) & hvalid[:, 0].bool()).to(pm.dtype)
    target = pm * new
    weight = 1 - (pm > 0).to(pm.dtype) * (1 - new)
    return target, weight


def masked_soft_iou_loss(logit, target, weight):
    """Soft-IoU over the pixels with weight 1 (weights are 0 or 1)."""
    p = torch.sigmoid(logit) * weight
    t = target * weight
    dims = tuple(range(1, logit.dim()))
    inter = (p * t).sum(dims)
    return 1 - (inter / (p.sum(dims) + t.sum(dims) - inter).clamp(min=1e-6)).mean()


def masked_focal_bce(logit, target, weight, gamma=2.0):
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
    mod = (target - torch.sigmoid(logit)).abs().pow(gamma)
    return (mod * ce * weight).sum() / weight.sum().clamp(min=1)
