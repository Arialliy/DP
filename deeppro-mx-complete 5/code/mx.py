"""DeepPro-MX: motion exemplars for targets the temporal profile cannot see.

Problem
    A temporal-profile detector finds a target by its anomaly in time. A target that does
    not move relative to its background has none: on the RGB-T test set the released
    DeepPro-Plus misses 1,362 of 11,303 targets even at threshold 1e-5 (authors' log), and
    89 % of the near-stationary validation boxes belong to tracks that never reach 0.5
    px/frame, so no window length, rate or History reaches them. What is left is
    appearance, and DeepPro-LV's level pathway cannot tell a parked target from a ground
    hot spot by local appearance alone.

Idea
    Within one clip, targets of the same kind look alike whether they move or not, and the
    temporal profile already finds the moving ones. DeepPro-MX uses them as exemplars:
      profile     mr_lv's variation pathway on V = x - L (masked median)   -> l_v(t)
      encoder A   per-frame embedding F_t = A(x_t), unit vectors (no normalisation layers,
                  so a patch and a full frame give the same vector at the same pixel)
      exemplars   patches of moving targets of the same clip (training: GT boxes from other
                  windows; inference: peaks of the network's own l_v from other windows),
                  embedded by A at their centre pixel -> e_j
      matching    M_t(p) = max_j <F_t(p), e_j>, M_x = masked median over every
                  `frame_step`-th window frame; M_g the same with learned global prototypes
                  (used when a clip offers no exemplars)
      level       LevelNet on [L, E, cover, M_x, A_x, M_g] (A_x = exemplars available)
      fusion      pointwise consistency gate on V and noisy-OR, as in mr_lv
    On frames without a valid exemplar M_x = 0 and A_x = 0, and the network is mr_lv plus a
    learned-prototype channel. That configuration (train with --ex-p-none 1, infer with
    --bank none) is the control that separates "exemplars from the clip" from "a larger
    appearance branch".

Inputs (needs): x (b,1,T,H,W); valid (b,T); pvalid (b,1,T,H,W) as mr_lv;
ex (b,K,1,S,S) exemplar patches, S = 2 * encoder.margin + 1; ex_valid (b,K) bool.
Registered: mx_lv (signed matching), mx_lv_abs (also matches the inverted frame, for
targets whose contrast polarity differs from the exemplars').
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # research_models package of the DP tree
    from .lv import Backbone, Gate, LevelNet, noisy_or_logit
    from .lvh import decompose_masked, masked_focal_bce, persistent_target
    from .models import register
except ImportError:  # code/ of this repository
    from lv import Backbone, Gate, LevelNet, noisy_or_logit
    from lvh import decompose_masked, masked_focal_bce, persistent_target
    from models import register

NEEDS = ('valid', 'pvalid', 'ex', 'ex_valid')


class LocalNorm(nn.Module):
    """Local contrast normalisation: (x - local mean) / sqrt(local var + floor^2) over a
    (2r+1)^2 window that ignores pixels outside the image (count_include_pad=False)."""

    def __init__(self, radius=7, floor=0.05):
        super().__init__()
        self.radius, self.floor = radius, floor

    def forward(self, x):
        k, r = 2 * self.radius + 1, self.radius
        mu = F.avg_pool2d(x, k, stride=1, padding=r, count_include_pad=False)
        var = F.avg_pool2d(x * x, k, stride=1, padding=r, count_include_pad=False) - mu * mu
        return (x - mu) / torch.sqrt(var.clamp(min=0) + self.floor ** 2)


class Encoder(nn.Module):
    """Per-frame appearance embedding (n,1,H,W) -> (n,dim,H,W), unit length per pixel.

    No batch or group normalisation: the embedding of a pixel depends only on the input
    within `margin` px, so an exemplar patch of size 2 * margin + 1 gives at its centre
    exactly the vector the full frame gives at that pixel (test_mx.py checks this).
    """

    def __init__(self, dim=32, width=32, dilations=(1, 1, 2, 4, 4), lcn_radius=7):
        super().__init__()
        self.norm = LocalNorm(lcn_radius)
        layers, cin = [], 1
        for d in dilations:
            layers += [nn.Conv2d(cin, width, 3, padding=d, dilation=d), nn.ReLU(inplace=True)]
            cin = width
        layers.append(nn.Conv2d(width, dim, 1))
        self.net = nn.Sequential(*layers)
        self.margin = int(sum(dilations)) + int(lcn_radius)
        self.dim = dim

    @property
    def patch(self):
        return 2 * self.margin + 1

    def forward(self, x):
        return F.normalize(self.net(self.norm(x)), dim=1, eps=1e-6)


def masked_median_t(m, valid):
    """m, valid: (b,T,H,W) -> (b,H,W) median over valid entries, 0 where none is valid."""
    mn = torch.where(valid, m, torch.full_like(m, float('nan')))
    med = torch.nanmedian(mn, dim=1).values
    return torch.where(valid.any(1), med, torch.zeros_like(med))


def erode_valid(pv, r):
    """(n,H,W) bool -> pixels whose (2r+1)^2 neighbourhood inside the image is all valid.
    Pixels outside the image do not count as invalid (max_pool2d pads with -inf)."""
    if r <= 0:
        return pv
    bad = F.max_pool2d((~pv).float()[:, None], 2 * r + 1, stride=1, padding=r)[:, 0]
    return pv & (bad == 0)


class DeepProMX(nn.Module):
    exemplar = True

    def __init__(self, seqlen=40, dim=32, n_global=8, frame_step=4, polarity='signed'):
        super().__init__()
        if polarity not in ('signed', 'both'):
            raise ValueError(polarity)
        self.var = Backbone(seqlen)
        self.enc = Encoder(dim)
        self.protos = nn.Parameter(torch.randn(n_global, dim) * 0.5)
        self.level = LevelNet(6)
        self.gate = Gate()
        self.match_scale = nn.Parameter(torch.tensor(8.0))   # auxiliary match logit s*M_x + c
        self.match_bias = nn.Parameter(torch.tensor(-4.0))
        self.frame_step, self.polarity = frame_step, polarity
        self.last = {}

    @property
    def patch(self):
        return self.enc.patch

    # ------------------------------------------------------------ pieces
    def embed_exemplars(self, ex):
        """(b,K,1,S,S) -> (b,K,dim): the encoder's vector at each patch centre."""
        b, K = ex.shape[:2]
        S, m = self.enc.patch, self.enc.margin
        if ex.shape[-2:] != (S, S):
            raise ValueError(f'exemplar patches must be {S} x {S}, got {tuple(ex.shape[-2:])}')
        z = self.enc(ex.reshape(b * K, 1, S, S).to(self.protos.dtype))[:, :, m, m]
        return z.view(b, K, -1)

    def profile(self, x, valid=None, pvalid=None):
        """Variation logits l_v (b,T,H,W) only: the exemplar source at inference."""
        pvalid = self._pvalid(x, valid, pvalid)
        _, V, _ = decompose_masked(x, pvalid)
        return self.var(V)[1]

    def _pvalid(self, x, valid, pvalid):
        if pvalid is None:
            pvalid = torch.ones_like(x, dtype=torch.bool)
        pvalid = pvalid.bool()
        if valid is not None:
            pvalid = pvalid & valid.bool()[:, None, :, None, None]
        return pvalid

    def match_maps(self, x, pvalid, ex=None, ex_valid=None):
        """-> M_x (b,H,W), M_g (b,H,W), available (b,) bool, per-frame indices used."""
        b, _, T, H, W = x.shape
        idx = list(range(0, T, self.frame_step))
        xs = x[:, 0, idx]                                             # (b,Tm,H,W)
        Tm = len(idx)
        pv = erode_valid(pvalid[:, 0, idx].reshape(b * Tm, H, W), self.enc.margin).view(b, Tm, H, W)
        inputs = [xs] if self.polarity == 'signed' else [xs, -xs]
        protos = F.normalize(self.protos, dim=1)
        if ex is not None and ex_valid is not None:
            avail = ex_valid.bool().any(1)                            # (b,)
        else:
            avail = torch.zeros(b, dtype=torch.bool, device=x.device)
        e = self.embed_exemplars(ex) if bool(avail.any()) else None   # (b,K,d)
        mx = mg = None
        for z in inputs:
            f = self.enc(z.reshape(b * Tm, 1, H, W)).view(b, Tm, -1, H, W)
            g = torch.einsum('btdhw,gd->btghw', f, protos).amax(2)    # (b,Tm,H,W)
            mg = g if mg is None else torch.maximum(mg, g)
            if e is not None:
                s = torch.einsum('btdhw,bkd->btkhw', f, e)
                s = s.masked_fill(~ex_valid.bool()[:, None, :, None, None], -2.0).amax(2)
                mx = s if mx is None else torch.maximum(mx, s)
        if mx is None:
            mx = torch.zeros_like(mg)
        mx = torch.where(avail[:, None, None, None], mx, torch.zeros_like(mx))
        return masked_median_t(mx, pv), masked_median_t(mg, pv), avail, idx

    # ------------------------------------------------------------ forward
    def forward(self, x, valid=None, pvalid=None, ex=None, ex_valid=None):
        b, _, T, H, W = x.shape
        pvalid = self._pvalid(x, valid, pvalid)
        L, V, count = decompose_masked(x, pvalid)
        f, lv = self.var(V)
        n = count[:, :, 0].clamp(min=1).to(x.dtype)                 # (b,1,H,W)
        E = V.abs().sum(2) / n
        cover = count[:, :, 0].to(x.dtype) / T
        Mx, Mg, avail, _ = self.match_maps(x, pvalid, ex, ex_valid)
        A = avail.to(x.dtype)[:, None, None, None].expand(b, 1, H, W)
        ll = self.level(torch.cat([L[:, :, 0], E, cover, Mx[:, None], A, Mg[:, None]], 1))[:, None]
        lq = F.logsigmoid(ll) + torch.log(self.gate(V).clamp(min=1e-6))
        self.last_ll = ll                                              # (b,1,H,W), with grad
        self.last_lm = (self.match_scale * Mx + self.match_bias)[:, None]
        self.last_avail = avail
        self.last = {'lv': lv.detach(), 'll': ll.detach(), 'lq': lq.detach(),
                     'mx': Mx.detach(), 'mg': Mg.detach(), 'avail': avail.detach()}
        return f, noisy_or_logit(lv, lq)


def match_loss(model, lab, pvalid, persist='hard'):
    """Focal BCE of the match logit s * M_x + c against the persistent-target mask, on the
    samples that had exemplars (0 when none had). Makes M_x itself discriminative, so the
    level pathway cannot ignore the exemplar channel."""
    lm = model.last_lm                                              # (b,1,H,W)
    pm = persistent_target(lab, pvalid, persist)[:, None]          # (b,1,H,W)
    w = model.last_avail.to(lm.dtype)[:, None, None, None].expand_as(lm)
    if not bool(model.last_avail.any()):
        return lm.sum() * 0.0
    return masked_focal_bce(lm, pm, w)


@register('mx_lv', needs=NEEDS)
def _mx(seqlen=40):
    return DeepProMX(seqlen)


@register('mx_lv_abs', needs=NEEDS)
def _mx_abs(seqlen=40):
    return DeepProMX(seqlen, polarity='both')
