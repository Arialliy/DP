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
targets whose contrast polarity differs from the exemplars'), mxc_lv (DeepPro-MXC below).

DeepPro-MXC: motion-calibrated level
    Problem. What separates a parked target from a hot spot in the level L is mostly its
    contrast, and the contrast of a target depends on the scene: on DJI_0319_2 the ships
    have 9-22 grey levels of local contrast and the hot spots a median of 37. A level
    pathway trained on other scenes learns one absolute contrast rule, which is wrong for
    such a scene. M0 measured this gap: [contrast, contrast^2] fitted on the other clips
    separates the near-stationary ships of DJI_0319_2 from hot spots with AUC 0.837, fitted
    within the clip with 0.995 (window-parity folds). The information is there, but only
    relative to the scene. mx_lv cannot use it: its encoder starts with LocalNorm, which
    removes contrast amplitude.
    Idea. The moving targets of the clip, which the temporal profile finds, are in-scene
    calibration samples: they show what contrast and polarity targets have in this scene.
    mxc_lv measures the local contrast c_e at each exemplar (centre 3 x 3 minus the ring
    at radii 4-7, as stabilize.local_contrast; signed value of largest magnitude within
    +-2 px of the patch centre) and the contrast c(p) of L, and gives LevelNet four
    calibrated channels (calibration_channels):
        R = log(|c| + eps) - median_j log(|c_e,j| + eps)    contrast in exemplar units
        |R|                                                  distance from the typical target
        D = min_j |log(|c| + eps) - log(|c_e,j| + eps)|      distance from the nearest
                                                             exemplar (clips with two kinds)
        P = sign(c) * mean_j sign(c_e,j)                     polarity agreement
    Floor eps. mxc_lv uses a floor relative to the exemplars, eps = FLOOR_REL * median_j
    |c_e,j| (at least EPS_MIN). A gain g > 0 on the clip's contrast multiplies c, every c_e
    and so eps alike, and the four channels are then exactly invariant (up to float
    rounding, and as long as FLOOR_REL * median |c_e| > EPS_MIN, i.e. 0.01 grey levels).
    mxc_lv_absfloor keeps a fixed eps of one grey level; it does not scale with the gain,
    so the channels are only approximately invariant: for g in [1/2, 2], R changes by at
    most 0.14 when every contrast lies in 5-60 grey levels, and by 0.27 in 2-60. L itself
    is not invariant. train.py's --gain-aug draws such a gain per sample (window and
    exemplars together), so that absolute contrast stops being a reliable cue in training
    and the level pathway has to use the calibrated one. Without exemplars the four
    channels are 0 (A_x = 0 already says so), as in mx_lv.
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
EPS_C = 1.0 / 47.4845      # one grey level in the normalised input (vtdata.RGBT_STD): the fixed floor
FLOOR_REL = 0.1            # relative floor: eps = FLOOR_REL * median exemplar |contrast|
EPS_MIN = 0.001 / 47.4845  # lower bound of the relative floor (0.001 grey levels)
CLIP_LOG = 4.0             # calibrated log-contrast channels are clipped to +-4 (a factor of 55)
SHIFT_C = 2                # exemplar contrast: largest magnitude within +-2 px of the centre


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


def box_sum(x, r):
    """(n,1,H,W) -> sum over the (2r+1)^2 window, zero outside the image (cv2.boxFilter with
    normalize=False and BORDER_CONSTANT, as stabilize.local_contrast)."""
    if r == 0:
        return x
    k = 2 * r + 1
    return F.avg_pool2d(x, k, stride=1, padding=r, count_include_pad=True) * float(k * k)


def local_contrast_t(img, ok, inner=1, outer=(4, 7)):
    """Torch twin of stabilize.local_contrast: (n,1,H,W) values and bool validity ->
    (contrast, defined), contrast = mean of the (2 inner + 1)^2 centre minus the mean of
    the ring between radii outer[0] and outer[1]; defined where the centre holds a valid
    pixel and the ring at least 4 (stabilize returns NaN elsewhere, here the value is 0)."""
    okf = ok.to(img.dtype)
    z = torch.where(ok, img, torch.zeros_like(img))
    s_in, n_in = box_sum(z, inner), box_sum(okf, inner)
    s_lo, n_lo = box_sum(z, outer[0] - 1), box_sum(okf, outer[0] - 1)
    s_hi, n_hi = box_sum(z, outer[1]), box_sum(okf, outer[1])
    ring_s, ring_n = s_hi - s_lo, n_hi - n_lo
    defined = (n_in > 0.5) & (ring_n > 3.5)
    c = s_in / n_in.clamp(min=1.0) - ring_s / ring_n.clamp(min=1.0)
    return torch.where(defined, c, torch.zeros_like(c)), defined


def exemplar_contrast(ex, shift=SHIFT_C):
    """(b,K,1,S,S) exemplar patches -> (b,K) signed local contrast of largest magnitude
    within +-shift px of the patch centre. The patch holds the whole centre-and-ring
    neighbourhood (S >= 2 * (7 + shift) + 1), so this is the value on the full frame."""
    b, K, _, S, _ = ex.shape
    m = S // 2
    if S < 2 * (7 + shift) + 1:
        raise ValueError(f'exemplar patches of {S} px are too small for the contrast ring')
    p = ex.reshape(b * K, 1, S, S)
    c, _ = local_contrast_t(p, torch.ones_like(p, dtype=torch.bool))
    w = c[:, 0, m - shift:m + shift + 1, m - shift:m + shift + 1].reshape(b * K, -1)
    i = w.abs().argmax(1, keepdim=True)
    return w.gather(1, i)[:, 0].view(b, K)


def calibration_channels(c, defined, ce, ce_valid, floor='relative', eps=EPS_C, clip=CLIP_LOG):
    """Exemplar-calibrated contrast of the level (DeepPro-MXC, module docstring).

    c, defined: (b,1,H,W) local contrast of L and where it is defined; ce, ce_valid: (b,K)
    exemplar contrasts and bool validity. floor 'relative': eps = FLOOR_REL * median
    |c_e| (exactly gain-invariant); 'absolute': the fixed eps. -> (b,4,H,W) [R, |R|, D, P],
    0 for samples without a valid exemplar and where c is undefined.
    """
    ce_valid = ce_valid.bool()
    avail = ce_valid.any(1)                                                   # (b,)
    nan = torch.full_like(ce, float('nan'))
    if floor == 'relative':
        s_e = torch.nanquantile(torch.where(ce_valid, ce.abs(), nan), 0.5, dim=1)
        s_e = torch.where(avail, s_e, torch.zeros_like(s_e))
        e = (FLOOR_REL * s_e).clamp(min=EPS_MIN)                              # (b,)
    elif floor == 'absolute':
        e = torch.full_like(ce[:, 0], float(eps))
    else:
        raise ValueError(floor)
    lc = torch.log(c.abs() + e[:, None, None, None])                          # (b,1,H,W)
    le = torch.log(ce.abs() + e[:, None])                                     # (b,K)
    ref = torch.nanquantile(torch.where(ce_valid, le, nan), 0.5, dim=1)
    ref = torch.where(avail, ref, torch.zeros_like(ref))                      # (b,)
    nv = ce_valid.sum(1).clamp(min=1).to(c.dtype)
    pol = (torch.sign(ce) * ce_valid.to(c.dtype)).sum(1) / nv                 # (b,) in [-1, 1]
    R = (lc - ref[:, None, None, None]).clamp(-clip, clip)
    d = (lc[:, None] - le[:, :, None, None, None]).abs()                      # (b,K,1,H,W)
    d = d.masked_fill(~ce_valid[:, :, None, None, None], float('inf')).amin(1)
    D = d.clamp(max=clip)
    P = torch.sign(c) * pol[:, None, None, None]
    keep = (avail[:, None, None, None] & defined).to(c.dtype)
    return torch.cat([R, R.abs(), D, P], 1) * keep


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
    n_extra = 0          # level-pathway channels added by a subclass (extra_level_inputs)

    def __init__(self, seqlen=40, dim=32, n_global=8, frame_step=4, polarity='signed'):
        super().__init__()
        if polarity not in ('signed', 'both'):
            raise ValueError(polarity)
        self.var = Backbone(seqlen)
        self.enc = Encoder(dim)
        self.protos = nn.Parameter(torch.randn(n_global, dim) * 0.5)
        self.level = LevelNet(6 + self.n_extra)
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

    def extra_level_inputs(self, L0, defined, ex, ex_valid):
        """Further LevelNet channels (b, n_extra, H, W); none for mx_lv."""
        return []

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
        extra = self.extra_level_inputs(L[:, :, 0], count[:, :, 0] > 0, ex, ex_valid)
        ll = self.level(torch.cat([L[:, :, 0], E, cover, Mx[:, None], A, Mg[:, None]] + list(extra), 1))[:, None]
        lq = F.logsigmoid(ll) + torch.log(self.gate(V).clamp(min=1e-6))
        self.last_ll = ll                                              # (b,1,H,W), with grad
        self.last_lm = (self.match_scale * Mx + self.match_bias)[:, None]
        self.last_avail = avail
        self.last = {'lv': lv.detach(), 'll': ll.detach(), 'lq': lq.detach(),
                     'mx': Mx.detach(), 'mg': Mg.detach(), 'avail': avail.detach()}
        return f, self.extra_fusion(noisy_or_logit(lv, lq), x, pvalid, ex, ex_valid)

    def extra_fusion(self, logit, x, pvalid, ex, ex_valid):
        """Further evidence fused into the per-frame logit (b,T,H,W); none for mx_lv / mxc_lv."""
        return logit


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


class DeepProMXC(DeepProMX):
    """DeepPro-MX with the motion-calibrated level channels (module docstring)."""
    n_extra = 4

    def __init__(self, seqlen=40, floor='relative', **kw):
        super().__init__(seqlen, **kw)
        if floor not in ('relative', 'absolute'):
            raise ValueError(floor)
        self.floor = floor

    def extra_level_inputs(self, L0, defined, ex, ex_valid):
        b, _, H, W = L0.shape
        if ex is None or ex_valid is None or not bool(ex_valid.bool().any()):
            cal = L0.new_zeros(b, 4, H, W)
        else:
            c, ok = local_contrast_t(L0, defined)
            ce = exemplar_contrast(ex.to(L0.dtype))
            cal = calibration_channels(c, ok, ce, ex_valid, self.floor)
        self.last_cal = cal.detach()
        return [cal]


class AppearanceHead(nn.Module):
    """Per-frame 2-D head (n, cin, H, W) -> logit (n, H, W): four 3 x 3 convolutions (receptive
    field 11 px, one dilated) and a 1 x 1 output, no normalisation layers; starts mostly off."""

    def __init__(self, cin, width=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, width, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(width, 1, 1))
        nn.init.constant_(self.net[-1].bias, -4.0)

    def forward(self, z):
        return self.net(z)[:, 0]


class DeepProMTA(DeepProMXC):
    """DeepPro-MTA: a single-frame appearance pathway taught by the temporal one.

    Problem (official RGB-T test, DeepPro-Plus, threshold 1e-5): the misses are not mainly
    near-stationary targets (25.8 %; isolated near-stationary boxes are missed as often as
    isolated moving ones). They are concentrated in time: the first 60 frames of the clips
    hold 41 % of them, and the same tracks are missed 26 % of the time there and 10 % later.
    The temporal evidence of a target varies along the clip, its appearance in a frame does
    not: in single frames the missed targets look like the targets the detector finds in
    the same video, often the same track in other windows.
    Design. mxc_lv (the temporal pathway, the level pathway with motion-calibrated
    channels) plus, for every frame x_t of the window, an appearance logit
        a_t = AppearanceHead([LocalNorm(x_t), S_t, R_t, |R_t|, D_t, P_t, A_x])
    where S_t(p) = max_j <A(x_t)(p), e_j> is the similarity of the frame to the exemplars
    (the encoder of mx_lv, read without gradient here), and R_t, |R_t|, D_t, P_t are the
    calibrated contrast of the frame's own local contrast against the exemplars' (exactly
    gain-invariant, relative floor). Exemplars are, at inference, the network's own
    confident detections in other windows (infer_mx.py, bank 'outside'): the temporal
    pathway teaches the appearance pathway, per video, without labels. The fused output is
    the noisy-OR of the temporal, level and appearance probabilities. Without exemplars
    (A_x = 0) the head is a plain single-frame detector: train.py --ex-p-none 1 is the
    'no teacher' control, which separates "taught by motion" from "one more CNN".
    """
    appearance = True
    app_chunk = 8           # frames per encoder call in the appearance pathway

    def __init__(self, seqlen=40, floor='relative', **kw):
        super().__init__(seqlen, floor=floor, **kw)
        self.app_norm = LocalNorm(7)
        self.app = AppearanceHead(7)

    def appearance_logits(self, x, pvalid, ex, ex_valid):
        """(b,T,H,W) appearance logits; FILL where a pixel is not valid in its frame."""
        b, _, T, H, W = x.shape
        xs = x[:, 0].reshape(b * T, 1, H, W)
        pv = pvalid[:, 0].reshape(b * T, 1, H, W)
        if ex is not None and ex_valid is not None:
            ev = ex_valid.bool()
        else:
            ev = torch.zeros(b, 1, dtype=torch.bool, device=x.device)
        avail = ev.any(1)
        S = xs.new_zeros(b * T, 1, H, W)
        cal = xs.new_zeros(b * T, 4, H, W)
        if bool(avail.any()):
            with torch.no_grad():                                                  # a feature, as M_x
                e = self.embed_exemplars(ex).repeat_interleave(T, 0)              # (bT,K,d)
                evt = ev.repeat_interleave(T, 0)
                parts = []
                for i in range(0, b * T, self.app_chunk):                         # bounds the memory
                    f = self.enc(xs[i:i + self.app_chunk])                        # (n,d,H,W)
                    s = torch.einsum('ndhw,nkd->nkhw', f, e[i:i + self.app_chunk])
                    parts.append(s.masked_fill(~evt[i:i + self.app_chunk, :, None, None], -2.0).amax(1))
                s = torch.cat(parts, 0)
                S = torch.where(avail.repeat_interleave(T)[:, None, None, None], s[:, None], S)
            ce = exemplar_contrast(ex.to(xs.dtype))                               # (b,K)
            c, ok = local_contrast_t(xs, pv)
            cal = calibration_channels(c, ok, ce.repeat_interleave(T, 0), ev.repeat_interleave(T, 0), self.floor)
        A = avail.repeat_interleave(T).to(xs.dtype)[:, None, None, None].expand(b * T, 1, H, W)
        z = torch.cat([self.app_norm(xs) * pv.to(xs.dtype), S, cal, A], 1)
        la = self.app(z).view(b, T, H, W)
        return torch.where(pvalid[:, 0], la, torch.full_like(la, -30.0))

    def extra_fusion(self, logit, x, pvalid, ex, ex_valid):
        la = self.appearance_logits(x, pvalid, ex, ex_valid)
        self.last_la = la                                                         # with grad
        self.last['la'] = la.detach()
        return noisy_or_logit(logit, F.logsigmoid(la))


def appearance_loss(model, lab, pvalid, focal=1.0):
    """Soft-IoU + focal BCE of the appearance logits against every frame's label, on the
    valid pixels: the head must find every target of a frame, moving or not."""
    la = model.last_la                                                            # (b,T,H,W)
    w = pvalid[:, 0].to(la.dtype)
    p = torch.sigmoid(la) * w
    inter = (p * lab).sum((1, 2, 3))
    iou = 1 - (inter / (p.sum((1, 2, 3)) + (lab * w).sum((1, 2, 3)) - inter + 1e-6)).mean()
    return iou + focal * masked_focal_bce(la, lab, w)


@register('mx_lv', needs=NEEDS)
def _mx(seqlen=40):
    return DeepProMX(seqlen)


@register('mx_lv_abs', needs=NEEDS)
def _mx_abs(seqlen=40):
    return DeepProMX(seqlen, polarity='both')


@register('mxc_lv', needs=NEEDS)
def _mxc(seqlen=40):
    return DeepProMXC(seqlen)


@register('mta_lv', needs=NEEDS)
def _mta(seqlen=40):
    return DeepProMTA(seqlen)


@register('mxc_lv_absfloor', needs=NEEDS)
def _mxc_absfloor(seqlen=40):
    """Ablation: the fixed floor of one grey level (only approximately gain-invariant)."""
    return DeepProMXC(seqlen, floor='absolute')
