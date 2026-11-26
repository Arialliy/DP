"""Level-Variation decomposition of temporal profiles (DeepPro-LV) and controls.

Every pixel's temporal profile x_p(t) over the T-frame window is split exactly into
    level      L_p   = median_t x_p(t)             (persistent component)
    variation  V_p(t) = x_p(t) - L_p               (transient component)
A target that covers p for fewer than T/2 frames leaves L_p unchanged and lies entirely in
V_p(t); a target that dwells longer moves into L_p. The two pathways are therefore
complementary in dwell time:
  * variation pathway: the DeepPro-Plus backbone (spatial/temporal difference convs + TPro)
    applied to V (background level removed) -> per-frame logit l_v(t)
  * level pathway: a 2-D encoder-decoder on [L, E], E_p = mean_t |V_p(t)| -> logit l_L
  * consistency gate g(t) = sigmoid(MLP[V(t), |V(t)|]) closes the level evidence at frames
    where the current observation departs from the level (target left / not yet arrived)
  * fusion (noisy-OR): p(t) = 1 - (1 - sigmoid(l_v(t))) * (1 - g(t) * sigmoid(l_L))
Variants registered here (see models.build):
  lv_full        V-path + level path + gate (proposed)
  lv_vonly       V-path only (decomposition without level pathway)
  lv_xlevel      DeepPro-Plus on x (unchanged) + level path + gate
  lv_frame       V-path + the same 2-D network applied to every current frame x(t) instead of
                 L (control: "just a spatial branch")
  lv_nogate      V-path + level path, noisy-OR without the consistency gate
  dpp_wide       DeepPro-Plus with widened channels (parameter-matched control)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import SDConv, STDResblock, TPro, cbr, register


def decompose(x, valid=None):
    """x: (b,1,T,H,W). valid: (b,T) bool or None. -> L (b,1,1,H,W), V (b,1,T,H,W)."""
    if valid is None:
        L = x.median(dim=2, keepdim=True).values
        return L, x - L
    m = valid[:, None, :, None, None].expand_as(x)
    xn = torch.where(m, x, torch.full_like(x, float('nan')))
    L = torch.nanmedian(xn, dim=2, keepdim=True).values
    V = torch.where(m, x - L, torch.zeros_like(x))
    return L, V


class Backbone(nn.Module):
    """DeepPro-Plus temporal backbone up to conv_out1 (8-channel per-frame features)."""

    def __init__(self, seqlen=40, in_ch=1, w=(8, 16, 32)):
        super().__init__()
        a, b, c = w
        self.conv_in = cbr(SDConv(in_ch, a, (5, 7, 7), padding=(2, 3, 3)), a)
        self.layer1 = nn.Sequential(STDResblock(a, b), STDResblock(b, c))
        self.TPro = TPro(c, 8, seqlen, seqlen)
        self.conv_out1 = cbr(nn.Conv3d(c, 8, 1), 8)
        self.conv_out2 = nn.Conv3d(8, 1, 1)

    def forward(self, x):
        f = self.conv_out1(self.TPro(self.layer1(self.conv_in(x))))
        return f, self.conv_out2(f).squeeze(1)


def c2(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class LevelNet(nn.Module):
    """2-D encoder-decoder (3 scales) on the level field. (b,cin,H,W) -> logit (b,H,W)."""

    def __init__(self, cin=2, w=(16, 32, 64)):
        super().__init__()
        a, b, c = w
        self.e1, self.e2, self.e3 = c2(cin, a), c2(a, b), c2(b, c)
        self.d2, self.d1 = c2(c + b, b), c2(b + a, a)
        self.head = nn.Conv2d(a, 1, 1)

    def forward(self, z):
        H, W = z.shape[-2:]
        ph, pw = (-H) % 4, (-W) % 4
        if ph or pw:
            z = F.pad(z, (0, pw, 0, ph), mode='replicate')
        e1 = self.e1(z)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        d2 = self.d2(torch.cat([F.interpolate(e3, scale_factor=2, mode='bilinear', align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False), e1], 1))
        return self.head(d1)[:, 0, :H, :W]


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(2, 8, 1), nn.ReLU(inplace=True), nn.Conv3d(8, 1, 1))
        nn.init.constant_(self.net[2].bias, 2.0)  # start mostly open

    def forward(self, V):
        return torch.sigmoid(self.net(torch.cat([V, V.abs()], 1))).squeeze(1)


def noisy_or_logit(lv, lq):
    """logit of 1-(1-sigmoid(lv))(1-q) given q in log-space: lq = log q."""
    # log(1-p) = log(1-sigmoid(lv)) + log(1-q)
    log1m_pv = -F.softplus(lv)
    log1m_q = torch.log1p(-torch.exp(lq).clamp(max=1 - 1e-6))
    log1m = log1m_pv + log1m_q
    # logit = log(1-exp(log1m)) - log1m
    return torch.log(-torch.expm1(log1m).clamp(max=-1e-12)) - log1m


class DeepProLV(nn.Module):
    def __init__(self, seqlen=40, var_input='V', level_input='L', gate=True):
        super().__init__()
        self.var_input, self.level_input, self.use_gate = var_input, level_input, gate
        self.var = Backbone(seqlen)
        self.level = LevelNet(2) if level_input else None
        self.gate = Gate() if (gate and level_input) else None
        self.last = {}

    def forward(self, x, valid=None):
        L, V = decompose(x, valid)
        f, lv = self.var(V if self.var_input == 'V' else x)
        if self.level is None:
            return f, lv
        b, _, T, H, W = x.shape
        E = V.abs().mean(2)  # (b,1,H,W)
        if self.level_input == 'L':
            ll = self.level(torch.cat([L[:, :, 0], E], 1))[:, None]          # (b,1,H,W)
        else:  # 'frame': apply the 2-D net to every current frame
            fr = x.permute(0, 2, 1, 3, 4).reshape(b * T, 1, H, W)
            ee = E[:, None].expand(b, T, 1, H, W).reshape(b * T, 1, H, W)
            ll = self.level(torch.cat([fr, ee], 1)).view(b, T, H, W)
        lq = F.logsigmoid(ll)
        if self.gate is not None:
            g = self.gate(V)
            lq = lq + torch.log(g.clamp(min=1e-6))
        else:
            lq = lq.expand(b, T, H, W)
        self.last_ll = ll  # kept with grad for the auxiliary level loss
        self.last = {'lv': lv.detach(), 'll': ll.detach(), 'lq': lq.detach()}
        return f, noisy_or_logit(lv, lq)


class DeepProPlusWide(nn.Module):
    def __init__(self, seqlen=40):
        super().__init__()
        self.var = Backbone(seqlen, w=(12, 32, 56))

    def forward(self, x, valid=None):
        return self.var(x)


@register('lv_full', needs=('valid',))
def _f(seqlen=40):
    return DeepProLV(seqlen, 'V', 'L', True)


@register('lv_vonly', needs=('valid',))
def _v(seqlen=40):
    return DeepProLV(seqlen, 'V', None, False)


@register('lv_xlevel', needs=('valid',))
def _x(seqlen=40):
    return DeepProLV(seqlen, 'x', 'L', True)


@register('lv_frame', needs=('valid',))
def _fr(seqlen=40):
    return DeepProLV(seqlen, 'V', 'frame', True)


@register('lv_nogate', needs=('valid',))
def _ng(seqlen=40):
    return DeepProLV(seqlen, 'V', 'L', False)


@register('dpp_wide', needs=('valid',))
def _w(seqlen=40):
    return DeepProPlusWide(seqlen)


def persistent_mask(lab, valid, frac=0.75):
    """(b,T,H,W) labels -> (b,H,W) pixels covered in >= frac of the valid frames."""
    v = valid[:, :, None, None].float()
    return ((lab * v).sum(1) >= frac * v.sum(1).clamp(min=1)).float()
