"""Network definitions.

`DeepProPlus` is an independent re-implementation of DeepPro-Plus (Li et al., TPAMI 2026,
arXiv:2506.12766) whose parameter names match the authors' checkpoints, so released
weights load unchanged.  Equivalence with the authors' code is tested in
code/test_equivalence.py (max |diff| of logits on real clips).
Further architectures used in this study are registered in `build`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TDConv(nn.Conv3d):
    """Temporal-difference conv: kernel w'_t = 2 w_t - w_{t+1} - w_{t-1} (edges clamped)."""

    def forward(self, x):
        w = self.weight
        wp = torch.cat([w[:, :, 1:], w[:, :, -1:]], 2)
        wm = torch.cat([w[:, :, :1], w[:, :, :-1]], 2)
        return F.conv3d(x, 2 * w - wp - wm, self.bias, self.stride, self.padding, self.dilation, self.groups)


class SDConv(nn.Conv3d):
    """Spatial-difference conv: sum_k w_k (x_centre - x_k) within each frame tap."""

    def forward(self, x):
        w = -self.weight
        c = self.weight.shape[-1] // 2
        s = self.weight.sum(dim=(3, 4))
        w = w.clone()
        w[:, :, :, c, c] = s
        return F.conv3d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


def cbr(conv, c):
    return nn.Sequential(conv, nn.BatchNorm3d(c), nn.ReLU(inplace=True))


class STDResblock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = TDConv(cin, cout, (3, 3, 3), padding=(2, 1, 1), dilation=(2, 1, 1))
        self.bn1 = nn.BatchNorm3d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SDConv(cout, cout, (3, 3, 3), padding=(1, 2, 2), dilation=(1, 2, 2))
        self.bn2 = nn.BatchNorm3d(cout)
        self.conv3 = nn.Conv3d(cout, cout, 1)
        self.bn3 = nn.BatchNorm3d(cout)
        self.shortcut = nn.Sequential(nn.Conv3d(cin, cout, 1), nn.BatchNorm3d(cout)) if cin != cout else None

    def forward(self, x):
        r = x if self.shortcut is None else self.shortcut(x)
        o = self.relu(self.bn1(self.conv1(x)))
        o = self.relu(self.bn2(self.conv2(o)))
        o = self.bn3(self.conv3(o))
        return self.relu(o + r)


class TPro(nn.Module):
    """Temporal probe: per pixel, m learnable T x T correlation matrices over channel groups."""

    def __init__(self, d_model=32, num_head=8, seqlen=40, out_len=40):
        super().__init__()
        self.seqlen, self.num_head, self.out_len = seqlen, num_head, out_len
        self.QK_heads = nn.Sequential(*[nn.Linear(seqlen, out_len) for _ in range(num_head)])
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm3d(d_model)
        self.conv = cbr(nn.Conv3d(d_model, d_model, 1), d_model)

    def forward(self, x):  # x: (b, c, t, h, w)
        b, c, t, h, w = x.shape
        g = self.num_head
        W = torch.stack([l.weight for l in self.QK_heads])        # (g, out, t)
        B = torch.stack([l.bias for l in self.QK_heads])          # (g, out)
        v = x.view(b, g, c // g, t, h, w) / (self.seqlen ** 0.5)
        y = torch.einsum('bgcthw,got->bgcohw', v, W) + B[None, :, None, :, None, None]
        y = y.reshape(b, c, self.out_len, h, w)
        return self.conv(self.relu(self.norm1(y)))


class DeepProPlus(nn.Module):
    """DeepPro-Plus. Input (b,1,T,H,W) normalised grey; output (features, logits (b,T,H,W))."""

    def __init__(self, seqlen=40, in_ch=1):
        super().__init__()
        self.conv_in = cbr(SDConv(in_ch, 8, (5, 7, 7), padding=(2, 3, 3)), 8)
        self.layer1 = nn.Sequential(STDResblock(8, 16), STDResblock(16, 32))
        self.TPro = TPro(32, 8, seqlen, seqlen)
        self.conv_out1 = cbr(nn.Conv3d(32, 8, 1), 8)
        self.conv_out2 = nn.Conv3d(8, 1, 1)

    def forward(self, x):
        f = self.TPro(self.layer1(self.conv_in(x)))
        f = self.conv_out1(f)
        return f, self.conv_out2(f).squeeze(1)


REGISTRY = {}


def register(name, needs=()):
    def deco(fn):
        REGISTRY[name] = (fn, tuple(needs))
        return fn
    return deco


@register('deeppro_plus')
def _dpp(seqlen=40):
    return DeepProPlus(seqlen)


def build(name, seqlen=40):
    import importlib
    for module in ('lv', 'lvh', 'mr', 'mx'):  # architectures defined in other modules register on import
        if name not in REGISTRY:
            importlib.import_module(f'.{module}', __package__)
    fn, needs = REGISTRY[name]
    return fn(seqlen=seqlen), needs


def extra_inputs(needs, clip, video):
    """Auxiliary per-clip inputs. 'valid' (frame validity of zero-padded training windows)
    is not needed at inference, where every window is complete."""
    needs = [n for n in needs if n != 'valid']
    if not needs:
        return None
    raise ValueError(needs)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
