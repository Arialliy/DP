"""Network of the multi-rate temporal-profile detector (MRTP, see multirate.py).

mr_lv is DeepPro-LV (variation pathway, level pathway on [L, E], pointwise consistency
gate, noisy-OR) with the masked decomposition of lvh.py, so the same weights score
recorded and camera-compensated windows: invalid pixels are left out of the median and a
`cover` channel (fraction of valid samples) is added to the level input. On recorded
frames cover = 1 and the network is lv_full with one constant extra input channel.
It has no History input. Rates and coordinates are chosen outside the network
(train.py --rates/--coords, infer_mr.py).
"""
from lvh import DeepProLVH
from models import register


@register('mr_lv', needs=('valid', 'pvalid'))
def _mr_lv(seqlen=40):
    return DeepProLVH(seqlen, history=False, gate='point')
