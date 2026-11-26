"""MRTP's mr_lv network on original contiguous inputs; not full multi-rate MRTP."""
import importlib
from research_models.models import build

SoftLoUloss = importlib.import_module('networks.models.DeepPro-Plus').SoftLoUloss


def detector(num_classes, seqlen=40, out_len=40):
    if num_classes != 1 or out_len != seqlen:
        raise ValueError('mr_lv requires one class and out_len == seqlen')
    model, _ = build('mr_lv', seqlen=seqlen)
    return model
