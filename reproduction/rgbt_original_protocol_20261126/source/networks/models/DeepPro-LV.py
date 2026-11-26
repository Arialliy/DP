"""DeepPro-LV model adapter for the unchanged original train/test interface."""
import importlib
from research_models.models import build

# Use exactly the baseline's loss implementation, with no auxiliary losses.
SoftLoUloss = importlib.import_module('networks.models.DeepPro-Plus').SoftLoUloss


def detector(num_classes, seqlen=40, out_len=40):
    if num_classes != 1 or out_len != seqlen:
        raise ValueError('DeepPro-LV requires one class and out_len == seqlen')
    model, _ = build('lv_full', seqlen=seqlen)
    return model
