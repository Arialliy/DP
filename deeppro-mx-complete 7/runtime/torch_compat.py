"""One missing PyTorch utility backport for the frozen research-model scripts.

Source: PyTorch v2.5.1, torch/nn/utils/memory_format.py,
https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/utils/memory_format.py

The installed NVIDIA PyTorch 2.2.0a0 exposes the equivalent Conv2d utility but
lacks the Conv3d helper invoked by code/train.py and the inference scripts.
This preserves the official operation order and Conv3d/ConvTranspose3d scope.
Only weight storage layout changes; parameter objects and values are preserved.
The upstream copyright and license are retained in torch_compat.PYTORCH_LICENSE.
"""
import torch

SOURCE_URL = 'https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/utils/memory_format.py'


def convert_conv3d_weight_memory_format(module, memory_format):
    """Recursively convert only Conv3d/ConvTranspose3d weights and return module."""
    if isinstance(module, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
        converted = module.weight.detach().clone()
        converted = converted.contiguous(memory_format=memory_format)
        module.weight.data = converted.resize_(converted.size(), memory_format=memory_format)
    for child in module.children():
        convert_conv3d_weight_memory_format(child, memory_format)
    return module


def install():
    """Add the helper only if absent; return whether this call installed it."""
    name = 'convert_conv3d_weight_memory_format'
    if hasattr(torch.nn.utils, name):
        return False
    setattr(torch.nn.utils, name, convert_conv3d_weight_memory_format)
    return True
