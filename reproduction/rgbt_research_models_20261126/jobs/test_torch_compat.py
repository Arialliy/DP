"""CPU semantic witness for the narrow Conv3d layout-helper backport."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch_compat


class Mixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.volume = nn.Sequential(nn.Conv3d(2, 3, (2, 3, 2)), nn.Tanh(),
                                    nn.Sequential(nn.ConvTranspose3d(3, 2, 2)))
        self.image = nn.Conv2d(2, 3, 2)
        self.other_five_dimensional = nn.Parameter(torch.randn(2, 3, 2, 2, 2))
        self.register_buffer('buffer', torch.randn(2, 3))

    def forward(self, volume, image):
        return self.volume(volume), self.image(image)


def compare(dtype):
    torch.manual_seed(46)
    reference = Mixed().to(dtype=dtype)
    converted = copy.deepcopy(reference)
    original_parameters = {name: id(p) for name, p in converted.named_parameters()}
    original_values = {name: p.detach().clone() for name, p in converted.named_parameters()}
    original_strides = {name: p.stride() for name, p in converted.named_parameters()}
    original_pointers = {name: p.data_ptr() for name, p in converted.named_parameters()}
    original_buffer = converted.buffer.detach().clone()
    utils_before = dict(vars(torch.nn.utils))
    installed = torch_compat.install()
    utils_after = dict(vars(torch.nn.utils))
    changed_api_names = {key for key in set(utils_before) | set(utils_after)
                         if utils_before.get(key) is not utils_after.get(key)}
    assert changed_api_names == ({'convert_conv3d_weight_memory_format'} if installed else set())
    helper = torch.nn.utils.convert_conv3d_weight_memory_format
    assert not torch_compat.install() and torch.nn.utils.convert_conv3d_weight_memory_format is helper
    assert helper(converted, torch.channels_last_3d) is converted
    expected_changed = {'volume.0.weight', 'volume.2.0.weight'}
    actual_changed = set()
    for name, parameter in converted.named_parameters():
        assert id(parameter) == original_parameters[name], ('Parameter replaced', name)
        assert torch.equal(parameter.detach(), original_values[name]), ('Value changed', name)
        if parameter.stride() != original_strides[name]:
            actual_changed.add(name)
        if name in expected_changed:
            assert parameter.is_contiguous(memory_format=torch.channels_last_3d), name
        else:
            assert parameter.data_ptr() == original_pointers[name], ('Unrelated storage changed', name)
    assert actual_changed == expected_changed, actual_changed
    assert torch.equal(converted.buffer, original_buffer)
    volume_ref = torch.randn(2, 2, 5, 6, 5, dtype=dtype, requires_grad=True)
    image_ref = torch.randn(2, 2, 5, 6, dtype=dtype, requires_grad=True)
    volume_new = volume_ref.detach().clone().requires_grad_(True)
    image_new = image_ref.detach().clone().requires_grad_(True)
    outputs_ref = reference(volume_ref, image_ref)
    outputs_new = converted(volume_new, image_new)
    tolerance = {'atol': 1e-6, 'rtol': 1e-5} if dtype == torch.float32 else {'atol': 1e-12, 'rtol': 1e-10}
    for left, right in zip(outputs_ref, outputs_new):
        torch.testing.assert_close(left, right, **tolerance)
    loss_ref = sum(t.square().mean() for t in outputs_ref) + reference.other_five_dimensional.square().mean()
    loss_new = sum(t.square().mean() for t in outputs_new) + converted.other_five_dimensional.square().mean()
    loss_ref.backward()
    loss_new.backward()
    torch.testing.assert_close(volume_ref.grad, volume_new.grad, **tolerance)
    torch.testing.assert_close(image_ref.grad, image_new.grad, **tolerance)
    grad_errors = {}
    for (name, left), (_, right) in zip(reference.named_parameters(), converted.named_parameters()):
        torch.testing.assert_close(left.grad, right.grad, **tolerance)
        grad_errors[name] = float((left.grad - right.grad).abs().max())
    assert helper(converted, torch.contiguous_format) is converted
    for name, parameter in converted.named_parameters():
        assert torch.equal(parameter.detach(), original_values[name]), name
        assert parameter.stride() == original_strides[name], name
    singleton = nn.Conv3d(1, 1, 1).to(dtype=dtype)
    singleton_value = singleton.weight.detach().clone()
    helper(singleton, torch.channels_last_3d)
    assert singleton.weight.is_contiguous(memory_format=torch.channels_last_3d)
    assert torch.equal(singleton.weight.detach(), singleton_value)
    return {'dtype': str(dtype), 'status': 'pass', 'installed_this_test': installed,
            'only_changed_strides': sorted(actual_changed), 'parameter_identity_preserved': True,
            'weight_values_exact_equal': True, 'unrelated_storage_unchanged': True,
            'round_trip_contiguous': 'pass', 'singleton_weight': 'pass',
            'max_output_absolute_error': max(float((a - b).abs().max()) for a, b in zip(outputs_ref, outputs_new)),
            'max_parameter_gradient_absolute_error': max(grad_errors.values()),
            'max_input_gradient_absolute_error': max(float((volume_ref.grad - volume_new.grad).abs().max()),
                                                      float((image_ref.grad - image_new.grad).abs().max())),
            'tolerance': tolerance}


if __name__ == '__main__':
    torch.set_num_threads(1)
    report = {'status': 'pass', 'torch': torch.__version__, 'device': 'cpu',
              'official_source_url': torch_compat.SOURCE_URL,
              'backport_sha256': hashlib.sha256((ROOT / 'torch_compat.py').read_bytes()).hexdigest(),
              'tests': [compare(torch.float32), compare(torch.float64)]}
    print(json.dumps(report, indent=2))
