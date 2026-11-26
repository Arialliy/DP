"""Standalone environment checks. This process never starts/resumes training."""
import argparse
import hashlib
import importlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
args = parser.parse_args()
root = Path('/official')
run = Path('/run')
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'networks/models'))
source = json.loads((run / 'SOURCE_SHA256.json').read_text())
# Published logs are hidden by the writable output mount; code is checked here.
checked = 0
for name, expected in source.items():
    if name.startswith('log/'):
        continue
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    checked += 1
spec = json.loads((run / 'env-spec.json').read_text())
for name, expected in spec['packages'].items():
    assert importlib.metadata.version(name) == expected, (name, expected)

import numpy as np
import torch
import train
import test  # Also checks final-evaluation dependencies.
from data_utils.TrainDataLoader import TrainIRSeqDataLoader
from data_utils.TestDataLoader import TestIRSeqDataLoader

assert torch.__version__ == spec['torch']
for split in ('train.txt', 'test.txt'):
    assert (Path('/data') / split).read_bytes() == (root / 'datasets/NUDT-MIRSDT' / split).read_bytes()

# Exercise the original ragged-list sampler, without adapting its container.
train.seed_everything(46)  # Only this disposable witness process is seeded.
dataset = TrainIRSeqDataLoader('NUDT-MIRSDT', '/data', 40, 0.1, 128)
assert len(dataset) == 768 and len(dataset.samplelist) == 7680
assert isinstance(dataset.samplelist, list)
batch = [dataset[i] for i in range(4)]
images = torch.stack([row[0] for row in batch]).float()
targets = torch.stack([row[1] for row in batch]).float()
assert images.shape == (4, 1, 40, 128, 128)
assert targets.shape == (4, 40, 128, 128)
assert torch.isfinite(images).all() and targets.sum() > 0
testset = TestIRSeqDataLoader('NUDT-MIRSDT', '/data', 40, 4)
assert len(testset) == 20
test_images, test_masks, centroids, ends = testset[0][0]
assert test_images.shape[1] == 40 and ends == [0, 39]
assert centroids.sum() > 0

result = {'device': args.device, 'source_files_checked': checked,
          'torch': torch.__version__, 'numpy': np.__version__,
          'training_samples': len(dataset), 'updates_per_epoch': 192,
          'test_sequences': len(testset), 'training_batch_shape': list(images.shape),
          'original_ragged_list_sampler': 'PASS'}
if args.device == 'cuda':
    model_module = importlib.import_module('DeepPro-Plus')
    model = model_module.detector(1, 40, 40).cuda()
    loss_fn = model_module.SoftLoUloss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    optimizer.zero_grad()
    _, scores = model(images.cuda())
    loss = loss_fn(scores, targets.cuda())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    # Test the unmodified torch.load call used by official test.py.
    buf = io.BytesIO()
    torch.save({'model_state_dict': model.state_dict(), 'class_avg_iou': np.float64(.1)}, buf)
    buf.seek(0)
    checkpoint = torch.load(buf)
    model.load_state_dict(checkpoint['model_state_dict'])
    result.update(loss=float(loss.detach()), gpu=torch.cuda.get_device_name(),
                  allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                  forward_backward_adam_checkpoint='PASS')
print('WITNESS ' + json.dumps(result, sort_keys=True), flush=True)
