"""Exercise the real RGB-T loaders, optimizer, and full-resolution checkpoint."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path[:0] = ['/work', '/work/networks/models']
import train
import test  # Verify all final evaluation dependencies.
from data_utils.TrainDataLoader import TrainIRSeqDataLoader
from data_utils.TestDataLoader import TestIRSeqDataLoader

run = Path('/run')
manifest = json.loads((run / 'source_manifest.json').read_text())
for name, record in manifest['files'].items():
    assert hashlib.sha256((Path('/work') / name).read_bytes()).hexdigest() == record['run_sha256'], name
train.seed_everything(46)  # Disposable witness only; does not seed the training run.
# train.main never calls seed_everything: use its actual default cuDNN settings.
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
started = time.monotonic()
dataset = TrainIRSeqDataLoader('RGB-T', '/data/RGB-T', 40, 0.05, 128)
assert len(dataset.samplelist) == 18955
assert len(dataset) == 947
batch = [dataset[i] for i in range(4)]
images = torch.stack([item[0] for item in batch]).float().cuda()
labels = torch.stack([item[1] for item in batch]).float().cuda()
assert images.shape == (4, 1, 40, 128, 128)
assert labels.shape == (4, 40, 128, 128)
assert torch.isfinite(images).all() and labels.sum() > 0
model_module = importlib.import_module('DeepPro-Plus')
model = model_module.detector(1, 40, 40).cuda()
criterion = model_module.SoftLoUloss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
_, scores = model(images)
loss = criterion(scores, labels)
loss.backward()
assert torch.isfinite(loss)
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
optimizer.step()
train_peak = torch.cuda.max_memory_allocated()
del optimizer, images, labels, scores, loss, model
torch.cuda.empty_cache()
model = model_module.detector(1, 40, 40).cuda().eval()
checkpoint_path = run / 'log/sem_seg/released_rgbt_deepproplus/checkpoints/best_model.pth'
assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == manifest['checkpoint_sha256']
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
testset = TestIRSeqDataLoader('RGB-T', '/data/RGB-T', 40, 4)
assert len(testset) == 15
total_windows = 0
for index in range(len(testset)):
    seq = testset[index]
    total_windows += len(seq)
    assert all(len(window) == 40 for window in seq.samplelist)
    covered = {row[2] for window in seq.samplelist for row in window}
    assert covered == set(range(max(covered) + 1)), testset.seq_names[index]
images, labels, centroids, ends = testset[0][0]
assert tuple(images.shape) == (1, 40, 512, 640) and ends == [0, 39]
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
eval_start = time.monotonic()
with torch.no_grad():
    _, logits = model(images.unsqueeze(0).float().cuda())
torch.cuda.synchronize()
assert tuple(logits.shape) == (1, 40, 512, 640)
assert torch.isfinite(logits).all()
result = {
    'status': 'passed', 'torch': torch.__version__, 'numpy': np.__version__,
    'gpu': torch.cuda.get_device_name(), 'training_windows_available': len(dataset.samplelist),
    'training_samples_per_epoch': len(dataset), 'training_updates_per_epoch': len(dataset) // 4,
    'test_sequences': len(testset), 'test_windows': total_windows,
    'full_resolution_shape': list(logits.shape),
    'full_resolution_forward_seconds': time.monotonic() - eval_start,
    'training_peak_allocated_bytes': train_peak,
    'evaluation_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
    'original_sampler_forward_backward_adam': 'passed',
    'released_checkpoint_strict_load': 'passed', 'test_frame_coverage': 'passed',
    'elapsed_seconds': time.monotonic() - started,
}
(run / 'preflight.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
