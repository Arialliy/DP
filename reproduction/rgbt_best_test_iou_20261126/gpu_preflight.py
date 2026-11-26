"""One real optimizer update plus two full-frame selection windows per model."""
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import traceback

import torch

from runlib import ROOT, read_config, save_json, train_command, verify_source, verify_artifacts
from torch_compat import install

install()

sys.path.insert(0, str(ROOT / 'code'))
import models
import train


def main():
    verify_source()
    artifacts = verify_artifacts()
    results = {'status': 'running', 'gpu': torch.cuda.get_device_name(), 'models': {}, 'artifacts': artifacts}
    path = ROOT / 'gpu_preflight.json'
    save_json(path, results)
    for model in read_config()['models']:
        command = train_command(model, smoke=True)
        command += ['--selection-max-windows', '2']
        out = ROOT / 'runs' / ('preflight_' + model['arch'])
        if out.exists():
            raise RuntimeError('Refusing to overwrite existing smoke run: ' + str(out))
        started = time.monotonic()
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            sys.argv = command[2:]
            train.main()
            log = (out / 'train_log.txt').read_text()
            assert 'epoch 1/1' in log and 'iters 1 ' in log, log
            loss = float(re.search(r'loss ([0-9.eE+-]+)', log).group(1))
            assert torch.isfinite(torch.tensor(loss))
            records = [json.loads(line) for line in (out / 'eval_log.jsonl').read_text().splitlines()]
            assert len(records) == 1 and records[0]['windows'] == 2, records
            assert records[0]['complete'] is False and records[0]['best_epoch'] is None
            assert not (out / 'best_model.pth').exists(), 'Partial preflight must not save formal best'
            assert 0 <= records[0]['iou'] <= 1
            assert records[0]['threshold'] == 0.5 and records[0]['split'] == 'official_test'
            sd = torch.load(out / 'final.pth', map_location='cpu', weights_only=False)
            loaded, needs = models.build(model['arch'])
            loaded.load_state_dict(sd, strict=True)
            assert all(torch.isfinite(t).all() for t in sd.values())
            results['models'][model['arch']] = {
                'status': 'passed', 'optimizer_updates': 1, 'batch': 4, 'frames': 40, 'crop': 128,
                'selection_iou': records[0]['iou'], 'selection_windows': 2,
                'eval_seconds': records[0]['elapsed_seconds'], 'loss': loss, 'parameters': models.n_params(loaded), 'needs': list(needs),
                'max_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                'elapsed_seconds': time.monotonic() - started,
                'checkpoint_sha256': hashlib.sha256((out / 'final.pth').read_bytes()).hexdigest(),
                'command': command,
            }
            del loaded, sd
        except BaseException as exc:
            results['status'] = 'failed'
            results['models'][model['arch']] = {'status': 'failed', 'error': repr(exc),
                                               'traceback': traceback.format_exc()}
            save_json(path, results)
            raise
        save_json(path, results)
    results['status'] = 'passed'
    save_json(path, results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
