"""One real optimizer update per architecture, using its unchanged training loop."""
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import traceback

import torch

from runlib import ROOT, read_config, save_json, train_command, verify_source
from torch_compat import install

install()

sys.path.insert(0, str(ROOT / 'code'))
import models
import train


def main():
    verify_source()
    assert json.loads((ROOT / 'cache_ready.json').read_text())['status'] in ('ready', 'passed')
    assert json.loads((ROOT / 'hotspots_ready.json').read_text())['status'] in ('ready', 'passed')
    results = {'status': 'running', 'gpu': torch.cuda.get_device_name(), 'models': {}}
    path = ROOT / 'gpu_preflight.json'
    save_json(path, results)
    for model in read_config()['models']:
        command = train_command(model, smoke=True)
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
            sd = torch.load(out / 'final.pth', map_location='cpu', weights_only=False)
            loaded, needs = models.build(model['arch'])
            loaded.load_state_dict(sd, strict=True)
            assert all(torch.isfinite(t).all() for t in sd.values())
            results['models'][model['arch']] = {
                'status': 'passed', 'optimizer_updates': 1, 'batch': 4, 'frames': 40, 'crop': 128,
                'loss': loss, 'parameters': models.n_params(loaded), 'needs': list(needs),
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
