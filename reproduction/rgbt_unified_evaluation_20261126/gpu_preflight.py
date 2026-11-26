"""Full-resolution inference and one real clip through the complete scorer."""
import gc
import json
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from common import ROOT, config, read_json, save_json, verify_source
from inference import load_model, benchmark_window


def main():
    verify_source()
    torch.set_num_threads(4)
    result = dict(status='running', models={}, device=torch.cuda.get_device_name())
    save_json(ROOT / 'preflight.json', result)
    try:
        test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                              capture_output=True, text=True, timeout=120)
        (ROOT / 'logs/unit_tests.log').write_text(test.stdout + test.stderr)
        assert test.returncode == 0, test.stderr[-3000:]
        result['unit_tests'] = 'passed; logs/unit_tests.log'
        state = read_json(ROOT / 'STATUS.json')
        video = np.asarray(np.load('/artifacts/cache/video/DJI_0049_2.npy', mmap_mode='r'), dtype=np.float32)
        for model_cfg in config()['models']:
            arch = model_cfg['arch']
            jobs = state['models'][model_cfg['id']]['jobs']
            if jobs:
                job = read_json(ROOT / 'jobs' / (jobs[0] + '.json'))
                checkpoint = ROOT / job['checkpoint']
                train_cfg = read_json(ROOT / job['train_config'])
            else:
                assert arch == 'lvh_dec'
                checkpoint = ROOT / 'checkpoints/preflight_lvh_dec.pth'
                train_cfg = dict(arch=arch, hist_len=80, hist_step=4, hist_mode='both', motion_gate=1.0)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model, needs = load_model(arch, checkpoint, 'cuda:0')
            record = benchmark_window(model, needs, video, 'DJI_0049_2', train_cfg, 'cuda:0')
            record['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            result['models'][arch] = record
            del model
            save_json(ROOT / 'preflight.json', result)
        gc.collect()
        torch.cuda.empty_cache()
        subprocess.run([sys.executable, '-u', 'evaluate.py', 'preflight_lv', '--max-clips', '1'], check=True)
        status = read_json(ROOT / 'jobs/preflight_lv.status.json')
        assert status['state'] == 'completed' and status['complete'] is False
        metrics = read_json(ROOT / status['results'][0])
        assert metrics['frames'] == 90 and len(metrics['thresholds']) == 27
        assert metrics['complete'] is False and 0 <= metrics['auc'] <= 1
        result.update(status='passed', full_clip_integration='passed; 90 frames, 27 thresholds; explicitly partial')
    except BaseException as exc:
        result.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        save_json(ROOT / 'preflight.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
