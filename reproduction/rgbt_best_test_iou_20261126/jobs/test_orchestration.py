"""CPU-only queue tests with all container/GPU calls replaced by in-memory fakes."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import runlib
import scheduler


def queue_witness():
    cfg = runlib.read_config()
    with tempfile.TemporaryDirectory(prefix='scheduler-test-', dir=ROOT / 'logs') as folder:
        root = Path(folder)
        for sub in ('jobs', 'logs', 'runs'):
            (root / sub).mkdir()
        states, launched = {}, []
        runlib.save_json(root / 'gpu_preflight.json', {'status': 'passed'})
        for model in cfg['models']:
            runlib.save_json(root / 'jobs' / (model['arch'] + '.json'), {'state': 'queued'})
            (root / 'logs' / (model['arch'] + '_console.log')).write_text('synthetic test log\n')

        def fake_run(command, **kwargs):
            assert command[:2] == ['docker', 'run'], command
            arch = command[-1]
            gpu = command[command.index('--gpus') + 1]
            assert gpu in ('device=0', 'device=2')
            launched.append((arch, gpu))
            states[runlib.container_name(arch)] = {'Running': True, 'Status': 'running', 'ExitCode': 0, 'OOMKilled': False}
            runlib.save_json(root / 'jobs' / (arch + '.json'), {'state': 'running'})
            return SimpleNamespace(returncode=0, stdout='synthetic-' + arch, stderr='')

        def complete(arch):
            states[runlib.container_name(arch)] = {'Running': False, 'Status': 'exited', 'ExitCode': 0, 'OOMKilled': False}
            runlib.save_json(root / 'jobs' / (arch + '.json'), {'state': 'completed'})

        with patch.object(scheduler, 'ROOT', root), patch.object(scheduler, 'inspect', side_effect=lambda name: states.get(name)), \
             patch.object(scheduler, 'idle', return_value=True), patch.object(scheduler.subprocess, 'run', side_effect=fake_run):
            scheduler.cycle(cfg, allow_launch=False)
            assert not launched, 'Status-only launched a job'
            scheduler.cycle(cfg, allow_launch=True)
            assert launched == [('lv_full', 'device=2'), ('lvh_full', 'device=0')], launched
            scheduler.cycle(cfg, allow_launch=True)
            assert len(launched) == 2, 'Duplicate launch while GPU occupied'
            complete('lv_full')
            scheduler.cycle(cfg, allow_launch=True)
            assert launched[-1] == ('mr_lv', 'device=2') and len(launched) == 3
            complete('lvh_full')
            scheduler.cycle(cfg, allow_launch=True)
            assert launched[-1] == ('lvh_dec', 'device=0') and len(launched) == 4
            complete('mr_lv')
            complete('lvh_dec')
            assert scheduler.cycle(cfg, allow_launch=True)
            runlib.save_json(root / 'gpu_preflight.json', {'status': 'failed'})
            states.clear()
            for model in cfg['models']:
                runlib.save_json(root / 'jobs' / (model['arch'] + '.json'), {'state': 'queued'})
            scheduler.cycle(cfg, allow_launch=True)
            assert len(launched) == 4, 'Failed preflight allowed a launch'
    return {'status': 'passed', 'scope': 'in-memory fakes only; no Docker/GPU execution',
            'checks': ['status-only never starts work', 'LV then MRTP on GPU2', 'LVH then dec on GPU0',
                       'no overlapping jobs per GPU', 'no GPU1 assignment', 'terminal detection',
                       'failed preflight prevents starts'], 'simulated_launch_order': launched}


if __name__ == '__main__':
    result = queue_witness()
    (ROOT / 'scheduler_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
