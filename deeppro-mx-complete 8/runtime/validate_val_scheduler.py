#!/usr/bin/env python3
"""Pure synthetic Docker/GPU state fixtures. Never invokes Docker, CUDA, or training."""
import copy
import json
from pathlib import Path
import tempfile

import val_scheduler as queue
from validate_val_summary import complete_fixture

HERE = Path(__file__).resolve().parent
HOST = HERE.parent
IMAGE = queue.template_prefix(HOST)[1]
CLIPS = ['fixture_val_A', 'fixture_val_B']


def record(epoch=1, complete=True):
    return dict(complete=complete, selection_mode='val_iou', split='val', threshold=.5,
                epoch=epoch, iou=.123, windows=81 if complete else 1, expected_windows=81,
                selection_max_windows=None, test_clips=CLIPS,
                protocol={'split': 'val', 'clips': CLIPS})


def set_barrier(root, valid=True):
    folder = root/'runs'/queue.MAIN
    folder.mkdir(parents=True, exist_ok=True)
    item = record(complete=valid)
    queue.atomic_json(folder/'best_selection.json', item)
    (folder/'best_model.pth').write_bytes(b'fixture only, not weights')
    (folder/'eval_log.jsonl').write_text(json.dumps(item)+'\n')


def finish(root, run):
    complete_fixture(root, run, CLIPS)
    folder = root/'runs'/run
    (folder/'final.pth').write_bytes(b'fixture final')
    (folder/'eval_log.jsonl').write_text(''.join(json.dumps(record(e))+'\n' for e in range(1, 33)))


def new_root():
    base = HERE/'test_fixtures/val_scheduler'
    base.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='validation_', dir=base))
    command = queue.read_json(HOST/'runtime/m0_val.command.json')
    command = [arg.replace(str(HOST), str(root)) for arg in command]
    queue.atomic_json(root/'runtime/m0_val.command.json', command)
    queue.atomic_json(root/'runtime/training_source_manifest.json', {'source_files': {}})
    queue.atomic_json(root/'configs/splits.json', {'val': CLIPS})
    queue.atomic_json(root/'results/calib_diag/summary.json', {'decision': {'go': False}, 'settings': {'split': ['train']}})
    queue.atomic_json(root/'results/miss_recovery/official_rgbt/test/summary.json', {'decision': {'go': False, 'diagnostic_only': True}})
    return root


class FakeBackend:
    def __init__(self):
        self.containers = {}
        self.created = []
        self.started = []
        self.devices = {0: dict(memory_mib=10, pids=['other-user-job']),
                        1: dict(memory_mib=500, pids=[]), 2: dict(memory_mib=0, pids=[])}

    def main(self):
        self.containers[queue.MAIN_CONTAINER] = {'Id': 'fixture-main', 'Image': IMAGE,
            'State': {'Status': 'running'}, 'Config': {'Labels': {}},
            'HostConfig': {'DeviceRequests': [{'DeviceIDs': ['2']}]}}

    def inspect(self, name):
        return copy.deepcopy(self.containers.get(name))

    def gpu_snapshot(self):
        return copy.deepcopy(self.devices)

    def create(self, command):
        name, config = queue.docker_config(command)
        assert name not in self.containers, 'must not duplicate containers'
        self.created.append(command)
        ident = 'fixture-'+name
        self.containers[name] = {'Id': ident, 'Image': config['Image'], 'Config': config,
                                'HostConfig': config['HostConfig'], 'State': {'Status': 'created'}}
        return ident

    def start(self, ident):
        self.started.append(ident)
        item = next(item for item in self.containers.values() if item['Id'] == ident)
        item['State']['Status'] = 'running'

    def exit(self, name, code=0):
        self.containers[name]['State'] = {'Status': 'exited', 'ExitCode': code, 'OOMKilled': False}


def main():
    root = new_root()
    backend = FakeBackend()
    scheduler = queue.Scheduler(root, backend, summarize=False)
    checks = []
    scheduler.tick()
    assert not backend.created and scheduler.state['jobs'][queue.MAIN]['status'] == 'waiting'
    backend.main()
    scheduler.tick()
    assert not backend.created
    checks.append('main container is adopted only; absent main and no val record launch nothing')
    set_barrier(root, False)
    scheduler.tick()
    assert not backend.created and not scheduler.state['barrier']['passed']
    checks.append('partial val selection cannot release controls')
    set_barrier(root)
    item = queue.read_json(root/'runs'/queue.MAIN/'best_selection.json')
    item['test_clips'] = ['incorrect-clip']
    queue.atomic_json(root/'runs'/queue.MAIN/'best_selection.json', item)
    scheduler.tick()
    assert not scheduler.state['barrier']['passed']
    checks.append('a complete selection with wrong clips cannot release controls')
    set_barrier(root)
    scheduler.tick()
    assert scheduler.state['barrier']['passed'] and not backend.created
    checks.append('compute process, memory=500 MiB, and reserved main GPU each block reuse')
    backend.devices[0] = dict(memory_mib=499, pids=[])
    scheduler.tick()
    assert len(backend.created) == 1 and scheduler.state['jobs'][queue.MAIN+'_noex']['gpu'] == 0
    first_command = backend.created[0]
    _, config = queue.docker_config(first_command)
    assert 'PYTHONPATH=/experiment/runtime' in config['Env']
    assert config['HostConfig']['RestartPolicy']['Name'] == 'no'
    assert config['HostConfig']['DeviceRequests'][0]['DeviceIDs'] == ['0']
    assert any(bind.endswith(':/data/RGB-T:ro') for bind in config['HostConfig']['Binds'])
    assert '--ex-p-none 1' in config['Cmd'][-1] and '--bank none' in config['Cmd'][-1]
    assert '--epochs 32 --batch 4' in config['Cmd'][-1]
    for filename in ('m0c.json', 'm1.json'):
        doc = queue.read_json(root/'runs'/(queue.MAIN+'_noex')/filename)
        assert doc['gate'] is False and doc['summary_sha256']
    checks.append('idle <500 MiB launches exact noex recipe once, records diagnostics and SHA')
    scheduler = queue.Scheduler(root, backend, summarize=False)
    scheduler.tick()
    assert len(backend.created) == 1
    checks.append('scheduler restart adopts running worker; no duplicate and no GPU reuse while initializing')
    backend.devices[1] = dict(memory_mib=0, pids=[])
    scheduler.tick()
    assert len(backend.created) == 2 and scheduler.state['jobs'][queue.MAIN+'_self']['gpu'] == 1
    assert '--ex-source self' in backend.created[-1][-1] and '--bank self' in backend.created[-1][-1]
    checks.append('second idle GPU launches self concurrently, preserves self bank')
    backend.exit(queue.worker_name(queue.SPECS[1]), 2)
    scheduler.tick()
    assert scheduler.state['jobs'][queue.MAIN+'_noex']['status'] == 'failed'
    assert len(backend.created) == 3 and '--gain-aug 1' in backend.created[-1][-1]
    del backend.containers[queue.worker_name(queue.SPECS[1])]
    scheduler = queue.Scheduler(root, backend, summarize=False)
    scheduler.tick()
    assert scheduler.state['jobs'][queue.MAIN+'_noex']['attempts'] == 1 and len(backend.created) == 3
    checks.append('failed worker is terminal and not retried; subsequent task can use its freed GPU')
    finish(root, queue.MAIN+'_self')
    backend.exit(queue.worker_name(queue.SPECS[2]))
    scheduler.tick()
    assert scheduler.state['jobs'][queue.MAIN+'_self']['status'] == 'completed'
    assert len(backend.created) == 4 and '--arch mx_lv' in backend.created[-1][-1] and '--gain-aug 2' in backend.created[-1][-1]
    checks.append('exit0 requires all32 selections and verified score; mx preserves gain2')
    backend.exit(queue.worker_name(queue.SPECS[3]))
    scheduler.tick()
    assert scheduler.state['jobs'][queue.MAIN+'_nogain']['status'] == 'failed'
    assert len(backend.created) == 5 and '--arch mxc_lv' in backend.created[-1][-1]
    checks.append('exit0 without final artifacts is failed; noapp uses mxc_lv')
    backend.exit(queue.MAIN_CONTAINER, 1)
    scheduler.tick()
    assert scheduler.state['status'] == 'main_failed_new_launches_halted'
    checks.append('main failure halts new launch requests without stopping existing workers')

    # Crash after Docker create but before start: same container, GPU recheck, no duplicate.
    root2 = new_root()
    backend2 = FakeBackend()
    backend2.main()
    set_barrier(root2)
    backend2.devices[0] = dict(memory_mib=0, pids=[])
    spec = queue.SPECS[1]
    ident = backend2.create(queue.docker_command(root2, spec, 0))
    backend2.devices[0]['pids'] = ['job-arrived-after-create']
    recovered = queue.Scheduler(root2, backend2, summarize=False)
    recovered.tick()
    assert not backend2.started
    backend2.devices[0]['pids'] = []
    recovered = queue.Scheduler(root2, backend2, summarize=False)
    recovered.tick()
    assert backend2.started == [ident] and len(backend2.created) == 1
    checks.append('created-container recovery waits for idle GPU then starts same container once')

    assert all('--split test' not in cmd[-1] and '--clips test' not in cmd[-1] for cmd in backend.created)
    assert all(spec['run'] != queue.MAIN for spec in queue.SPECS[1:])
    checks.append('six-run plan has no official/test stage and never creates main')
    report = dict(status='passed', checks=checks, gpu_calls=0, docker_calls=0, training_runs=0,
                  fixture_roots=[str(root), str(root2)], note='Synthetic state and score fixtures only; not experimental results.')
    queue.atomic_json(HERE/'val_scheduler_validation.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
