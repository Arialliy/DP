#!/usr/bin/env python3
"""Durable six-run val queue. Standard library only; no model/training imports.

`plan` writes the reviewable launch command. `start` starts/resumes the background
scheduler container. `run` is its foreground loop. `once` is for manual diagnostics.
The main run is ALWAYS adopted, never launched by this scheduler. Worker containers
have restart=no; failed jobs remain failed. No official-test stage is implemented.
"""
import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import http.client
import io
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import quote

VERSION = 1
QUEUE = '20261126-s46-val'
MAIN = 'mta_val_s46'
MAIN_CONTAINER = 'dp-mta-v8-val-main-20261126'
SCHEDULER_CONTAINER = 'dp-mta-v8-val-scheduler-20261126'
ALLOWED_GPUS = (0, 1, 2)
TERMINAL = {'completed', 'failed'}
SPECS = [
    dict(run=MAIN, arch='mta_lv', extra=[], bank=None, main=True),
    dict(run=MAIN+'_noex', arch='mta_lv', extra=['--ex-p-none', '1'], bank='none'),
    dict(run=MAIN+'_self', arch='mta_lv', extra=['--ex-source', 'self'], bank='self'),
    dict(run=MAIN+'_nogain', arch='mta_lv', extra=['--gain-aug', '1'], bank=None),
    dict(run=MAIN+'_mx', arch='mx_lv', extra=['--gain-aug', '2'], bank=None),
    dict(run=MAIN+'_noapp', arch='mxc_lv', extra=[], bank=None),
]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def source_provenance(root):
    """Record runtime provenance and enforce the same source as the main training."""
    baseline_path = root/'runtime/training_source_manifest.json'
    baseline = read_json(baseline_path)
    if not isinstance(baseline, dict) or not isinstance(baseline.get('source_files'), dict):
        raise ValueError('main training source manifest is missing')
    observed = {name: sha256(root/name) for name in baseline['source_files']}
    changed = [name for name, digest in observed.items() if digest != baseline['source_files'][name]]
    if changed:
        raise ValueError('source changed since main training: '+', '.join(changed))
    runtime_names = ['val_scheduler.py', 'summarize_val.py', 'sitecustomize.py', 'torch_compat.py',
                     'm0_val.command.json', 'main_val.command.json', 'training_source_manifest.json']
    return dict(recorded_at=now(), main_source_matches=True, source_files=observed,
                runtime_files={name: sha256(root/'runtime'/name) for name in runtime_names
                               if (root/'runtime'/name).is_file()},
                docker_image=template_prefix(root)[1])


def commands(spec):
    run = spec['run']
    train = ['python', 'code/train.py', '--run', run, '--arch', spec['arch'],
             '--split', 'train', '--seed', '46', '--epochs', '32', '--batch', '4',
             '--checkpoint-selection', 'val_iou', '--selection-split', 'val',
             '--label-rule', 'paper_k6_floor', '--device', 'cuda'] + spec['extra']
    infer = ['python', 'code/infer_mx.py', '--run', run, '--ckpt', f'runs/{run}/best_model.pth',
             '--clips', 'val', '--device', 'cuda', '--save-arms']
    if spec['bank']:
        infer += ['--bank', spec['bank']]
    score = ['python', 'code/score.py', '--run', run, '--split', 'val', '--author',
             '--label-rule', 'paper_k6_floor']
    return [train, infer, score]


def worker_name(spec):
    return MAIN_CONTAINER if spec.get('main') else 'dp-mta-v8-val-'+spec['run'][len(MAIN)+1:]+'-20261126'


def template_prefix(root):
    command = read_json(root/'runtime/m0_val.command.json')
    if command[:3] != ['docker', 'run', '--detach']:
        raise ValueError('unexpected m0 Docker command template')
    at = command.index('--entrypoint')
    prefix, image = command[:at], command[at+2]
    if not image.startswith('sha256:'):
        raise ValueError('Docker image must be pinned by sha256')
    return prefix, image


def docker_command(root, spec, gpu):
    if gpu not in ALLOWED_GPUS:
        raise ValueError('GPU outside allowed physical indices 0/1/2')
    prefix, image = template_prefix(root)
    prefix = list(prefix)
    prefix[prefix.index('--name')+1] = worker_name(spec)
    body = ['set -euo pipefail', 'unset HOMOGRAPHY_DIR VT_ANN_DIR VT_OLD DET_RUN',
            'mkdir -p logs']
    for stage, command in zip(('train', 'infer', 'score'), commands(spec)):
        body.append(shlex.join(command)+' 2>&1 | tee '+shlex.quote('logs/'+stage+'.'+spec['run']+'.log'))
    prefix += ['--restart', 'no', '--label', 'dp.val.queue='+QUEUE,
               '--label', 'dp.val.run='+spec['run'], '--gpus', 'device='+str(gpu),
               '-e', 'PYTHONPATH=/experiment/runtime', '-e', 'M0_GATE=0',
               '--entrypoint', 'bash', image, '-c', '\n'.join(body)+'\n']
    return prefix


def docker_config(command):
    """Translate the audited CLI template to Docker Engine create JSON."""
    config = {'Env': [], 'Labels': {}, 'HostConfig': {'Binds': []}}
    host = config['HostConfig']
    name = None
    index = 2
    while index < len(command):
        arg = command[index]
        if arg == '--detach':
            index += 1
            continue
        if not arg.startswith('-'):
            config['Image'] = arg
            config['Cmd'] = command[index+1:]
            break
        val = command[index+1]
        index += 2
        if arg == '--name': name = val
        elif arg == '--network': host['NetworkMode'] = val
        elif arg == '--user': config['User'] = val
        elif arg == '--cpus': host['NanoCpus'] = int(float(val)*1_000_000_000)
        elif arg in ('--memory', '--shm-size'):
            units = {'k': 1024, 'm': 1024**2, 'g': 1024**3}
            size = int(float(val[:-1])*units[val[-1].lower()]) if val[-1].isalpha() else int(val)
            host['Memory' if arg == '--memory' else 'ShmSize'] = size
        elif arg == '-v': host['Binds'].append(val)
        elif arg == '-w': config['WorkingDir'] = val
        elif arg == '-e': config['Env'].append(val)
        elif arg == '--entrypoint': config['Entrypoint'] = [val]
        elif arg == '--label':
            key, value = val.split('=', 1)
            config['Labels'][key] = value
        elif arg == '--restart': host['RestartPolicy'] = {'Name': val}
        elif arg == '--gpus':
            ids = val.removeprefix('device=').split(',')
            host['DeviceRequests'] = [{'Driver': 'nvidia', 'Count': 0, 'DeviceIDs': ids, 'Capabilities': [['gpu']], 'Options': {}}]
        else: raise ValueError('unsupported Docker template argument '+arg)
    if not name or 'Image' not in config:
        raise ValueError('incomplete Docker template')
    return name, config


class UnixHTTP(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect('/var/run/docker.sock')


class DockerBackend:
    def request(self, method, path, body=None, allow404=False):
        connection = UnixHTTP('localhost', timeout=45)
        data = json.dumps(body).encode() if body is not None else None
        try:
            connection.request(method, '/v1.41'+path, body=data, headers={'Content-Type': 'application/json'})
            response = connection.getresponse()
            content = response.read()
            if response.status == 404 and allow404:
                return None
            if response.status >= 300:
                raise RuntimeError(f'Docker {method} {path}: HTTP {response.status}: '+content.decode(errors='replace')[:1000])
            return json.loads(content) if content else {}
        finally:
            connection.close()

    def inspect(self, name):
        return self.request('GET', '/containers/'+quote(name, safe='')+'/json', allow404=True)

    def create(self, command):
        name, config = docker_config(command)
        return self.request('POST', '/containers/create?name='+quote(name, safe=''), config)['Id']

    def start(self, ident):
        return self.request('POST', '/containers/'+quote(ident, safe='')+'/start')

    def gpu_snapshot(self):
        args = ['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits']
        rows = csv.reader(io.StringIO(subprocess.check_output(args, text=True, timeout=15)))
        result, by_uuid = {}, {}
        for row in rows:
            index, uuid, memory = [cell.strip() for cell in row]
            index = int(index)
            if index in ALLOWED_GPUS:
                result[index] = dict(index=index, uuid=uuid, memory_mib=int(memory), pids=[])
                by_uuid[uuid] = index
        args = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits']
        rows = csv.reader(io.StringIO(subprocess.check_output(args, text=True, timeout=15)))
        for row in rows:
            if not row:
                continue
            uuid, pid = [cell.strip() for cell in row]
            if uuid in by_uuid:
                # PID text is enough: inaccessible process details must not look idle.
                result[by_uuid[uuid]]['pids'].append(pid)
        return result


def valid_selection(record):
    return (isinstance(record, dict) and record.get('complete') is True
            and record.get('selection_mode') == 'val_iou' and record.get('split') == 'val'
            and record.get('threshold') == 0.5 and isinstance(record.get('epoch'), int)
            and record['epoch'] >= 1 and isinstance(record.get('iou'), (float, int))
            and math.isfinite(record['iou']) and record.get('windows', 0) > 0
            and record.get('windows') == record.get('expected_windows')
            and record.get('selection_max_windows') is None)


def selection_rows(root, run):
    path = root/'runs'/run/'eval_log.jsonl'
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # Only a trailing partially written line is ignored.
            if line != path.read_text().splitlines()[-1]:
                raise ValueError(f'{path}: malformed eval log')
    return rows


def first_selection(root):
    best = read_json(root/'runs'/MAIN/'best_selection.json')
    if not valid_selection(best) or not (root/'runs'/MAIN/'best_model.pth').is_file():
        return None
    expected = set(read_json(root/'configs/splits.json')['val'])
    if set(best.get('test_clips', [])) != expected or set(best.get('protocol', {}).get('clips', [])) != expected:
        return None
    rows = [row for row in selection_rows(root, MAIN) if valid_selection(row)]
    rows = [row for row in rows if set(row.get('test_clips', [])) == expected
            and set(row.get('protocol', {}).get('clips', [])) == expected]
    return min(rows, key=lambda r: r['epoch']) if rows else None


def complete_artifacts(root, run):
    progress = read_json(root/'runs'/run/'progress.json', {})
    epochs = {row['epoch'] for row in selection_rows(root, run) if valid_selection(row)}
    if progress.get('stage') != 'completed' or progress.get('epoch') != 32 or progress.get('epoch_total') != 32:
        return False, 'training progress is not completed at epoch 32/32'
    if not set(range(1, 33)).issubset(epochs):
        return False, 'missing full val selection records for one or more epochs 1..32'
    for rel in [f'runs/{run}/best_model.pth', f'runs/{run}/final.pth', f'runs/{run}/best_selection.json',
                f'results/scores/{run}/val/summary.json']:
        if not (root/rel).is_file():
            return False, 'missing '+rel
    from summarize_val import collect_run
    spec = next(spec for spec in SPECS if spec['run'] == run)
    clips = read_json(root/'configs/splits.json')['val']
    report = collect_run(root, run, spec['arch'], clips)
    if report['status'] != 'complete':
        return False, 'score/checkpoint validation: '+report['detail']
    return True, ''


def write_diagnostics(root, spec):
    run_dir = root/'runs'/spec['run']
    run_dir.mkdir(parents=True, exist_ok=True)
    source = root/'results/calib_diag/summary.json'
    m0 = read_json(source)
    if not isinstance(m0, dict) or 'decision' not in m0:
        raise ValueError('current M0 summary missing or malformed')
    record = {key: m0.get(key) for key in ('decision', 'decision_all', 'settings')}
    record.update(summary='results/calib_diag/summary.json', summary_sha256=sha256(source),
                  gate=False, diagnostic_only=True, user_instruction='M0/M1 只记录，继续 val 训练和对照', recorded_at=now())
    atomic_json(run_dir/'m0c.json', record)
    source = root/'results/miss_recovery/official_rgbt/test/summary.json'
    m1 = read_json(source)
    if not isinstance(m1, dict) or 'decision' not in m1:
        raise ValueError('current M1 summary missing or malformed')
    atomic_json(run_dir/'m1.json', dict(summary=str(source.relative_to(root)), summary_sha256=sha256(source),
        decision=m1['decision'], gate=False, diagnostic_only=True,
        user_instruction='M1 只记录，取消门槛，继续训练和对照', recorded_at=now()))


class Scheduler:
    def __init__(self, root, backend, summarize=True):
        self.root = Path(root).resolve()
        self.backend = backend
        self.do_summary = summarize
        self.base = self.root/'runtime/val_queue'
        self.base.mkdir(parents=True, exist_ok=True)
        self.path = self.base/'state.json'
        self.state = read_json(self.path)
        if self.state is None:
            self.state = dict(version=VERSION, queue=QUEUE, created_at=now(), status='waiting_for_main',
                barrier=dict(passed=False), jobs={s['run']: dict(status='pending', container_name=worker_name(s),
                    gpu=2 if s.get('main') else None, arch=s['arch'], attempts=0) for s in SPECS})
        if self.state.get('queue') != QUEUE or set(self.state['jobs']) != {s['run'] for s in SPECS}:
            raise ValueError('state queue/recipe mismatch; refusing to reuse')

    def save(self):
        self.state['updated_at'] = now()
        atomic_json(self.path, self.state)

    def event(self, message, **fields):
        record = dict(time=now(), message=message, **fields)
        with open(self.base/'events.jsonl', 'a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False)+'\n')
        with open(self.base/'scheduler.log', 'a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False)+'\n')
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def fail(self, job, message):
        job.update(status='failed', error=message, finished_at=now())
        self.event('task_failed', container=job['container_name'], error=message)
        self.save()

    def adopt(self, spec, job, info):
        if not spec.get('main'):
            labels = info.get('Config', {}).get('Labels') or {}
            if labels.get('dp.val.queue') != QUEUE or labels.get('dp.val.run') != spec['run']:
                self.fail(job, 'existing container name belongs to a different task; not adopted')
                return False
        expected_image = template_prefix(self.root)[1]
        if info.get('Image') != expected_image:
            self.fail(job, 'container image does not match the pinned M0 image')
            return False
        job.update(container_id=info['Id'])
        if job.get('gpu') is None:
            requests = info.get('HostConfig', {}).get('DeviceRequests', [])
            ids = [i for r in requests for i in r.get('DeviceIDs', [])]
            if len(ids) != 1 or ids[0] not in {'0', '1', '2'}:
                self.fail(job, 'adopted worker has an unexpected GPU request')
                return False
            job['gpu'] = int(ids[0])
        return True

    def summarize(self):
        if not self.do_summary:
            return
        script = self.root/'runtime/summarize_val.py'
        if not script.exists():
            raise RuntimeError('runtime/summarize_val.py is missing')
        # sitecustomize imports torch for workers; the scheduler and summary remain CPU/std-lib.
        command = [sys.executable, '-S', str(script), '--root', str(self.root)]
        result = subprocess.run(command, text=True, capture_output=True, timeout=60)
        if result.returncode:
            raise RuntimeError('summary failed: '+result.stderr[-2000:])

    def tick(self):
        self.state.pop('scheduler_error', None)
        for spec in SPECS:
            job = self.state['jobs'][spec['run']]
            if job['status'] in TERMINAL:
                continue
            info = self.backend.inspect(job['container_name'])
            if info is None:
                if spec.get('main'):
                    if job.get('container_id'):
                        self.fail(job, 'previously observed main container disappeared; no automatic restart')
                    else:
                        job['status'] = 'waiting'
                elif job['status'] == 'running':
                    self.fail(job, 'worker container disappeared; no automatic retry')
                continue
            if not self.adopt(spec, job, info):
                continue
            status = info['State']['Status']
            if status == 'created':
                if spec.get('main'):
                    job['status'] = 'waiting'  # root owns starting the main container
                else:
                    try:
                        snapshot = self.backend.gpu_snapshot()
                        device = snapshot.get(job['gpu'])
                        if device is None or device['pids'] or device['memory_mib'] >= 500:
                            job['status'] = 'waiting'
                            job['wait_reason'] = 'created container is waiting for an idle GPU'
                            continue
                        self.backend.start(info['Id'])
                        job.update(status='running', started_at=job.get('started_at', now()))
                        self.event('resumed_created_container', run=spec['run'])
                    except Exception as exc:
                        self.fail(job, 'container start failed: '+str(exc))
            elif status == 'running':
                job.update(status='running', started_at=job.get('started_at', info['State'].get('StartedAt', now())))
            elif status in ('exited', 'dead'):
                code = info['State'].get('ExitCode')
                if status == 'dead' or code != 0 or info['State'].get('OOMKilled'):
                    self.fail(job, f'container {status}, exit={code}, OOMKilled={info["State"].get("OOMKilled")}; '+info['State'].get('Error', ''))
                else:
                    ok, reason = complete_artifacts(self.root, spec['run'])
                    if ok:
                        job.update(status='completed', finished_at=info['State'].get('FinishedAt', now()), exit_code=0,
                                   checkpoint_sha256=sha256(self.root/'runs'/spec['run']/'best_model.pth'))
                        self.event('task_completed', run=spec['run'])
                    else:
                        self.fail(job, 'container exited 0 but '+reason)
            elif status in ('paused', 'restarting', 'removing'):
                job['status'] = 'waiting'
                job['container_status'] = status
            else:
                self.fail(job, 'unknown container status '+str(status))
        main = self.state['jobs'][MAIN]
        if not self.state['barrier']['passed']:
            evidence = first_selection(self.root)
            if evidence is not None and main['status'] in ('running', 'completed'):
                self.state['barrier'] = dict(passed=True, at=now(), epoch=evidence['epoch'], iou=evidence['iou'],
                    windows=evidence['windows'], source='runs/'+MAIN+'/eval_log.jsonl')
                self.event('first_full_val_selection_passed', epoch=evidence['epoch'], iou=evidence['iou'])
        # A later main failure also halts new launches; already running controls continue.
        can_launch = self.state['barrier']['passed'] and main['status'] in ('running', 'completed')
        if can_launch:
            for spec in SPECS[1:]:
                job = self.state['jobs'][spec['run']]
                if job['status'] not in ('pending', 'launching'):
                    continue
                if job['status'] == 'launching':
                    # A crash before create can safely finish the same named request once.
                    gpu = job['gpu']
                else:
                    gpu = None
                snapshot = self.backend.gpu_snapshot()
                self.state['gpu_snapshot'] = dict(at=now(), devices=snapshot)
                reserved = {j['gpu'] for j in self.state['jobs'].values()
                            if j['status'] in ('running', 'waiting', 'launching') and j is not job and j.get('container_id')}
                available = [i for i in ALLOWED_GPUS if i in snapshot and i not in reserved
                             and not snapshot[i]['pids'] and snapshot[i]['memory_mib'] < 500]
                if gpu is not None and gpu not in available:
                    continue
                if gpu is None:
                    if not available:
                        break
                    gpu = available[0]
                try:
                    if job['status'] != 'launching':
                        # Refuse accidental train.py auto-resume of an unowned experiment.
                        old = self.root/'runs'/spec['run']
                        if any((old/name).exists() for name in ('config.json', 'last.pth', 'best_model.pth', 'progress.json')):
                            raise RuntimeError('run artifacts already exist without an owned container; manual review required')
                        provenance = source_provenance(self.root)
                        atomic_json(self.base/'provenance'/(spec['run']+'.json'), provenance)
                        write_diagnostics(self.root, spec)
                        job.update(status='launching', gpu=gpu, attempts=1, started_at=now())
                    command = docker_command(self.root, spec, gpu)
                    atomic_json(self.base/'commands'/(spec['run']+'.json'), command)
                    self.save()  # intent durable before Docker mutation
                    ident = self.backend.create(command)
                    job['container_id'] = ident
                    self.save()
                    self.backend.start(ident)
                    job['status'] = 'running'
                    self.event('task_started', run=spec['run'], gpu=gpu, container_id=ident)
                    self.save()
                except Exception as exc:
                    self.fail(job, 'launch failed: '+str(exc))
        statuses = [job['status'] for job in self.state['jobs'].values()]
        self.state['status'] = ('finished_with_errors' if 'failed' in statuses else 'completed') if all(s in TERMINAL for s in statuses) else ('running' if can_launch else 'waiting_for_main')
        if main['status'] == 'failed':
            self.state['status'] = 'main_failed_new_launches_halted'
        self.save()
        self.summarize()
        return self.state


def launch_command(root):
    prefix, image = template_prefix(root)
    prefix = list(prefix)
    prefix[prefix.index('--name')+1] = SCHEDULER_CONTAINER
    # This monitor performs no CUDA computation. Device visibility is solely for nvidia-smi.
    prefix += ['--restart', 'unless-stopped', '--gpus', '"device=0,1,2"',
               '--pid', 'host', '-e', 'NVIDIA_DRIVER_CAPABILITIES=utility',
               '-v', '/var/run/docker.sock:/var/run/docker.sock',
               '--group-add', str(os.stat('/var/run/docker.sock').st_gid),
               '--label', 'dp.val.queue='+QUEUE, '--label', 'dp.val.role=scheduler',
               '--entrypoint', 'python', image, '-S', '/experiment/runtime/val_scheduler.py',
               'run', '--root', '/experiment']
    return prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'start', 'run', 'once'])
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--interval', type=float, default=30)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.interval < 5:
        parser.error('--interval must be >=5 seconds')
    base = root/'runtime/val_queue'
    base.mkdir(parents=True, exist_ok=True)
    if args.action in ('plan', 'start'):
        command = launch_command(root)
        atomic_json(base/'scheduler.command.json', command)
        atomic_json(base/'source_manifest.json', source_provenance(root))
        atomic_json(base/'recipe.json', dict(queue=QUEUE, specs=SPECS, commands={s['run']: commands(s) for s in SPECS},
            gpu_rule='physical GPU 0/1/2 only; no compute processes AND memory.used < 500 MiB',
            barrier='main run has at least one complete finite full-val selection at threshold 0.5',
            main_container=MAIN_CONTAINER, main_is_adopted_only=True,
            m0_gate=False, m1_gate=False, official_phase=False, created_at=now()))
        for spec in SPECS[1:]:
            # Explicit alternative commands remain reviewable before a GPU is assigned.
            atomic_json(base/'planned_commands'/(spec['run']+'.json'),
                        {str(gpu): docker_command(root, spec, gpu) for gpu in ALLOWED_GPUS})
        print(shlex.join(command))
        if args.action == 'start':
            result = subprocess.run(['docker', 'inspect', SCHEDULER_CONTAINER], capture_output=True, text=True)
            if result.returncode == 0:
                info = json.loads(result.stdout)[0]
                labels = info.get('Config', {}).get('Labels') or {}
                if labels.get('dp.val.queue') != QUEUE or labels.get('dp.val.role') != 'scheduler':
                    raise RuntimeError('scheduler name conflict')
                if info['State']['Running']:
                    print('Scheduler already running; no duplicate launched.')
                else:
                    subprocess.run(['docker', 'start', SCHEDULER_CONTAINER], check=True)
            else:
                subprocess.run(command, check=True)
        return
    with open(base/'scheduler.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('another scheduler owns this queue lock')
        scheduler = Scheduler(root, DockerBackend())
        while True:
            try:
                scheduler.tick()
            except Exception as exc:
                scheduler.state['scheduler_error'] = dict(at=now(), message=str(exc))
                scheduler.save()
                scheduler.event('scheduler_poll_error_no_new_launch', error=str(exc))
                if args.action == 'once':
                    raise
            if args.action == 'once':
                return
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
