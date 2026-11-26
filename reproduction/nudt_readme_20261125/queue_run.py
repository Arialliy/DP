"""Reserve an idle GPU and execute untouched official CLI in Docker.

This scheduler does not import Torch, models, datasets, or training code.
"""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

RUN = Path(__file__).resolve().parent
SOURCE = Path('/home/ly/DeepPro-main')
DATA = Path('/home/ly/DeepPro-main/datasets/NUDT-MIRSDT')
NAME = 'deeppro-nudt-readme-20261125'
SPEC = json.loads((RUN / 'env-spec.json').read_text())
IMAGE = (RUN / 'image-id.txt').read_text().strip()
state = {'pid': os.getpid(), 'container': NAME, 'image_id': IMAGE,
         'training_command': SPEC['training_command'], 'source_edits': []}


def save(status, **extra):
    state.update(status=status, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
    path = RUN / 'STATUS.json'
    path.with_suffix('.tmp').write_text(json.dumps(state, indent=2) + '\n')
    path.with_suffix('.tmp').replace(path)


def query_gpus():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True)
    rows = []
    for line in output.splitlines():
        index, uuid, memory, utilization = [x.strip() for x in line.split(',')]
        rows.append({'index': int(index), 'uuid': uuid, 'memory_mib': int(memory),
                     'utilization': int(utilization)})
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                      '--format=csv,noheader'], text=True).splitlines()
    return rows, {x.strip() for x in active}


def source_check():
    for name, expected in json.loads((RUN / 'SOURCE_SHA256.json').read_text()).items():
        if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Source changed: ' + name)
    for split in ('train.txt', 'test.txt'):
        assert (DATA / split).read_bytes() == (SOURCE / 'datasets/NUDT-MIRSDT' / split).read_bytes()


lock = (RUN / 'queue.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
try:
    source_check()
    logdir = RUN / 'log/sem_seg' / SPEC['training_command'][-1]
    if logdir.exists():
        raise RuntimeError('Refusing accidental resume: output directory already exists')
    while True:
        rows, active = query_gpus()
        available = [g for g in rows if g['memory_mib'] < 500 and g['uuid'] not in active]
        if available:
            chosen = available[0]
            # Recheck after a brief delay so a transient process gap is less likely.
            time.sleep(5)
            rows, active = query_gpus()
            ready = next(g for g in rows if g['uuid'] == chosen['uuid'])
            if ready['memory_mib'] < 500 and ready['uuid'] not in active:
                break
        save('waiting_for_idle_gpu', gpu_snapshot=rows)
        time.sleep(30)
    source_check()
    command = ['docker', 'run', '--name', NAME, '--gpus', 'device=' + chosen['uuid'],
               '--network', 'none', '--cpus', '8', '--memory', '32g', '--shm-size', '2g',
               '--user', f'{os.getuid()}:{os.getgid()}', '--entrypoint', 'bash',
               '-v', f'{SOURCE}:/official:ro', '-v', f'{DATA}:/data:ro',
               '-v', f'{RUN}:/run', '-v', f'{RUN}/log:/official/log', '-w', '/official']
    for key, value in SPEC['env'].items():
        command.extend(['-e', f'{key}={value}'])
    # Witness has its own process; its seed has no effect on train.py.
    script = ('python /run/preflight.py --device cuda > /run/gpu_witness.log 2>&1'
              ' && touch /run/GPU_WITNESS_OK'
              ' && exec ' + shlex.join(SPEC['training_command']))
    command.extend([IMAGE, '-lc', script])
    (RUN / 'LAUNCH.json').write_text(json.dumps({'command': command, 'gpu': chosen,
                                                'source_edits': []}, indent=2) + '\n')
    save('gpu_preflight', gpu=chosen)
    with (RUN / 'console.log').open('w') as output:
        proc = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            if (RUN / 'GPU_WITNESS_OK').exists():
                save('training_or_official_final_test', docker_client_pid=proc.pid)
            time.sleep(10)
        exitcode = proc.returncode
    source_check()
    eval_file = logdir / 'eval.txt'
    console = (RUN / 'console.log').read_text(errors='replace')
    finished = (exitcode == 0 and eval_file.exists() and 'Done!' in console
                and 'Epoch 32 (32/32)' in console and (logdir / 'checkpoints/best_model.pth').exists())
    if not finished:
        raise RuntimeError(f'Incomplete official run or test; Docker exit={exitcode}; inspect logs')
    evaluation = eval_file.read_text()
    if evaluation.count('Th_Seg = ') != 27 * 23:
        raise RuntimeError('Official evaluation log does not contain all 20 clips and 3 aggregates')
    groups = {}
    group = None
    for line in evaluation.splitlines():
        header = re.search(r'(Low SNR|High SNR|Final) results:\s*AUC:([0-9.]+)', line)
        if header:
            group = {'Low SNR': 'low_snr', 'High SNR': 'high_snr', 'Final': 'all'}[header[1]]
            groups[group] = {'auc_printed': float(header[2]), 'thresholds': []}
        row = re.search(r'Th_Seg = ([\deE+.-]+):\s*PD:\[(\d+)/(\d+), ([\d.]+)\]\s*FA:\[(\d+), ([\deE+.-]+)\]', line)
        if row and group:
            groups[group]['thresholds'].append({'threshold': float(row[1]), 'true_positive': int(row[2]),
                'ground_truth': int(row[3]), 'pd': int(row[2])/int(row[3]),
                'false_pixels': int(row[5]), 'fa_printed': float(row[6])})
    assert set(groups) == {'low_snr', 'high_snr', 'all'}
    assert all(len(g['thresholds']) == 27 for g in groups.values())
    results = {'status': 'complete', 'groups': groups, 'training_epochs': 32,
               'training_seed': None, 'protocol': 'unmodified current official README defaults',
               'source_edits': [], 'checkpoint_sha256': hashlib.sha256(
                   (logdir / 'checkpoints/best_model.pth').read_bytes()).hexdigest(),
               'note': 'Counts parsed from official log; Fa/AUC retain author print precision.'}
    (RUN / 'RESULTS.json').write_text(json.dumps(results, indent=2) + '\n')
    save('complete', exit_code=exitcode, eval_file=str(eval_file),
         best_checkpoint=str(logdir / 'checkpoints/best_model.pth'))
except Exception as error:
    save('failed', error=repr(error))
    raise
