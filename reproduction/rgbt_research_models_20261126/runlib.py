"""Shared configuration for this four-model RGB-T training batch."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_config():
    return json.loads((ROOT / 'experiment.json').read_text())


def save_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.%d.tmp' % os.getpid())
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def verify_source():
    manifest = json.loads((ROOT / 'source_manifest.json').read_text())
    for name, sha in manifest['files'].items():
        if name == 'configs/splits.json':
            continue  # Only the two explicitly documented split aliases were added.
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
    cfg = read_config()
    splits = json.loads((ROOT / 'configs/splits.json').read_text())
    assert splits['official_train'] == cfg['train_clips']
    assert splits['official_test'] == cfg['test_clips']
    assert len(cfg['train_clips']) == 45 and len(cfg['test_clips']) == 15
    assert not set(cfg['train_clips']) & set(cfg['test_clips'])


def train_command(model, *, smoke=False):
    cfg = read_config()
    command = ['python', '-u', 'train_entry.py', '--run',
               'preflight_' + model['arch'] if smoke else model['run'],
               '--arch', model['arch'], '--epochs', '1' if smoke else str(cfg['epochs']),
               '--seed', str(cfg['seed']), '--batch', str(cfg['batch']),
               '--lr', str(cfg['lr']), '--sample_rate', '0.00025' if smoke else str(cfg['sample_rate']),
               '--label-rule', cfg['label_rule'], '--split', cfg['train_split'],
               '--device', 'cuda:0', '--workers', '0' if smoke else '3', '--threads', '4',
               '--aux', '0.5', '--main-focal', '0',
               '--hist-len', '80', '--hist-step', '4', '--hist-mode', 'both']
    return command + model['extra']


def docker_command(name, gpu, script, script_args=(), detach=False):
    cfg = read_config()
    cmd = ['docker', 'run', '--name', name, '--network', 'none', '--gpus', 'device=' + str(gpu),
           '--user', '%d:%d' % (os.getuid(), os.getgid()), '--cpus', '8', '--memory', '48g',
           '--shm-size', '8g', '-v', str(ROOT) + ':/experiment',
           '-v', str(ROOT / 'code') + ':/experiment/code:ro',
           '-v', str(ROOT / 'configs') + ':/experiment/configs:ro',
           '-v', cfg['dataset'] + ':/data/RGB-T:ro', '-w', '/experiment']
    for value in ['VT_ROOT=/data/RGB-T', 'OFFICIAL_LISTS=/data/RGB-T',
                  'VT_ANN_DIR=/data/RGB-T/annotations_coco',
                  'HOMOGRAPHY_DIR=/experiment/results/profile/homographies',
                  'OMP_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=1',
                  'OPENCV_FOR_THREADS_NUM=1', 'PYTHONDONTWRITEBYTECODE=1',
                  'PYTHONUNBUFFERED=1', 'MPLCONFIGDIR=/tmp/mpl']:
        cmd += ['-e', value]
    cmd += ['--detach'] if detach else ['--rm']
    return cmd + ['--entrypoint', 'python', cfg['image'], '-u', '/experiment/' + script, *script_args]
