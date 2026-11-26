"""Configuration and read-only artifact mounts for the best-test-IoU rerun."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_config():
    return json.loads((ROOT / 'experiment.json').read_text())


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.%d.tmp' % os.getpid())
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source():
    manifest = read_json(ROOT / 'source_manifest.json')
    if not manifest or not manifest.get('files'):
        raise RuntimeError('The reviewed code snapshot/source_manifest.json is not ready')
    for name, expected in manifest['files'].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError('Source hash mismatch: ' + name)
    if 'code/train.py' not in manifest['files'] or 'code/checkpoint_selection.py' not in manifest['files']:
        raise RuntimeError('Manifest must cover trainer and checkpoint-selection module')
    cfg = read_config()
    split = read_json(ROOT / 'configs/splits.json')
    assert split['official_train'] == cfg['train_clips']
    assert split['official_test'] == cfg['test_clips']
    assert len(cfg['train_clips']) == 45 and len(cfg['test_clips']) == 15
    assert not set(cfg['train_clips']) & set(cfg['test_clips'])
    assert set(cfg['gpu_assignment'].values()) == {0, 2}


def verify_artifacts():
    cfg = read_config()
    reused = Path('/reused') if ROOT == Path('/experiment') else Path(cfg['artifact_sources']['root'])
    data = read_json(reused / 'data_ready.json')
    hot = read_json(reused / 'hotspots_ready.json')
    assert data['status'] == hot['status'] == 'ready'
    assert data['clips'] == 60 and data['frames'] == 25111 and hot['clips'] == 45
    expected_clips = set(cfg['train_clips'] + cfg['test_clips'])
    assert {row['clip'] for row in data['records']} == expected_clips
    for row in data['records']:
        assert (reused / row['cache']).stat().st_size == row['cache_bytes'], row['clip']
        assert sha256(reused / row['homography']) == row['homography_sha256'], row['clip']
    assert {row['clip'] for row in hot['records']} == set(cfg['train_clips'])
    for row in hot['records']:
        assert sha256(reused / row['path']) == row['sha256'], row['clip']
    return {'cache_clips': data['clips'], 'frames': data['frames'], 'hotspot_clips': hot['clips'],
            'hotspot_candidates': hot['candidates'], 'video_check': 'size against prior full-SHA256 witness',
            'homography_and_hotspot_check': 'SHA256', 'reuse_mode': 'Docker read-only bind mounts'}


def train_command(model, *, smoke=False):
    # These training options are identical to the previous research workflow.
    cfg = read_config()
    command = ['python', '-u', 'train_entry.py', '--run',
               'preflight_' + model['arch'] if smoke else model['run'],
               '--arch', model['arch'], '--epochs', '1' if smoke else str(cfg['epochs']),
               '--seed', str(cfg['seed']), '--batch', str(cfg['batch']),
               '--lr', str(cfg['lr']), '--sample_rate', '0.00025' if smoke else str(cfg['sample_rate']),
               '--label-rule', cfg['label_rule'], '--split', cfg['train_split'],
               '--device', 'cuda:0', '--workers', '0' if smoke else '3', '--threads', '4',
               '--aux', '0.5', '--main-focal', '0',
               '--checkpoint-selection', 'original_test_iou',
               '--hist-len', '80', '--hist-step', '4', '--hist-mode', 'both']
    return command + model['extra']


def container_name(arch):
    return 'dp-rgbt-best-test-iou-' + arch.replace('_', '-') + '-20261126'


def docker_command(name, gpu, script, script_args=(), detach=False):
    cfg = read_config()
    if int(gpu) not in (0, 2):
        raise ValueError('Only GPUs 0 and 2 are assigned to this batch')
    artifacts = cfg['artifact_sources']
    mounts = [(str(ROOT), '/experiment', False), (str(ROOT / 'code'), '/experiment/code', True),
              (str(ROOT / 'configs'), '/experiment/configs', True),
              (cfg['dataset'], '/data/RGB-T', True), (artifacts['root'], '/reused', True),
              (artifacts['video_cache'], '/experiment/cache/video', True),
              (artifacts['hotspots'], '/experiment/cache/hotspots', True),
              (artifacts['homographies'], '/experiment/results/profile/homographies', True)]
    cmd = ['docker', 'run', '--name', name, '--network', 'none', '--gpus', 'device=' + str(gpu),
           '--user', '%d:%d' % (os.getuid(), os.getgid()), '--cpus', '8', '--memory', '48g',
           '--shm-size', '8g']
    for source, target, readonly in mounts:
        cmd += ['-v', source + ':' + target + (':ro' if readonly else '')]
    cmd += ['-w', '/experiment']
    for value in ['VT_ROOT=/data/RGB-T', 'OFFICIAL_LISTS=/data/RGB-T',
                  'VT_ANN_DIR=/data/RGB-T/annotations_coco',
                  'HOMOGRAPHY_DIR=/experiment/results/profile/homographies',
                  'OMP_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=1',
                  'OPENCV_FOR_THREADS_NUM=1', 'PYTHONDONTWRITEBYTECODE=1',
                  'PYTHONUNBUFFERED=1', 'MPLCONFIGDIR=/tmp/mpl']:
        cmd += ['-e', value]
    cmd += ['--detach'] if detach else ['--rm']
    return cmd + ['--entrypoint', 'python', cfg['image'], '-u', '/experiment/' + script, *script_args]
