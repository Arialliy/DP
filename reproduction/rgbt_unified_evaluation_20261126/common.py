"""Small host/container helpers for the isolated evaluation queue."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.%d.tmp' % os.getpid())
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def config():
    return read_json(ROOT / 'config.json')


def verify_source():
    manifest = read_json(ROOT / 'source_manifest.json')
    if not manifest:
        raise RuntimeError('Evaluation source manifest missing')
    for name, digest in manifest['files'].items():
        assert sha256(ROOT / name) == digest, 'Evaluation source changed: ' + name


def docker_command(name, script, arguments=(), *, gpu=None, detach=False):
    cfg = config()
    if gpu is not None and int(gpu) not in cfg['allowed_gpus']:
        raise ValueError('GPU not authorized for evaluation')
    command = ['docker', 'run', '--name', name, '--network', 'none',
               '--user', f'{os.getuid()}:{os.getgid()}', '--cpus', '8', '--memory', '48g',
               '--shm-size', '8g']
    if gpu is not None:
        command += ['--gpus', 'device=' + str(gpu)]
    mounts = [(str(ROOT), '/evaluation', False), (str(ROOT / 'code'), '/evaluation/code', True),
              (str(ROOT / 'official'), '/evaluation/official', True),
              (cfg['dataset'], '/data/RGB-T', True), (cfg['artifacts'], '/artifacts', True),
              (cfg['official_source'], '/official-source', True)]
    for source, target, readonly in mounts:
        command += ['-v', source + ':' + target + (':ro' if readonly else '')]
    for value in ['PYTHONDONTWRITEBYTECODE=1', 'PYTHONUNBUFFERED=1', 'OMP_NUM_THREADS=4',
                  'MKL_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=1', 'OPENCV_FOR_THREADS_NUM=1',
                  'VT_ROOT=/data/RGB-T', 'OFFICIAL_LISTS=/data/RGB-T',
                  'VT_ANN_DIR=/data/RGB-T/annotations_coco', 'OFFICIAL_SOURCE=/official-source',
                  'HOMOGRAPHY_DIR=/artifacts/results/profile/homographies',
                  'PYTHONPATH=/evaluation:/evaluation/code', 'MPLCONFIGDIR=/tmp/mpl', 'TZ=Asia/Taipei']:
        command += ['-e', value]
    command += ['-w', '/evaluation']
    command += ['--detach'] if detach else ['--rm']
    return command + ['--entrypoint', 'python', cfg['image'], '-u', '/evaluation/' + script, *arguments]
