"""Single-use, check-gated v7 CPU profile. Never launches GPU work or training."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'runtime'
DATA = Path('/home/md0/ly/DP/datasets/RGB-T')
IMAGE = 'sha256:ea2c24a3a72cad22047223725869da1044de36feb1c5d0a19f23544a9cc3e594'
CPU_NAME = 'dp-mxc-v7-prepare-20261126'
MANIFEST = RUNTIME / 'source_manifest.json'
CHECK = RUNTIME / 'check_passed.json'
STATUS = RUNTIME / 'status.json'


def now():
    return datetime.now(ZoneInfo('Asia/Taipei')).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def source_files():
    paths = list((ROOT / 'code').glob('*.py'))
    paths += [ROOT / 'run_mx.sh', ROOT / 'configs/splits.json']
    paths += [RUNTIME / name for name in ('run_v7.py', 'sitecustomize.py', 'torch_compat.py',
                                         'torch_compat.PYTORCH_LICENSE', 'torch_compat_provenance.json')]
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def verify_source():
    expected = json.loads(MANIFEST.read_text())
    if expected['files'] != source_files() or expected['image'] != IMAGE:
        raise RuntimeError('Source/image differs from source_manifest.json; no launch')
    return digest(MANIFEST)


def verify_check():
    manifest_sha = verify_source()
    record = json.loads(CHECK.read_text())
    if record.get('status') != 'passed' or record.get('exit_code') != 0:
        raise RuntimeError('A successful complete run_mx.sh check is required')
    if record.get('source_manifest_sha256') != manifest_sha:
        raise RuntimeError('Check evidence belongs to a different source manifest')
    log = ROOT / record['log']
    if digest(log) != record['log_sha256']:
        raise RuntimeError('Recorded check log changed')


def command(args, **kwargs):
    return subprocess.run(args, capture_output=True, text=True, timeout=30, **kwargs)


def inspect(name):
    result = command(['docker', 'inspect', '--format', '{{json .State}}', name])
    if result.returncode:
        if 'No such object' in result.stderr or 'No such container' in result.stderr:
            return None
        raise RuntimeError('docker inspect failed: ' + result.stderr[-2000:])
    return json.loads(result.stdout)


def docker_command(name):
    args = ['docker', 'run', '--detach', '--name', name, '--network', 'none',
            '--user', f'{os.getuid()}:{os.getgid()}', '--cpus', '8', '--memory', '48g', '--shm-size', '8g']
    # configs remains writable for profile's generated folds.json; supplied splits are read-only.
    for host, target, ro in ((ROOT, '/experiment', False), (ROOT / 'code', '/experiment/code', True),
                             (ROOT / 'run_mx.sh', '/experiment/run_mx.sh', True),
                             (ROOT / 'configs/splits.json', '/experiment/configs/splits.json', True),
                             (RUNTIME, '/experiment/runtime', True), (DATA, '/data/RGB-T', True)):
        args += ['-v', f'{host}:{target}' + (':ro' if ro else '')]
    args += ['-w', '/experiment']
    env = ['VT_ROOT=/data/RGB-T', 'OFFICIAL_LISTS=/data/RGB-T', 'OMP_NUM_THREADS=4',
           'MKL_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=1', 'OPENCV_FOR_THREADS_NUM=1',
           'PYTHONDONTWRITEBYTECODE=1', 'PYTHONUNBUFFERED=1', 'MPLCONFIGDIR=/tmp/mpl',
           'TZ=Asia/Taipei', 'NPROC=8', 'FOLDS=1', 'PY=python', 'DEVICE=cpu']
    for entry in env:
        args += ['-e', entry]
    verify = 'python /experiment/runtime/run_v7.py verify-source'
    clear = 'unset HOMOGRAPHY_DIR VT_ANN_DIR VT_OLD DET_RUN; '
    stages = f'{verify} && bash run_mx.sh profile && {verify}'
    return args + ['--entrypoint', 'bash', IMAGE, '-c', clear + stages]


def execute_container(name, state):
    verify_check()
    if inspect(name) is not None:
        raise RuntimeError(f'Container {name} already exists; refusing relaunch')
    args = docker_command(name)
    save(RUNTIME / (name + '.command.json'), args)
    result = command(args)
    if result.returncode:
        raise RuntimeError('Container launch failed: ' + result.stderr[-4000:])
    state.update(stage='profile', container=name,
                 container_id=result.stdout.strip(), updated_at=now())
    save(STATUS, state)
    with (RUNTIME / (name + '.log')).open('w') as stream:
        logger = subprocess.Popen(['docker', 'logs', '--follow', name], stdout=stream, stderr=subprocess.STDOUT)
        try:
            while True:
                container = inspect(name)
                if container is None:
                    raise RuntimeError(f'Container disappeared: {name}')
                state.update(updated_at=now(), container_state=container)
                save(STATUS, state)
                if not container['Running']:
                    logger.wait(timeout=30)
                    if container.get('ExitCode') != 0:
                        raise RuntimeError(f'{name} exited {container.get("ExitCode")}; see its runtime log')
                    return
                time.sleep(20)
        finally:
            if logger.poll() is None:
                logger.terminate()  # Stop only this log reader; never stop a training container.


def start():
    with (RUNTIME / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if STATUS.exists():
            raise RuntimeError('runtime/status.json exists; this runner is single-use')
        verify_check()
        if inspect(CPU_NAME) is not None:
            raise RuntimeError(f'Container already exists: {CPU_NAME}')
        state = dict(status='running', stage='starting', started_at=now(), updated_at=now(),
                     pid=os.getpid(), source_manifest_sha256=digest(MANIFEST), image=IMAGE)
        save(STATUS, state)
        try:
            execute_container(CPU_NAME, state)
            artifacts = ['results/profile/targets.csv', 'results/profile/targets.json',
                         'results/motion/clips.csv', 'results/motion/motion.json', 'configs/folds.json']
            for path in artifacts:
                if not (ROOT / path).is_file():
                    raise RuntimeError('Missing profile output: ' + path)
            state.update(status='completed', stage='profile_completed', updated_at=now(), artifacts=artifacts)
            save(STATUS, state)
        except BaseException as error:
            state.update(status='failed', error=repr(error), updated_at=now())
            save(STATUS, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'record-check', 'verify-source', 'preview', 'start'))
    parser.add_argument('--log', help='Completed check log, relative to project root')
    parser.add_argument('--exit-code', type=int, help='Observed full check process exit code')
    args = parser.parse_args()
    if args.action == 'prepare':
        if MANIFEST.exists() or CHECK.exists() or STATUS.exists():
            raise RuntimeError('Manifest/evidence/status already exists; refusing to replace it')
        save(MANIFEST, dict(created_at=now(), image=IMAGE, files=source_files()))
    elif args.action == 'record-check':
        if args.exit_code != 0 or not args.log:
            raise RuntimeError('Provide the observed successful full check exit code and its completed log')
        if CHECK.exists() or STATUS.exists():
            raise RuntimeError('Check record or launch status already exists')
        manifest_sha = verify_source()
        log = (ROOT / args.log).resolve()
        relative = str(log.relative_to(ROOT))
        save(CHECK, dict(status='passed', exit_code=0, recorded_at=now(), log=relative,
                         log_sha256=digest(log), source_manifest_sha256=manifest_sha,
                         command='bash run_mx.sh check', complete_suite=True))
    elif args.action == 'verify-source':
        print(verify_source())
    elif args.action == 'preview':
        print(json.dumps(dict(cpu=docker_command(CPU_NAME)), indent=2))
    else:
        start()


if __name__ == '__main__':
    main()
