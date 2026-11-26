"""Run one CPU validation stage in the recorded Docker environment."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REL = HERE.relative_to(ROOT)
DATA = Path('/home/md0/ly/DP/datasets/RGB-T')
IMAGE = 'sha256:8470a68886ff2f480d76858d930fb72d0e680e69276fc1d068fe8b6625b4cb9f'
COMMANDS = {
    'check_labels': ['python', 'code/check_labels.py', '--nproc', '8', '--out', str(REL / 'labels.json')],
    'profile': ['python', 'code/profile_motion.py'],
    'folds': ['python', 'code/folds.py'],
    'm0': ['python', 'code/exemplar_diag.py', '--split', 'val', '--nproc', '4'],
}


def now():
    return datetime.now(ZoneInfo('Asia/Taipei')).isoformat()


def update_manifest(stage, record):
    path = HERE / 'run_manifest.json'
    data = json.loads(path.read_text())
    data['stages'][stage] = record
    data['status'] = stage + '_' + record['status']
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=COMMANDS)
    stage = ap.parse_args().stage
    cmd = ['docker', 'run', '--rm', '--name', 'dp-mx-v5-' + stage.replace('_', '-') + '-20261126',
           '--network', 'none', '--user', f'{os.getuid()}:{os.getgid()}',
           '--cpus', '8', '--memory', '24g', '--shm-size', '2g']
    env = dict(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OPENBLAS_NUM_THREADS='1',
               OPENCV_FOR_THREADS_NUM='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NPROC='8',
               VT_ROOT='/data/RGB-T', OFFICIAL_LISTS='/data/RGB-T', VT_ANN_DIR='/data/RGB-T/annotations_coco')
    for k, v in env.items():
        cmd += ['-e', f'{k}={v}']
    cmd += ['-v', f'{ROOT}:/work', '-v', f'{DATA}:/data/RGB-T:ro', '-w', '/work', IMAGE]
    cmd += COMMANDS[stage]
    record = dict(status='running', started_at=now(), command=cmd, log=f'{stage}.log')
    if stage == 'profile' and (ROOT / 'results/profile').exists():
        backup = ROOT / 'results' / ('profile.before.' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        shutil.move(str(ROOT / 'results/profile'), str(backup))
        record['backup'] = str(backup)
    if stage == 'folds' and (ROOT / 'configs/folds.json').exists():
        backup = HERE / ('folds.before.' + datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.json')
        shutil.copy2(ROOT / 'configs/folds.json', backup)
        record['backup'] = str(backup)
    update_manifest(stage, record)
    with (HERE / f'{stage}.log').open('w') as log:
        result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    record.update(ended_at=now(), exit_code=result.returncode,
                  status='passed' if result.returncode == 0 else ('no_go' if stage == 'm0' and result.returncode == 3 else 'failed'))
    update_manifest(stage, record)
    print(json.dumps(record, indent=2))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
