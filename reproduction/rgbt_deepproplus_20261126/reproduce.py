"""Host launcher: immutable source/data mounts, isolated outputs, checked jobs."""
import argparse
import json
import os
from pathlib import Path
import subprocess

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['preflight', 'released', 'train', 'status'])
    parser.add_argument('--gpu', default='1')
    args = parser.parse_args()
    if args.phase == 'status':
        for phase in ['released', 'train']:
            name = 'deeppro-rgbt-' + phase + '-20261126'
            inspect = subprocess.run(['docker', 'inspect', '--format', '{{json .State}}', name], capture_output=True, text=True)
            path = RUN / (phase + '_status.json')
            print(json.dumps({'phase': phase, 'pipeline': json.loads(path.read_text()) if path.exists() else None,
                              'container': json.loads(inspect.stdout) if inspect.returncode == 0 else None}, indent=2))
        return
    name = 'deeppro-rgbt-' + args.phase + '-20261126'
    image_record = RUN / 'runtime/image-id.txt'
    image = image_record.read_text().strip() if image_record.exists() else 'deeppro-rgbt:20261126'
    command = ['docker', 'run', '--name', name, '--network', 'none', '--gpus', 'device=' + args.gpu,
               '--cpus', '8', '--memory', '48g', '--shm-size', '8g',
               '--user', str(os.getuid()) + ':' + str(os.getgid()),
               '-v', str(RUN / 'source') + ':/work:ro',
               '-v', str(ROOT / 'datasets/RGB-T') + ':/data/RGB-T:ro',
               '-v', str(RUN) + ':/run', '-w', '/work']
    for variable in ['OMP_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4',
                     'PYTHONDONTWRITEBYTECODE=1', 'PYTHONUNBUFFERED=1', 'MPLCONFIGDIR=/tmp/mpl',
                     'PYTHONPATH=/work', 'TZ=Asia/Taipei']:
        command += ['-e', variable]
    if args.phase != 'preflight':
        command += ['--detach']
    else:
        command += ['--rm']
    command += ['--entrypoint', 'python', image, '-u',
                '/run/preflight.py' if args.phase == 'preflight' else '/run/pipeline.py']
    if args.phase != 'preflight':
        command.append(args.phase)
    (RUN / (args.phase + '_launch.json')).write_text(json.dumps(command, indent=2) + '\n')
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
