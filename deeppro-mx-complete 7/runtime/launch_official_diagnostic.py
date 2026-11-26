"""Run only the frozen official checkpoint's inference and scoring on GPU 2."""
import json
import subprocess
from pathlib import Path

import run_v7

ROOT = Path(__file__).resolve().parents[1]
NAME = 'dp-mxc-v7-official-rgbt-20261126'


def main():
    run_v7.verify_check()
    if run_v7.inspect(NAME) is not None:
        raise RuntimeError(f'{NAME} already exists; inspect it before any retry')
    query = subprocess.check_output(
        ['nvidia-smi', '-i', '2', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'],
        text=True).strip().split(',')
    uuid, free_mib = query[0].strip(), int(query[1])
    processes = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    if uuid in processes or free_mib < 22000:
        raise RuntimeError('GPU 2 is not available; no existing job will be stopped')
    args = run_v7.docker_command(NAME)
    args[2:2] = ['--gpus', 'device=2']
    args[args.index('DEVICE=cpu')] = 'DEVICE=cuda'
    entry = args.index('--entrypoint')
    args[entry:entry] = ['-e', 'PYTHONPATH=/experiment/runtime']
    args[-1] = '''set -euo pipefail
unset HOMOGRAPHY_DIR VT_ANN_DIR VT_OLD DET_RUN
python runtime/run_v7.py verify-source
python code/infer.py --run official_rgbt --clips test --device cuda --threads 4 --ckpt inputs/official_rgbt_best_model.pth 2>&1 | tee logs/official_infer.20261126.log
python code/score.py --run official_rgbt --split test --author --nproc 4 2>&1 | tee logs/official_score.20261126.log
'''
    run_v7.save(ROOT / 'runtime/official_diagnostic.command.json', args)
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    run_v7.save(ROOT / 'runtime/official_diagnostic.json', dict(
        status='running', container=NAME, container_id=result.stdout.strip(),
        started_at=run_v7.now(), stages=['official_checkpoint_inference', 'score'],
        training=False, gpu=2, source_manifest_sha256=run_v7.digest(run_v7.MANIFEST),
        launcher_sha256=run_v7.digest(Path(__file__))))
    print(result.stdout.strip())


if __name__ == '__main__':
    main()
