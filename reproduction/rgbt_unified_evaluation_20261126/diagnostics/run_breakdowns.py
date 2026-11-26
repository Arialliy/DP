"""Run three isolated GPU2 diagnostics, then restore the regular evaluator."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import config, docker_command, save_json

JOBS = ['author_released_reference', 'deeppro_plus_416648f66cbee93f',
        'deeppro_plus_c1c818212eb0c1d8']


def main():
    own_lock = (ROOT / 'diagnostics/breakdown_queue.lock').open('a')
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = dict(pid=os.getpid(), jobs=JOBS, state='waiting', gpu=2)
    path = ROOT / 'diagnostics/breakdown_queue.status.json'

    def update(**kwargs):
        status.update(kwargs, updated_at=datetime.now(timezone.utc).isoformat())
        save_json(path, status)

    assert config()['allowed_gpus'] == [2]
    update()
    # Do not interrupt an inference container or compete with other GPU users.
    while True:
        running = subprocess.check_output(['docker', 'ps', '--format', '{{.Names}}'], text=True)
        mem = int(subprocess.check_output(['nvidia-smi', '-i', '2', '--query-gpu=memory.used',
                                         '--format=csv,noheader,nounits'], text=True).strip())
        if mem < 2000 and 'dp-rgbt-unified-eval-' not in running:
            break
        time.sleep(20)
    pid = int((ROOT / 'scheduler.pid').read_text())
    procfile = Path(f'/proc/{pid}/cmdline')
    if procfile.exists():
        assert str(ROOT / 'scheduler.py').encode() in procfile.read_bytes()
        os.kill(pid, signal.SIGTERM)
        for _ in range(50):
            if not procfile.exists():
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Regular evaluator did not stop')
    lock = (ROOT / 'scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    update(state='running', paused_scheduler_pid=pid)
    try:
        for job in JOBS:
            # Recheck after acquiring the scheduler lock: a regular evaluation
            # might have launched between the initial observation and shutdown.
            while True:
                mem = int(subprocess.check_output(['nvidia-smi', '-i', '2', '--query-gpu=memory.used',
                                                 '--format=csv,noheader,nounits'], text=True).strip())
                if mem < 2000:
                    break
                update(stage='waiting_for_gpu2', current_job=job)
                time.sleep(20)
            update(stage='inference_and_diagnostics', current_job=job)
            command = docker_command('dp-rgbt-breakdown-' + job.replace('_', '-'),
                                     'breakdown.py', [job, '--threshold', '0.99'], gpu=2)
            save_json(ROOT / 'diagnostics' / (job + '.breakdown.launch.json'), command)
            with (ROOT / 'logs' / (job + '.breakdown.log')).open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{job} breakdown failed with {result.returncode}')
            update(completed_jobs=status.get('completed_jobs', []) + [job])
        compare = ROOT.parents[1] / 'code/fa_compare.py'
        command = [sys.executable, str(compare), '--runs', 'official_rgbt',
                   'deeppro_plus', 'deeppro_plus_interim', '--split', 'test',
                   '--fa', '4.467025818116215e-5', '4.5281018278844545e-4',
                   '6.136428240631799e-4', '--breakdown', '--threshold', '0.99']
        with (ROOT / 'diagnostics/fa_compare.final.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        update(state='completed')
    except BaseException as error:
        update(state='failed', error=repr(error))
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        with (ROOT / 'logs/scheduler.log').open('ab') as log:
            proc = subprocess.Popen([sys.executable, '-u', str(ROOT / 'scheduler.py')],
                                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        update(restored_scheduler_pid=proc.pid)


if __name__ == '__main__':
    main()
