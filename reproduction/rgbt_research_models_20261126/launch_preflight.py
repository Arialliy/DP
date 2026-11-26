"""Wait for verified CPU assets, then run the real GPU checks before dispatch."""
import json
import subprocess
import time
import traceback

from runlib import ROOT, docker_command, save_json


def main():
    while True:
        ready = True
        for name in ('cache_ready.json', 'hotspots_ready.json'):
            path = ROOT / name
            if not path.exists():
                ready = False
            elif json.loads(path.read_text())['status'] not in ('ready', 'passed'):
                raise RuntimeError('Preprocessing prerequisite failed: ' + name)
        result = subprocess.run(['nvidia-smi', '-i', '2', '--query-gpu=memory.used',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
        gpu_ready = result.returncode == 0 and int(result.stdout.strip()) < 2000
        if ready and gpu_ready:
            break
        time.sleep(10)
    command = docker_command('dp-rgbt-four-preflight-20261126', 2, 'gpu_preflight.py')
    save_json(ROOT / 'gpu_preflight_launch.json', command)
    subprocess.run(command, check=True, cwd=ROOT)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        path = ROOT / 'gpu_preflight.json'
        previous = json.loads(path.read_text()) if path.exists() else {}
        if previous.get('status') != 'failed':
            save_json(path, {'status': 'failed', 'error': repr(exc), 'traceback': traceback.format_exc()})
        raise
