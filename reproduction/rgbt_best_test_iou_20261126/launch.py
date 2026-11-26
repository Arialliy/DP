"""Print the launch plan, or explicitly start the reviewed scheduler with --start."""
import argparse
import fcntl
import json
import subprocess
import sys

from runlib import ROOT, container_name, docker_command, read_config, read_json, verify_artifacts, verify_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', action='store_true', help='Start detached scheduler after source/data/preflight checks')
    args = parser.parse_args()
    cfg = read_config()
    plan = [{'arch': model['arch'], 'gpu': cfg['gpu_assignment'][model['arch']],
             'container': container_name(model['arch']),
             'command': docker_command(container_name(model['arch']), cfg['gpu_assignment'][model['arch']],
                                       'job.py', [model['arch']], detach=True)} for model in cfg['models']]
    if not args.start:
        print(json.dumps({'mode': 'preview_only', 'output_root': str(ROOT), 'jobs': plan}, indent=2))
        return
    verify_source()
    verify_artifacts()
    if read_json(ROOT / 'gpu_preflight.json', {}).get('status') != 'passed':
        raise SystemExit('The new code must pass gpu_preflight.json before starting this batch')
    lock = (ROOT / 'scheduler.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('This batch already has a live scheduler')
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    with (ROOT / 'logs/scheduler.log').open('ab') as log:
        proc = subprocess.Popen([sys.executable, '-u', str(ROOT / 'scheduler.py')], cwd=ROOT,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    print(json.dumps({'scheduler_pid': proc.pid, 'report': str(ROOT / 'REPORT.txt'),
                      'status': str(ROOT / 'STATUS.json')}, indent=2))


if __name__ == '__main__':
    main()
