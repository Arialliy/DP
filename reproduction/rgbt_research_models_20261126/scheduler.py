"""One training job per GPU (0 and 2); GPU 1 stays with the existing Plus run."""
from datetime import datetime
import fcntl
import json
import os
import re
import subprocess
import time
from zoneinfo import ZoneInfo

from runlib import ROOT, docker_command, read_config, save_json

ASSIGN = {'lv_full': 2, 'lvh_full': 0, 'mr_lv': 2, 'lvh_dec': 0}


def inspect(name):
    result = subprocess.run(['docker', 'inspect', '--format', '{{json .State}}', name],
                            capture_output=True, text=True, timeout=20)
    return json.loads(result.stdout) if result.returncode == 0 else None


def idle(gpu):
    result = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
    return result.returncode == 0 and int(result.stdout.strip()) < 2000


def main():
    lock = (ROOT / 'scheduler.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (ROOT / 'scheduler.pid').write_text(str(os.getpid()) + '\n')
    cfg = read_config()
    while True:
        snapshot = {'updated_at': datetime.now(ZoneInfo('Asia/Taipei')).isoformat(), 'jobs': {}}
        busy = set()
        for model in cfg['models']:
            arch = model['arch']
            name = 'dp-rgbt-' + arch.replace('_', '-') + '-20261126'
            container = inspect(name)
            path = ROOT / 'jobs' / (arch + '.json')
            status = json.loads(path.read_text()) if path.exists() else {'state': 'queued'}
            if container and container['Running']:
                busy.add(ASSIGN[arch])
            elif container and status['state'] not in ('completed', 'failed'):
                status.update(state='failed', error='Container exited before completion',
                              exit_code=container['ExitCode'], oom_killed=container['OOMKilled'])
                save_json(path, status)
            if container and not container['Running'] and not (ROOT / 'logs' / (arch + '_console.log')).exists():
                with (ROOT / 'logs' / (arch + '_console.log')).open('w') as file:
                    subprocess.run(['docker', 'logs', name], stdout=file, stderr=subprocess.STDOUT,
                                   check=False, timeout=30)
            snapshot['jobs'][arch] = {'name': model['name'], 'gpu': ASSIGN[arch], 'container_name': name,
                                      'container': container, **status}
        ready_path = ROOT / 'gpu_preflight.json'
        preflight = json.loads(ready_path.read_text()) if ready_path.exists() else {}
        ready = preflight.get('status') == 'passed'
        if preflight.get('status') == 'failed':
            for arch, entry in snapshot['jobs'].items():
                if entry['state'] == 'queued':
                    entry.update(state='failed', error='GPU preflight failed; training was not started. See gpu_preflight.json')
                    save_json(ROOT / 'jobs' / (arch + '.json'), entry)
        for model in cfg['models']:
            arch = model['arch']
            entry = snapshot['jobs'][arch]
            gpu = ASSIGN[arch]
            if entry['state'] != 'queued' or not ready or gpu in busy or not idle(gpu):
                continue
            name = entry['container_name']
            command = docker_command(name, gpu, 'job.py', [arch], detach=True)
            save_json(ROOT / 'jobs' / (arch + '_launch.json'), command)
            process = subprocess.run(command, capture_output=True, text=True, timeout=60)
            if process.returncode:
                entry.update(state='failed', error=process.stderr[-4000:])
                save_json(ROOT / 'jobs' / (arch + '.json'), entry)
            else:
                entry.update(state='starting', container_id=process.stdout.strip())
                busy.add(gpu)
        lines = ['RGB-T 四模型训练', '更新：' + snapshot['updated_at'],
                 '45训练序列 / 15测试序列；32 epochs；seed46；batch4；sample_rate .05；paper_k6_floor。',
                 'GPU 1保留给既有DeepPro-Plus；每张新增GPU一次运行一个模型。',
                 '采用最后一轮final.pth，不使用测试集选择checkpoint。', '']
        for model in cfg['models']:
            arch = model['arch']
            entry = snapshot['jobs'][arch]
            path = ROOT / 'runs' / model['run'] / 'train_log.txt'
            history = path.read_text() if path.exists() else ''
            rows = re.findall(r'^epoch (\d+)/32 .*?loss (\S+).*?iters (\d+) (\d+)s', history, re.M)
            entry['epochs_completed'] = int(rows[-1][0]) if rows else entry.get('epochs_completed', 0)
            if rows:
                entry['last_loss'] = float(rows[-1][1])
                entry['last_epoch_seconds'] = int(rows[-1][3])
            lines.append('%s | GPU %d | %s | %d/32 epochs | %s' %
                         (model['name'], entry['gpu'], entry['state'], entry['epochs_completed'],
                          'loss=' + str(entry['last_loss']) if rows else '等待首轮日志'))
            if 'error' in entry:
                lines.append('  ' + entry['error'])
        save_json(ROOT / 'STATUS.json', snapshot)
        temp = ROOT / 'REPORT.txt.tmp'
        temp.write_text('\n'.join(lines) + '\n')
        temp.replace(ROOT / 'REPORT.txt')
        if all(e['state'] in ('completed', 'failed') for e in snapshot['jobs'].values()):
            return
        time.sleep(20)


if __name__ == '__main__':
    main()
