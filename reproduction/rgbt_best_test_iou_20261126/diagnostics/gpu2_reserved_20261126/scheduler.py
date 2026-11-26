"""GPU 2: LV then MRTP; GPU 0: LVH then LVH-dec. GPU 1 is excluded."""
import argparse
from datetime import datetime
import fcntl
import json
import math
import os
import re
import subprocess
import time
from zoneinfo import ZoneInfo

from runlib import ROOT, container_name, docker_command, read_config, read_json, save_json

TERMINAL = {'completed', 'failed'}


def inspect(name):
    result = subprocess.run(['docker', 'inspect', '--format', '{{json .State}}', name],
                            capture_output=True, text=True, timeout=20)
    return json.loads(result.stdout) if result.returncode == 0 else None


def idle(gpu):
    result = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
    return result.returncode == 0 and int(result.stdout.strip()) < 2000


def run_progress(model):
    out = ROOT / 'runs' / model['run']
    history_path = out / 'train_log.txt'
    history = history_path.read_text() if history_path.exists() else ''
    rows = re.findall(r'^epoch (\d+)/(\d+) lr \S+ loss (\S+) iters (\d+) (\d+)s', history, re.M)
    result = {'train_epochs_completed': len(rows), 'eval_epochs_completed': 0}
    if rows:
        result.update(last_loss=float(rows[-1][2]), last_train_seconds=int(rows[-1][4]))
    eval_path = out / 'eval_log.jsonl'
    if eval_path.exists():
        evaluations = []
        for line in eval_path.read_text().splitlines():
            try:
                evaluations.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # A writer may currently be appending the final line.
        complete = [row for row in evaluations if row.get('complete', True)]
        result['eval_epochs_completed'] = len(complete)
        if complete:
            best = max(complete, key=lambda row: (row['iou'], row['epoch']))
            result.update(last_eval_iou=complete[-1]['iou'], best_iou=best['iou'], best_epoch=best['epoch'])
    progress_path = out / 'progress.json'
    if progress_path.exists():
        try:
            result.update(json.loads(progress_path.read_text()))
        except json.JSONDecodeError:
            result['progress_warning'] = 'progress.json is currently being written'
    return result


def write_report(snapshot, cfg):
    lines = ['RGB-T 四模型：每轮测试 IoU 选择最佳权重', '更新：' + snapshot['updated_at'],
             '保留研究数据流程、模型和训练配置；45训练 / 15测试；32 epochs；seed46；batch4。',
             '每轮完整测试172窗口，sigmoid > 0.5，全局IoU；同分保存较晚轮次。',
             'GPU 2: LV → MRTP；GPU 0: LVH → LVH-dec；GPU 1未使用。', '']
    for model in cfg['models']:
        entry = snapshot['jobs'][model['arch']]
        extra = ''
        if entry.get('stage') == 'eval':
            extra = ' | 评估窗口 %s/%s | %s' % (entry.get('eval_windows_done', 0),
                                               entry.get('eval_windows_total', 172), entry.get('eval_clip') or '')
        best = '未产生' if entry.get('best_epoch') is None else '%.6f @ epoch %s' % (
            entry['best_iou'], entry['best_epoch'])
        lines.append('%s | GPU %s | %s/%s | 训练 %s/32 | 评估 %s/32 | best %s%s' % (
            model['name'], entry['gpu'], entry['state'], entry.get('stage', 'waiting'),
            entry.get('train_epochs_completed', 0), entry.get('eval_epochs_completed', 0), best, extra))
        if entry.get('error'):
            lines.append('  ' + entry['error'])
    if snapshot.get('readiness_message'):
        lines.extend(['', snapshot['readiness_message']])
    save_json(ROOT / 'STATUS.json', snapshot)
    temp = ROOT / 'REPORT.txt.tmp'
    temp.write_text('\n'.join(lines) + '\n')
    temp.replace(ROOT / 'REPORT.txt')


def cycle(cfg, allow_launch):
    snapshot = {'updated_at': datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
                'checkpoint_selection': cfg['checkpoint_selection'], 'jobs': {}}
    busy = set()
    for model in cfg['models']:
        arch = model['arch']
        name = container_name(arch)
        gpu = cfg['gpu_assignment'][arch]
        container = inspect(name)
        path = ROOT / 'jobs' / (arch + '.json')
        status = read_json(path, {'state': 'queued'})
        if container and container['Running']:
            busy.add(gpu)
        elif container and container['Status'] in ('exited', 'dead'):
            if status['state'] not in TERMINAL or container['ExitCode'] != 0:
                status.update(state='failed', error='Container exited before successful completion',
                              exit_code=container['ExitCode'], oom_killed=container['OOMKilled'])
                save_json(path, status)
            console = ROOT / 'logs' / (arch + '_console.log')
            if not console.exists():
                with console.open('w') as stream:
                    subprocess.run(['docker', 'logs', name], stdout=stream, stderr=subprocess.STDOUT,
                                   check=False, timeout=30)
        progress = run_progress(model)
        entry = {'name': model['name'], 'gpu': gpu, 'container_name': name,
                 'container': container, **status, **progress}
        if status['state'] in TERMINAL:
            entry['stage'] = status['state']
        snapshot['jobs'][arch] = entry
    preflight = read_json(ROOT / 'gpu_preflight.json', {})
    ready = preflight.get('status') == 'passed'
    if not ready:
        snapshot['readiness_message'] = '等待新代码的GPU预检通过；未启动排队任务。'
    if preflight.get('status') == 'failed':
        snapshot['readiness_message'] = 'GPU预检失败；排队任务保留，等待修复并重新预检。'
    if allow_launch and ready:
        for model in cfg['models']:
            arch = model['arch']
            entry = snapshot['jobs'][arch]
            gpu = entry['gpu']
            if entry['state'] != 'queued' or gpu in busy or not idle(gpu):
                continue
            if entry['container'] is not None:
                entry.update(state='failed', error='Existing container name; refusing automatic replacement')
                save_json(ROOT / 'jobs' / (arch + '.json'), entry)
                continue
            command = docker_command(entry['container_name'], gpu, 'job.py', [arch], detach=True)
            save_json(ROOT / 'jobs' / (arch + '_launch.json'), command)
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            if result.returncode:
                state = {'state': 'failed', 'arch': arch, 'error': result.stderr[-4000:]}
                save_json(ROOT / 'jobs' / (arch + '.json'), state)
                entry.update(state)
            else:
                entry.update(state='starting', container_id=result.stdout.strip())
                busy.add(gpu)
                # job.py owns its status file once the container starts; avoid a write race.
    write_report(snapshot, cfg)
    return all(entry['state'] in TERMINAL and not (entry.get('container') or {}).get('Running', False)
               for entry in snapshot['jobs'].values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status-only', action='store_true', help='Refresh reports once; never launch jobs')
    args = parser.parse_args()
    cfg = read_config()
    if args.status_only:
        cycle(cfg, allow_launch=False)
        return
    lock = (ROOT / 'scheduler.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (ROOT / 'scheduler.pid').write_text(str(os.getpid()) + '\n')
    while not cycle(cfg, allow_launch=True):
        time.sleep(20)


if __name__ == '__main__':
    main()
