"""Evaluate current best snapshots, then automatically finalize all 32-epoch runs."""
from datetime import datetime, timezone
import fcntl
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
import zlib

from common import ROOT, config, docker_command, read_json, save_json, sha256, verify_source


def now():
    return datetime.now(timezone.utc).isoformat()


def training_state(model):
    source = Path(model['source_root'])
    if model['source_kind'] == 'baseline':
        status = read_json(source / 'train_status.json', {})
        log = source / 'log/sem_seg/retrained_rgbt_deepproplus/logs/DeepPro-Plus.txt'
        count = len(re.findall(r'Eval avg class IoU of prediction:', log.read_text())) if log.exists() else 0
        # The original pipeline runs test.py after train.main returns. That
        # separate test can fail even though all 32 training epochs completed.
        # Its saved checkpoint digest is evidence of entering post-train test;
        # keep the original failure state while accepting verified final weights.
        complete = status.get('state') == 'completed'
        test_command = status.get('test_command')
        if (not complete and count == 32 and status.get('state') in ('evaluating', 'failed')
                and isinstance(test_command, list)
                and any(Path(str(arg)).name == 'test.py' for arg in test_command)
                and status.get('checkpoint_sha256')):
            checkpoint = source / model['checkpoint']
            complete = checkpoint.is_file() and sha256(checkpoint) == status['checkpoint_sha256']
    else:
        status = read_json(source / 'jobs' / (model['arch'] + '.json'), {})
        log = source / 'runs' / model['run'] / 'eval_log.jsonl'
        records = []
        if log.exists():
            for line in log.read_text().splitlines():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        count = len([row for row in records if row.get('complete')])
        complete = status.get('state') == 'completed'
    return count, complete, status.get('state', 'queued')


def valid_checkpoint_archive(data):
    """Check torch.save ZIP metadata and every member's CRC without unpickling."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
            metadata = [name for name in names if name.endswith('/data.pkl')]
            if len(metadata) != 1 or len(set(names)) != len(names):
                return False
            prefix = metadata[0][:-len('data.pkl')]
            if prefix + 'version' not in names:
                return False
            if not any(name.startswith(prefix + 'data/') and not name.endswith('/') for name in names):
                return False
            if any(archive.getinfo(name).file_size == 0 for name in (metadata[0], prefix + 'version')):
                return False
            return archive.testzip() is None
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, EOFError,
            RuntimeError, NotImplementedError, zlib.error):
        return False


def snapshot(model, count, final):
    source = Path(model['source_root']) / model['checkpoint']
    if not source.exists():
        return None
    # Original Plus does not save atomically. Matching reads may still both be
    # truncated, so require a complete central directory and valid member CRCs.
    first = source.read_bytes()
    second = source.read_bytes()
    if first != second or not valid_checkpoint_archive(first):
        return None
    import hashlib
    digest = hashlib.sha256(first).hexdigest()
    job_id = model['id'] + '_' + digest[:16]
    job_path = ROOT / 'jobs' / (job_id + '.json')
    existing = read_json(job_path)
    if existing:
        if final and not existing['final_checkpoint']:
            existing.update(final_checkpoint=True, training_epochs_completed=count, finalized_at=now())
            save_json(job_path, existing)
        return job_id
    folder = ROOT / 'checkpoints' / job_id
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / 'best_model.pth'
    temp = folder / 'best_model.pth.tmp'
    temp.write_bytes(first)
    temp.replace(checkpoint)
    assert sha256(checkpoint) == digest
    shutil.copy2(Path(model['source_root']) / model['train_config'], folder / 'train_config.json')
    job = dict(job_id=job_id, model=model['model'], model_id=model['id'], arch=model['arch'],
               created_at=now(), checkpoint=str(checkpoint.relative_to(ROOT)),
               checkpoint_sha256=digest, train_config=str((folder / 'train_config.json').relative_to(ROOT)),
               training_epochs_completed=count, final_checkpoint=final,
               variants=model['variants'], primary_variant=model['primary_variant'],
               source_checkpoint=str(source))
    save_json(job_path, job)
    return job_id


def inspect(name):
    result = subprocess.run(['docker', 'inspect', '--format', '{{json .State}}', name],
                            text=True, capture_output=True, timeout=15)
    return json.loads(result.stdout) if result.returncode == 0 else None


def gpu_free(gpu):
    result = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used',
                             '--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=15)
    return result.returncode == 0 and int(result.stdout.strip()) < 2000


def render():
    name = 'dp-rgbt-unified-report-' + str(time.time_ns())
    result = subprocess.run(docker_command(name, 'report.py'), text=True, capture_output=True, timeout=120)
    (ROOT / 'logs/report.log').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError('Report generation failed; see logs/report.log')


def cycle(state):
    cfg = config()
    for model in cfg['models']:
        count, final, train_status = training_state(model)
        item = state['models'].setdefault(model['id'], dict(jobs=[]))
        item.update(training_epochs_completed=count, training_final=final, training_state=train_status)
        if not item['jobs'] or (final and not item.get('final_captured')):
            job_id = snapshot(model, count, final)
            if job_id:
                if job_id not in item['jobs']:
                    item['jobs'].append(job_id)
                if final:
                    item['final_captured'] = True
    busy = False
    active = list(cfg.get('reference_results', []))
    pending = []
    final_done = []
    queue = []
    statuses = {}
    for model in cfg['models']:
        item = state['models'][model['id']]
        completed = []
        failures = []
        for job_id in item['jobs']:
            job = read_json(ROOT / 'jobs' / (job_id + '.json'))
            status_path = ROOT / 'jobs' / (job_id + '.status.json')
            status = read_json(status_path, {'state': 'queued'})
            name = 'dp-rgbt-unified-eval-' + job_id.replace('_', '-')
            container = inspect(name)
            if container and container['Running']:
                busy = True
            elif container and container['Status'] in ('exited', 'dead'):
                if status['state'] != 'completed' or container['ExitCode'] != 0:
                    status.update(state='failed', error=status.get('error') or 'Evaluation container exited without success',
                                  exit_code=container['ExitCode'], oom_killed=container['OOMKilled'])
                    save_json(status_path, status)
                console = ROOT / 'logs' / (job_id + '.log')
                if not console.exists():
                    result = subprocess.run(['docker', 'logs', name], text=True, capture_output=True, timeout=30)
                    console.write_text(result.stdout + result.stderr)
            statuses[job_id] = status
            if status['state'] == 'failed':
                detail = job_id + ': ' + str(status.get('error') or 'unspecified evaluation error')
                if 'exit_code' in status:
                    detail += ' (exit_code=%s, oom_killed=%s)' % (status['exit_code'], status.get('oom_killed'))
                failures.append(detail)
            if status['state'] == 'completed' and status.get('complete'):
                paths = []
                for path in status['results']:
                    record = read_json(ROOT / path)
                    assert record['checkpoint_sha256'] == job['checkpoint_sha256']
                    if job['final_checkpoint'] and not record['final_checkpoint']:
                        record.update(final_checkpoint=True, checkpoint_status='final',
                                      training_epochs_completed=job['training_epochs_completed'])
                        save_json(ROOT / path, record)
                    paths.append(dict(path=path, checkpoint_sha256=job['checkpoint_sha256']))
                completed.append((job, paths))
            elif status['state'] == 'queued' and container is None:
                queue.append((model, job, name))
        if completed:
            job, paths = max(completed, key=lambda pair: (pair[0]['final_checkpoint'], pair[0]['created_at']))
            active.extend(paths)
            is_final = job['final_checkpoint']
        else:
            is_final = False
        final_done.append(is_final)
        if not is_final:
            reason = (
                'Final 32-epoch checkpoint ready; unified evaluation queued or in progress' if item['training_final'] else
                'Current best snapshot evaluated; waiting for final 32-epoch checkpoint' if completed else
                'Training/evaluation queued; no completed unified metrics yet')
            if failures:
                reason = 'Evaluation failed: ' + '; '.join(failures) + '. No automatic retry. ' + reason
            pending.append(dict(model=model['model'], reason=reason))
    # Completed LV first, then existing current-best snapshots in source order.
    queue.sort(key=lambda value: (not value[1]['final_checkpoint'], value[1]['created_at']))
    if queue and not busy:
        ready = read_json(ROOT / 'preflight.json', {}).get('status') == 'passed'
        if ready:
            for gpu in cfg['gpu_priority']:
                if gpu in cfg['allowed_gpus'] and gpu_free(gpu):
                    model, job, name = queue[0]
                    command = docker_command(name, 'evaluate.py', [job['job_id']], gpu=gpu, detach=True)
                    save_json(ROOT / 'jobs' / (job['job_id'] + '.launch.json'), command)
                    result = subprocess.run(command, text=True, capture_output=True, timeout=60)
                    if result.returncode:
                        save_json(ROOT / 'jobs' / (job['job_id'] + '.status.json'),
                                  dict(state='failed', error=result.stderr[-4000:]))
                    else:
                        busy = True
                    break
    cfg['active_results'] = active
    cfg['pending_results'] = pending
    save_json(ROOT / 'config.json', cfg)
    signature = json.dumps(dict(active=active, pending=pending,
                                final_flags=[read_json(ROOT / row['path']).get('final_checkpoint') for row in active]),
                           sort_keys=True)
    if state.get('report_signature') != signature:
        render()
        state['report_signature'] = signature
    state.update(updated_at=now(), jobs=statuses, all_final_complete=all(final_done),
                 evaluation_gpu=cfg['gpu_priority'], currently_running=busy)
    save_json(ROOT / 'STATUS.json', state)
    return all(final_done) and not busy


def main():
    lock = (ROOT / 'scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (ROOT / 'scheduler.pid').write_text(str(os.getpid()) + '\n')
    verify_source()
    state = read_json(ROOT / 'STATUS.json', dict(models={}))
    while True:
        try:
            done = cycle(state)
        except Exception:
            (ROOT / 'logs/scheduler_error.txt').write_text(traceback.format_exc())
            print(traceback.format_exc(), flush=True)
            done = False
        if done:
            return
        time.sleep(20)


if __name__ == '__main__':
    main()
