"""Wait for the user's copy; verify data, then dispatch the queued official run.

Never writes to the dataset or imports training/model code.
"""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

RUN = Path(__file__).resolve().parent
DATA = Path('/home/ly/DeepPro-main/datasets/NUDT-MIRSDT')
SOURCE_RECORDS = json.loads((RUN / 'data_audit/members.json').read_text())
# The reference manifest was checked directly against the retained official ZIP.
if isinstance(SOURCE_RECORDS, dict):
    raise RuntimeError('Unexpected reference manifest format')
FILES = [(row['member'].split('/', 1)[1], row['bytes'], row['archive_sha256'])
         for row in SOURCE_RECORDS]
assert len(FILES) == 48122 and len({row[0] for row in FILES}) == 48122


def status(value, **extra):
    state = {'status': value, 'pid': os.getpid(), 'data_path': str(DATA),
             'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra}
    path = RUN / 'STATUS.json'
    path.with_suffix('.tmp').write_text(json.dumps(state, indent=2) + '\n')
    path.with_suffix('.tmp').replace(path)


lock = (RUN / 'data_wait.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
try:
    while True:
        missing = []
        wrong_size = []
        for name, size, _ in FILES:
            p = DATA / name
            try:
                if p.stat().st_size != size:
                    wrong_size.append(name)
            except FileNotFoundError:
                missing.append(name)
        if missing or wrong_size:
            status('waiting_for_dataset_copy', expected_files=len(FILES),
                   present_files=len(FILES)-len(missing), missing_files=len(missing),
                   incomplete_files=len(wrong_size))
            time.sleep(30)
            continue
        status('verifying_completed_dataset', expected_files=len(FILES))
        mismatches = []
        for name, _, digest in FILES:
            try:
                actual = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
            except FileNotFoundError:
                actual = None
            if actual != digest:
                mismatches.append(name)
        if mismatches:
            status('waiting_for_dataset_copy', hash_mismatches=len(mismatches),
                   mismatch_examples=mismatches[:10])
            time.sleep(30)
            continue
        official_names = {row[0] for row in FILES}
        extras = sorted(str(p.relative_to(DATA)) for p in DATA.rglob('*')
                        if p.is_file() and str(p.relative_to(DATA)) not in official_names)
        unsafe_extras = [name for name in extras if Path(name).parent.name in
                         {'images', 'masks', 'masks_centroid'}]
        if unsafe_extras:
            status('waiting_for_dataset_cleanup', unexpected_loader_files=unsafe_extras)
            time.sleep(30)
            continue
        audit = {'status': 'PASS', 'dataset_root': str(DATA),
                 'official_files_checked': len(FILES), 'missing_files': [], 'hash_mismatches': [],
                 'extra_files_outside_loader_directories': extras,
                 'reference_manifest': str(RUN / 'data_audit/members.json'),
                 'reference_manifest_sha256': hashlib.sha256((RUN / 'data_audit/members.json').read_bytes()).hexdigest(),
                 'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        (RUN / 'DATA_READY.json').write_text(json.dumps(audit, indent=2) + '\n')
        break
    status('checking_official_loader')
    cmd = json.loads((RUN / 'cpu-preflight-command.json').read_text())
    started = time.monotonic()
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (RUN / 'cpu_after_copy.log').write_text(result.stdout + '\n' + result.stderr)
    (RUN / 'cpu_after_copy.json').write_text(json.dumps({'command': cmd, 'returncode': result.returncode,
        'stdout': result.stdout, 'stderr': result.stderr, 'elapsed_seconds': time.monotonic()-started}, indent=2) + '\n')
    if result.returncode or 'WITNESS ' not in result.stdout:
        raise RuntimeError('Official loader preflight failed after copy; see cpu_after_copy.log')
    status('dispatching_gpu_queue')
    # Replace this OS-only waiting process; training still runs as original CLI.
    (RUN / 'scheduler.pid').write_text(str(os.getpid()) + '\n')
    os.execv(sys.executable, [sys.executable, str(RUN / 'queue_run.py')])
except Exception as exc:
    status('failed', error=repr(exc))
    raise
