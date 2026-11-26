#!/usr/bin/env python3
"""Isolated shell orchestration checks; no project Python or real training is run.

Copies run_mx.sh unchanged into a temporary directory whose path contains spaces.
Project entry points are stand-ins selected through the script's PY environment hook.
The inline require_go Python snippet is executed unchanged with the host interpreter.
Run from any directory: python failure_gates.py
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone


STANDIN = r'''import json
import os
from pathlib import Path
import subprocess
import sys

args = sys.argv[1:]
name = 'require_go' if args[0] == '-' else Path(args[0]).name
case = os.environ['GATE_CASE']
with open('calls.jsonl', 'a') as out:
    out.write(json.dumps({'name': name, 'args': args[1:]}) + '\n')
failures = {
    'first_test_fails': ('test_001_first.py', 41),
    'second_test_fails': ('test_002_second.py', 42),
    'labels_differ': ('check_labels.py', 43),
    'compare_diff': ('compare_datasets.py', 1),
    'compare_unconfirmed': ('compare_datasets.py', 2),
    'profile_fails': ('profile_motion.py', 47),
    'folds_fails': ('folds.py', 48),
    'm0_no_go': ('exemplar_diag.py', 3),
    'm0_error': ('exemplar_diag.py', 49),
    'training_fails': ('train.py', 51),
    'inference_fails': ('infer_mx.py', 52),
    'scoring_fails': ('score.py', 53),
}
if name == 'require_go':
    # Execute only the unchanged inline JSON gate, never any project Python file.
    result = subprocess.run([sys.executable, *args], input=sys.stdin.read(), text=True)
    sys.exit(result.returncode)
if name == 'profile_motion.py':
    root = Path('results/profile')
    root.mkdir(parents=True, exist_ok=True)
    target = root / 'targets.csv'
    target.write_text('stand-in,not-real-data\n')
    os.utime(target, ns=(1600000000000000000, 1600000000000000000))
    if case != 'profile_missing_metadata':
        (root / 'targets.json').write_text(json.dumps({'stand_in': True}))
elif name == 'exemplar_diag.py' and case not in ('m0_error', 'm0_missing_summary'):
    root = Path('results/exemplar_diag')
    root.mkdir(parents=True, exist_ok=True)
    summary = root / 'summary.json'
    summary.write_text(json.dumps({'decision': {'go': case not in ('m0_no_go', 'm0_false_go_exit_zero')}}))
    stamp = 1599999999000000000 if case == 'm0_stale_summary' else 1600000002000000000
    os.utime(summary, ns=(stamp, stamp))
elif name == 'train.py' and case not in ('training_fails', 'training_missing_checkpoint'):
    run = args[args.index('--run') + 1]
    root = Path('runs') / run
    root.mkdir(parents=True, exist_ok=True)
    (root / 'final.pth').write_text('stand-in,not-a-model\n')
if case in failures and name == failures[case][0]:
    sys.exit(failures[case][1])
sys.exit(0)
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path,
                    default=Path(__file__).resolve().parents[2] / 'run_mx.sh')
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    source = args.source.resolve()
    source_bytes = source.read_bytes()
    common = ['test_001_first.py', 'test_002_second.py', 'check_labels.py', 'compare_datasets.py',
              'profile_motion.py', 'folds.py', 'exemplar_diag.py', 'require_go', 'train.py',
              'infer_mx.py', 'score.py']
    # Expected exact call prefix and shell status; every later call must be absent.
    cases = [
        ('first_test_fails', 'all', 1, common[:1]),
        ('second_test_fails', 'all', 1, common[:2]),
        ('labels_differ', 'all', 1, common[:3]),
        ('compare_diff', 'all', 1, common[:4]),
        ('compare_unconfirmed', 'all', 1, common[:4]),
        ('profile_fails', 'all', 47, common[:5]),
        ('profile_missing_metadata', 'all', 1, common[:5]),
        ('folds_fails', 'all', 48, common[:6]),
        ('m0_no_go', 'all', 3, common[:7]),
        ('m0_error', 'all', 1, common[:7]),
        ('m0_missing_summary', 'all', 1, common[:7]),
        ('m0_stale_summary', 'all', 1, common[:7]),
        ('m0_false_go_exit_zero', 'all', 3, common[:8]),
        ('training_fails', 'all', 51, common[:9]),
        ('training_missing_checkpoint', 'all', 1, common[:9]),
        ('inference_fails', 'all', 52, common[:10]),
        ('scoring_fails', 'all', 53, common),
        ('positive_go', 'all', 0, common),
        ('positive_without_compare', 'all', 0, [c for c in common if c != 'compare_datasets.py']),
        ('first_test_fails', 'check', 1, common[:1]),
        ('labels_differ', 'check', 1, common[:3]),
        ('compare_unconfirmed', 'check', 1, common[:4]),
        ('positive_go', 'check', 0, common[:4]),
    ]
    results = []
    for case, stage, expected_status, expected_calls in cases:
        with tempfile.TemporaryDirectory(prefix='mx failure gates ') as tmp:
            work = Path(tmp) / 'bundle with spaces'
            code = work / 'code'
            code.mkdir(parents=True)
            script = work / 'run_mx.sh'
            shutil.copyfile(source, script)
            assert script.read_bytes() == source_bytes
            for name in common:
                if name.endswith('.py'):
                    (code / name).write_text('# Placeholder: handled by the PY stand-in.\n')
            py = work / 'python stand-in'
            py.write_text('#!' + sys.executable + '\n' + STANDIN)
            py.chmod(0o755)
            env = {k: v for k, v in os.environ.items()
                   if k not in ('VT_ANN_DIR', 'OFFICIAL_LISTS', 'VT_OLD', 'RUN', 'SEED', 'PY',
                                'DEVICE', 'M0_SPLIT', 'FOLDS', 'NPROC')}
            env.update(PY=str(py), VT_ROOT=str(work / 'fake RGB-T'),
                       VT_OLD=str(work / 'fake older copy'), GATE_CASE=case, RUN='gate_run',
                       DEVICE='cpu', FOLDS='1', M0_SPLIT='val', NPROC='1')
            if case == 'positive_without_compare':
                env.pop('VT_OLD')
            done = subprocess.run(['bash', str(script), stage], cwd=work, env=env,
                                  text=True, capture_output=True, timeout=30)
            calls_path = work / 'calls.jsonl'
            details = [json.loads(line) for line in calls_path.read_text().splitlines()] if calls_path.exists() else []
            called = [d['name'] for d in details]
            passed = done.returncode == expected_status and called == expected_calls
            # The positive full path must also retain the requested label protocol.
            if passed and case.startswith('positive') and stage == 'all':
                for d in details:
                    if d['name'] in ('train.py', 'score.py'):
                        passed = passed and '--label-rule' in d['args']
                        passed = passed and d['args'][d['args'].index('--label-rule') + 1] == 'paper_k6_floor'
                    if d['name'] == 'score.py':
                        passed = passed and '--author' in d['args']
            results.append(dict(case=case, stage=stage, passed=bool(passed),
                                expected_status=expected_status, actual_status=done.returncode,
                                expected_calls=expected_calls, actual_calls=called,
                                call_arguments=details, stdout=done.stdout, stderr=done.stderr))
            print(f"{'PASS' if passed else 'FAIL'} {stage}: {case} (status {done.returncode})", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), source=str(source),
                  source_sha256=hashlib.sha256(source_bytes).hexdigest(), python=sys.version,
                  scope='Isolated shell orchestration only. Project Python entry points, data, and model training were stand-ins; inline require_go JSON code was unchanged.',
                  cases=len(results), passed=sum(r['passed'] for r in results),
                  failed=sum(not r['passed'] for r in results), results=results)
    (args.out / 'failure_gates.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"{report['passed']}/{report['cases']} passed; {args.out / 'failure_gates.json'}")
    return 0 if report['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
