"""Checks of run_mx.sh: every stop condition (numpy only; training is replaced by a stand-in).

Run: python code/test_run_mx.py
"""
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fake_dataset  # noqa: E402
import test_exemplar_diag as ted  # noqa: E402

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run_mx.sh')

# Stands in for python: train.py only records that it was started; everything else runs.
WRAPPER = """#!/usr/bin/env bash
case "$1" in
  */train.py) echo "$@" > train_started; mkdir -p runs/mx_s46; touch runs/mx_s46/final.pth; exit 0 ;;
esac
exec "{python}" "$@"
"""


def _setup(d, hot_shape):
    video, boxes, _, _, _ = ted.scene(hot_shape=hot_shape)
    root = os.path.join(d, 'RGB-T')
    fake_dataset.write(root, {'syn': (video, boxes, 'train2017')}, ann_dir=os.path.join(root, 'annotations_coco'))
    open(os.path.join(root, 'train.txt'), 'w').write('syn\n')
    open(os.path.join(root, 'test.txt'), 'w').write('\n')
    work = os.path.join(d, 'work')
    os.makedirs(work)
    py = os.path.join(d, 'py')
    open(py, 'w').write(WRAPPER.replace('{python}', sys.executable))
    os.chmod(py, 0o755)
    env = {k: v for k, v in os.environ.items() if k not in ('VT_ANN_DIR', 'OFFICIAL_LISTS', 'VT_OLD', 'RUN', 'SEED')}
    env.update(VT_ROOT=root, OFFICIAL_LISTS=root, PY=py, FOLDS='0', M0_SPLIT='syn', NPROC='1')  # not the shell's
    return work, env


def run(stage, work, env):
    return subprocess.run(['bash', SCRIPT, stage], cwd=work, env=env, capture_output=True, text=True)


def test_go_path_trains_and_no_go_path_stops():
    for shape, m0_status in (('round', 0), ('car', 3)):
        with tempfile.TemporaryDirectory() as d:
            work, env = _setup(d, shape)
            r = run('train', work, env)                          # nothing yet: refused
            assert r.returncode == 1 and 'profile' in r.stderr and not os.path.exists(f'{work}/train_started')
            r = run('profile', work, env)
            assert r.returncode == 0, r.stderr[-2000:]
            assert os.path.exists(f'{work}/results/profile/targets.json')
            r = run('train', work, env)                          # no M0 yet: refused
            assert r.returncode == 1 and 'm0' in r.stderr and not os.path.exists(f'{work}/train_started')
            r = run('m0', work, env)
            assert r.returncode == m0_status, (shape, r.returncode, r.stdout[-1500:], r.stderr[-1500:])
            r = run('train', work, env)
            if m0_status == 0:
                assert r.returncode == 0, r.stderr[-1500:]
                started = open(f'{work}/train_started').read()
                assert '--label-rule paper_k6_floor' in started and '--arch mx_lv' in started
            else:
                assert r.returncode == 3 and 'NO-GO' in r.stderr and not os.path.exists(f'{work}/train_started')


def test_stale_m0_and_old_profile_are_handled():
    with tempfile.TemporaryDirectory() as d:
        work, env = _setup(d, 'round')
        assert run('profile', work, env).returncode == 0
        assert run('m0', work, env).returncode == 0
        time.sleep(1.1)
        assert run('profile', work, env).returncode == 0            # new targets.csv, the old folder kept aside
        assert any(n.startswith('profile.before.') for n in os.listdir(f'{work}/results'))
        r = run('train', work, env)                                  # M0 predates these speeds
        assert r.returncode == 1 and 'older' in r.stderr and not os.path.exists(f'{work}/train_started')
        assert run('m0', work, env).returncode == 0
        assert run('train', work, env).returncode == 0


def test_compare_stage_stops_on_differences_and_on_unconfirmed_tracks():
    """run_mx.sh compare: 0 when the copies agree; stops when they differ, and when two
    targets only ever share one position so that their track relation cannot be confirmed."""
    import json
    import numpy as np
    empty = dict(images=[], annotations=[], categories=[])
    for case, expect in (('same', None), ('unconfirmed', 'cannot be confirmed'), ('moved', 'differ')):
        with tempfile.TemporaryDirectory() as d:
            work, env = _setup(d, 'round')
            root, old = env['VT_ROOT'], os.path.join(d, 'OLD')
            video, boxes, _, _, _ = ted.scene()                    # the clip _setup wrote into root
            new_boxes, old_boxes = boxes, boxes
            if case == 'unconfirmed':                              # two parked boxes at one position
                new_boxes = [np.vstack([b, [(70, 30, 7, 3, 1, 901), (70, 30, 7, 3, 1, 902)]]) for b in boxes]
                old_boxes = [np.vstack([b, [(70, 30, 7, 3, 1, 911), (70, 30, 7, 3, 1, 912)]]) for b in boxes]
                fake_dataset.write(root, {'syn': (video, new_boxes, 'train2017')},
                                   ann_dir=os.path.join(root, 'annotations_coco'))
            fake_dataset.write(old, {'syn': (video, old_boxes, 'train2017')})
            if case == 'moved':
                p = os.path.join(root, 'annotations_coco', 'instances_01_train2017.json')
                ann = json.load(open(p))
                ann['annotations'][0]['bbox'][0] += 2.0
                json.dump(ann, open(p, 'w'))
            for f in (os.path.join(old, 'annotations'), os.path.join(root, 'annotations_coco')):
                json.dump(empty, open(os.path.join(f, 'instances_01_test2017.json'), 'w'))
            env['VT_OLD'] = old
            r = run('compare', work, env)
            if expect is None:
                assert r.returncode == 0, (case, r.stdout[-800:], r.stderr[-800:])
            else:
                assert r.returncode == 1 and expect in r.stderr, (case, r.returncode, r.stderr[-800:])


def test_usage_and_missing_root():
    with tempfile.TemporaryDirectory() as d:
        work, env = _setup(d, 'round')
        assert run('nonsense', work, env).returncode == 2
        env.pop('VT_ROOT')
        r = run('m0', work, env)
        assert r.returncode != 0 and 'VT_ROOT' in r.stderr


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
