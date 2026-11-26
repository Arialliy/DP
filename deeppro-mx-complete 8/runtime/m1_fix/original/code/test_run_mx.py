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
import test_calib_diag as tcd  # noqa: E402
import test_exemplar_diag as ted  # noqa: E402

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run_mx.sh')

# Stands in for python: train.py only records that it was started (and writes the checkpoints
# of its --run); everything else runs.
WRAPPER = """#!/usr/bin/env bash
case "$1" in
  */train.py)
    echo "$@" >> train_started; run=""; prev=""
    for a in "$@"; do [ "$prev" = "--run" ] && run="$a"; prev="$a"; done
    mkdir -p "runs/$run"; touch "runs/$run/final.pth" "runs/$run/best_model.pth"; exit 0 ;;
esac
exec "{python}" "$@"
"""
CLIPS = ','.join(tcd.SCENES)


def _setup(d, hot_shape, single=False):
    """Three scenes of test_calib_diag; hot_shape 'round' keeps their exemplars matched to the
    parked cars (M0c says GO), 'car' gives the moving cars another contrast (NO-GO).
    single: the one clip 'syn' of test_exemplar_diag instead (for compare)."""
    clips = {}
    if single:
        video, boxes, _, _, _ = ted.scene(hot_shape=hot_shape)
        clips['syn'] = (video, boxes, 'train2017')
    else:
        for c in tcd.SCENES:
            video, boxes, _ = tcd.scene(c, moving=None if hot_shape == 'round' else 90)
            clips[c] = (video, boxes, 'train2017')
    root = os.path.join(d, 'RGB-T')
    fake_dataset.write(root, clips, ann_dir=os.path.join(root, 'annotations_coco'))
    open(os.path.join(root, 'train.txt'), 'w').write('\n'.join(clips) + '\n')
    open(os.path.join(root, 'test.txt'), 'w').write('\n')
    work = os.path.join(d, 'work')
    os.makedirs(work)
    py = os.path.join(d, 'py')
    open(py, 'w').write(WRAPPER.replace('{python}', sys.executable))
    os.chmod(py, 0o755)
    env = {k: v for k, v in os.environ.items() if k not in ('VT_ANN_DIR', 'OFFICIAL_LISTS', 'VT_OLD', 'RUN', 'SEED')}
    env.update(VT_ROOT=root, OFFICIAL_LISTS=root, PY=py, FOLDS='0', M0_SPLIT=CLIPS, NPROC='1')  # not the shell's
    for k in ('ARCH', 'SELECT', 'SELECT_SPLIT', 'TRAIN_SPLIT', 'EVAL_SPLIT', 'DET_RUN'):
        env.pop(k, None)
    return work, env


def run(stage, work, env):
    return subprocess.run(['bash', SCRIPT, stage], cwd=work, env=env, capture_output=True, text=True)


def test_go_and_no_go_are_recorded_and_the_gate_stops():
    """M0c is a diagnostic: GO and NO-GO both let training start, and the verdict is saved
    with the run. With M0_GATE=1 a NO-GO stops training (status 3)."""
    import json
    for shape, go in (('round', True), ('car', False)):
        with tempfile.TemporaryDirectory() as d:
            work, env = _setup(d, shape)
            r = run('train', work, env)                          # nothing yet: refused
            assert r.returncode == 1 and 'profile' in r.stderr and not os.path.exists(f'{work}/train_started')
            r = run('profile', work, env)
            assert r.returncode == 0, r.stderr[-2000:]
            assert os.path.exists(f'{work}/results/profile/targets.json')
            assert os.path.exists(f'{work}/results/motion/motion.json')            # annotation-free motion
            assert os.path.exists(f'{work}/results/motion/homographies/DJI_9001_1.npy')
            r = run('train', work, env)                          # no M0c yet: refused
            assert r.returncode == 1 and 'm0' in r.stderr and not os.path.exists(f'{work}/train_started')
            r = run('m0', work, env)
            assert r.returncode == 0, (shape, r.returncode, r.stdout[-1500:], r.stderr[-1500:])
            assert go or 'diagnostic' in r.stderr
            gate = dict(env, M0_GATE='1')
            assert run('m0', work, gate).returncode == (0 if go else 3)
            r = run('train', work, gate)
            if not go:
                assert r.returncode == 3 and 'NO-GO' in r.stderr and not os.path.exists(f'{work}/train_started')
            r = run('train', work, env)                          # without the gate it trains either way
            assert r.returncode == 0, r.stderr[-1500:]
            started = open(f'{work}/train_started').read()
            assert '--label-rule paper_k6_floor' in started and '--arch mxc_lv' in started
            assert '--checkpoint-selection original_test_iou' in started and '--split official_train' in started
            assert '--run mxc_s46' in started and os.path.exists(f'{work}/runs/mxc_s46/best_model.pth')
            rec = json.load(open(f'{work}/runs/mxc_s46/m0c.json'))
            assert rec['decision']['go'] is go


def test_stale_m0_and_old_profile_are_handled():
    with tempfile.TemporaryDirectory() as d:
        work, env = _setup(d, 'round')
        assert run('profile', work, env).returncode == 0
        assert run('m0', work, env).returncode == 0
        time.sleep(1.1)
        assert run('profile', work, env).returncode == 0            # new targets.csv, the old folder kept aside
        assert any(n.startswith('profile.before.') for n in os.listdir(f'{work}/results'))
        r = run('train', work, env)                                  # M0c predates these speeds
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
            work, env = _setup(d, 'round', single=True)
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


def test_validation_protocol_and_controls():
    """SELECT=val_iou passes the selection split; controls trains the three ablations."""
    with tempfile.TemporaryDirectory() as d:
        work, env = _setup(d, 'round')
        assert run('profile', work, env).returncode == 0
        assert run('m0', work, env).returncode == 0
        env.update(SELECT='val_iou', TRAIN_SPLIT='train', SELECT_SPLIT='val')
        r = run('train', work, env)
        assert r.returncode == 0, r.stderr[-1500:]
        started = open(f'{work}/train_started').read()
        assert '--checkpoint-selection val_iou --selection-split val' in started and '--split train' in started
        os.remove(f'{work}/train_started')
        env['EVAL_SPLIT'] = 'nonexistent_split_for_this_test'      # eval is reached only after each training
        r = run('controls', work, env)
        started = open(f'{work}/train_started').read().splitlines()
        assert len(started) == 1 and '--ex-p-none 1' in started[0] and r.returncode != 0
        env['SELECT'] = 'best'
        assert run('train', work, env).returncode == 2


def test_all_controls_keep_selection_and_main_gain():
    """Run all four controls with stand-in training/evaluation and inspect their arguments."""
    import json
    with tempfile.TemporaryDirectory() as d:
        work, env = _setup(d, 'round')
        wrapper = WRAPPER.replace('  */train.py)',
                                  '  */infer_mx.py|*/score.py) echo "$@" >> eval_started; exit 0 ;;\n'
                                  '  */train.py)')
        open(env['PY'], 'w').write(wrapper.replace('{python}', sys.executable))
        for folder in ('profile', 'motion', 'calib_diag'):
            os.makedirs(f'{work}/results/{folder}')
        for name in ('profile/targets.csv', 'motion/motion.json'):
            path = f'{work}/results/{name}'
            open(path, 'w').write('{}\n')
            os.utime(path, (1, 1))
        json.dump({'decision': {'go': False}}, open(f'{work}/results/calib_diag/summary.json', 'w'))
        env.update(RUN='control_fixture', SELECT='val_iou', TRAIN_SPLIT='train', SELECT_SPLIT='val',
                   EVAL_SPLIT='val', M0_GATE='0')
        for arch, gain in (('mxc_lv', '2'), ('mxc_lv_absfloor', '2'), ('mx_lv', '1')):
            env['ARCH'] = arch
            r = run('controls', work, env)
            assert r.returncode == 0, (arch, r.stderr[-1500:])
            commands = [row.split() for row in open(f'{work}/train_started')]
            assert len(commands) == 4
            by_run = {row[row.index('--run') + 1]: row for row in commands}
            for row in commands:
                assert row[row.index('--checkpoint-selection') + 1] == 'val_iou'
                assert row[row.index('--selection-split') + 1] == 'val'
                assert row[row.index('--split') + 1] == 'train'
            for suffix in ('noex', 'self', 'nogain'):
                row = by_run[f'control_fixture_{suffix}']
                assert row[row.index('--arch') + 1] == arch
            noex, own = by_run['control_fixture_noex'], by_run['control_fixture_self']
            assert noex[noex.index('--ex-p-none') + 1] == '1'
            assert own[own.index('--ex-source') + 1] == 'self'
            nogain, mx = by_run['control_fixture_nogain'], by_run['mx_s46']
            assert nogain[nogain.index('--gain-aug') + 1] == '1'
            assert mx[mx.index('--arch') + 1] == 'mx_lv'
            assert mx[mx.index('--gain-aug') + 1] == gain
            evaluations = [row.split() for row in open(f'{work}/eval_started')]
            assert len(evaluations) == 8
            infer = {row[row.index('--run') + 1]: row for row in evaluations[::2]}
            for name, bank in (('control_fixture_noex', 'none'), ('control_fixture_self', 'self')):
                assert infer[name][infer[name].index('--bank') + 1] == bank
            os.remove(f'{work}/train_started')
            os.remove(f'{work}/eval_started')


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
