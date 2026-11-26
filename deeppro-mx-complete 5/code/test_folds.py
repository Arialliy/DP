"""Checks of the grouped cross-validation folds (no data needed).

Run: python code/test_folds.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import folds  # noqa: E402
import splits  # noqa: E402


def _inputs():
    s = json.load(open(splits.SPLIT_FILE))
    clips = sorted(s['train'] + s['val'])
    cam = s['camera_speed_px_per_frame']
    near = {c: (1945 if c == 'DJI_0319_2' else (100 * (i % 4))) for i, c in enumerate(clips)}
    boxes = {c: 300 + 10 * i for i, c in enumerate(clips)}
    return s, clips, cam, near, boxes


def test_every_clip_once_and_groups_intact():
    s, clips, cam, near, boxes = _inputs()
    fc, stats = folds.build_folds(clips, cam, near, boxes, 5)
    flat = [c for f in fc for c in f]
    assert sorted(flat) == clips and len(flat) == len(set(flat)) == 40
    assert not set(flat) & set(s['test'])
    owner = {c: i for i, f in enumerate(fc) for c in f}
    for g, members in folds.groups_of(clips).items():
        assert len({owner[c] for c in members}) == 1, g          # a recording never straddles folds
    assert owner['DJI_0043_2'] == owner['DJI_0057_1'] and owner['DJI_0113_3'] == owner['DJI_0115_2']
    assert len({owner[c] for c in ('DJI_0253_1', 'DJI_0253_2', 'DJI_0253_3')}) == 1


def test_moving_groups_and_near_stationary_are_spread():
    _, clips, cam, near, boxes = _inputs()
    fc, stats = folds.build_folds(clips, cam, near, boxes, 5)
    mv = [f['moving'] for f in stats]
    assert max(mv) - min(mv) <= 1 and sum(mv) == 11, mv
    owner = {c: i for i, f in enumerate(fc) for c in f}
    big = owner['DJI_0319_2']
    others = [f['near'] for i, f in enumerate(stats) if i != big]
    assert stats[big]['near'] - 1945 <= min(others), [f['near'] for f in stats]


def test_fold_names_resolve(tmp_name='folds_test.json'):
    _, clips, cam, near, boxes = _inputs()
    fc, _ = folds.build_folds(clips, cam, near, boxes, 5)
    path = os.path.join(os.path.dirname(splits.FOLD_FILE), tmp_name)
    old = splits.FOLD_FILE
    try:
        json.dump(dict(k=5, folds=fc), open(path, 'w'))
        splits.FOLD_FILE = path
        for i in range(5):
            val, train = splits.clips(f'fold{i}/val'), splits.clips(f'fold{i}/train')
            assert sorted(val + train) == clips and not set(val) & set(train)
    finally:
        splits.FOLD_FILE = old
        os.remove(path)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
