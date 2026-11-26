"""Checks of compare_datasets.py on two small copies on disk (numpy only).

Run: python code/test_compare_datasets.py
"""
import os
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_datasets as cd  # noqa: E402
import fake_dataset  # noqa: E402


def _copies(d):
    rng = np.random.default_rng(0)
    video = rng.normal(80, 5, (6, 30, 40))
    raw = [np.array([(5 + t, 5, 4, 3, 3, 0), (20, 15, 5, 4, 5, 0)], float) for t in range(6)]   # per-category ids
    unique = [b.copy() for b in raw]
    for b in unique:
        b[:, 5] = [1903000, 2507000]                                    # clip-unique ids, another numbering
    a, b = os.path.join(d, 'VT'), os.path.join(d, 'RGB-T')
    fake_dataset.write(a, {'C_1': (video, unique, 'train2017'), 'D_2': (video[:3], unique[:3], 'test2017')},
                       track_key='track_id')
    fake_dataset.write(b, {'C_1': (video, raw, 'train2017'), 'D_2': (video[:3], raw[:3], 'test2017')},
                       track_key='tracking_id', ann_dir=os.path.join(b, 'annotations_coco'))
    open(os.path.join(b, 'train.txt'), 'w').write('C_1\n')
    open(os.path.join(b, 'test.txt'), 'w').write('D_2\n')
    return a, b


def test_same_data_with_other_track_numbering_is_the_same():
    with tempfile.TemporaryDirectory() as d:
        a, b = _copies(d)
        assert cd.main(['--a', a, '--b', b]) == 0
        A = dict(root=a, boxes={s: cd.load_boxes(a, None, s) for s in ('train2017',)})
        B = dict(root=b, boxes={s: cd.load_boxes(b, None, s) for s in ('train2017',)})
        r = cd.compare_clip('C_1', A, B)
        assert r['same'] and r['tracks_a'] == r['tracks_b'] == 2 and r['boxes_a'] == 12


def test_a_moved_box_and_a_changed_image_are_found():
    with tempfile.TemporaryDirectory() as d:
        a, b = _copies(d)
        import json
        p = os.path.join(b, 'annotations_coco', 'instances_01_train2017.json')
        ann = json.load(open(p))
        ann['annotations'][3]['bbox'][0] += 1.0
        json.dump(ann, open(p, 'w'))
        img = os.path.join(b, 'train2017', 'C_1', '01', '000002.jpg')
        Image.fromarray(np.zeros((30, 40), np.uint8)).save(img, quality=100)
        A = dict(root=a, boxes={'train2017': cd.load_boxes(a, None, 'train2017')})
        B = dict(root=b, boxes={'train2017': cd.load_boxes(b, None, 'train2017')})
        r = cd.compare_clip('C_1', A, B, pixels=True)
        assert not r['same'] and r['frames_with_box_differences'] == 1 and r['images_differing'] == 1
        assert r['max_pixel_diff'] > 50
        assert cd.main(['--a', a, '--b', b]) == 1


def test_two_targets_merged_into_one_track_are_a_conflict():
    with tempfile.TemporaryDirectory() as d:
        a, b = _copies(d)
        import json
        p = os.path.join(a, 'annotations', 'instances_01_train2017.json')
        ann = json.load(open(p))
        for x in ann['annotations']:
            x['track_id'] = 7                                  # one id for both targets
        json.dump(ann, open(p, 'w'))
        A = dict(root=a, boxes={'train2017': cd.load_boxes(a, None, 'train2017')})
        B = dict(root=b, boxes={'train2017': cd.load_boxes(b, None, 'train2017')})
        r = cd.compare_clip('C_1', A, B)
        assert not r['same'] and r['track_conflicts'] == 1


def _pair(d, a_tracks, b_tracks):
    """Two one-clip copies: frame t holds boxes (x, y, w, h, cat, track) given per copy."""
    video = np.random.default_rng(1).normal(80, 5, (len(a_tracks), 30, 40))
    a, b = os.path.join(d, 'A'), os.path.join(d, 'B')
    fake_dataset.write(a, {'C_1': (video, [np.array(f, float) for f in a_tracks], 'train2017')}, track_key='track_id')
    fake_dataset.write(b, {'C_1': (video, [np.array(f, float) for f in b_tracks], 'train2017')}, track_key='track_id')
    A = dict(root=a, boxes={'train2017': cd.load_boxes(a, None, 'train2017')})
    B = dict(root=b, boxes={'train2017': cd.load_boxes(b, None, 'train2017')})
    return cd.compare_clip('C_1', A, B)


def test_boxes_at_a_shared_position_need_their_pairing_confirmed():
    """Two targets at the same position and category (e.g. a duplicated annotation) cannot be
    told apart in that frame. Their pairing counts as confirmed only through other frames."""
    with tempfile.TemporaryDirectory() as d:
        # frame 0 separates tracks 1/2 (A) = 11/12 (B); in frame 1 they share a position
        r = _pair(d, [[(0, 0, 4, 4, 1, 1), (9, 9, 4, 4, 1, 2)], [(5, 5, 4, 4, 1, 1), (5, 5, 4, 4, 1, 2)]],
                  [[(0, 0, 4, 4, 1, 11), (9, 9, 4, 4, 1, 12)], [(5, 5, 4, 4, 1, 12), (5, 5, 4, 4, 1, 11)]])
        assert r['status'] == 'SAME' and r['duplicate_position_boxes'] == 2 and r['unconfirmed_boxes'] == 0
    with tempfile.TemporaryDirectory() as d:
        # the two tracks only ever appear at a shared position: the relation cannot be confirmed
        r = _pair(d, [[(5, 5, 4, 4, 1, 1), (5, 5, 4, 4, 1, 2)]] * 2, [[(5, 5, 4, 4, 1, 11), (5, 5, 4, 4, 1, 12)]] * 2)
        assert r['status'] == 'UNCONFIRMED' and not r['same'] and r['unconfirmed_boxes'] == 4
    with tempfile.TemporaryDirectory() as d:
        # paired elsewhere as 1->11 and 2->12, but at the shared position B holds 11 and 13: a conflict
        r = _pair(d, [[(0, 0, 4, 4, 1, 1), (9, 9, 4, 4, 1, 2)], [(5, 5, 4, 4, 1, 1), (5, 5, 4, 4, 1, 2)]],
                  [[(0, 0, 4, 4, 1, 11), (9, 9, 4, 4, 1, 12)], [(5, 5, 4, 4, 1, 11), (5, 5, 4, 4, 1, 13)]])
        assert r['status'] == 'DIFF' and r['track_conflicts'] >= 1


def test_exit_status_two_when_only_unconfirmed():
    with tempfile.TemporaryDirectory() as d:
        _pair(d, [[(5, 5, 4, 4, 1, 1), (5, 5, 4, 4, 1, 2)]] * 2, [[(5, 5, 4, 4, 1, 11), (5, 5, 4, 4, 1, 12)]] * 2)
        b = os.path.join(d, 'B')
        open(os.path.join(b, 'train.txt'), 'w').write('C_1\n')
        open(os.path.join(b, 'test.txt'), 'w').write('\n')
        for root in (os.path.join(d, 'A'), b):                       # an empty test split in both copies
            fake_dataset.write(root, {}, track_key='track_id')
            import json
            json.dump(dict(images=[], annotations=[], categories=[]),
                      open(os.path.join(root, 'annotations', 'instances_01_test2017.json'), 'w'))
        assert cd.main(['--a', os.path.join(d, 'A'), '--b', b]) == 2


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
