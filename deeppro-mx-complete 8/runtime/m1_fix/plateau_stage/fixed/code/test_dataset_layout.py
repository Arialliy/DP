"""Checks of the official RGB-T layout: boxes from VT_ANN_DIR, labels against the PNGs,
and the guard against a targets.csv from other annotations (numpy only).

Run: python code/test_dataset_layout.py
"""
import os
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_masks  # noqa: E402
import fake_dataset  # noqa: E402
import mx_data  # noqa: E402
import vtdata  # noqa: E402
import check_labels  # noqa: E402


def _clip(n=12, H=40, W=60, seed=0):
    rng = np.random.default_rng(seed)
    video = rng.normal(80, 3, (n, H, W))
    boxes = []
    for t in range(n):
        boxes.append(np.array([(5 + t, 10, 6, 4, 1, 3), (40, 25.5, 3.5, 2.5, 3, 4)], float))
        video[t, 10:14, 5 + t:11 + t] += 50
    return video, boxes


class Layout:
    """An official-layout RGB-T root (train2017/, test2017/, segmentations/, lists) and a
    separate annotations_coco/ folder, as on the training machine."""

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.root, self.ann = os.path.join(d, 'RGB-T'), os.path.join(d, 'RGBT-Tiny', 'annotations_coco')
        self.clips = {'A_1': _clip(seed=1) + ('train2017',), 'B_2': _clip(seed=2) + ('test2017',)}
        fake_dataset.write(self.root, self.clips, track_key='tracking_id', ann_dir=self.ann)
        for name, split in (('train', 'A_1'), ('test', 'B_2')):
            open(os.path.join(self.root, f'{name}.txt'), 'w').write(split + '\n')
        self.saved = vtdata.DATA_ROOT, vtdata.ANN_DIR, vtdata.OFFICIAL_LISTS
        vtdata.DATA_ROOT, vtdata.ANN_DIR, vtdata.OFFICIAL_LISTS = self.root, self.ann, self.root
        vtdata._ANN_CACHE.clear()
        for c in self.clips:
            export_masks.export_clip((c, os.path.join(self.root, 'segmentations'), 6.0, 'floor', False))
        return self

    def __exit__(self, *exc):
        vtdata.DATA_ROOT, vtdata.ANN_DIR, vtdata.OFFICIAL_LISTS = self.saved
        vtdata._ANN_CACHE.clear()
        self.tmp.cleanup()


def test_boxes_come_from_the_annotation_folder():
    with Layout() as L:
        sc = vtdata.Scene('A_1')
        B = vtdata.TRACK_BASE                                      # category * B + tracking_id
        assert len(sc) == 12 and sc.boxes[3].shape == (2, 6) and list(sc.boxes[3][:, 5]) == [1 * B + 3, 3 * B + 4]
        assert not os.path.exists(os.path.join(L.root, 'annotations'))
        vtdata.ANN_DIR = None                                    # official layout alone: a clear error
        vtdata._ANN_CACHE.clear()
        try:
            vtdata.Scene('A_1')
            raise AssertionError('no boxes, no error')
        except FileNotFoundError as e:
            assert 'VT_ANN_DIR' in str(e)
        os.symlink(L.ann, os.path.join(L.root, 'annotations_coco'))   # or the folder inside VT_ROOT
        vtdata._ANN_CACHE.clear()
        assert len(vtdata.Scene('B_2')) == 12


def test_paper_labels_equal_the_exported_pngs():
    with Layout() as L:
        assert check_labels.main(['--nproc', '1']) == 0
        assert check_labels.main(['--nproc', '1', '--rule', 'sigma_sixth']) == 1   # the earlier masks differ
        stem = os.path.splitext(vtdata.Scene('A_1').frames[5])[0]
        p = os.path.join(L.root, 'segmentations', 'A_1', stem + '.png')
        img = np.array(Image.open(p))
        img[0, 0] = 255                                           # one wrong pixel
        Image.fromarray(img).save(p)
        _, r = check_labels.check_clip(('A_1', 'paper_k6_floor', os.path.join(L.root, 'segmentations')))
        assert not r['ok'] and r['differing_frames'] == 1 and r['differing_pixels'] == 1
        assert r['first_difference']['frame'] == stem + '.jpg'
        os.remove(os.path.join(L.root, 'segmentations', 'B_2', stem + '.png'))
        _, r = check_labels.check_clip(('B_2', 'paper_k6_floor', os.path.join(L.root, 'segmentations')))
        assert r['missing'] == 1 and not r['ok']


def test_tracking_ids_within_categories_become_distinct_tracks():
    """RGBT-Tiny numbers tracks within each category: a pedestrian and a drone can both be
    track 0. They must stay two tracks, and every box must get its own speed row."""
    import profile_motion
    B = vtdata.TRACK_BASE
    assert vtdata.track_of({'tracking_id': 0, 'category_id': 3}) == 3 * B
    assert vtdata.track_of({'tracking_id': 0, 'category_id': 5}) == 5 * B
    assert vtdata.track_of({'track_id': 1903000, 'tracking_id': 4, 'category_id': 6}) == 1903000   # older copy
    assert vtdata.split_track(5 * B + 17) == (5, 17)
    try:
        vtdata.track_of({'tracking_id': B, 'category_id': 1})
        raise AssertionError('an id that does not fit was accepted')
    except ValueError:
        pass
    n = 30
    video = np.random.default_rng(0).normal(80, 3, (n, 40, 60))
    boxes = []
    for t in range(n):                       # pedestrian 0 walks 1 px/frame; drone 0 hovers
        video[t, 5:9, 5 + t:8 + t] += 40
        video[t, 25:28, 40:44] += 40
        boxes.append(np.array([(5 + t, 5, 3, 4, 3, 0), (40, 25, 4, 3, 5, 0)], float))
    with tempfile.TemporaryDirectory() as d:
        saved = vtdata.DATA_ROOT, vtdata.ANN_DIR
        try:
            fake_dataset.write(d, {'P_1': (video, boxes, 'train2017')}, track_key='tracking_id')
            vtdata.DATA_ROOT, vtdata.ANN_DIR = d, None
            vtdata._ANN_CACHE.clear()
            sc = vtdata.Scene('P_1')
        finally:
            vtdata.DATA_ROOT, vtdata.ANN_DIR = saved
            vtdata._ANN_CACHE.clear()
    assert vtdata.duplicate_keys(sc.boxes) == []
    assert {int(k) for b in sc.boxes for k in b[:, 5]} == {3 * B, 5 * B}
    rows, n_tracks, n_dup = profile_motion.target_rows('P_1', sc.boxes, np.repeat(np.eye(3)[None], n - 1, 0))
    assert len(rows) == 2 * n and n_tracks == 2 and n_dup == 0
    walk = [r[10] for r in rows if r[2] == 3 * B]
    hover = [r[10] for r in rows if r[2] == 5 * B]
    assert np.allclose(walk, 1.0) and np.allclose(hover, 0.0)
    # keyed by tracking_id alone (the earlier mapping), the two would collide in every frame
    raw = [b.copy() for b in sc.boxes]
    for b in raw:
        b[:, 5] = b[:, 5] % B
    assert len(vtdata.duplicate_keys(raw)) == n
    rows, n_tracks, n_dup = profile_motion.target_rows('P_1', raw, np.repeat(np.eye(3)[None], n - 1, 0))
    assert len(rows) == 2 * n and n_dup == 2 * n and all(np.isnan(r[10]) for r in rows)   # kept, marked ambiguous


def _refused(boxes_by_clip, speed, path, **kw):
    try:
        mx_data.require_speeds(boxes_by_clip, speed, path, **kw)
    except ValueError as e:
        assert 'profile_motion.py' in str(e)
        return True
    return False


def test_targets_from_other_annotations_are_refused():
    import json
    B = vtdata.TRACK_BASE
    _, boxes = _clip()
    for b in boxes:                                      # clip-unique ids, as vtdata gives them
        b[:, 5] = b[:, 4] * B + b[:, 5]
    speed = {('A_1', t, int(r[5])): 0.7 for t, b in enumerate(boxes) for r in b}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'targets.csv')
        assert _refused({'A_1': boxes}, speed, path)                       # no targets.json: an old file
        json.dump(dict(track_ids='tracking_id'), open(os.path.join(d, 'targets.json'), 'w'))
        assert _refused({'A_1': boxes}, speed, path)                       # another scheme
        json.dump(dict(track_ids=vtdata.TRACK_SCHEME), open(os.path.join(d, 'targets.json'), 'w'))
        assert mx_data.speed_coverage('A_1', boxes, speed) == 1.0
        assert not _refused({'A_1': boxes}, speed, path)
        shifted = {(c, t, k + 1000): v for (c, t, k), v in speed.items()}  # other track ids
        assert mx_data.speed_coverage('A_1', boxes, shifted) == 0.0 and _refused({'A_1': boxes}, shifted, path)
        # coverage is compared unrounded: 89.96 % is not 90 %
        many = [np.array([(i, 0, 2, 2, 1, B + i) for i in range(10000)], float)]
        sp = {('M_1', 0, B + i): 0.1 for i in range(8996)}
        assert abs(mx_data.speed_coverage('M_1', many, sp) - 0.8996) < 1e-12
        assert _refused({'M_1': many}, sp, path, min_cover=0.9)
        assert _refused({'M_1': many}, sp, path)                          # the default asks for every box
        # a file keyed by the raw tracking_id matches every ship (category 0: id unchanged) but no other
        # box, so a 95 % ship clip would pass a 90 % threshold; asking for every box refuses it, and a file
        # from before the fix has no targets.json either
        ships = [np.array([(i, 0, 2, 2, 0, i) for i in range(95)] + [(i, 9, 2, 2, 1, B + i) for i in range(5)],
                          float)]
        raw = {('S_1', 0, i): 0.0 for i in range(95)} | {('S_1', 0, i): 0.0 for i in range(5)}
        assert mx_data.speed_coverage('S_1', ships, raw) == 0.95 and _refused({'S_1': ships}, raw, path)


def test_m0_exit_status_stops_a_pipeline():
    """exemplar_diag.py exits 0 on GO and 3 on NO-GO."""
    import subprocess
    import test_exemplar_diag as ted
    code = os.path.dirname(os.path.abspath(__file__))
    for shape, expect in (('round', 0), ('car', 3)):
        video, boxes, speed, _, _ = ted.scene(hot_shape=shape)
        with tempfile.TemporaryDirectory() as d:
            fake_dataset.write(d, {'syn': (video, boxes, 'train2017')})
            fake_dataset.write_targets(os.path.join(d, 'targets.csv'), speed, {'syn': boxes})
            env = dict(os.environ, VT_ROOT=d, VT_ANN_DIR='')
            env.pop('VT_ANN_DIR')
            r = subprocess.run([sys.executable, os.path.join(code, 'exemplar_diag.py'), '--clips', 'syn',
                                '--targets', 'targets.csv', '--coords', 'raw', '--nproc', '1', '--key-clips', 'syn'],
                               cwd=d, env=env, capture_output=True, text=True)
            assert r.returncode == expect, (shape, r.returncode, r.stdout[-500:], r.stderr[-2000:])


def test_profile_motion_feeds_m0_with_annotations_inside_the_root():
    """The setup of the training machine: RGB-T/annotations_coco, no VT_ANN_DIR.
    profile_motion.py writes targets.csv and homographies; M0 reads them (auto coordinates)."""
    import exemplar_diag
    import profile_motion
    import test_stabilize as ts
    pan_video, pan_boxes, _ = ts.scene()                       # 1.5 px/frame pan, parked car, textured ground
    rng = np.random.default_rng(3)                              # static camera: a driving car and a parked one
    stat = rng.normal(80, 4, (120, 96, 128)).astype(np.float32)
    stat_boxes = []
    for t in range(120):
        x = 5 + t % 100
        stat[t, 10:14, x:x + 6] += 50
        stat[t, 60:63, 40:47] += 40
        stat_boxes.append(np.array([(x, 10, 6, 4, 1, 3), (40, 60, 7, 3, 1, 4)], float))
    tmp = tempfile.TemporaryDirectory()
    root = os.path.join(tmp.name, 'RGB-T')
    saved = vtdata.DATA_ROOT, vtdata.ANN_DIR, vtdata.OFFICIAL_LISTS, os.getcwd()
    try:
        fake_dataset.write(root, {'pan_1': (pan_video, pan_boxes, 'train2017'),
                                  'stat_1': (stat, stat_boxes, 'test2017')},
                           ann_dir=os.path.join(root, 'annotations_coco'))
        open(os.path.join(root, 'train.txt'), 'w').write('pan_1\n')
        open(os.path.join(root, 'test.txt'), 'w').write('stat_1\n')
        vtdata.DATA_ROOT, vtdata.ANN_DIR, vtdata.OFFICIAL_LISTS = root, None, root
        vtdata._ANN_CACHE.clear()
        os.chdir(tmp.name)
        os.environ['NPROC'] = '2'
        profile_motion.main()
        assert os.path.exists('results/profile/targets.csv') and os.path.exists('results/profile/clips.csv')
        H = np.load('results/profile/homographies/pan_1.npy')
        assert H.shape == (159, 3, 3) and abs(np.median(H[:, 0, 2]) + 1.5) < 0.2     # the pan is recovered
        import motion_image
        motion_image.main(['--nproc', '1'])                       # the annotation-free homographies
        Hi = np.load('results/motion/homographies/pan_1.npy')      # that compensated windows read
        assert Hi.shape == (159, 3, 3) and abs(np.median(Hi[:, 0, 2]) + 1.5) < 0.2
        sp = mx_data.read_speeds('results/profile/targets.csv')
        assert mx_data.speed_coverage('stat_1', vtdata.Scene('stat_1').boxes, sp) == 1.0
        parked = [v for (c, t, k), v in sp.items() if c == 'pan_1' and np.isfinite(v)]
        assert parked and np.median(parked) < 0.3                  # the parked car is near-stationary
        summary = exemplar_diag.main(['--clips', 'pan_1,stat_1', '--coords', 'auto', '--nproc', '1',
                                      '--min-pos', '1', '--out', 'diag'])
        assert summary['per_clip']['pan_1']['warped_windows'] > 0   # auto compensates the pan
        assert summary['per_clip']['stat_1']['mean_exemplars'] > 0  # the driving car gives exemplars
    finally:
        vtdata.DATA_ROOT, vtdata.ANN_DIR, vtdata.OFFICIAL_LISTS = saved[:3]
        vtdata._ANN_CACHE.clear()
        os.chdir(saved[3])
        tmp.cleanup()


def test_score_author_protocol_uses_the_paper_masks_by_default():
    import json
    import score
    with Layout():
        d, saved_argv = os.getcwd(), sys.argv
        work = tempfile.TemporaryDirectory()
        try:
            os.chdir(work.name)
            os.makedirs('cache/preds/r')
            n = len(vtdata.Scene('A_1'))
            np.save('cache/preds/r/A_1.npy', np.full((n, 40, 60), -20.0, np.float16))
            for extra, rule in (([], 'paper_k6_floor'), (['--label-rule', 'sigma_sixth'], 'sigma_sixth')):
                sys.argv = ['score.py', '--run', 'r', '--split', 'A_1', '--author', '--nproc', '1'] + extra
                score.main()
                summ = json.load(open('results/scores/r/A_1/summary.json'))
                assert summ['author_label_rule'] == rule
        finally:
            sys.argv = saved_argv
            os.chdir(d)
            work.cleanup()


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
