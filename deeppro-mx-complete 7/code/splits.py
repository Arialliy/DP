"""Recording-independent train / val / test split of the DeepPro RGB-T clips.

Rules (fixed before any model was trained; written to configs/splits.json):
 1. test  = the authors' 15 test clips, unchanged.
 2. Clips in the authors' 45-clip training list whose drone recording id (DJI_XXXX) also
    occurs in the test list are excluded from train AND val (5 clips) so that no recording
    is shared between training data and the test set.
 3. The remaining 40 clips are grouped by recording id. Recordings are stratified into
    static camera (mean background motion < 0.3 px/frame) and moving camera, ordered by
    SHA-256 of the recording id, and the first 20% (rounded up) of each stratum is
    validation.
"""
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import vtdata  # noqa: E402

SPLIT_FILE = os.path.join(os.path.dirname(__file__), '..', 'configs', 'splits.json')
FOLD_FILE = os.path.join(os.path.dirname(__file__), '..', 'configs', 'folds.json')  # code/folds.py


def build(clip_csv='results/profile/clips.csv'):
    import csv
    cam = {r['scene']: float(r['cam_speed_mean']) for r in csv.DictReader(open(clip_csv))}
    train45, test = vtdata.scene_list('train'), vtdata.scene_list('test')
    test_rec = {vtdata.recording_id(c) for c in test}
    excluded = [c for c in train45 if vtdata.recording_id(c) in test_rec]
    pool = [c for c in train45 if c not in excluded]
    recs = {}
    for c in pool:
        recs.setdefault(vtdata.recording_id(c), []).append(c)
    rec_cam = {r: max(cam[c] for c in cs) for r, cs in recs.items()}
    val = []
    strata = {}
    for name, members in (('static', [r for r in recs if rec_cam[r] < 0.3]),
                          ('moving', [r for r in recs if rec_cam[r] >= 0.3])):
        order = sorted(members, key=lambda r: hashlib.sha256(r.encode()).hexdigest())
        k = math.ceil(0.2 * len(order))
        strata[name] = {'recordings': order, 'val': order[:k]}
        for r in order[:k]:
            val += recs[r]
    train = [c for c in pool if c not in val]
    out = {'rule': __doc__.strip(), 'train': sorted(train), 'val': sorted(val), 'test': sorted(test),
           'excluded_shared_recording': sorted(excluded), 'strata': strata,
           'camera_speed_px_per_frame': {c: round(cam[c], 4) for c in train45 + test}}
    json.dump(out, open(SPLIT_FILE, 'w'), indent=1)
    return out


def clips(name):
    """Split name ('train', 'val', 'test', 'all'), cross-validation fold ('fold2/val' is
    fold 2, 'fold2/train' the other non-test clips; configs/folds.json), or a comma list."""
    s = json.load(open(SPLIT_FILE))
    if name in s:
        return s[name]
    if name == 'all':
        return s['train'] + s['val'] + s['test']
    if name.startswith('fold') and '/' in name:
        idx, part = name[4:].split('/', 1)
        folds = json.load(open(FOLD_FILE))['folds']
        i = int(idx)
        if part == 'val':
            return list(folds[i])
        if part == 'train':
            return sorted(c for j, f in enumerate(folds) if j != i for c in f)
        raise ValueError(name)
    return name.split(',')


def selection_plan(split, mode, selection_split, train_clips):
    """How train.py chooses best_model.pth. -> (mode, selection split, selection clips or None).

    mode 'auto': the authors' per-epoch test-window IoU (original_test_iou), except for the
    research splits, which select on their validation clips: split 'train' on 'val',
    'foldK/train' on 'foldK/val' (val_iou). selection_split None means that default.
    Raises ValueError if the selection would see training clips, or test clips under
    val_iou, or if original_test_iou would select on clips that are trained on.
    """
    fold = split.startswith('fold') and split.endswith('/train')
    if mode == 'auto':
        mode = 'val_iou' if (fold or split == 'train') else 'original_test_iou'
    if mode not in ('final', 'original_test_iou', 'val_iou'):
        raise ValueError(f'unknown checkpoint selection {mode!r}')
    if selection_split is None:
        selection_split = split[:-len('train')] + 'val' if fold else 'val'
    test = set(clips('test'))
    if mode == 'original_test_iou':
        trained = sorted(set(train_clips) & test)
        if trained:
            raise ValueError(f'the training clips include test clips {trained}: no test-set selection')
    if mode != 'val_iou':
        return mode, selection_split, None
    chosen = clips(selection_split)
    shared = sorted(set(chosen) & set(train_clips))
    if shared:
        raise ValueError(f'--selection-split {selection_split} overlaps the training clips: {shared}')
    tested = sorted(set(chosen) & test)
    if tested:
        raise ValueError(f'val_iou must not select on test clips ({tested}); use original_test_iou')
    return mode, selection_split, chosen


if __name__ == '__main__':
    o = build()
    for k in ('train', 'val', 'test', 'excluded_shared_recording'):
        print(k, len(o[k]), o[k])
