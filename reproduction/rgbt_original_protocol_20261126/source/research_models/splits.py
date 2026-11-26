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
from pathlib import Path

from . import vtdata

CONFIG_DIR = Path(__file__).with_name('configs')
SPLIT_FILE = CONFIG_DIR / 'splits.json'
FOLD_FILE = CONFIG_DIR / 'folds.json'


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


if __name__ == '__main__':
    o = build()
    for k in ('train', 'val', 'test', 'excluded_shared_recording'):
        print(k, len(o[k]), o[k])
