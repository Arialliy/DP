"""VT-Tiny-MOT (thermal channel '01') access for the DeepPro 'RGB-T' protocol.

DeepPro / DeepPro-Plus (Li et al., TPAMI 2026) call this data 'RGB-T': 45 training and
15 test clips selected from the RGBT-Tiny / VT-Tiny-MOT release (train2017 / test2017).
The paper (Sec. 4.6) says only: "For training, we generated rough Gaussian masks as labels
according to the bounding boxes provided by its authors." The masks were not released.
code/gaussian_masks.py implements that sentence with every unstated choice as a parameter;
code/export_masks.py writes the masks in the layout of the official code, and
code/check_mask_protocol.py compares the parameters with the per-clip target counts of the
authors' released evaluation log. `gaussian_mask` below is the earlier reconstruction that
the training code in this repository uses.
"""
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

DATA_ROOT = os.environ.get('VT_ROOT', '/data/VT-Tiny-MOT')
OFFICIAL_LISTS = os.environ.get('OFFICIAL_LISTS', '/official/datasets/RGB-T')
# Folder of instances_01_{train,test}2017.json. The official DeepPro 'RGB-T' layout
# (train2017/, test2017/, segmentations/, train.txt, test.txt) holds no boxes, so point this
# at the RGBT-Tiny release's annotations_coco/ when VT_ROOT is such a folder. Unset, the
# boxes are looked for in <VT_ROOT>/annotations and then <VT_ROOT>/annotations_coco.
ANN_DIR = os.environ.get('VT_ANN_DIR')
CATS = {0: 'ship', 1: 'car', 2: 'cyclist', 3: 'pedestrian', 4: 'bus', 5: 'drone', 6: 'plane'}
# Normalisation constants used by the authors for 'RGB-T' (TrainDataLoader.py)
RGBT_MEAN, RGBT_STD = 85.0799, 47.4845
# Track ids. The VT-Tiny-MOT copy used earlier gives a clip-unique 'track_id'. The RGBT-Tiny
# release gives 'tracking_id', numbered within each category (a pedestrian and a drone can
# both be track 0), so a track there is (category, tracking_id): track = category *
# TRACK_BASE + tracking_id. Keying by tracking_id alone merges different targets into one
# track and lets boxes of one frame overwrite each other.
TRACK_BASE = 1_000_000
TRACK_SCHEME = f'track_id, else category * {TRACK_BASE} + tracking_id'  # recorded next to targets.csv

_ANN_CACHE = {}


def track_of(a):
    """Clip-unique track id of one COCO annotation (see TRACK_BASE)."""
    if 'track_id' in a:
        return int(a['track_id'])
    raw, cat = int(a['tracking_id']), int(a['category_id'])
    if not 0 <= raw < TRACK_BASE or cat < 0:
        raise ValueError(f'tracking_id {raw} of category {cat} does not fit track = category * {TRACK_BASE} + id')
    return cat * TRACK_BASE + raw


def split_track(track):
    """Inverse of track_of for RGBT-Tiny ids: -> (category, tracking_id)."""
    return int(track) // TRACK_BASE, int(track) % TRACK_BASE


def duplicate_keys(boxes):
    """Boxes that share their frame's (track) key with another box: [(t, track, count)]."""
    out = []
    for t, b in enumerate(boxes):
        if len(b):
            ids, counts = np.unique(np.asarray(b)[:, 5].astype(np.int64), return_counts=True)
            out += [(t, int(i), int(n)) for i, n in zip(ids, counts) if n > 1]
    return out


def scene_list(name):
    """name in {'train','test'} -> the authors' literal clip lists."""
    return open(os.path.join(OFFICIAL_LISTS, f'{name}.txt')).read().split()


def split_of(scene):
    """train2017 / test2017 directory that stores a clip."""
    for s in ('train2017', 'test2017'):
        if os.path.isdir(os.path.join(DATA_ROOT, s, scene)):
            return s
    raise FileNotFoundError(scene)


def annotation_path(split):
    """instances_01_<split>.json from VT_ANN_DIR, else <VT_ROOT>/annotations(_coco)."""
    dirs = [ANN_DIR] if ANN_DIR else [os.path.join(DATA_ROOT, d) for d in ('annotations', 'annotations_coco')]
    for d in dirs:
        p = os.path.join(d, f'instances_01_{split}.json')
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f'instances_01_{split}.json not found in {dirs}. The official RGB-T layout has frames and masks but no '
        f'boxes; set VT_ANN_DIR to the annotations_coco folder of the RGBT-Tiny release.')


def _annotations(split):
    if split not in _ANN_CACHE:
        ann = json.load(open(annotation_path(split)))
        imgs = {im['id']: im for im in ann['images']}
        per = defaultdict(lambda: defaultdict(list))
        for a in ann['annotations']:
            if 'track_id' not in a and 'tracking_id' in a:  # RGBT-Tiny: ids within each category
                a['track_id'] = track_of(a)
            im = imgs[a['image_id']]
            scene, _, fname = im['file_name'].split('/')
            per[scene][fname].append(a)
        _ANN_CACHE[split] = per
    return _ANN_CACHE[split]


class Scene:
    """One clip: sorted frame paths, per-frame boxes (x, y, w, h, cat, track)."""

    def __init__(self, scene):
        self.name = scene
        self.split = split_of(scene)
        d = os.path.join(DATA_ROOT, self.split, scene, '01')
        self.frames = sorted(f for f in os.listdir(d) if f.endswith('.jpg'))
        self.paths = [os.path.join(d, f) for f in self.frames]
        per = _annotations(self.split)[scene]
        self.boxes = []
        for f in self.frames:
            rows = [(*a['bbox'], a['category_id'], a['track_id']) for a in per.get(f, [])
                    if not a.get('ignore', 0)]
            self.boxes.append(np.array(rows, dtype=np.float64).reshape(-1, 6))
        im0 = Image.open(self.paths[0])
        self.W, self.H = im0.size

    def __len__(self):
        return len(self.frames)

    def image(self, i):
        im = np.asarray(Image.open(self.paths[i]), dtype=np.float32)
        if im.ndim == 3:
            im = im[:, :, 0]
        return im

    def video(self):
        return np.stack([self.image(i) for i in range(len(self))])


def gaussian_mask(boxes, H, W, rule='sigma_sixth', quant=True):
    """Earlier reconstruction of the DeepPro masks, used by the training code here.

    rule 'sigma_<x>': sigma = box size / {sixth: 6, quarter: 4, third: 3, half: 2}, peak 1 at
    the box centre, per-pixel maximum over targets. With quant=True the result is binary:
    rint(255 g) > 0, the authors' loader applied to an 8-bit image.
    Two additions are not in the paper: sigma is floored at 0.5 px, and the Gaussian is cut
    to a square of ceil(4 sigma) + 1 px around the centre. For boxes of at least 3 px on both
    sides (sigma >= 0.5) and quant=True this equals
    gaussian_masks.label(boxes, H, W, k=6, storage='round') pixel for pixel; smaller boxes get
    a larger mask here. Masks written to the paper: code/gaussian_masks.py.

    rule 'paper_k6_floor': gaussian_masks with k = 6 and floor storage, the setting that
    reproduces the authors' per-clip test counts most closely (docs/MASKS.md) and the masks
    export_masks.py writes for the official code. With quant=False the float Gaussian.
    """
    if rule == 'paper_k6_floor':
        import gaussian_masks as gm  # local: research_models in the DP tree has no such module
        if quant:
            return gm.label(boxes, H, W, 6.0, 'floor')
        return gm.heatmap(boxes, H, W, 6.0).astype(np.float32)
    m = np.zeros((H, W), np.float32)
    if len(boxes) == 0:
        return m
    yy, xx = np.mgrid[0:H, 0:W]
    div = {'sigma_quarter': 4.0, 'sigma_third': 3.0, 'sigma_half': 2.0, 'sigma_sixth': 6.0}[rule]
    for x, y, w, h, *_ in boxes:
        cx, cy = x + w / 2.0 - 0.5, y + h / 2.0 - 0.5
        sx, sy = max(w / div, 0.5), max(h / div, 0.5)
        r = int(np.ceil(4 * max(sx, sy))) + 1
        x0, x1 = max(int(cx) - r, 0), min(int(cx) + r + 1, W)
        y0, y1 = max(int(cy) - r, 0), min(int(cy) + r + 1, H)
        g = np.exp(-((xx[y0:y1, x0:x1] - cx) ** 2 / (2 * sx ** 2) + (yy[y0:y1, x0:x1] - cy) ** 2 / (2 * sy ** 2)))
        m[y0:y1, x0:x1] = np.maximum(m[y0:y1, x0:x1], g)
    if quant:
        m = (np.round(m * 255.0) > 0).astype(np.float32)
    return m


def recording_id(scene):
    """'DJI_0257_2' -> 'DJI_0257' (clips cut from the same drone recording)."""
    return '_'.join(scene.split('_')[:2])
