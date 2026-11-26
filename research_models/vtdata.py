"""VT-Tiny-MOT (thermal channel '01') access for the DeepPro 'RGB-T' protocol.

DeepPro / DeepPro-Plus (Li et al., TPAMI 2026) call this data 'RGB-T': 45 training and
15 test clips selected from the RGBT-Tiny / VT-Tiny-MOT release (train2017 / test2017).
The authors trained with 'rough Gaussian masks generated from the bounding boxes'
(masks were not released).  `gaussian_mask` reconstructs them; the reconstruction rule is
checked against per-clip target counts printed in the authors' released evaluation log
(see code/check_mask_protocol.py).
"""
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

DATA_ROOT = os.environ.get('VT_ROOT', '/data/VT-Tiny-MOT')
OFFICIAL_LISTS = os.environ.get('OFFICIAL_LISTS', '/official/datasets/RGB-T')
CATS = {0: 'ship', 1: 'car', 2: 'cyclist', 3: 'pedestrian', 4: 'bus', 5: 'drone', 6: 'plane'}
# Normalisation constants used by the authors for 'RGB-T' (TrainDataLoader.py)
RGBT_MEAN, RGBT_STD = 85.0799, 47.4845

_ANN_CACHE = {}


def scene_list(name):
    """name in {'train','test'} -> the authors' literal clip lists."""
    return open(os.path.join(OFFICIAL_LISTS, f'{name}.txt')).read().split()


def split_of(scene):
    """train2017 / test2017 directory that stores a clip."""
    for s in ('train2017', 'test2017'):
        if os.path.isdir(os.path.join(DATA_ROOT, s, scene)):
            return s
    raise FileNotFoundError(scene)


def _annotations(split):
    if split not in _ANN_CACHE:
        ann = json.load(open(os.path.join(DATA_ROOT, 'annotations', f'instances_01_{split}.json')))
        imgs = {im['id']: im for im in ann['images']}
        per = defaultdict(lambda: defaultdict(list))
        for a in ann['annotations']:
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


def gaussian_mask(boxes, H, W, rule='sigma_quarter', quant=True):
    """Rough Gaussian target masks from boxes (reconstruction of the DeepPro protocol).

    rule 'sigma_quarter': sigma_x = w/4, sigma_y = h/4, peak 1 at the box centre.
    The image is the per-pixel max over targets, stored as uint8 (x255) and binarised with
    > 0 like the authors' loader (label[label > 0] = 1), unless quant=False.
    """
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
