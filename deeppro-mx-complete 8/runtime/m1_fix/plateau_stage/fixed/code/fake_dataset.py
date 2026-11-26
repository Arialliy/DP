"""A tiny VT-Tiny-MOT / RGBT-Tiny style dataset on disk, for end-to-end tests.

write(root, clips) creates
    root/<split>/<clip>/01/000000.jpg ...           grey frames (JPEG, quality 100)
    root/annotations/instances_01_<split>.json      COCO boxes with category_id and a track id
under the key 'tracking_id' (as in the RGBT-Tiny release) or 'track_id'.
"""
import json
import os

import numpy as np
from PIL import Image


def write(root, clips, track_key='tracking_id', ann_dir=None):
    """clips: {name: (video (n,H,W) array, boxes per frame (x, y, w, h, cat, track), split)}.
    ann_dir: folder of the json files (default root/annotations)."""
    per_split = {}
    for name, (video, boxes, split) in clips.items():
        d = os.path.join(root, split, name, '01')
        os.makedirs(d, exist_ok=True)
        ann = per_split.setdefault(split, dict(images=[], annotations=[], categories=[]))
        for t, frame in enumerate(np.asarray(video)):
            fname = f'{t:06d}.jpg'
            img = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8), mode='L')
            img.save(os.path.join(d, fname), quality=100)
            img_id = len(ann['images']) + 1
            ann['images'].append(dict(id=img_id, file_name=f'{name}/01/{fname}', width=frame.shape[1],
                                      height=frame.shape[0]))
            for x, y, w, h, c, tid in np.asarray(boxes[t]).reshape(-1, 6):
                ann['annotations'].append({'id': len(ann['annotations']) + 1, 'image_id': img_id,
                                           'bbox': [float(x), float(y), float(w), float(h)],
                                           'category_id': int(c), track_key: int(tid), 'ignore': 0})
    ann_dir = ann_dir or os.path.join(root, 'annotations')
    os.makedirs(ann_dir, exist_ok=True)
    for split, ann in per_split.items():
        with open(os.path.join(ann_dir, f'instances_01_{split}.json'), 'w') as f:
            json.dump(ann, f)


def write_targets(path, speed, boxes_by_clip, track_key='tracking_id'):
    """A targets.csv (profile_motion.py columns) from {(clip, t, raw track): rel_speed}.
    With track_key 'tracking_id' (ids within each category, as write() stores them) the file
    holds vtdata's clip-unique ids, category * vtdata.TRACK_BASE + id, as profile_motion.py
    would write them."""
    import vtdata
    with open(os.path.splitext(path)[0] + '.json', 'w') as f:      # as profile_motion.py writes it
        json.dump(dict(track_ids=vtdata.TRACK_SCHEME), f)
    with open(path, 'w') as f:
        f.write('scene,t,track,cat,cx,cy,w,h,image_speed,camera_speed,rel_speed,span\n')
        for clip, boxes in boxes_by_clip.items():
            for t, b in enumerate(boxes):
                for x, y, w, h, c, tid in np.asarray(b).reshape(-1, 6):
                    v = speed.get((clip, t, int(tid)), float('nan'))
                    key = vtdata.track_of({track_key: int(tid), 'category_id': int(c)})
                    f.write(f'{clip},{t},{key},{int(c)},{x + w / 2},{y + h / 2},{w},{h},{v},0,{v},5\n')
