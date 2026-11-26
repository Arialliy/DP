"""Read-only, full-frame audit against COCO boxes; no exporter imports."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy import ndimage

OUT = Path(__file__).resolve().parent
DATA = OUT.parents[1] / 'datasets/RGB-T'
SOURCE = OUT.parents[1] / 'datasets/RGBT-Tiny'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    tasks, checks = [], {}
    for split in ('train', 'test'):
        clips = (DATA / (split + '.txt')).read_text().split()
        assert len(clips) == len(set(clips))
        path = SOURCE / 'annotations_coco' / f'instances_01_{split}2017.json'
        source_sha = digest(path)
        assert digest(DATA / 'annotations_coco' / path.name) == source_sha
        doc = json.loads(path.read_text())
        images, filenames = {}, set()
        by_clip = defaultdict(dict)
        for im in doc['images']:
            assert im['id'] not in images, 'Duplicate image ID'
            assert im['file_name'] not in filenames, 'Duplicate image filename'
            filenames.add(im['file_name'])
            clip, channel, name = im['file_name'].split('/')
            assert channel == '01', 'Mixed modality'
            images[im['id']] = (clip, name)
            if clip in clips:
                assert (im['width'], im['height']) == (640, 512)
                by_clip[clip][name] = []
        ids = set()
        skipped = dict(ignored=0, invalid=0)
        for ann in doc['annotations']:
            assert ann['id'] not in ids, 'Duplicate annotation ID'
            ids.add(ann['id'])
            assert ann['image_id'] in images, 'Orphan annotation'
            clip, name = images[ann['image_id']]
            if clip not in clips:
                continue
            box = tuple(float(x) for x in ann['bbox'])
            assert len(box) == 4 and np.isfinite(box).all()
            if ann.get('ignore', 0):
                skipped['ignored'] += 1
            elif box[2] <= 0 or box[3] <= 0:
                skipped['invalid'] += 1
            else:
                by_clip[clip][name].append(box)
        checks[split] = dict(annotation_file=str(path), annotation_sha256=source_sha,
                            all_image_ids_unique=True, all_filenames_unique=True,
                            all_annotation_ids_unique=True, orphan_annotations=0,
                            channel='01', selected_clips=len(clips), skipped=skipped)
        for clip in clips:
            image_dir = DATA / (split + '2017') / clip / '01'
            mask_dir = DATA / 'segmentations' / clip
            names = sorted(by_clip[clip])
            assert names == sorted(f.name for f in image_dir.iterdir())
            assert [Path(n).with_suffix('.png').name for n in names] == sorted(f.name for f in mask_dir.iterdir())
            tasks.append((split, clip, sorted(by_clip[clip].items())))
    return tasks, checks


def expected_mask(boxes):
    """Evaluate the documented formula independently on pixel-center coordinates."""
    mask = np.zeros((512, 640), dtype=np.uint8)
    vanished = 0
    for x, y, w, h in boxes:
        cx, cy, sx, sy = x + w / 2, y + h / 2, w / 6, h / 6
        x0 = max(0, math.floor(cx - 6 * sx - .5))
        x1 = min(640, math.ceil(cx + 6 * sx - .5) + 1)
        y0 = max(0, math.floor(cy - 6 * sy - .5))
        y1 = min(512, math.ceil(cy + 6 * sy - .5) + 1)
        if x0 >= x1 or y0 >= y1:
            vanished += 1
            continue
        dx = (np.arange(x0, x1, dtype=np.float64) + .5 - cx) / sx
        dy = (np.arange(y0, y1, dtype=np.float64) + .5 - cy) / sy
        # Direct exp of summed exponents, unlike the exporter's separable product.
        patch = np.floor(255 * np.exp(-.5 * (dy[:, None] ** 2 + dx[None, :] ** 2))).astype(np.uint8)
        vanished += int(not patch.any())
        np.maximum(mask[y0:y1, x0:x1], patch, out=mask[y0:y1, x0:x1])
    return mask, vanished


def audit_clip(task):
    split, clip, rows = task
    result = dict(split=split, clip=clip, frames=len(rows), boxes=0,
                  exact_gray_frames=0, exact_binary_frames=0, different_gray_pixels=0,
                  different_binary_pixels=0, max_gray_error=0, vanished_boxes=0,
                  targets=0, mismatch_examples=[])
    for name, boxes in rows:
        with Image.open(DATA / (split + '2017') / clip / '01' / name) as im:
            assert im.size == (640, 512) and im.mode == 'L'
        path = DATA / 'segmentations' / clip / Path(name).with_suffix('.png')
        with Image.open(path) as im:
            assert im.size == (640, 512) and im.mode == 'L'
            actual = np.asarray(im, dtype=np.uint8)
        expected, vanished = expected_mask(boxes)
        gray_diff = int(np.count_nonzero(actual != expected))
        binary_diff = int(np.count_nonzero((actual > 0) != (expected > 0)))
        result['boxes'] += len(boxes)
        result['vanished_boxes'] += vanished
        result['exact_gray_frames'] += int(gray_diff == 0)
        result['exact_binary_frames'] += int(binary_diff == 0)
        result['different_gray_pixels'] += gray_diff
        result['different_binary_pixels'] += binary_diff
        if gray_diff:
            result['max_gray_error'] = max(result['max_gray_error'], int(np.abs(actual.astype(np.int16) - expected.astype(np.int16)).max()))
            if len(result['mismatch_examples']) < 5:
                result['mismatch_examples'].append(dict(frame=name, gray_pixels=gray_diff, binary_pixels=binary_diff))
        if split == 'test':
            result['targets'] += int(ndimage.label(actual > 0, structure=np.ones((3, 3), np.uint8))[1])
    return result


def main():
    started = time.monotonic()
    tasks, checks = prepare()
    print('Checked COCO identity, modality and frame mapping; starting all PNGs', flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(audit_clip, tasks):
            results.append(result)
            print(f"{result['clip']}: {result['frames']} frames, binary differences={result['different_binary_pixels']}, gray differences={result['different_gray_pixels']}", flush=True)
    keys = ['frames', 'boxes', 'exact_gray_frames', 'exact_binary_frames',
            'different_gray_pixels', 'different_binary_pixels', 'vanished_boxes']
    totals = {key: sum(row[key] for row in results) for key in keys}
    report = dict(completed_at=datetime.now(timezone.utc).isoformat(), dataset=str(DATA),
                  status='passed' if totals['different_gray_pixels'] == 0 else 'review_required',
                  formula='independent direct-exp k=6, floor, COCO xywh, pixel centers +0.5, max over boxes',
                  scope='All selected COCO annotations and all 25111 stored masks; does not certify equality to author masks or object silhouettes',
                  annotation_checks=checks, totals=totals,
                  test_components=sum(r['targets'] for r in results),
                  elapsed_seconds=time.monotonic() - started, per_clip=results)
    (OUT / 'alignment_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'per_clip'}), flush=True)


if __name__ == '__main__':
    main()
