"""Render real RGB-T images, original COCO boxes, and saved PNG > 0 contours.

Figure contract: a diagnostic image plate compares annotation and mask locations
in two real examples. The full frame establishes context; an explicitly marked
crop exposes pixel alignment. It is not a full-dataset verification. Python and
matplotlib only; fixed 0..255 grayscale, nearest-neighbor display, no mask
generation, no image enhancement, and no writes to the dataset.

Outputs: alignment_examples.png and alignment_examples.sources.json.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.patheffects as effects
import numpy as np
from PIL import Image


BOX_COLOR = '#27C7E8'
MASK_COLOR = '#FFA94D'
SAMPLES = (
    {'scene': 'DJI_0109_1', 'stem': '00000', 'focus_annotation_ids': [673292],
     'description': 'Single plane', 'zoom_side_pixels': 56},
    {'scene': 'DJI_0049_2', 'stem': '00000', 'focus_annotation_ids': [626242, 626243],
     'description': 'Adjacent pedestrians', 'zoom_side_pixels': 56},
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_sample(root, spec, images, annotations, categories):
    relative = f'{spec["scene"]}/01/{spec["stem"]}.jpg'
    records = [image for image in images if image['file_name'] == relative]
    if len(records) != 1:
        raise ValueError(f'Expected exactly one COCO image record: {relative}')
    image_record = records[0]
    image_path = root / 'test2017' / relative
    mask_path = root / 'segmentations' / spec['scene'] / (spec['stem'] + '.png')
    with Image.open(image_path) as image:
        pixels = np.array(image)
        image_mode = image.mode
    with Image.open(mask_path) as image:
        mask = np.array(image)
        mask_mode = image.mode
    if pixels.dtype != np.uint8 or mask.dtype != np.uint8 or pixels.ndim != 2 or mask.ndim != 2:
        raise ValueError('These examples require untouched 2-D uint8 image and mask arrays')
    if pixels.shape != mask.shape or pixels.shape != (image_record['height'], image_record['width']):
        raise ValueError('Image, mask and COCO dimensions disagree')
    boxes = [annotation for annotation in annotations if annotation['image_id'] == image_record['id']]
    if any(annotation.get('ignore', 0) for annotation in boxes):
        raise ValueError('An ignored annotation requires an explicit separate display style')
    focus = [a for a in boxes if a['id'] in spec['focus_annotation_ids']]
    if len(focus) != len(spec['focus_annotation_ids']):
        raise ValueError('Missing predeclared focus annotations')
    left = min(a['bbox'][0] for a in focus)
    top = min(a['bbox'][1] for a in focus)
    right = max(a['bbox'][0] + a['bbox'][2] for a in focus)
    bottom = max(a['bbox'][1] + a['bbox'][3] for a in focus)
    side = spec['zoom_side_pixels']
    height, width = pixels.shape
    x0 = max(0, min(width - side, int(np.floor((left + right - side) / 2))))
    y0 = max(0, min(height - side, int(np.floor((top + bottom - side) / 2))))
    crop = (x0, y0, x0 + side, y0 + side)
    foreground = mask > 0
    rows, columns = np.nonzero(foreground)
    provenance = dict(
        scene=spec['scene'], frame_stem=spec['stem'], split='test', modality='01 / infrared',
        description=spec['description'], image_record=image_record,
        image_path=str(image_path), image_sha256=digest(image_path), image_mode=image_mode,
        mask_path=str(mask_path), mask_sha256=digest(mask_path), mask_mode=mask_mode,
        source_shape_hw=list(pixels.shape), annotations=boxes,
        category_names={str(a['category_id']): categories[a['category_id']] for a in boxes},
        focus_annotation_ids=spec['focus_annotation_ids'],
        focus_tracking_ids=[a['tracking_id'] for a in focus],
        zoom_xyxy_exclusive=list(crop), nonzero_mask_pixels=int(foreground.sum()),
        foreground_extent_xyxy_exclusive=[int(columns.min()), int(rows.min()),
                                         int(columns.max()) + 1, int(rows.max()) + 1],
        mask_value_range=[int(mask.min()), int(mask.max())],
    )
    return pixels, foreground, boxes, crop, provenance


def image_overlay(axis, pixels, foreground, annotations, crop=None):
    height, width = pixels.shape
    axis.imshow(pixels, cmap='gray', vmin=0, vmax=255, origin='upper', interpolation='nearest',
                extent=(0, width, height, 0))
    # Use COCO continuous coordinates: pixel (u,v) is centered at (u+.5,v+.5).
    # The source boxes, image pixels and contour must share this convention.
    axis.contour(np.arange(width) + .5, np.arange(height) + .5, foreground.astype(np.uint8),
                 levels=[.5], colors=[MASK_COLOR], linewidths=1.0, zorder=3)
    for annotation in annotations:
        x, y, w, h = annotation['bbox']
        axis.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=BOX_COLOR,
                                 linewidth=1.1, zorder=4))
    if crop is None:
        axis.set_xlim(0, width)
        axis.set_ylim(height, 0)
        axis.set_xticks([0, 160, 320, 480, 639])
        axis.set_yticks([0, 128, 256, 384, 511])
    else:
        x0, y0, x1, y1 = crop
        axis.set_xlim(x0, x1)
        axis.set_ylim(y1, y0)
        axis.set_xticks([x0, x0 + 14, x0 + 28, x0 + 42, x1 - 1])
        axis.set_yticks([y0, y0 + 14, y0 + 28, y0 + 42, y1 - 1])
    axis.set_aspect('equal')
    axis.set_xlabel('Original image x (pixels)', labelpad=3)
    axis.set_ylabel('Original image y (pixels)', labelpad=3)
    axis.tick_params(length=3, pad=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/md0/ly/DP/datasets/RGB-T'))
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    annotation_path = root / 'annotations_coco/instances_01_test2017.json'
    raw = annotation_path.read_bytes()
    annotation_hash = hashlib.sha256(raw).hexdigest()
    coco = json.loads(raw)
    del raw
    categories = {category['id']: category['name'] for category in coco['categories']}
    samples = [load_sample(root, spec, coco['images'], coco['annotations'], categories) for spec in SAMPLES]
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 9, 'axes.titlesize': 11, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.spines.right': False, 'axes.spines.top': False,
        'axes.linewidth': .7, 'legend.frameon': False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(12.3, 10.4),
                             gridspec_kw={'width_ratios': [1.25, 1]})
    fig.subplots_adjust(left=.07, right=.975, bottom=.11, top=.86, hspace=.35, wspace=.18)
    fig.suptitle('RGB-T annotation and saved-mask alignment', y=.975, fontsize=16, fontweight='bold')
    fig.text(.5, .943, 'Original infrared images with COCO boxes and existing PNG foreground contours',
             ha='center', fontsize=10, color='#414141')
    handles = [Line2D([0], [0], color=BOX_COLOR, lw=1.8, label='Original COCO bounding box'),
               Line2D([0], [0], color=MASK_COLOR, lw=1.8, label='Saved PNG > 0 contour'),
               Line2D([0], [0], color='#777777', lw=1.2, ls='--', label='Area enlarged at right')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .923),
               ncol=3, fontsize=9, columnspacing=2.5)
    for row, (pixels, foreground, boxes, crop, source) in enumerate(samples):
        full, zoom = axes[row]
        image_overlay(full, pixels, foreground, boxes)
        image_overlay(zoom, pixels, foreground, boxes, crop=crop)
        x0, y0, x1, y1 = crop
        crop_box = Rectangle((x0, y0), x1 - x0, y1 - y0,
                             fill=False, edgecolor='white', linestyle='--', linewidth=1.0,
                             path_effects=[effects.Stroke(linewidth=2, foreground='#333333'), effects.Normal()])
        full.add_patch(crop_box)
        full.set_title(f'{source["scene"]} / {source["frame_stem"]}  |  full frame', loc='left', pad=9)
        zoom.set_title(source['description'] + '  |  56 x 56 px', loc='left', pad=9)
        for column, axis in enumerate((full, zoom)):
            axis.text(-.105, 1.055, chr(ord('a') + row * 2 + column),
                      transform=axis.transAxes, fontsize=12, fontweight='bold', va='top')
        track_text = ', '.join(map(str, source['focus_tracking_ids']))
        zoom.text(.5, -.17, f'COCO tracking_id: {track_text}', transform=zoom.transAxes,
                  ha='center', fontsize=9, color='#414141')
    fig.text(.07, .045, 'No brightness, contrast, gamma, or mask regeneration applied. Grayscale fixed at 0–255; nearest-neighbor display.',
             fontsize=9, color='#414141')
    fig.text(.07, .023, 'The mask is the existing uint8 PNG thresholded at > 0. The orange boundary is its 0.5 contour; two examples are shown.',
             fontsize=9, color='#414141')
    destination = output / 'alignment_examples.png'
    fig.savefig(destination, dpi=220, facecolor='white')
    plt.close(fig)
    # Ensure that rendering changed none of the source image/mask bytes.
    for *_, source in samples:
        assert digest(source['image_path']) == source['image_sha256']
        assert digest(source['mask_path']) == source['mask_sha256']
    metadata = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        figure=str(destination), figure_sha256=digest(destination),
        script=str(Path(__file__).resolve()), script_sha256=digest(__file__),
        annotation_path=str(annotation_path), annotation_sha256=annotation_hash,
        samples=[sample[-1] for sample in samples],
        integrity=dict(source_images_modified=False, source_masks_modified=False,
                       masks_regenerated=False, grayscale_vmin=0, grayscale_vmax=255,
                       gamma_applied=False, display_interpolation='nearest',
                       contour='matplotlib contour at 0.5 of the existing PNG > 0 boolean array',
                       crop_coordinates='COCO continuous coordinates; pixel centers u+0.5,v+0.5; crop upper bounds exclusive'),
        figure_contract=dict(archetype='diagnostic image plate', backend='Python/matplotlib',
                             conclusion='Compare original boxes and saved mask locations in two actual frames.',
                             limitation='Visual examples do not establish full-dataset correspondence or author-mask equivalence.'),
        runtime=dict(matplotlib=matplotlib.__version__, numpy=np.__version__, gpu_used=False),
    )
    (output / 'alignment_examples.sources.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(figure=str(destination), sources=str(output / 'alignment_examples.sources.json'),
                         source_files_unchanged=True, samples=len(samples)), ensure_ascii=False))


if __name__ == '__main__':
    main()
