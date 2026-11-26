"""Audit exported RGB-T Gaussian PNGs and the official on-disk data layout.

Decode every PNG, match every JPEG name/size, recompute three frames per clip
from the original COCO boxes, and count all test labels with 8-connectivity.
Run only after the image copy and mask export have completed.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image
from scipy import ndimage

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "datasets/paper-masks/code"))
import gaussian_masks as gm

VARIANT = "k6_floor_without_ignored"
STRUCTURE = np.ones((3, 3), dtype=np.uint8)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def regular_directory(path):
    require(path.is_dir() and not path.is_symlink(), f"Not a real directory: {path}")


def prepare_tasks(root, source, metadata, protocol):
    tasks = []
    all_clips = []
    for split, expected_clips, expected_frames in (("train", 45, 19135), ("test", 15, 5976)):
        list_path = root / f"{split}.txt"
        require(list_path.read_bytes() == (source / f"{split}.txt").read_bytes(),
                f"Split list differs from previously checked official list: {list_path}")
        clips = list_path.read_text().split()
        require(len(clips) == len(set(clips)) == expected_clips, f"Unexpected {split} clips")
        all_clips.extend(clips)
        split_dir = root / f"{split}2017"
        regular_directory(split_dir)
        require({p.name for p in split_dir.iterdir()} == set(clips), f"Unexpected {split} directory entries")
        frames = {}
        samples = {}
        for clip in clips:
            clip_dir = split_dir / clip
            regular_directory(clip_dir)
            require({p.name for p in clip_dir.iterdir()} == {"01"}, f"Unexpected channel directory: {clip_dir}")
            image_dir = clip_dir / "01"
            regular_directory(image_dir)
            entries = sorted(image_dir.iterdir())
            require(entries and all(p.is_file() and not p.is_symlink() and p.suffix == ".jpg" for p in entries),
                    f"Unexpected/nonregular image file in {image_dir}")
            source_names = sorted(p.name for p in (source / "images" / clip / "01").glob("*.jpg"))
            names = [p.name for p in entries]
            require(names == source_names, f"Images differ from source frame names: {clip}")
            frames[clip] = names
            samples[clip] = {names[0], names[len(names) // 2], names[-1]}
        require(sum(map(len, frames.values())) == expected_frames, f"Unexpected {split} frame count")

        annotation_path = source / "annotations_coco" / f"instances_01_{split}2017.json"
        annotation = json.loads(annotation_path.read_text())
        images = {}
        coco_names = defaultdict(set)
        for im in annotation["images"]:
            clip, channel, name = im["file_name"].split("/")
            if clip in frames:
                require(channel == "01" and (im["width"], im["height"]) == (640, 512),
                        f"Unexpected COCO image geometry: {im['file_name']}")
                images[im["id"]] = (clip, name)
                coco_names[clip].add(name)
        for clip in clips:
            require(coco_names[clip] == set(frames[clip]), f"COCO frame names mismatch: {clip}")
        counters = {c: {"boxes": 0, "ignored_skipped": 0, "empty_skipped": 0} for c in clips}
        sample_boxes = {c: {f: [] for f in samples[c]} for c in clips}
        for ann in annotation["annotations"]:
            if ann["image_id"] not in images:
                continue
            clip, name = images[ann["image_id"]]
            stats = counters[clip]
            if ann.get("ignore", 0):
                stats["ignored_skipped"] += 1
                continue
            box = ann["bbox"]
            if not (box[2] > 0 and box[3] > 0):
                stats["empty_skipped"] += 1
                continue
            stats["boxes"] += 1
            if name in sample_boxes[clip]:
                sample_boxes[clip][name].append(box)
        for clip in clips:
            recorded = metadata["clips"][clip]
            require(recorded["frames"] == len(frames[clip]), f"Metadata frame count: {clip}")
            for name, count in counters[clip].items():
                require(recorded[name] == count, f"Metadata {name} disagrees with COCO: {clip}")
            require(0 <= recorded["vanished"] <= recorded["boxes"], f"Invalid vanished count: {clip}")
            tasks.append({"root": str(root), "split": split, "clip": clip,
                          "frames": frames[clip], "samples": sample_boxes[clip],
                          "annotation_counts": counters[clip], "export_vanished": recorded["vanished"],
                          "expected_targets": protocol["per_clip"][clip]["counts"][VARIANT] if split == "test" else None,
                          "author_targets": protocol["reference"][clip] if split == "test" else None})
        del annotation, images
    require(len(set(all_clips)) == 60, "Train/test clips overlap")
    mask_root = root / "segmentations"
    regular_directory(mask_root)
    require({p.name for p in mask_root.iterdir()} == set(all_clips) | {"MASKS.json"},
            "Unexpected/missing mask directory entries")
    require(set(metadata["clips"]) == set(all_clips), "Metadata clip set mismatch")
    return tasks


def verify_clip(task):
    root = Path(task["root"])
    split, clip = task["split"], task["clip"]
    image_dir = root / f"{split}2017" / clip / "01"
    mask_dir = root / "segmentations" / clip
    regular_directory(mask_dir)
    names = task["frames"]
    expected_masks = [Path(f).with_suffix(".png").name for f in names]
    entries = sorted(mask_dir.iterdir())
    require([p.name for p in entries] == expected_masks, f"Mask/image frame names mismatch: {clip}")
    require(all(p.is_file() and not p.is_symlink() for p in entries), f"Nonregular masks: {clip}")
    targets = 0
    gaussian_frames = 0
    empty_masks = 0
    byte_count = 0
    sample_results = []
    for image_name, mask_path in zip(names, entries):
        image_path = image_dir / image_name
        with Image.open(image_path) as image:
            require(image.format == "JPEG" and image.mode == "L" and image.size == (640, 512),
                    f"Unexpected JPEG mode/size: {image_path}")
            image_size = image.size
        with Image.open(mask_path) as mask:
            require(mask.format == "PNG" and mask.mode == "L" and mask.size == image_size,
                    f"Unexpected PNG mode/size: {mask_path}")
            mask.load()
            array = np.array(mask)
        require(array.dtype == np.uint8 and array.shape == (512, 640), f"Unexpected mask dtype/shape: {mask_path}")
        byte_count += mask_path.stat().st_size
        gaussian_frames += int(np.any((array > 0) & (array < 255)))
        empty_masks += int(not array.any())
        if image_name in task["samples"]:
            boxes = task["samples"][image_name]
            expected = gm.store(gm.heatmap(boxes, 512, 640, 6), "floor")
            require(np.array_equal(array, expected), f"Gaussian pixel mismatch: {mask_path}")
            sample_results.append({"frame": image_name, "boxes": len(boxes), "pixels_equal": True,
                                   "png_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest()})
        if split == "test":
            targets += int(ndimage.label(array > 0, structure=STRUCTURE)[1])
    if split == "test":
        require(targets == task["expected_targets"], f"Actual PNG target count mismatches scan: {clip}")
    return clip, {"split": split, "frames": len(names), "mask_bytes": byte_count,
                  "fully_decoded_masks": len(names), "image_mask_names_match": True,
                  "mode": "L", "dtype": "uint8", "size_wh": [640, 512],
                  "images_are_regular_files": True, "masks_with_intermediate_gray_values": gaussian_frames,
                  "empty_masks": empty_masks, "sample_recomputations": sample_results,
                  **task["annotation_counts"], "export_vanished": task["export_vanished"],
                  "test_targets": targets if split == "test" else None,
                  "expected_scan_targets": task["expected_targets"],
                  "author_targets": task["author_targets"],
                  "difference_from_author": targets - task["author_targets"] if split == "test" else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT / "datasets/RGB-T")
    parser.add_argument("--source", type=Path, default=PROJECT / "datasets/RGBT-Tiny")
    parser.add_argument("--nproc", type=int, default=8)
    args = parser.parse_args()
    root, source = args.root.resolve(), args.source.resolve()
    metadata_path = root / "segmentations/MASKS.json"
    metadata = json.loads(metadata_path.read_text())
    require(metadata["k"] == 6 and metadata["storage"] == "floor" and metadata["ignored_drawn"] is False,
            "Wrong export parameters")
    require(np.isclose(metadata["label_semi_axes_in_box_units"], gm.semi_axis("floor") / 6),
            "Wrong support radius metadata")
    protocol_path = source / "mask_protocol/check.json"
    protocol = json.loads(protocol_path.read_text())
    tasks = prepare_tasks(root, source, metadata, protocol)
    results = {}
    with ProcessPoolExecutor(max_workers=max(1, min(args.nproc, len(tasks))),
                             mp_context=multiprocessing.get_context("fork")) as pool:
        for clip, result in pool.map(verify_clip, tasks):
            results[clip] = result
            suffix = f", test targets={result['test_targets']}" if result["split"] == "test" else ""
            print(f"PASS {clip}: {result['frames']} masks{suffix}", flush=True)
    counts = {split: {"clips": sum(r["split"] == split for r in results.values()),
                      "frames": sum(r["frames"] for r in results.values() if r["split"] == split)}
              for split in ("train", "test")}
    test_results = {c: r for c, r in results.items() if r["split"] == "test"}
    total_targets = sum(r["test_targets"] for r in test_results.values())
    exact = sum(r["difference_from_author"] == 0 for r in test_results.values())
    abs_diff = sum(abs(r["difference_from_author"]) for r in test_results.values())
    require(total_targets == 11300 and exact == 13 and abs_diff == 5, "Unexpected final target totals")
    report = {"status": "passed", "completed_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
              "root": str(root), "source": str(source), "k": 6, "storage": "floor", "ignored_drawn": False,
              "counts": counts, "total_pngs": sum(r["frames"] for r in results.values()),
              "mask_bytes": sum(r["mask_bytes"] for r in results.values()),
              "sample_frames_recomputed": sum(len(r["sample_recomputations"]) for r in results.values()),
              "ignored_skipped": sum(r["ignored_skipped"] for r in results.values()),
              "empty_boxes_skipped": sum(r["empty_skipped"] for r in results.values()),
              "export_vanished_boxes": sum(r["export_vanished"] for r in results.values()),
              "test_targets": total_targets, "author_test_targets": sum(protocol["reference"].values()),
              "test_clips_matching_author": exact, "absolute_count_difference": abs_diff,
              "all_test_counts_match_selected_scan": True, "counting_backend": "scipy.ndimage.label",
              "connectivity": 8, "reference_protocol": str(protocol_path),
              "MASKS_json_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
              "per_clip": dict(sorted(results.items()))}
    output = root / "mask_protocol/verification.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_clip"}, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
