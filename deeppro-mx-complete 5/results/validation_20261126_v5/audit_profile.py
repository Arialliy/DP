#!/usr/bin/env python3
"""Independent COCO-to-profile audit. Does not import any project modules.

Prepare: python audit_profile.py --prepare
Audit:   python audit_profile.py
Only writes expected_profile.json / profile_audit.json next to this script.
"""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


TRACK_BASE = 1_000_000
TRACK_SCHEME = "track_id, else category * 1000000 + tracking_id"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def reference(root):
    """Read source annotations, frame directories and lists, independently of vtdata."""
    lists = {name: (root / f"{name}.txt").read_text().split() for name in ("train", "test")}
    errors = []
    selected = lists["train"] + lists["test"]
    if len(selected) != len(set(selected)):
        errors.append("Official lists contain duplicate or overlapping clip names")
    selected_set = set(selected)
    hashes = {name + ".txt": sha256(root / (name + ".txt")) for name in lists}
    clips = {}
    frame_indices = {}
    for label, names in lists.items():
        split = label + "2017"
        for name in names:
            folder = root / split / name / "01"
            frames = sorted(p.name for p in folder.glob("*.jpg"))
            if not frames:
                errors.append(f"No frames in {folder}")
            frame_indices[name] = {frame: i for i, frame in enumerate(frames)}
            clips[name] = dict(list=label, split=split, frames=len(frames), boxes=0,
                               ignored=0, tracks=0, duplicate_keys=0)
    records = []
    original_tracks = defaultdict(set)
    raw_classes = defaultdict(set)
    explicit_track_fields = 0
    raw_min = raw_max = None
    category_values = set()
    for split in ("train2017", "test2017"):
        path = root / "annotations_coco" / f"instances_01_{split}.json"
        hashes[str(path.relative_to(root))] = sha256(path)
        ann = json.loads(path.read_text())
        images = {}
        selected_names = defaultdict(Counter)
        for im in ann["images"]:
            parts = im["file_name"].split("/")
            if len(parts) != 3:
                continue
            name, channel, frame = parts
            if name not in selected_set:
                continue
            if clips[name]["split"] != split or channel != "01":
                errors.append(f"Unexpected split/channel for {im['file_name']}: {split}")
                continue
            selected_names[name][frame] += 1
            if frame not in frame_indices[name]:
                errors.append(f"COCO image absent from image directory: {name}/{frame}")
                continue
            images[im["id"]] = (name, frame_indices[name][frame])
        for name in lists[split.removesuffix("2017")]:
            counts = selected_names[name]
            if set(counts) != set(frame_indices[name]):
                errors.append(f"COCO/image-directory frame set mismatch: {name}")
            if any(n != 1 for n in counts.values()):
                errors.append(f"COCO contains duplicate image names: {name}")
        for a in ann["annotations"]:
            if a["image_id"] not in images:
                continue
            name, t = images[a["image_id"]]
            if a.get("ignore", 0):
                clips[name]["ignored"] += 1
                continue
            cat = int(a["category_id"])
            category_values.add(cat)
            raw = int(a["tracking_id"]) if "tracking_id" in a else None
            if raw is not None:
                raw_min = raw if raw_min is None else min(raw_min, raw)
                raw_max = raw if raw_max is None else max(raw_max, raw)
                raw_classes[(name, raw)].add(cat)
            if "track_id" in a:
                track = int(a["track_id"])
                explicit_track_fields += 1
            else:
                if raw is None or not 0 <= raw < TRACK_BASE or cat < 0:
                    errors.append(f"Invalid category/raw tracking ID: {name}, {a['id']}")
                    continue
                track = cat * TRACK_BASE + raw
            x, y, w, h = map(float, a["bbox"])
            if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
                errors.append(f"Invalid bbox: {name}, {a['id']}")
            records.append((name, t, track, cat, x + w / 2, y + h / 2, w, h))
            clips[name]["boxes"] += 1
            original_tracks[name].add((track, cat, raw))
        del ann
    keys = Counter(r[:3] for r in records)
    for (name, _t, _track), count in keys.items():
        if count > 1:
            clips[name]["duplicate_keys"] += count
    for name, stats in clips.items():
        stats["tracks"] = len({r[0] for r in original_tracks[name]})
        if stats["tracks"] != len(original_tracks[name]):
            errors.append(f"Canonical track maps to multiple raw/category identities: {name}")
    h = hashlib.sha256()
    for row in sorted(records):
        h.update((json.dumps(row, separators=(",", ":")) + "\n").encode())
    totals = {field: sum(c[field] for c in clips.values())
              for field in ("frames", "boxes", "ignored", "tracks", "duplicate_keys")}
    totals["clips"] = len(clips)
    split_totals = {label: {field: sum(c[field] for c in clips.values() if c["list"] == label)
                            for field in ("frames", "boxes", "ignored", "tracks", "duplicate_keys")}
                    for label in lists}
    for label, names in lists.items():
        split_totals[label]["clips"] = len(names)
    summary = dict(status="PASS" if not errors else "FAIL", source_root=str(root),
                   track_ids=TRACK_SCHEME, totals=totals, splits=split_totals,
                   raw_tracking_id_range=[raw_min, raw_max], categories=sorted(category_values),
                   explicit_track_id_fields=explicit_track_fields,
                   cross_category_raw_id_groups=sum(len(c) > 1 for c in raw_classes.values()),
                   source_sha256=hashes, reference_rows_sha256=h.hexdigest(), clips=clips, errors=errors)
    return records, summary


def audit(profile, records, expected, saved_reference):
    import numpy as np
    errors = list(expected["errors"])
    if saved_reference is not None:
        for field in ("source_sha256", "reference_rows_sha256"):
            if saved_reference[field] != expected[field]:
                errors.append(f"Source changed since prepare: {field}")
    with (profile / "targets.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    parsed = [(r["scene"], int(r["t"]), int(r["track"]), int(r["cat"]),
               *[float(r[k]) for k in ("cx", "cy", "w", "h")]) for r in rows]
    expected_keys = Counter(r[:3] for r in records)
    actual_keys = Counter(r[:3] for r in parsed)
    missing_keys, extra_keys = expected_keys - actual_keys, actual_keys - expected_keys
    missing_records, extra_records = Counter(records) - Counter(parsed), Counter(parsed) - Counter(records)
    if missing_keys or extra_keys:
        errors.append("targets.csv key multiset differs from raw COCO reference")
    if missing_records or extra_records:
        errors.append("targets.csv full rows (category/centre/size) differ from raw COCO reference")
    metadata = json.loads((profile / "targets.json").read_text())
    expected_clip_counts = {c: s["boxes"] for c, s in expected["clips"].items()}
    expected_dup = {c: s["duplicate_keys"] for c, s in expected["clips"].items() if s["duplicate_keys"]}
    for key, value in dict(track_ids=TRACK_SCHEME, boxes=len(records), rows=len(rows),
                           clips=expected_clip_counts, duplicate_keys=expected_dup).items():
        if metadata.get(key) != value:
            errors.append(f"targets.json mismatch: {key}")
    with (profile / "clips.csv").open(newline="") as f:
        clip_rows = list(csv.DictReader(f))
    if Counter(r["scene"] for r in clip_rows) != Counter(expected["clips"].keys()):
        errors.append("clips.csv clip set/multiplicity mismatch")
    for row in clip_rows:
        c = row["scene"]
        if c not in expected["clips"]:
            continue
        for key in ("frames", "boxes", "tracks", "duplicate_keys"):
            if int(row[key]) != expected["clips"][c][key]:
                errors.append(f"clips.csv mismatch: {c}, {key}")
        for key in ("split", "list"):
            if row[key] != expected["clips"][c][key]:
                errors.append(f"clips.csv mismatch: {c}, {key}")
    homographies = {}
    h_dir = profile / "homographies"
    found = {p.stem for p in h_dir.glob("*.npy")}
    if found != set(expected["clips"]):
        errors.append("Homography clip set mismatch")
    for c, stats in expected["clips"].items():
        path = h_dir / f"{c}.npy"
        if not path.is_file():
            continue
        arr = np.load(path, allow_pickle=False)
        finite = bool(np.isfinite(arr).all())
        if tuple(arr.shape) != (stats["frames"] - 1, 3, 3) or not finite:
            errors.append(f"Invalid homography shape/nonfinite values: {c}")
        homographies[c] = dict(shape=list(arr.shape), all_finite=finite, sha256=sha256(path))
    nan_rows = []
    nonfinite_bad = []
    negative_bad = []
    span_bad = []
    image_speed_bad = []
    max_image_speed_error = 0.0
    unique_reference = {r[:3]: r for r in records if expected_keys[r[:3]] == 1}
    track_times = defaultdict(list)
    for scene, t, track in unique_reference:
        track_times[(scene, track)].append(t)
    for times in track_times.values():
        times.sort()
    per_clip_finite = Counter()
    for row in rows:
        speeds = [float(row[k]) for k in ("image_speed", "camera_speed", "rel_speed")]
        span = int(row["span"])
        key = (row["scene"], int(row["t"]), int(row["track"]))
        if span == 0:
            if not all(math.isnan(v) for v in speeds):
                nonfinite_bad.append(key)
            nan_rows.append(dict(scene=key[0], t=key[1], track=key[2], span=span))
        elif not all(math.isfinite(v) for v in speeds):
            nonfinite_bad.append(key)
        else:
            per_clip_finite[key[0]] += 1
        if any(math.isfinite(v) and v < 0 for v in speeds):
            negative_bad.append(key)
        expected_span = 0
        times = track_times.get((key[0], key[2]), [])
        if expected_keys[key] == 1:
            lo = bisect_left(times, key[1] - 5)
            hi = bisect_right(times, key[1] + 5)
            t0, t1 = times[lo], times[hi - 1]
            expected_span = t1 - t0
            if expected_span:
                first = unique_reference[(key[0], t0, key[2])]
                last = unique_reference[(key[0], t1, key[2])]
                expected_speed = math.hypot(last[4] - first[4], last[5] - first[5]) / expected_span
                err = abs(speeds[0] - expected_speed)
                if math.isfinite(err):
                    max_image_speed_error = max(max_image_speed_error, err)
                if not math.isclose(speeds[0], expected_speed, rel_tol=1e-10, abs_tol=1e-10):
                    image_speed_bad.append(key)
        if span != expected_span:
            span_bad.append(key)
    if nonfinite_bad or negative_bad or span_bad or image_speed_bad:
        errors.append("Invalid speed/span fields")
    checks = dict(expected_rows=len(records), actual_rows=len(rows),
                  expected_unique_keys=len(expected_keys), actual_unique_keys=len(actual_keys),
                  missing_key_occurrences=sum(missing_keys.values()), extra_key_occurrences=sum(extra_keys.values()),
                  missing_full_row_occurrences=sum(missing_records.values()),
                  extra_full_row_occurrences=sum(extra_records.values()),
                  missing_full_row_examples=list(missing_records)[:5], extra_full_row_examples=list(extra_records)[:5],
                  clip_rows=len(clip_rows), homography_files=len(found),
                  homography_matrices=sum(v["shape"][0] for v in homographies.values()),
                  nan_speed_rows=len(nan_rows), nan_speed_examples=nan_rows[:20],
                  invalid_nonfinite_speed_rows=len(nonfinite_bad), negative_speed_rows=len(negative_bad),
                  invalid_span_rows=len(span_bad), incorrect_image_speed_rows=len(image_speed_bad),
                  max_image_speed_absolute_error=max_image_speed_error,
                  finite_speed_rows_by_clip=dict(per_clip_finite))
    return dict(status="PASS" if not errors else "FAIL", audited_at_utc=datetime.now(timezone.utc).isoformat(),
                profile=str(profile), checks=checks, expected=expected, homographies=homographies,
                output_sha256={name: sha256(profile / name) for name in ("targets.csv", "targets.json", "clips.csv")},
                errors=errors)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/md0/ly/DP/datasets/RGB-T"))
    parser.add_argument("--profile", type=Path, default=here.parent / "profile")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    records, summary = reference(args.root)
    reference_path = here / "expected_profile.json"
    if args.prepare:
        write_json(reference_path, summary)
        print(json.dumps(dict(status=summary["status"], totals=summary["totals"], splits=summary["splits"],
                              errors=summary["errors"], report=str(reference_path)), indent=2))
        return 0 if summary["status"] == "PASS" else 1
    saved = json.loads(reference_path.read_text()) if reference_path.exists() else None
    report = audit(args.profile, records, summary, saved)
    path = here / "profile_audit.json"
    write_json(path, report)
    print(json.dumps(dict(status=report["status"], checks=report["checks"], errors=report["errors"],
                          report=str(path)), indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
