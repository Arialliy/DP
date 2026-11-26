"""Copy the official DeepPro RGB-T thermal subset, verifying every SHA256.

Images are real copies, not links. Existing matching files are retained; existing
different files and symlink destinations are rejected without replacement.
"""
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / "datasets/RGBT-Tiny/images"
TARGET = PROJECT / "datasets/RGB-T"
REPORT = TARGET / "mask_protocol"
EXPECTED = {"train": (45, 19135), "test": (15, 5976)}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_real(path):
    current = path
    while current != TARGET.parent:
        if current.is_symlink():
            raise RuntimeError(f"Destination must not be a symlink: {current}")
        current = current.parent


def copy_clip(task):
    split, clip, names = task
    source = SOURCE / clip / "01"
    target = TARGET / f"{split}2017" / clip / "01"
    require_real(target)
    target.mkdir(parents=True, exist_ok=True)
    extra = {p.name for p in target.iterdir()} - set(names)
    if extra:
        raise RuntimeError(f"Unexpected destination files in {target}: {sorted(extra)}")
    rows = []
    for name in names:
        src, dst = source / name, target / name
        require_real(dst)
        source_hash = digest(src)
        action = "retained"
        if not dst.exists():
            with src.open("rb") as incoming, dst.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
            shutil.copystat(src, dst)
            action = "copied"
        target_hash = digest(dst)
        if source_hash != target_hash:
            raise RuntimeError(f"SHA256 mismatch; destination was not overwritten: {dst}")
        if src.stat().st_ino == dst.stat().st_ino and src.stat().st_dev == dst.stat().st_dev:
            raise RuntimeError(f"Destination must be an independent copy: {dst}")
        rows.append({"split": split, "clip": clip, "frame": name,
                     "source": str(src.relative_to(PROJECT)),
                     "target": str(dst.relative_to(PROJECT)), "bytes": dst.stat().st_size,
                     "source_sha256": source_hash, "target_sha256": target_hash,
                     "action": action})
    if sorted(p.name for p in target.iterdir()) != names:
        raise RuntimeError(f"Frame-name mismatch: {target}")
    return rows


def main():
    tasks, clip_sets = [], {}
    for split, (expected_clips, expected_frames) in EXPECTED.items():
        clips = (TARGET / f"{split}.txt").read_text().split()
        if len(clips) != expected_clips or len(set(clips)) != len(clips):
            raise ValueError(f"Unexpected {split} sequence list")
        clip_sets[split] = set(clips)
        total = 0
        for clip in clips:
            if Path(clip).name != clip:
                raise ValueError(f"Invalid sequence name: {clip}")
            source = SOURCE / clip / "01"
            names = sorted(p.name for p in source.glob("*.jpg") if p.is_file())
            if not names:
                raise FileNotFoundError(f"No JPG frames in {source}")
            tasks.append((split, clip, names))
            total += len(names)
        if total != expected_frames:
            raise ValueError(f"Unexpected {split} frame count: {total}")
    if clip_sets["train"] & clip_sets["test"]:
        raise ValueError("Train/test sequence overlap")
    REPORT.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=4) as workers:
        for clip_rows in workers.map(copy_clip, tasks):
            rows.extend(clip_rows)
            print(f"{clip_rows[0]['split']} {clip_rows[0]['clip']}: {len(clip_rows)} copies verified", flush=True)
    manifest = REPORT / "images_sha256.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "passed",
        "completed_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
        "source": str(SOURCE), "target": str(TARGET), "channel": "01 (thermal)",
        "copy_type": "independent regular files; no symlinks or source hardlinks",
        "all_file_names_and_sha256_match": True,
        "frames": len(rows), "bytes": sum(row["bytes"] for row in rows),
        "copied": sum(row["action"] == "copied" for row in rows),
        "retained": sum(row["action"] == "retained" for row in rows),
        "splits": {
            split: {"clips": len(clip_sets[split]),
                    "frames": sum(row["split"] == split for row in rows),
                    "bytes": sum(row["bytes"] for row in rows if row["split"] == split)}
            for split in EXPECTED
        },
        "sha256_manifest": str(manifest.relative_to(TARGET)),
        "sha256_manifest_sha256": digest(manifest),
        "script_sha256": digest(Path(__file__)),
    }
    (REPORT / "images_copy.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
