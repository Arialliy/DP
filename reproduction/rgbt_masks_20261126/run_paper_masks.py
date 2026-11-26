"""Run the supplied paper-masks scripts against the local RGBT-Tiny release.

Only adapt directories and tracking_id -> track_id in memory. The supplied
Gaussian formula, parameter sweep, target counting and PNG writer are unchanged.
"""
import importlib
import multiprocessing
import os
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "datasets/RGBT-Tiny"
LISTS = PROJECT / "datasets/RGB-T"
SOURCE = PROJECT / "datasets/paper-masks/code"


def link(path, target):
    if path.exists() or path.is_symlink():
        if path.resolve() != target.resolve():
            raise FileExistsError(f"Refusing to replace {path}")
    else:
        path.symlink_to(os.path.relpath(target, path.parent), target_is_directory=True)


def prepare_layout():
    link(ROOT / "annotations", ROOT / "annotations_coco")
    for split in ("train", "test"):
        source_list = LISTS / f"{split}.txt"
        clips = source_list.read_text().split()
        destination = ROOT / f"{split}2017"
        destination.mkdir(exist_ok=True)
        for clip in clips:
            images = ROOT / "images" / clip
            if not (images / "01").is_dir():
                raise FileNotFoundError(images / "01")
            link(destination / clip, images)
        output_list = ROOT / f"{split}.txt"
        if output_list.exists():
            if output_list.read_bytes() != source_list.read_bytes():
                raise FileExistsError(f"Different split already exists: {output_list}")
        else:
            shutil.copyfile(source_list, output_list)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"check", "export"}:
        raise SystemExit("Usage: run_paper_masks.py {check|export} [original script arguments]")
    action, arguments = sys.argv[1], sys.argv[2:]
    prepare_layout()
    os.environ["VT_ROOT"] = str(ROOT)
    os.environ["OFFICIAL_LISTS"] = str(LISTS)
    sys.path.insert(0, str(SOURCE))
    import vtdata

    # Preload once before fork so workers share the annotation cache.
    splits = ("test2017",) if action == "check" else ("train2017", "test2017")
    for split in splits:
        cache = vtdata._annotations(split)
        adapted = 0
        for frames in cache.values():
            for rows in frames.values():
                for annotation in rows:
                    if "track_id" not in annotation:
                        annotation["track_id"] = int(annotation["tracking_id"])
                        adapted += 1
        print(f"{split}: adapted {adapted} tracking_id fields in memory", flush=True)
    multiprocessing.set_start_method("fork", force=True)
    module_name = "check_mask_protocol" if action == "check" else "export_masks"
    sys.argv = [str(SOURCE / f"{module_name}.py"), *arguments]
    importlib.import_module(module_name).main()


if __name__ == "__main__":
    main()
