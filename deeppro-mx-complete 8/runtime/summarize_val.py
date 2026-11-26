#!/usr/bin/env python3
"""Validate and summarize the six authorized val runs; standard library only.

Usage: python3 runtime/summarize_val.py [--root /experiment] [--strict]
Missing/in-progress artifacts produce pending rows with empty metrics. Invalid
artifacts produce error rows with empty metrics. --strict returns 2 for errors.
No torch imports, checkpoint deserialization, GPU calls, or training are used.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

RUNS = (
    ("mta_val_s46", "mta_lv"),
    ("mta_val_s46_noex", "mta_lv"),
    ("mta_val_s46_self", "mta_lv"),
    ("mta_val_s46_nogain", "mta_lv"),
    ("mta_val_s46_mx", "mx_lv"),
    ("mta_val_s46_noapp", "mxc_lv"),
)
# Exactly code/evaluate.py:TH_SEG; no optional scientific packages are needed.
THRESHOLDS = (0, 1e-20, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4,
              1e-3, 1e-2, .1, .2, .3, .35, .4, .45, .5, .55,
              .6, .65, .7, .8, .85, .9, .95, .99, 1)
METRICS = ("IoU_at_0_5", "Pd_at_0_5", "Fa_at_0_5", "AUC_27_thresholds")
FIELDS = ("run", "architecture", "status", *METRICS, "threshold_count",
          "best_epoch", "selection_mode", "selection_split", "selection_IoU",
          "checkpoint_sha256", "selection_sha256", "score_sha256",
          "checkpoint_path", "selection_path", "score_path", "detail")


class Invalid(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Invalid(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def number(value, name):
    require(isinstance(value, (float, int)) and not isinstance(value, bool), name + " is not numeric")
    require(math.isfinite(value) and 0 <= value <= 1, name + " must be finite and in [0,1]")
    return value


def same_clips(value, clips, label):
    require(isinstance(value, (list, dict)), label + " is missing")
    require(len(value) == len(clips) and set(value) == set(clips), label + " differs from configs/splits.json val")


def check_close(a, b, label):
    require(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), label + " is inconsistent")


def validate_metrics(score, clips):
    require(score.get("author_label_rule") == "paper_k6_floor", "author_label_rule must be paper_k6_floor")
    require(score.get("author_edges") == "official", "author_edges must be official")
    headline = score.get("author_headline", {})
    require(headline.get("threshold") == .5, "author_headline threshold must be 0.5")
    require(headline.get("thresholds") == list(THRESHOLDS), "author_headline must contain exactly the 27 official thresholds")
    table = score.get("author_protocol", {})
    require(isinstance(table, dict) and len(table) == 27, "author_protocol must have 27 thresholds")
    try:
        by_threshold = {float(key): value for key, value in table.items()}
    except (ValueError, TypeError) as error:
        raise Invalid("invalid author_protocol threshold") from error
    require(set(by_threshold) == set(THRESHOLDS), "author_protocol thresholds differ from official thresholds")
    same_clips(score.get("per_clip"), clips, "score per_clip")
    same_clips(score.get("author_protocol_per_clip"), clips, "author_protocol_per_clip")
    for threshold in THRESHOLDS:
        point = by_threshold[threshold]
        require(isinstance(point, dict), "invalid author_protocol point")
        for key in ("IoU", "Pd", "Fa"):
            number(point.get(key), f"author_protocol[{threshold}].{key}")
    at_half = by_threshold[.5]
    for key in ("IoU", "Pd", "Fa"):
        check_close(number(headline.get(key), "author_headline." + key), at_half[key], "headline " + key)
    fa = [by_threshold[threshold]["Fa"] for threshold in THRESHOLDS]
    pd = [by_threshold[threshold]["Pd"] for threshold in THRESHOLDS]
    delta = [right - left for left, right in zip(fa, fa[1:])]
    if all(value <= 0 for value in delta):
        direction = -1
    elif all(value >= 0 for value in delta):
        direction = 1
    else:
        raise Invalid("official curve Fa is not monotonic")
    area = direction * sum(dx * (left + right) / 2 for dx, left, right in zip(delta, pd, pd[1:]))
    check_close(number(headline.get("AUC"), "author_headline.AUC"), area, "27-threshold AUC")
    return dict(IoU_at_0_5=headline["IoU"], Pd_at_0_5=headline["Pd"],
                Fa_at_0_5=headline["Fa"], AUC_27_thresholds=headline["AUC"], threshold_count=27)


def collect_run(root, run, architecture, clips, scheduler_job=None):
    row = {field: "" for field in FIELDS}
    row.update(run=run, architecture=architecture, status="pending")
    run_dir = root / "runs" / run
    checkpoint = run_dir / "best_model.pth"
    selection_path = run_dir / "best_selection.json"
    score_path = root / "results" / "scores" / run / "val" / "summary.json"
    progress_path = run_dir / "progress.json"
    row.update(checkpoint_path=str(checkpoint.relative_to(root)),
               selection_path=str(selection_path.relative_to(root)), score_path=str(score_path.relative_to(root)))
    try:
        if scheduler_job and scheduler_job.get("status") in ("failed", "error"):
            raise Invalid("scheduler: " + str(scheduler_job.get("error") or scheduler_job.get("detail") or "failed"))
        # A live best checkpoint is provisional until all 32 val selections finish.
        if not progress_path.exists():
            row["detail"] = "pending: training progress is not available"
            return row
        progress = read_json(progress_path)
        if progress.get("stage") != "completed":
            row["detail"] = f"pending: training {progress.get('stage', 'unknown')}, epoch {progress.get('epoch', '?')}/{progress.get('epoch_total', '?')}"
            return row
        require(progress.get("epoch") == 32 and progress.get("epoch_total") == 32,
                "training must complete the authorized 32 epochs")
        missing = [str(path.relative_to(root)) for path in (checkpoint, selection_path, score_path) if not path.is_file()]
        if missing:
            row["detail"] = "pending: " + ", ".join(missing)
            return row
        before = checkpoint.stat()
        checkpoint_sha = sha256(checkpoint)
        row.update(checkpoint_sha256=checkpoint_sha, selection_sha256=sha256(selection_path), score_sha256=sha256(score_path))
        selection = read_json(selection_path)
        require(selection.get("selection_mode") == "val_iou", "best_selection selection_mode must be val_iou")
        require(selection.get("split") == "val", "best_selection split must be val")
        require(selection.get("threshold") == .5, "best_selection threshold must be 0.5")
        require(selection.get("complete") is True, "best_selection is incomplete")
        require(isinstance(selection.get("epoch"), int) and 1 <= selection["epoch"] <= 32,
                "best_selection epoch must be in [1,32]")
        same_clips(selection.get("test_clips"), clips, "best_selection clips")
        protocol = selection.get("protocol", {})
        require(protocol.get("split") == "val", "best_selection protocol split must be val")
        same_clips(protocol.get("clips"), clips, "best_selection protocol clips")
        row.update(best_epoch=selection["epoch"], selection_mode="val_iou", selection_split="val",
                   selection_IoU=number(selection.get("iou"), "best_selection.iou"))
        score = read_json(score_path)
        require(score.get("run") == run and score.get("split") == "val", "score run/split mismatch")
        predictions = score.get("predictions", {})
        same_clips(predictions, clips, "prediction fingerprints")
        missing_meta = []
        for clip in clips:
            prediction = predictions[clip]
            require(isinstance(prediction, dict), clip + ": missing score prediction fingerprint")
            require(prediction.get("ckpt_sha256") == checkpoint_sha, clip + ": score checkpoint SHA does not match best_model.pth")
            metadata_path = root / "cache" / "preds" / (run + "__meta") / (clip + ".json")
            if not metadata_path.is_file():
                missing_meta.append(str(metadata_path.relative_to(root)))
                continue
            fingerprint = read_json(metadata_path).get("fingerprint")
            require(isinstance(fingerprint, dict), clip + ": cache fingerprint is missing")
            require(fingerprint.get("clip") == clip, clip + ": cache fingerprint clip mismatch")
            require(fingerprint.get("ckpt_sha256") == checkpoint_sha, clip + ": cache checkpoint SHA does not match best_model.pth")
            digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
            require(prediction.get("fingerprint") == digest, clip + ": scored fingerprint differs from current prediction cache")
        if missing_meta:
            row["detail"] = "pending: " + ", ".join(missing_meta)
            return row
        metrics = validate_metrics(score, clips)
        after = checkpoint.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "checkpoint changed during summary")
        row.update(metrics, status="complete", detail="32 epochs completed; val_iou selected; all val prediction SHA verified")
    except (Invalid, OSError, ValueError, KeyError, TypeError) as error:
        row.update(status="error", detail=str(error))
        for name in (*METRICS, "threshold_count"):
            row[name] = ""
    return row


def scheduler_jobs(path):
    if not path or not path.is_file():
        return {}
    state = read_json(path)
    jobs = state.get("jobs", state.get("runs", {}))
    if isinstance(jobs, list):
        return {job["run"]: job for job in jobs if isinstance(job, dict) and "run" in job}
    return jobs if isinstance(jobs, dict) else {}


def summarize(root, state_path=None):
    root = Path(root).resolve()
    clips = read_json(root / "configs" / "splits.json")["val"]
    require(isinstance(clips, list) and clips and len(set(clips)) == len(clips), "invalid val split")
    jobs = scheduler_jobs(state_path or root / "runtime" / "val_queue" / "state.json")
    rows = [collect_run(root, run, arch, clips, jobs.get(run)) for run, arch in RUNS]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    output = root / "results" / "comparison"
    atomic_text(output / "val.csv", buffer.getvalue())
    lines = ["六组 val 对照：仅汇总已完成且权重一致的结果", "",
             "Pd、Fa、IoU 均用官方评估实现、同一 best_model.pth、概率阈值 0.5：Pd 按合并后的连通组件计，Fa 与 IoU 按像素计。",
             "Fa 单位是错误前景像素/总像素；AUC 使用官方 27 个阈值。数值均为 0–1 原始比例。",
             "权重按每轮 val IoU 选取；val 同时参与选权重，结果属于验证集比较，不是独立 test 结果。",
             "不使用固定 Fa 下的 Pd 替代直接 Pd；未完成/异常项留空，不填零。", "",
             "run | 状态 | IoU@0.5 | Pd@0.5 | Fa@0.5 | AUC(27) | best epoch"]
    for row in rows:
        values = [format(row[field], ".9g") if row[field] != "" else "—" for field in METRICS]
        lines.append(" | ".join([row["run"], row["status"], *values, str(row["best_epoch"] or "—")]))
    lines.extend(["", "状态与来源："])
    for row in rows:
        lines.extend([row["run"] + ": " + row["detail"],
                      "  " + row["score_path"], "  " + row["selection_path"],
                      "  " + row["checkpoint_path"],
                      "  checkpoint SHA-256: " + (row["checkpoint_sha256"] or "pending")])
    lines.extend(["", "生成时间 UTC: " + datetime.now(timezone.utc).isoformat(),
                  "完整 SHA、选权重 IoU 与来源列见 val.csv。", ""])
    atomic_text(output / "val.txt", "\n".join(lines))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--state", type=Path, help="optional scheduler JSON (jobs/runs mapping or list)")
    parser.add_argument("--strict", action="store_true", help="exit 2 if any row is error; pending is normal")
    args = parser.parse_args()
    rows = summarize(args.root, args.state)
    print(json.dumps({"outputs": [str(args.root / "results/comparison/val.csv"), str(args.root / "results/comparison/val.txt")],
                      "status": {row["run"]: row["status"] for row in rows}}, ensure_ascii=False))
    return 2 if args.strict and any(row["status"] == "error" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
