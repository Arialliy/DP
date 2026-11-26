#!/usr/bin/env python3
"""Pure-file fixtures: no torch, models, GPU, dataset, or training access."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile

import summarize_val as summary


HERE = Path(__file__).resolve().parent


def write_json(path, value):
    summary.atomic_text(path, json.dumps(value, indent=2) + "\n")


def complete_fixture(root, run, clips):
    run_dir = root / "runs" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best_model.pth"
    checkpoint.write_bytes(b"fixture-only; this is not a torch checkpoint\n")
    ckpt_sha = summary.sha256(checkpoint)
    write_json(run_dir / "progress.json", {"stage": "completed", "epoch": 32, "epoch_total": 32})
    write_json(run_dir / "best_selection.json", {
        "selection_mode": "val_iou", "split": "val", "threshold": .5,
        "complete": True, "epoch": 23, "iou": .456,
        "test_clips": clips, "protocol": {"split": "val", "clips": clips}})
    predictions = {}
    for clip in clips:
        fp = {"ckpt": str(checkpoint), "ckpt_sha256": ckpt_sha, "clip": clip,
              "settings": {"bank": "outside"}, "code_sha256": "fixture-code"}
        write_json(root / "cache/preds" / (run + "__meta") / (clip + ".json"), {"fingerprint": fp})
        predictions[clip] = {"ckpt": str(checkpoint), "ckpt_sha256": ckpt_sha,
                             "fingerprint": hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]}
    protocol = {str(th): {"IoU": .432, "Pd": .57 * (1 - th) / .5,
                          "Fa": .001 * (1 - th) / .5} for th in summary.THRESHOLDS}
    # Keep every Pd valid without obscuring direct Pd@0.5 = .57.
    for point in protocol.values():
        point["Pd"] = min(point["Pd"], 1)
    points = [protocol[str(th)] for th in summary.THRESHOLDS]
    auc = -sum((b["Fa"] - a["Fa"]) * (a["Pd"] + b["Pd"]) / 2 for a, b in zip(points, points[1:]))
    score = {"run": run, "split": "val", "per_clip": dict.fromkeys(clips, {}),
             "predictions": predictions, "overall": {"pd@1": .999, "pd@0.25": .888, "cpm": .777},
             "author_label_rule": "paper_k6_floor", "author_edges": "official",
             "author_protocol": protocol, "author_protocol_per_clip": dict.fromkeys(clips, protocol),
             "author_headline": {"IoU": .432, "Pd": .57, "Fa": .001, "AUC": auc,
                                 "threshold": .5, "thresholds": list(summary.THRESHOLDS)}}
    write_json(root / "results/scores" / run / "val/summary.json", score)
    return score


def main():
    base = HERE / "test_fixtures/val_summary"
    base.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="validation_", dir=base))
    clips = ["fixture_val_A", "fixture_val_B"]
    write_json(root / "configs/splits.json", {"val": clips})
    checks = []
    for run, _ in summary.RUNS:
        complete_fixture(root, run, clips)
    rows = summary.summarize(root)
    assert all(row["status"] == "complete" for row in rows)
    assert all(row["Pd_at_0_5"] == .57 and row["Pd_at_0_5"] != .999 for row in rows)
    assert all(row["IoU_at_0_5"] == .432 and row["Fa_at_0_5"] == .001 for row in rows)
    assert all(row["threshold_count"] == 27 and row["checkpoint_sha256"] for row in rows)
    checks.append("complete six runs: direct headline metrics, 27-threshold AUC, checkpoint SHA")
    checks.append("direct Pd=0.57 is not overall pd@1=0.999 or pd@0.25=0.888")

    noex = summary.RUNS[1][0]
    (root / "results/scores" / noex / "val/summary.json").unlink()
    self_run = summary.RUNS[2][0]
    score_path = root / "results/scores" / self_run / "val/summary.json"
    score = summary.read_json(score_path)
    score["predictions"][clips[1]]["ckpt_sha256"] = "0" * 64
    write_json(score_path, score)
    nogain = summary.RUNS[3][0]
    fp_path = root / "cache/preds" / (nogain + "__meta") / (clips[0] + ".json")
    fp = summary.read_json(fp_path)
    fp["fingerprint"]["settings"]["bank"] = "none"
    write_json(fp_path, fp)
    mx = summary.RUNS[4][0]
    selection_path = root / "runs" / mx / "best_selection.json"
    selection = summary.read_json(selection_path)
    selection["selection_mode"] = "original_test_iou"
    write_json(selection_path, selection)
    noapp = summary.RUNS[5][0]
    write_json(root / "runs" / noapp / "progress.json", {"stage": "eval", "epoch": 5, "epoch_total": 32})
    rows = summary.summarize(root)
    expected = ["complete", "pending", "error", "error", "error", "pending"]
    assert [row["status"] for row in rows] == expected
    assert all(all(row[key] == "" for key in summary.METRICS) for row in rows[1:])
    checks.extend(["missing summary -> pending with blank metrics", "mixed checkpoint weights -> error with blank metrics",
                   "current cache fingerprint changed -> error with blank metrics", "test-based selection -> error with blank metrics",
                   "in-progress training provisional best -> pending with blank metrics"])

    primary = summary.RUNS[0][0]
    score_path = root / "results/scores" / primary / "val/summary.json"
    original_score = summary.read_json(score_path)
    damaged = copy.deepcopy(original_score)
    damaged["author_headline"]["thresholds"].pop()
    write_json(score_path, damaged)
    assert summary.collect_run(root, primary, "mta_lv", clips)["status"] == "error"
    damaged = copy.deepcopy(original_score)
    damaged["author_headline"]["AUC"] += .1
    write_json(score_path, damaged)
    assert summary.collect_run(root, primary, "mta_lv", clips)["status"] == "error"
    write_json(score_path, original_score)
    checks.extend(["26 thresholds -> error", "AUC inconsistent with 27 curve points -> error"])
    state_path = root / "runtime/val_queue/state.json"
    write_json(state_path, {"version": 1, "jobs": {primary: {"status": "failed", "error": "fixture expected failure"}}})
    rows = summary.summarize(root)
    assert rows[0]["status"] == "error" and rows[0]["Pd_at_0_5"] == ""
    checks.append("scheduler failed overrides stale completed artifacts with blank metrics")
    # Leave readable mixed-state fixture outputs showing one valid complete row.
    write_json(state_path, {"version": 1, "jobs": {primary: {"status": "completed"}}})
    final_rows = summary.summarize(root)
    report = {"status": "passed", "gpu_calls": 0, "training_runs": 0,
              "fixture_root": str(root), "checks": checks,
              "final_status": {row["run"]: row["status"] for row in final_rows},
              "note": "Synthetic fixture values, never experimental results."}
    write_json(HERE / "val_summary_validation.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
