#!/usr/bin/env python3
"""Compare saved RGB-T ROC samples without inference or synthesized endpoints.

Examples (from DP):
  python code/fa_compare.py --runs official_rgbt deeppro_plus --split test
  python code/fa_compare.py --runs official_rgbt deeppro_plus_interim --fa 0.0018
  python code/fa_compare.py --runs official_rgbt mr_lv:rate1_auto --breakdown --threshold .99

Uses only the Python standard library. All probabilities are fractions, not percent.
Completed exact_matched_fa diagnostics from real FP32 scores are always read.
--breakdown additionally displays fixed-threshold spatial geometry diagnostics.
An exact budget optimum may have achieved Fa below the budget because tied scores
cannot be split by a single threshold. These measurements never replace the 27-point
curve or its separately labeled estimates. This script does not run a model.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATION = PROJECT / "reproduction/rgbt_unified_evaluation_20261126"
DEFAULT_OUTPUT = PROJECT / "reproduction/rgbt_fa_comparison_20261127"
DEFAULT_FA = [4.467025818116215e-5, 0.00045281018278844545, 6.136428240631799e-4]
INTERIM_PATH = "results/deeppro_plus_c1c818212eb0c1d8/native/metrics.json"
ALIASES = {
    "official_rgbt": "released_reference", "official": "released_reference",
    "author_released_reference": "released_reference", "lv": "lv_full",
    "lvh": "lvh_full", "mrtp": "mr_lv", "lvh-dec": "lvh_dec",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def equal(a, b):
    # Tolerates textual float round trips; does not conflate small nonzero Fa with zero.
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0)


def point_order(point):
    return (point["pd"], -point["fa"], point["threshold"])


def validate_points(metrics):
    if metrics.get("complete") is not True:
        raise ValueError("metrics must explicitly have complete=true")
    pixels = metrics.get("pixels")
    if not isinstance(pixels, (int, float)) or pixels <= 0:
        raise ValueError("metrics must contain a positive pixel denominator")
    points = []
    for index, row in enumerate(metrics.get("thresholds", [])):
        point = dict(row)
        for key in ("threshold", "fa", "pd"):
            value = float(point[key])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"invalid {key} in ROC row {index}")
            point[key] = value
        if point.get("targets", 0) <= 0:
            raise ValueError(f"invalid targets in ROC row {index}")
        for numerator, denominator, ratio in (
            (point["false_pixels"], pixels, "fa"),
            (point["true_targets"], point["targets"], "pd"),
        ):
            if numerator < 0 or numerator > denominator:
                raise ValueError(f"invalid counts for {ratio} in row {index}")
            expected = numerator / denominator
            if not math.isclose(expected, point[ratio], rel_tol=1e-10, abs_tol=1e-15):
                raise ValueError(f"{ratio} disagrees with counts in row {index}")
            point[ratio] = expected
        points.append(point)
    if not points:
        raise ValueError("empty ROC grid")
    if len({p["targets"] for p in points}) != 1:
        raise ValueError("target denominator changes across ROC grid")
    return points


def groups_by(points, axis):
    groups = {}
    for point in points:
        groups.setdefault(point[axis], []).append(point)
    return sorted(groups.items())


def compare_at_fa(points, target):
    """Retain exact samples, budget-best sample and adjacent distinct-Fa groups.

    For duplicate Fa, all samples are preserved; interpolation uses the highest-Pd
    sample at each adjacent Fa. This explicit tie policy avoids division by zero.
    """
    if not math.isfinite(target) or not 0 <= target <= 1:
        raise ValueError("target Fa must be a finite fraction in [0,1]")
    groups = groups_by(points, "fa")
    low_fa, high_fa = groups[0][0], groups[-1][0]
    exact = [p for p in points if equal(p["fa"], target)]
    eligible = [p for p in points if p["fa"] <= target or equal(p["fa"], target)]
    out = {
        "target_fa": target,
        "measured_fa_range": [low_fa, high_fa],
        "exact_measured_points": exact,
        "best_measured_within_budget": max(eligible, key=point_order) if eligible else None,
        "bracket": None,
        "linear_estimate": None,
        "duplicate_fa_policy": "Preserve all points; select maximum Pd per Fa for interpolation.",
    }
    if exact:
        out.update(status="measured", selected_measured_point=max(exact, key=point_order))
        return out
    if target < low_fa or target > high_fa:
        out["status"] = "unavailable"
        out["reason"] = "below_measured_fa_range" if target < low_fa else "above_measured_fa_range"
        out["explanation"] = "No measured point/bracket at this Fa; no extrapolation or added endpoints."
        saturated = [p for p in points if p["threshold"] == 1.0 and p["false_pixels"] > 0]
        if target < low_fa and saturated:
            out["threshold_one_points"] = saturated
            out["explanation"] += (
                " Threshold 1 still has false positives under the saved >= rule, consistent with "
                "FP32 probabilities saturated at 1; a lower-Fa point is unavailable in this grid."
            )
        return out
    lower = max((g for g in groups if g[0] < target), key=lambda g: g[0])
    upper = min((g for g in groups if g[0] > target), key=lambda g: g[0])
    left, right = max(lower[1], key=point_order), max(upper[1], key=point_order)
    weight = (target - lower[0]) / (upper[0] - lower[0])
    out["status"] = "bracketed"
    out["bracket"] = {
        "lower_fa_points": lower[1], "upper_fa_points": upper[1],
        "selected_lower": left, "selected_upper": right,
        "fa_range": [lower[0], upper[0]],
        "pd_endpoint_range": [min(p["pd"] for p in lower[1] + upper[1]),
                              max(p["pd"] for p in lower[1] + upper[1])],
        "range_meaning": "Measured adjacent-point range; not a confidence interval or a new measurement.",
    }
    out["linear_estimate"] = {
        "fa": target, "pd": left["pd"] + weight * (right["pd"] - left["pd"]),
        "measured": False, "interpolation_weight": weight,
        "method": "Linear Pd versus Fa between selected adjacent measured Fa groups.",
        "warning": "Estimate only; sparse ROC samples do not establish actual performance at this Fa.",
    }
    return out


def estimate_fa_at_pd(points, target):
    """Inverse interpolation solely to audit a numeric claim; not a measured point."""
    groups = groups_by(points, "pd")
    exact = [p for p in points if equal(p["pd"], target)]
    if exact:
        return {"status": "measured", "target_pd": target, "exact_measured_points": exact}
    if target < groups[0][0] or target > groups[-1][0]:
        return {"status": "unavailable", "target_pd": target, "reason": "outside_measured_pd_range"}
    lower = max((g for g in groups if g[0] < target), key=lambda g: g[0])
    upper = min((g for g in groups if g[0] > target), key=lambda g: g[0])
    left, right = min(lower[1], key=lambda p: p["fa"]), min(upper[1], key=lambda p: p["fa"])
    weight = (target - lower[0]) / (upper[0] - lower[0])
    return {
        "status": "estimated_not_measured", "target_pd": target,
        "fa": left["fa"] + weight * (right["fa"] - left["fa"]),
        "lower_pd_point": left, "upper_pd_point": right,
        "duplicate_pd_policy": "Minimum Fa at each distinct Pd.",
        "warning": "Linear Fa versus Pd on saved samples; no inference or threshold search was performed.",
    }


def resolve_run(spec, evaluation, config):
    explicit = Path(spec).expanduser()
    if explicit.is_dir():
        explicit = explicit / "metrics.json"
    if explicit.is_file():
        path, expected_hash = explicit.resolve(), None
    elif spec == "deeppro_plus_interim":
        path, expected_hash = evaluation / INTERIM_PATH, None
    else:
        model_spec, sep, variant = spec.partition(":")
        if not sep and "/" in spec:
            model_spec, variant = spec.rsplit("/", 1)
        model_spec = ALIASES.get(model_spec.lower(), model_spec)
        for model in config.get("models", []):
            if model_spec == model.get("run"):
                model_spec = model["id"]
        matches = []
        for entry in config.get("active_results", []):
            candidate = evaluation / entry["path"]
            if not candidate.is_file():
                continue
            data = read_json(candidate)
            names = {data.get("model_id"), data.get("job_id"), data.get("model"), data.get("label")}
            chosen_variant = variant or data.get("primary_variant", "native")
            if model_spec in names and data.get("variant") == chosen_variant:
                matches.append((candidate, entry.get("checkpoint_sha256")))
        if len(matches) != 1:
            raise ValueError(f"{spec!r}: expected one active result, found {len(matches)}; use model:variant or metrics.json path")
        path, expected_hash = matches[0]
    data = read_json(path)
    if expected_hash and data.get("checkpoint_sha256") != expected_hash:
        raise ValueError(f"active checkpoint hash mismatch: {path}")
    if data.get("split", "test") != "test":
        raise ValueError(f"not a test-split result: {path}")
    points = validate_points(data)
    info = {key: data.get(key) for key in (
        "job_id", "model", "model_id", "variant", "primary_variant", "checkpoint_sha256",
        "checkpoint_status", "final_checkpoint", "best_epoch", "training_epochs_completed", "frames", "pixels",
    )}
    info.update(requested_run=spec, metrics_path=str(path.resolve()),
                metrics_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                targets=points[0]["targets"], measured_point_count=len(points), split="test")
    return info, points


def load_breakdown(info, threshold, directory):
    """Read completed identity-matched diagnostics only; never launch inference.

    threshold=None allows score-based exact-Fa statistics to be read independently
    of the threshold used for the diagnostic file's spatial geometry.
    """
    matches = []
    if directory.is_dir():
        for path in directory.glob("**/*.json"):
            try:
                data = read_json(path)
            except (OSError, ValueError):
                continue
            record = data.get("results", {}).get(info["variant"])
            if (data.get("checkpoint_sha256") == info["checkpoint_sha256"]
                    and isinstance(record, dict)
                    and data.get("split") in {"test", "official_test"}
                    and isinstance(data.get("threshold"), (float, int))
                    and (threshold is None or equal(float(data["threshold"]), threshold))):
                matches.append((path, data, record))
    completed = [(p, d, r) for p, d, r in matches
                 if r.get("status") in {"complete", "mismatch"} and r.get("complete") is True]
    if len(completed) == 1:
        path, data, record = completed[0]
        if (record.get("pixels") != info["pixels"] or record.get("frames") != info["frames"]
                or record.get("targets") != info["targets"]):
            return {"status": "unavailable", "reason": "breakdown frame/pixel/target denominator mismatch", "path": str(path)}
        return {"status": record["status"], "path": str(path),
                "threshold": float(data["threshold"]),
                "geometry_scope": "Fixed-threshold geometry only; not recomputed at exact matched-Fa thresholds.",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "job_status": data.get("status"), "result": record,
                "reference_comparison": data.get("reference_comparison", record.get("reference_comparison"))}
    if len(completed) > 1:
        return {"status": "unavailable", "reason": "multiple matching completed breakdowns", "paths": [str(p) for p, _, _ in completed]}
    return {"status": "pending", "threshold": threshold, "searched_directory": str(directory),
            "reason": "No completed matching prediction breakdown is saved. ROC points alone cannot recover pixel error locations.",
            "matching_records": [{"path": str(p), "status": r.get("status")} for p, _, r in matches]}


def exact_matches(info, breakdown, targets):
    """Read measured budget optima, keeping them separate from saved ROC estimates."""
    record = breakdown.get("result", {})
    payload = record.get("exact_matched_fa", record.get("exact_fa"))
    output = []
    for target in targets:
        result = {"target_fa": target, "status": "pending", "reachable": None,
                  "best_point": None, "reference_status": breakdown["status"],
                  "source_path": breakdown.get("path"),
                  "method": "Exact threshold statistics from actual FP32 predictions; not 27-point interpolation."}
        if breakdown["status"] not in {"complete", "mismatch"}:
            result["reason"] = "Completed matching prediction diagnostics are unavailable."
            output.append(result)
            continue
        if not isinstance(payload, dict) or not payload.get("complete", payload.get("status") == "complete"):
            result["reason"] = "The completed breakdown has no finalized exact same-Fa score statistics."
            output.append(result)
            continue
        if payload.get("method") != "exact_empirical_scores_no_interpolation":
            raise ValueError("unknown exact same-Fa method; refusing to label it a real-score measurement")
        for field in ("pixels", "frames", "targets"):
            if field in payload and payload[field] != info[field]:
                raise ValueError(f"exact same-Fa {field} denominator mismatch")
        budgets = [entry for entry in payload.get("budgets", [])
                   if equal(float(entry["fa_budget"]), target)]
        if len(budgets) != 1:
            result.update(status="unavailable", reason=f"Expected one exact result for this budget; found {len(budgets)}.")
            output.append(result)
            continue
        budget = budgets[0]
        best = budget.get("best_point")
        feasible = budget.get("feasible", budget.get("reachable"))
        result.update(fp_budget=budget.get("fp_budget"), source_result=budget,
                      lower_fa_point=budget.get("lower_fa_point"),
                      upper_fa_point=budget.get("upper_fa_point"),
                      minimum_attainable_point=payload.get("minimum_attainable_point"))
        if feasible is False:
            if best is not None:
                raise ValueError("unreachable exact same-Fa result unexpectedly has a best point")
            result.update(status="unreachable", reachable=False,
                          reason=budget.get("reason", "No admissible FP32 threshold achieves this false-pixel budget."))
            if result["reason"] == "threshold_one_still_exceeds_budget_due_to_float32_scores_equal_one":
                result["status"] = "unreachable_saturated"
                result["explanation"] = "FP32 scores equal to 1 still produce too many false positives at threshold=1; no threshold in [0,1] meets this budget."
            output.append(result)
            continue
        if feasible is not True or not isinstance(best, dict):
            raise ValueError("exact same-Fa result must state feasible and provide a best point when feasible")
        point = validate_points({"complete": True, "pixels": info["pixels"], "thresholds": [best]})[0]
        if point["targets"] != info["targets"]:
            raise ValueError("exact same-Fa target denominator mismatch")
        if point["fa"] > target and not equal(point["fa"], target):
            raise ValueError("exact same-Fa best point exceeds its requested budget")
        if budget.get("fp_budget") is not None and point["false_pixels"] > budget["fp_budget"]:
            raise ValueError("exact same-Fa false pixels exceed integer budget")
        result.update(status="measured_budget_optimum", reachable=True, best_point=point,
                      exact_budget_match=equal(point["fa"], target),
                      note="Achieved Fa is reported explicitly; score ties can leave unused budget.")
        if breakdown["status"] == "mismatch":
            result["warning"] = "This recomputation differs from saved 27-point counts; it is not a verification of those counts."
        output.append(result)
    return output


def csv_rows(report):
    for run in report["runs"]:
        for comparison in run["comparisons"]:
            row = {key: run[key] for key in ("requested_run", "job_id", "variant", "checkpoint_status", "checkpoint_sha256")}
            row.update(target_fa=comparison["target_fa"], status=comparison["status"],
                       reason=comparison.get("reason", ""))
            for prefix, point in (("measured", comparison.get("selected_measured_point")),
                                  ("budget_best", comparison["best_measured_within_budget"])):
                for key in ("threshold", "fa", "pd", "false_pixels", "true_targets"):
                    row[f"{prefix}_{key}"] = point.get(key) if point else None
            bracket = comparison["bracket"]
            for prefix, point in (("lower", bracket["selected_lower"] if bracket else None),
                                  ("upper", bracket["selected_upper"] if bracket else None)):
                for key in ("threshold", "fa", "pd"):
                    row[f"{prefix}_{key}"] = point.get(key) if point else None
            row["linear_pd_estimate_NOT_MEASURED"] = (comparison["linear_estimate"] or {}).get("pd")
            exact = next((item for item in run.get("exact_matched_fa", [])
                          if equal(item["target_fa"], comparison["target_fa"])), {})
            row.update(exact_status=exact.get("status", "not_requested"),
                       exact_reachable=exact.get("reachable"),
                       exact_budget_match=exact.get("exact_budget_match"),
                       exact_reference_status=exact.get("reference_status"),
                       exact_reason=exact.get("reason"),
                       exact_minimum_attainable_fa=(exact.get("minimum_attainable_point") or {}).get("fa"))
            point = exact.get("best_point") or {}
            for key in ("threshold", "fa", "pd", "false_pixels", "true_targets", "targets"):
                row[f"exact_{key}"] = point.get(key)
            yield row


def write_report(report, directory):
    directory.mkdir(parents=True, exist_ok=True)
    stem = "comparison_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    json_path, csv_path = directory / (stem + ".json"), directory / (stem + ".csv")
    with json_path.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    rows = list(csv_rows(report))
    with csv_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def format_point(point):
    if not point:
        return "unavailable"
    return f"Pd={point['pd'] * 100:.6f}%, Fa={point['fa']:.10g}, threshold={point['threshold']:.10g}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=["official_rgbt", "deeppro_plus"])
    parser.add_argument("--split", choices=["test"], default="test", help="Only saved test ROC results are available.")
    parser.add_argument("--fa", nargs="+", type=float, default=DEFAULT_FA, help="Target false-pixel fractions; replaces defaults.")
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--breakdown", action="store_true", help="Additionally display fixed-threshold spatial geometry; exact same-Fa results are always read.")
    parser.add_argument("--threshold", type=float, default=.99, help="Measured operating point and breakdown threshold.")
    parser.add_argument("--breakdown-dir", type=Path, default=None,
                        help="Defaults to evaluation-root/diagnostics/breakdown.")
    args = parser.parse_args(argv)
    if args.breakdown_dir is None:
        args.breakdown_dir = args.evaluation_root / "diagnostics/breakdown"
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0,1]")
    config_path = args.evaluation_root / "config.json"
    try:
        config = read_json(config_path)
        resolved = [resolve_run(spec, args.evaluation_root, config) for spec in args.runs]
        signatures = {(r[0]["frames"], r[0]["pixels"], r[0]["targets"]) for r in resolved}
        if len(signatures) != 1:
            raise ValueError("cannot compare different frame/pixel/target denominators")
        report = {
            "schema_version": 2, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "arguments": list(argv) if argv is not None else sys.argv[1:],
            "split": args.split, "config_path": str(config_path.resolve()),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "protocol": {
                "source": "Saved complete ROC results and optional completed FP32-score diagnostics; no model execution by this CLI.",
                "fa": "False-positive pixels divided by all test image pixels; fractions, not percentages.",
                "pd": "Detected target components divided by target components; fractions.",
                "budget_best": "Highest Pd among measured points with Fa <= requested budget; may use a lower Fa.",
                "linear_estimate": "Descriptive interpolation only, not a measured point or inferred threshold.",
                "outside_range": "Unavailable at target; no extrapolation or added (0,0)/(1,1) endpoints.",
                "exact_matched_fa": "Separate real-score budget-optimal threshold measurements; achieved Fa can be below budget. Original 27-point arrays remain unchanged.",
                "comparison_limit": "Retrospective test-set operating-point analysis: exact same-Fa statistics search test-score thresholds under a false-pixel budget. This is not held-out validation.",
            },
            "runs": [],
        }
        for info, points in resolved:
            info["comparisons"] = [compare_at_fa(points, target) for target in args.fa]
            info["measured_at_requested_threshold"] = [p for p in points if equal(p["threshold"], args.threshold)]
            info["numerical_claim_checks"] = {
                "pd_at_fa_0_0018": compare_at_fa(points, .0018),
                "fa_at_pd_0_888": estimate_fa_at_pd(points, .888),
                "fa_at_interim_pd_0_9034513274336283": estimate_fa_at_pd(points, .9034513274336283),
            }
            exact_source = load_breakdown(info, None, args.breakdown_dir)
            info["exact_source"] = {key: value for key, value in exact_source.items() if key != "result"}
            info["exact_matched_fa"] = exact_matches(info, exact_source, args.fa)
            if args.breakdown:
                info["breakdown"] = (exact_source if exact_source.get("threshold") is not None
                                     and equal(exact_source["threshold"], args.threshold)
                                     else load_breakdown(info, args.threshold, args.breakdown_dir))
            report["runs"].append(info)
        outputs = write_report(report, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print("All Pd values below are percentages. Interpolated estimates are NOT measurements.")
    print(report["protocol"]["comparison_limit"])
    for run in report["runs"]:
        print(f"\n{run['requested_run']} -> {run['job_id']}/{run['variant']} [{run['checkpoint_status']}]")
        print("  Exact matched-Fa from real FP32 scores:")
        for exact in run["exact_matched_fa"]:
            print(f"    Budget Fa={exact['target_fa']:.12g}: {exact['status']}, reachable={exact['reachable']}")
            if exact["best_point"]:
                print("      Measured budget optimum: " + format_point(exact["best_point"]))
                if not exact["exact_budget_match"]:
                    print("      Achieved Fa is below the budget; it is not an exact equality of Fa.")
            if exact.get("reason"):
                print("      " + exact["reason"])
            if exact.get("explanation"):
                print("      " + exact["explanation"])
            if exact["reachable"] is False and exact.get("minimum_attainable_point"):
                print("      Minimum attainable point: " + format_point(exact["minimum_attainable_point"]))
            if exact.get("warning"):
                print("      " + exact["warning"])
        print("  Supporting saved 27-point ROC (sparse-grid measurements and estimates):")
        if run["measured_at_requested_threshold"]:
            for point in run["measured_at_requested_threshold"]:
                print("  Measured operating point: " + format_point(point))
        else:
            print(f"  No saved measurement at requested threshold {args.threshold:g}.")
        for result in run["comparisons"]:
            print(f"  Target Fa={result['target_fa']:.12g}: {result['status']}")
            if result.get("selected_measured_point"):
                print("    Measured at target: " + format_point(result["selected_measured_point"]))
            print("    Best measured within budget: " + format_point(result["best_measured_within_budget"]))
            if result["bracket"]:
                print("    Lower neighbor: " + format_point(result["bracket"]["selected_lower"]))
                print("    Upper neighbor: " + format_point(result["bracket"]["selected_upper"]))
                print(f"    Linear estimate (NOT measured): Pd={100 * result['linear_estimate']['pd']:.6f}%")
            if result.get("explanation"):
                print("    " + result["explanation"])
        if args.breakdown:
            geometry_threshold = run["breakdown"].get("threshold", args.threshold)
            print(f"  Fixed-threshold geometry breakdown (threshold={geometry_threshold:g}): " + run["breakdown"]["status"])
            print("    Geometry is not recomputed at the exact matched-Fa thresholds.")
            if run["breakdown"]["status"] in {"complete", "mismatch"}:
                result = run["breakdown"]["result"]
                print(f"    TP={result['true_targets']}/{result['targets']}, FP={result['false_pixels']}")
                for key in ("foreground_fp_pixels", "near_fp_pixels", "far_fp_pixels", "empty_gt_fp_pixels"):
                    print(f"    {key}: {result.get(key, 'not recorded')}")
                if run["breakdown"]["status"] == "mismatch":
                    print("    Diagnostic recomputation differs from saved ROC counts; see reference_comparison.")
            else:
                print("    " + run["breakdown"].get("reason", "Not complete."))
    print("\nSaved JSON: " + str(outputs[0]))
    print("Saved CSV: " + str(outputs[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
