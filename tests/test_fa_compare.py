import importlib.util
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("fa_compare", Path(__file__).parents[1] / "code/fa_compare.py")
fa = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fa)


def point(threshold, false_pixels, true_targets, pixels=1000, targets=100):
    return dict(threshold=threshold, false_pixels=false_pixels, true_targets=true_targets,
                targets=targets, fa=false_pixels / pixels, pd=true_targets / targets)


class FaComparisonTest(unittest.TestCase):
    def test_sparse_segment_is_not_measured(self):
        points = [point(1e-5, 1, 80), point(1e-6, 101, 90)]
        out = fa.compare_at_fa(points, .011)
        self.assertEqual(out["status"], "bracketed")
        self.assertEqual(out["exact_measured_points"], [])
        self.assertAlmostEqual(out["linear_estimate"]["pd"], .81)
        self.assertFalse(out["linear_estimate"]["measured"])
        self.assertEqual(out["best_measured_within_budget"], points[0])
        self.assertEqual(out["bracket"]["pd_endpoint_range"], [.8, .9])

    def test_exact_and_duplicate_fa(self):
        points = [point(.9, 1, 70), point(.8, 1, 80), point(.5, 11, 90)]
        exact = fa.compare_at_fa(points, .001)
        self.assertEqual(len(exact["exact_measured_points"]), 2)
        self.assertEqual(exact["selected_measured_point"]["pd"], .8)
        self.assertIsNone(exact["linear_estimate"])
        out = fa.compare_at_fa(points, .006)
        self.assertAlmostEqual(out["linear_estimate"]["pd"], .85)
        self.assertEqual(len(out["bracket"]["lower_fa_points"]), 2)

    def test_saturated_threshold_one_does_not_get_zero_endpoint(self):
        points = [point(1, 10, 70), point(.99, 30, 90)]
        out = fa.compare_at_fa(points, .005)
        self.assertEqual(out["status"], "unavailable")
        self.assertIsNone(out["best_measured_within_budget"])
        self.assertIsNone(out["linear_estimate"])
        self.assertIn("saturated", out["explanation"])
        self.assertEqual(out["measured_fa_range"], [.01, .03])

    def test_above_range_keeps_budget_sample_without_extrapolation(self):
        points = [point(.99, 1, 70), point(.1, 30, 90)]
        out = fa.compare_at_fa(points, .5)
        self.assertEqual(out["status"], "unavailable")
        self.assertIsNone(out["linear_estimate"])
        self.assertEqual(out["best_measured_within_budget"], points[1])

    def test_zero_is_not_equal_to_small_positive_fa(self):
        out = fa.compare_at_fa([point(1, 1, 70, pixels=10**15)], 0)
        self.assertEqual(out["status"], "unavailable")

    def test_inverse_sparse_interpolation(self):
        points = [point(.99, 10, 80), point(.1, 30, 90)]
        out = fa.estimate_fa_at_pd(points, .85)
        self.assertEqual(out["status"], "estimated_not_measured")
        self.assertAlmostEqual(out["fa"], .02)
        self.assertEqual(fa.estimate_fa_at_pd(points, .95)["status"], "unavailable")

    def test_counts_are_validated(self):
        p = point(.5, 10, 80)
        self.assertEqual(fa.validate_points(dict(complete=True, pixels=1000, thresholds=[p])), [p])
        with self.assertRaises(ValueError):
            fa.validate_points(dict(complete=False, pixels=1000, thresholds=[p]))
        p["fa"] = .2
        with self.assertRaises(ValueError):
            fa.validate_points(dict(complete=True, pixels=1000, thresholds=[p]))

    def test_breakdown_requires_identity_completion_and_denominators(self):
        info = dict(checkpoint_sha256="abc", variant="native", pixels=1000, frames=2, targets=100)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(fa.load_breakdown(info, .99, root)["status"], "pending")
            record = dict(status="complete", complete=True, pixels=1000, frames=2, targets=100)
            data = dict(checkpoint_sha256="abc", split="official_test", threshold=.99,
                        status="complete", results={"native": record})
            path = root / "job.json"
            path.write_text(json.dumps(data))
            loaded = fa.load_breakdown(info, .99, root)
            self.assertEqual(loaded["status"], "complete")
            self.assertEqual(loaded["threshold"], .99)
            self.assertIn("Fixed-threshold", loaded["geometry_scope"])
            self.assertEqual(fa.load_breakdown(info, .9, root)["status"], "pending")
            self.assertEqual(fa.load_breakdown(info, None, root)["status"], "complete")
            record["status"] = "mismatch"
            path.write_text(json.dumps(data))
            self.assertEqual(fa.load_breakdown(info, .99, root)["status"], "mismatch")
            record["pixels"] = 999
            path.write_text(json.dumps(data))
            self.assertEqual(fa.load_breakdown(info, .99, root)["status"], "unavailable")

    def test_active_alias_does_not_silently_select_interim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "final.json"
            path.write_text(json.dumps(dict(complete=True, pixels=1000, frames=2,
                model_id="deeppro_plus", job_id="final", variant="native", primary_variant="native",
                checkpoint_sha256="finalhash", checkpoint_status="final", thresholds=[point(.99, 1, 80)])))
            config = {"active_results": [{"path": "final.json", "checkpoint_sha256": "finalhash"}]}
            info, points = fa.resolve_run("deeppro_plus", root, config)
            self.assertEqual(info["job_id"], "final")
            config["active_results"][0]["checkpoint_sha256"] = "otherhash"
            with self.assertRaises(ValueError):
                fa.resolve_run("deeppro_plus", root, config)

    def exact_fixture(self):
        info = dict(pixels=1000, frames=2, targets=100)
        best, upper = point(.71, 4, 80), point(.7, 7, 90)
        budget = dict(fa_budget=.005, fp_budget=5, feasible=True, best_point=best,
                      lower_fa_point=best, upper_fa_point=upper, exact_fp_budget_reached=False)
        payload = dict(complete=True, method="exact_empirical_scores_no_interpolation",
                       pixels=1000, frames=2, targets=100, budgets=[budget],
                       minimum_attainable_point=point(1, 1, 70))
        breakdown = dict(status="complete", path="synthetic.json", result={"exact_matched_fa": payload})
        return info, breakdown, payload

    def test_exact_real_score_point_keeps_achieved_fa_and_original_curve_separate(self):
        info, breakdown, payload = self.exact_fixture()
        original_curve = [point(1, 1, 70), point(.5, 20, 95)]
        saved = copy.deepcopy(original_curve)
        estimated = fa.compare_at_fa(original_curve, .005)
        exact = fa.exact_matches(info, breakdown, [.005])[0]
        self.assertEqual(exact["status"], "measured_budget_optimum")
        self.assertTrue(exact["reachable"])
        self.assertFalse(exact["exact_budget_match"])
        self.assertEqual(exact["best_point"]["fa"], .004)
        self.assertEqual(exact["best_point"]["pd"], .8)
        self.assertNotEqual(exact["best_point"]["pd"], estimated["linear_estimate"]["pd"])
        self.assertEqual(original_curve, saved)

    def test_exact_unreachable_saturation_remains_unreachable(self):
        info, breakdown, payload = self.exact_fixture()
        payload["budgets"] = [dict(fa_budget=0, fp_budget=0, feasible=False, best_point=None,
            lower_fa_point=None, upper_fa_point=payload["minimum_attainable_point"],
            reason="threshold_one_still_exceeds_budget_due_to_float32_scores_equal_one")]
        exact = fa.exact_matches(info, breakdown, [0])[0]
        self.assertEqual(exact["status"], "unreachable_saturated")
        self.assertFalse(exact["reachable"])
        self.assertIsNone(exact["best_point"])
        self.assertEqual(exact["minimum_attainable_point"]["fa"], .001)

    def test_exact_requires_completed_scores_and_requested_budget(self):
        info, breakdown, payload = self.exact_fixture()
        self.assertEqual(fa.exact_matches(info, dict(status="pending"), [.005])[0]["status"], "pending")
        self.assertEqual(fa.exact_matches(info, breakdown, [.007])[0]["status"], "unavailable")
        payload["complete"] = False
        self.assertEqual(fa.exact_matches(info, breakdown, [.005])[0]["status"], "pending")

    def test_exact_reference_mismatch_is_disclosed_and_bad_budget_rejected(self):
        info, breakdown, payload = self.exact_fixture()
        breakdown["status"] = "mismatch"
        exact = fa.exact_matches(info, breakdown, [.005])[0]
        self.assertEqual(exact["reference_status"], "mismatch")
        self.assertIn("differs", exact["warning"])
        payload["budgets"][0]["best_point"] = point(.7, 7, 90)
        with self.assertRaises(ValueError):
            fa.exact_matches(info, breakdown, [.005])

    def test_csv_contains_real_counts_alongside_separate_estimate(self):
        info, breakdown, payload = self.exact_fixture()
        run = dict(requested_run="synthetic", job_id="synthetic", variant="native",
                   checkpoint_status="test_fixture", checkpoint_sha256="test_fixture",
                   comparisons=[fa.compare_at_fa([point(1, 1, 70), point(.5, 20, 95)], .005)],
                   exact_matched_fa=fa.exact_matches(info, breakdown, [.005]))
        row = list(fa.csv_rows({"runs": [run]}))[0]
        self.assertEqual(row["exact_false_pixels"], 4)
        self.assertEqual(row["exact_status"], "measured_budget_optimum")
        self.assertEqual(row["exact_fa"], .004)
        self.assertIsNotNone(row["linear_pd_estimate_NOT_MEASURED"])

    def test_cli_always_shows_exact_before_sparse_roc_geometry_is_optional(self):
        info, breakdown, payload = self.exact_fixture()
        info.update(requested_run="synthetic", job_id="synthetic", variant="native", model_id="synthetic",
                    checkpoint_status="fixture", checkpoint_sha256="fixture")
        breakdown.update(threshold=.99)
        breakdown["result"].update(true_targets=80, targets=100, false_pixels=4)
        points = [point(.99, 1, 70), point(.5, 20, 95)]
        for geometry in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "config.json").write_text('{}')
                output = io.StringIO()
                argv = ["--runs", "synthetic", "--evaluation-root", str(root), "--fa", ".005"]
                if geometry:
                    argv.append("--breakdown")
                with patch.object(fa, "resolve_run", return_value=(copy.deepcopy(info), points)), \
                     patch.object(fa, "load_breakdown", return_value=copy.deepcopy(breakdown)) as loader, \
                     patch.object(fa, "write_report", return_value=(Path("fixture.json"), Path("fixture.csv"))), \
                     contextlib.redirect_stdout(output):
                    self.assertEqual(fa.main(argv), 0)
                text = output.getvalue()
                self.assertIn("Measured budget optimum: Pd=80.000000%", text)
                self.assertLess(text.index("Exact matched-Fa"), text.index("Supporting saved 27-point"))
                self.assertEqual("Fixed-threshold geometry breakdown" in text, geometry)
                self.assertNotIn("Claim check", text)
                self.assertIsNone(loader.call_args_list[0].args[1])


if __name__ == "__main__":
    unittest.main()
