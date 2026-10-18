import copy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pipeline.force_history import (
    CSV_NAME, RAW_NAME, assess_force_stability, assess_saved_force_history,
    configure_history, finish_history, parse_report_file, read_saved_force_history,
    settings,
)
from pipeline.temperature_export import temperature_conditions
from pipeline.utils import atomic_json, file_sha256


NAMES = {"drag": "nose-study-drag", "lift": "nose-study-lift"}


def config(work):
    return dict(project=dict(name="nose", work_dir=str(work)),
                flight=dict(mach=8., static_pressure_pa=1090., static_temperature_k=226.5,
                            flow_direction=[-1., 0.]),
                geometry=dict(body_wall_name="body_wall", input_file="nose.x_t"),
                thermal=dict(mode="radiative_equilibrium", emissivity=.9,
                             external_radiation_temperature_k=3.),
                aero=dict(enabled=True, force_history=dict(enabled=True)))


def rows(drags, start=25, step=25):
    return [dict(iteration=start+i*step, drag_per_unit_span_N_m=drag,
                 lift_per_unit_span_N_m=-2.) for i, drag in enumerate(drags)]


def native_file(path, records):
    # Deliberately change column order and add Fluent's title/metadata lines.
    path.write_text('"nose-study-force-history"\n'
                    '"Iteration" "nose-study-lift" "nose-study-drag"\n'
                    + "\n".join(f'{row["iteration"]} {row["lift_per_unit_span_N_m"]} {row["drag_per_unit_span_N_m"]}'
                                for row in records) + "\n", encoding="utf-8")


def final_solver(iteration, drag=100., lift=-2., expose_counter=True):
    def compute(report_defs):
        name = report_defs[0]
        value = drag if name == NAMES["drag"] else lift
        return {name: [value, iteration] if expose_counter else [value]}
    return SimpleNamespace(settings=SimpleNamespace(solution=SimpleNamespace(
        report_definitions=SimpleNamespace(compute=compute))))


def context(cfg, raw):
    return dict(status="configured", settings=settings(cfg), raw_path=raw,
                report=SimpleNamespace(active=True), names=NAMES,
                reference_state=dict(report_depth_m=1.), body_wall_zones=["body_wall"])


class ForceHistoryTests(unittest.TestCase):
    def test_settings_validate_samples_tolerances_and_boolean_controls(self):
        self.assertEqual(settings({})["sample_every_iterations"], 25)
        for key, value in (("sample_every_iterations", 0), ("recent_samples", 1),
                           ("recent_samples", 1.5), ("min_samples", True),
                           ("relative_tolerance", 0), ("relative_tolerance", float("nan")),
                           ("absolute_drag_tolerance_N_m", -1), ("enabled", "true"),
                           ("require_stable", 1), ("unknown", 3)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                settings({"aero": {"force_history": {key: value}}})
        with self.assertRaisesRegex(ValueError, "min_samples"):
            settings({"aero": {"force_history": dict(recent_samples=10, min_samples=5)}})

    def test_stability_requires_enough_samples_and_rejects_mean_cancelling_oscillations(self):
        passed = assess_force_stability(rows([100., 100.1, 100., 99.9, 100.]), {})
        self.assertEqual(passed["status"], "passed")
        self.assertAlmostEqual(passed["relative_final_to_recent_mean_change"], 0)
        oscillating = assess_force_stability(rows([90., 110., 90., 110., 100.]), {})
        self.assertEqual(oscillating["relative_final_to_recent_mean_change"], 0)
        self.assertEqual(oscillating["status"], "failed")
        drifting = assess_force_stability(rows([90., 92., 95., 98., 100.]), {})
        self.assertEqual(drifting["status"], "failed")
        self.assertEqual(assess_force_stability(rows([100.]), {})["status"], "insufficient_history")

    def test_zero_drag_uses_explicit_absolute_criterion_and_negative_drag_stays_signed(self):
        stable = assess_force_stability(rows([0., 1e-8, 0., -1e-8, 0.]), {})
        self.assertEqual(stable["status"], "passed")
        self.assertIsNone(stable["relative_final_to_recent_mean_change"])
        unstable = assess_force_stability(rows([0., .1, 0., -.1, 0.]), {})
        self.assertEqual(unstable["status"], "failed")
        negative = assess_force_stability(rows([-100.] * 5), {})
        self.assertEqual(negative["status"], "passed")
        self.assertEqual(negative["final_drag_per_unit_span_N_m"], -100)

    def test_native_parser_matches_names_and_actual_iterations_not_column_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / RAW_NAME
            native_file(path, rows([200., 198.], start=475, step=7))
            measured = parse_report_file(path, NAMES, report_depth_m=2.)
            self.assertEqual([row["iteration"] for row in measured], [475, 482])
            self.assertEqual(measured[-1]["drag_per_unit_span_N_m"], 99.)
            self.assertEqual(measured[-1]["lift_per_unit_span_N_m"], -1.)
            for record in ([dict(iteration=1, drag_per_unit_span_N_m=float("nan"), lift_per_unit_span_N_m=1)],
                           rows([1, 2], start=1, step=0), rows([1], start=-1)):
                native_file(path, record)
                with self.assertRaises(ValueError):
                    parse_report_file(path, NAMES)

    def test_disabled_and_cold_calibration_configuration_never_touch_native_api(self):
        cfg = config(Path("unused"))
        cfg["aero"]["force_history"]["enabled"] = False
        self.assertEqual(configure_history(object(), cfg, ["body_wall"])["status"], "disabled")
        cfg["aero"]["force_history"]["enabled"] = True
        cfg["_calibration_kind"] = "cold_wall"
        self.assertEqual(configure_history(object(), cfg, ["body_wall"])["status"], "disabled")

    def test_configure_records_every_native_iteration_without_changing_solver_controls(self):
        class Reports(dict):
            def get_object_names(self):
                return list(self)
            def create(self, name):
                self[name] = SimpleNamespace(frequency_of=SimpleNamespace(allowed_values=lambda: ["iteration"]))
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(Path(directory))
            collection = Reports()
            solver = SimpleNamespace(settings=SimpleNamespace(solution=SimpleNamespace(
                monitor=SimpleNamespace(report_files=collection))))
            original = copy.deepcopy(cfg)
            with patch("pipeline.aero.configure_force_reports", return_value=(NAMES, dict(report_depth_m=1.))) as reports:
                current = configure_history(solver, cfg, ["body_wall"])
            report = current["report"]
            self.assertEqual(report.frequency, 1)
            self.assertEqual(report.frequency_of, "iteration")
            self.assertEqual(report.report_defs, [NAMES["drag"], NAMES["lift"]])
            self.assertTrue(report.active)
            self.assertEqual(cfg, original)
            reports.assert_called_once_with(solver, cfg, ["body_wall"])

    def test_finish_retains_early_actual_final_iteration_and_usable_raw_history(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            cfg = config(work)
            raw = work / RAW_NAME
            native_file(raw, rows([100.] * 109, start=1, step=1))
            current = context(cfg, raw)
            metadata = finish_history(final_solver(109), cfg, current)
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["final_iteration"], 109)
            self.assertFalse(current["report"].active)
            self.assertEqual(metadata["force_stability"]["status"], "passed")
            with (work / CSV_NAME).open() as handle:
                recorded = list(csv.DictReader(handle))
            self.assertEqual([int(row["iteration"]) for row in recorded], [25, 50, 75, 100, 109])
            self.assertEqual(metadata["raw_report_sha256"], file_sha256(raw))
            self.assertEqual(metadata["conditions"], temperature_conditions(cfg))

    def test_finish_never_appends_samples_from_undocumented_compute_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            cfg = config(work)
            cfg["solver"] = dict(max_iterations=4800)
            raw = work / RAW_NAME
            native_file(raw, rows([100.] * 100, start=1, step=1))
            metadata = finish_history(final_solver(103), cfg, context(cfg, raw))
            self.assertEqual(metadata["final_iteration"], 100)
            with (work / CSV_NAME).open() as handle:
                recorded = list(csv.DictReader(handle))
            self.assertEqual(int(recorded[-1]["iteration"]), 100)
            self.assertEqual(metadata["native_sample_count"], 100)
            for counter in (0, 1, 99999):
                with self.subTest(compute_integer=counter):
                    verified = finish_history(final_solver(counter), cfg, context(cfg, raw))
                    self.assertEqual(verified["status"], "complete")
                    self.assertEqual(verified["final_iteration"], 100)

    def test_native_counter_with_matching_final_forces_works_without_compute_counter(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            cfg = config(work)
            raw = work / RAW_NAME
            native_file(raw, rows([100.] * 125, start=1, step=1))
            metadata = finish_history(final_solver(125, expose_counter=False), cfg, context(cfg, raw))
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["final_iteration"], 125)
            self.assertEqual(metadata["force_stability"]["status"], "passed")
            self.assertIn("matched", metadata["final_state_verification_method"])
            metadata = finish_history(final_solver(125, drag=120., expose_counter=False), cfg, context(cfg, raw))
            self.assertEqual(metadata["status"], "partial")
            self.assertEqual(metadata["force_stability"]["status"], "not_verified")
            metadata = finish_history(final_solver(125, drag=120.), cfg, context(cfg, raw))
            self.assertEqual(metadata["status"], "partial")
            self.assertIn("differs", metadata["force_stability"]["reason"])
            self.assertTrue(raw.is_file())

    def test_two_user_samples_verify_the_state_but_cannot_pass_the_stability_window(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            cfg = config(work)
            raw = work / RAW_NAME
            raw.write_text('"nose-study-force-history"\n"Iteration" "nose-study-drag etc.."\n'
                '("Iteration" "nose-study-drag" "nose-study-lift")\n'
                '2649 330.3499266915852 -327.0710832079477\n'
                '2650 330.348354062727 -327.079415710345\n')
            solver = final_solver(0, drag=330.348354062727, lift=-327.079415710345)
            metadata = finish_history(solver, cfg, context(cfg, raw))
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["final_iteration"], 2650)
            self.assertEqual(metadata["native_sample_count"], 2)
            self.assertEqual(metadata["force_stability"]["status"], "insufficient_history")

    def test_saved_history_checks_current_source_hashes_conditions_and_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            cfg = config(work)
            raw = work / RAW_NAME
            native_file(raw, rows([100.] * 125, start=1, step=1))
            metadata = finish_history(final_solver(125), cfg, context(cfg, raw))
            for name in ("04_final.cas.h5", "04_final.dat.h5"):
                (work / name).write_bytes(name.encode())
            metadata["source_sha256"] = {"final_case": file_sha256(work / "04_final.cas.h5"),
                                          "final_data": file_sha256(work / "04_final.dat.h5")}
            atomic_json(work / "summary.json", dict(force_history=metadata))
            measured, found = read_saved_force_history(cfg)
            self.assertEqual(measured[-1]["iteration"], 125)
            self.assertEqual(found["csv_sha256"], metadata["csv_sha256"])
            self.assertEqual(assess_saved_force_history(cfg)["status"], "passed")
            cfg["aero"]["reference_area_per_unit_span_m"] = .02
            self.assertEqual(assess_saved_force_history(cfg)["status"], "passed")
            for key, value in (("body_wall_zones", []), ("body_wall_zones", ["farfield"]),
                               ("native_frequency_iterations", 25), ("report_depth_m", 2.)):
                changed = dict(metadata, **{key: value})
                atomic_json(work / "summary.json", dict(force_history=changed))
                self.assertEqual(assess_saved_force_history(cfg)["status"], "not_verified")
            atomic_json(work / "summary.json", dict(force_history=metadata, body_wall_zones=["body_wall.1"]))
            self.assertEqual(assess_saved_force_history(cfg)["status"], "not_verified")
            atomic_json(work / "summary.json", dict(force_history=metadata))
            cfg["aero"]["force_history"]["sample_every_iterations"] = 10
            self.assertEqual(assess_saved_force_history(cfg)["status"], "not_verified")
            cfg["aero"]["force_history"].pop("sample_every_iterations")
            cfg["flight"]["mach"] = 7
            self.assertEqual(assess_saved_force_history(cfg)["status"], "not_verified")
            cfg["flight"]["mach"] = 8
            (work / "04_final.dat.h5").write_bytes(b"replaced native data")
            self.assertEqual(assess_saved_force_history(cfg)["status"], "not_verified")
            metadata["source_sha256"]["final_data"] = file_sha256(work / "04_final.dat.h5")
            atomic_json(work / "summary.json", dict(force_history=metadata))
            raw.write_text(raw.read_text() + "# changed\n")
            self.assertEqual(assess_saved_force_history(cfg)["status"], "not_verified")

    def test_legacy_missing_history_and_missing_workcontext_stay_unverified(self):
        self.assertEqual(assess_saved_force_history({})["status"], "not_verified")
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(Path(directory))
            atomic_json(Path(directory) / "summary.json", dict(solution_status="iterations_completed"))
            self.assertEqual(assess_saved_force_history(cfg)["status"], "not_verified")


if __name__ == "__main__":
    unittest.main()
