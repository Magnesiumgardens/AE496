import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from pipeline.aero import aerodynamic_summary, reference_state, report_scalar, validate_saved_conditions
from pipeline.trade_study import (
    aero_complete, create_trade_outputs, pareto_cases, thermal_screening_metrics,
)
import run_batch
import run_thermal_analysis
import compare_nose_trade
import yaml


def config(direction=(1, 0), width=None):
    return {
        "flight": {"flow_direction": list(direction), "mach": 8,
                   "static_pressure_pa": 1090, "static_temperature_k": 226.5},
        "geometry": {"body_wall_name": "body_wall"},
        "aero": {"reference_area_per_unit_span_m": width},
    }


class Report:
    report_output_type = SimpleNamespace(allowed_values=lambda: ["Coefficient", "Force"])


class Reports(dict):
    def get_object_names(self):
        return list(self)

    def create(self, name):
        self[name] = Report()

    def delete(self, names):
        for name in names:
            del self[name]


def fluent():
    definitions = SimpleNamespace(drag=Reports(), lift=Reports())

    def compute(report_defs):
        # Project a known (12, 3) N/m force vector onto each requested direction.
        values = {}
        for kind in ("drag", "lift"):
            for name, report in getattr(definitions, kind).items():
                if name not in report_defs:
                    continue
                assert report.report_output_type == "Force"
                assert report.zones == ["body_wall.1", "body_wall.2"]
                vector = report.force_vector
                values[name] = [12 * vector[0] + 3 * vector[1]]
        return values, {name: [1200] for name in report_defs}

    definitions.compute = compute
    solver = SimpleNamespace(settings=SimpleNamespace(
        setup=SimpleNamespace(reference_values=SimpleNamespace(),
            general=SimpleNamespace(solver=SimpleNamespace(
                two_dim_space=SimpleNamespace(get_state=lambda: "planar")))),
        solution=SimpleNamespace(report_definitions=definitions),
    ))
    return solver


class AeroTests(unittest.TestCase):
    def test_saved_case_conditions_must_match_requested_case(self):
        setting = lambda value: SimpleNamespace(get_state=lambda: value)
        cfg = config()
        cfg["physics"] = {"viscous_model": "laminar", "air": {}}
        cfg["thermal"] = {"mode": "adiabatic"}
        far = SimpleNamespace(momentum=SimpleNamespace(
            mach_number=setting(8), gauge_pressure=setting(1090), flow_direction=setting([1, 0])),
            thermal=SimpleNamespace(temperature=setting(226.5)))
        walls = Reports({"body_wall.1": SimpleNamespace(thermal=SimpleNamespace(
            thermal_condition=setting("Heat Flux"), heat_flux=setting(0)))})
        setup = SimpleNamespace(
            general=SimpleNamespace(operating_conditions=SimpleNamespace(operating_pressure=setting(0))),
            models=SimpleNamespace(viscous=SimpleNamespace(model=setting("laminar")),
                energy=SimpleNamespace(enabled=setting(True)), radiation=SimpleNamespace(model=setting("none"))),
            materials=SimpleNamespace(fluid={"air": SimpleNamespace(
                density=SimpleNamespace(option=setting("ideal-gas")),
                viscosity=SimpleNamespace(option=setting("sutherland")))}),
            boundary_conditions=SimpleNamespace(pressure_far_field=Reports({"farfield": far}), wall=walls))
        solver = SimpleNamespace(settings=SimpleNamespace(setup=setup))
        validate_saved_conditions(solver, cfg)
        cfg["flight"]["mach"] = 7
        with self.assertRaisesRegex(ValueError, "rerun CFD"):
            validate_saved_conditions(solver, cfg)
        cfg["flight"]["mach"] = 8
        walls["body_wall.1"].thermal.heat_flux = setting(10)
        with self.assertRaisesRegex(ValueError, "wall heat flux"):
            validate_saved_conditions(solver, cfg)
        walls["body_wall.1"].thermal.heat_flux = setting(0)
        cfg["physics"]["air"]["sutherland_reference_viscosity"] = 1.716e-5
        setup.materials.fluid["air"].viscosity.sutherland = SimpleNamespace(
            reference_viscosity=setting(2e-5))
        with self.assertRaisesRegex(ValueError, "Sutherland"):
            validate_saved_conditions(solver, cfg)

    def test_dimensional_forces_without_invented_reference(self):
        solver = fluent()
        result = aerodynamic_summary(solver, config(), ["body_wall.1", "body_wall.2"])
        self.assertEqual(result["drag_per_unit_span_N_m"], 12)
        self.assertEqual(result["lift_per_unit_span_N_m"], 3)
        self.assertEqual(result["L_over_D"], 0.25)
        self.assertIsNone(result["Cd"])
        self.assertEqual(solver.settings.setup.reference_values.depth, 1)

    def test_rotated_flow_and_coefficients(self):
        result = aerodynamic_summary(fluent(), config((0, 2), 0.1), ["body_wall.1", "body_wall.2"])
        self.assertEqual(result["drag_per_unit_span_N_m"], 3)
        self.assertEqual(result["lift_per_unit_span_N_m"], -12)
        self.assertAlmostEqual(result["Cd"], 3 / (0.5 * 1.4 * 1090 * 64 * 0.1))

    def test_zero_drag_ratio_is_undefined(self):
        solver = fluent()
        solver.settings.solution.report_definitions.compute = lambda report_defs: {
            "nose-study-drag": [0], "nose-study-lift": [3]}
        result = aerodynamic_summary(solver, config(), ["body_wall.1", "body_wall.2"])
        self.assertIsNone(result["L_over_D"])

    def test_zone_scope_and_axisymmetry_rejected(self):
        with self.assertRaises(ValueError):
            aerodynamic_summary(fluent(), config(), ["farfield"])
        cfg = config()
        cfg["aero"]["model_kind"] = "axisymmetric"
        with self.assertRaises(ValueError):
            reference_state(cfg)
        solver = fluent()
        solver.settings.setup.general.solver.two_dim_space.get_state = lambda: "axisymmetric"
        with self.assertRaises(ValueError):
            aerodynamic_summary(solver, config(), ["body_wall.1", "body_wall.2"])

    def test_invalid_reference_inputs(self):
        for direction in ((0, 0), (float("nan"), 1), (1, float("inf"))):
            with self.assertRaises(ValueError):
                reference_state(config(direction))
        for width in (-1, 0, float("inf")):
            with self.assertRaises(ValueError):
                reference_state(config(width=width))

    def test_no_iteration_number_misread_as_force(self):
        self.assertEqual(report_scalar(({"drag": [7]}, {"drag": [1200]}), "drag"), 7)
        with self.assertRaises(ValueError):
            report_scalar(({}, {"drag": [1200]}), "drag")
        for raw in ({"drag": [7, 8, 9]}, {"drag": [float("nan")]}, {"drag": True}):
            with self.assertRaises(ValueError):
                report_scalar(raw, "drag")

    def test_resume_detects_changed_reference_and_results(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ("04_final.cas.h5", "04_final.dat.h5"):
                (root / name).write_text("fixture")
            result = aerodynamic_summary(fluent(), config(), ["body_wall.1", "body_wall.2"])
            (root / "aero_summary.json").write_text(json.dumps(result))
            self.assertTrue(aero_complete(root, config()))
            self.assertFalse(aero_complete(root, config(width=0.1)))
            stamp = (root / "aero_summary.json").stat().st_mtime_ns + 1000000000
            os.utime(root / "04_final.dat.h5", ns=(stamp, stamp))
            self.assertFalse(aero_complete(root, config()))


class ThermalTests(unittest.TestCase):
    xy = [(0, 0), (1, 0), (1, 1), (0, 1)]
    settings = {"duration": 10, "rho": 2, "initial_t": 300}
    history = [{"time_s": 5, "maximum_K": 700}, {"time_s": 10, "maximum_K": 300}]

    def test_polygon_load_mass_peak_and_threshold_crossings(self):
        result = thermal_screening_metrics(self.xy, [10] * 4, self.history, self.settings, 500)
        self.assertEqual(result["cold_wall_heating_power_per_unit_span_W_m"], 40)
        self.assertEqual(result["applied_heat_load_per_unit_span_J_m"], 400)
        self.assertEqual(result["solid_mass_per_unit_span_kg_m"], 2)
        self.assertEqual(result["peak_temperature_K"], 700)
        self.assertEqual(result["time_above_temperature_limit_s"], 5)
        self.assertEqual(result["temperature_limit_margin_K"], -200)
        self.assertEqual(result["temperature_limit_status"], "exceeded")

    def test_limits_are_not_invented(self):
        result = thermal_screening_metrics(self.xy, [10] * 4, self.history, self.settings)
        self.assertIsNone(result["temperature_limit_margin_K"])
        self.assertEqual(result["temperature_limit_status"], "not_configured")

    def test_missing_or_unordered_history_rejected(self):
        for history in ([], self.history[:1], list(reversed(self.history))):
            with self.assertRaises(ValueError):
                thermal_screening_metrics(self.xy, [10] * 4, history, self.settings)

    def test_endpoint_mode_cannot_silently_run_fixed_heat_flux(self):
        cfg = {"loading": {"mode": "convective_from_cfd"}}
        with self.assertRaises(run_thermal_analysis.ThermalAnalysisError):
            run_thermal_analysis.validate_settings({}, cfg)
        with self.assertRaises(run_batch.BatchError):
            run_batch.validate_thermal_inputs(cfg)


class BatchTests(unittest.TestCase):
    def test_read_only_comparison_uses_peak_history_without_rerunning_thermal(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            case = root / "baseline"
            (case / "_batch").mkdir(parents=True)
            (case / "thermal_analysis").mkdir()
            cfg = config()
            cfg["geometry"]["input_file"] = "baseline.x_t"
            (case / "_batch" / "config.yaml").write_text(yaml.safe_dump(cfg))
            for name in ("04_final.cas.h5", "04_final.dat.h5"):
                (case / name).write_text("fixture")
            result = aerodynamic_summary(fluent(), cfg, ["body_wall.1", "body_wall.2"])
            (case / "aero_summary.json").write_text(json.dumps(result))
            thermal = case / "thermal_analysis"
            original = json.dumps({"status": "complete", "final_temperature": {"maximum_K": 400},
                "analysis": {"initial_temperature_k": 300}})
            (thermal / "thermal_summary.json").write_text(original)
            (thermal / "temperature_history.csv").write_text("time_s,maximum_K\n5,700\n10,400\n")
            rows, created = compare_nose_trade.compare(root, root / "comparison", reference_dir=root / "no_extra_references")
            self.assertEqual(rows[0]["peak_temperature_K"], 700)
            self.assertEqual(rows[0]["final_max_temperature_K"], 400)
            self.assertEqual(rows[0]["applied_heat_load_per_unit_span_J_m"], "")
            self.assertTrue(Path(created["nose_trade_screening"]).is_file())
            self.assertEqual((thermal / "thermal_summary.json").read_text(), original)

    def test_results_flow_to_csv_and_json(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = aerodynamic_summary(fluent(), config(), ["body_wall.1", "body_wall.2"])
            (root / "aero_summary.json").write_text(json.dumps(result))
            thermal = root / "thermal_analysis"
            thermal.mkdir()
            metrics = thermal_screening_metrics(ThermalTests.xy, [10] * 4,
                ThermalTests.history, ThermalTests.settings, 500)
            (thermal / "thermal_summary.json").write_text(json.dumps({
                "screening_metrics": metrics, "loading_mode": "fixed_heat_flux"}))
            row = run_batch.build_result_row(Path("test.x_t"), "test", root, "complete", 1)
            csv_path, json_path = run_batch.save_batch_summary(root / "comparison", [row])
            with csv_path.open() as f:
                saved = next(csv.DictReader(f))
            self.assertEqual(saved["drag_per_unit_span_N_m"], "12.0")
            self.assertEqual(saved["peak_temperature_K"], "700.0")
            self.assertEqual(json.loads(json_path.read_text())[0]["temperature_limit_status"], "exceeded")

    def test_retry_only_after_boundary_layer_stage(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "batch_logs").mkdir()
            log = root / "batch_logs" / "01_cfd.log"
            for text in ("[mesh] Launching\nLaunchFluentError", "[mesh] Curvature\nAttributeError",
                         "[mesh] boundary layer\nlicense unavailable", "[mesh] importing CAD"):
                log.write_text(text)
                self.assertFalse(run_batch._failed_during_meshing(root))
            log.write_text("[mesh] Adding boundary layers\nStage inactivity timeout")
            self.assertTrue(run_batch._failed_during_meshing(root))

    def test_pareto_ties_failures_and_limits(self):
        def row(name, drag, temp, **kw):
            return {"case": name, "drag_per_unit_span_N_m": drag, "peak_temperature_K": temp,
                    "status": "complete", "aero_status": "complete", **kw}
        rows = [row("A", 10, 500), row("B", 12, 400), row("C", 12, 500),
                row("tie", 10, 500), row("failed", 1, 1, status="failed"),
                row("hot", 1, 1000, temperature_limit_status="exceeded"),
                row("negative", -1, 200)]
        self.assertEqual(pareto_cases(rows), ["A", "B", "tie"])

    def test_trade_plots_and_incomparable_conditions(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = []
            for name, drag, temp in (("A", 10, 500), ("B", 12, 400)):
                (root / name).mkdir()
                aero = aerodynamic_summary(fluent(), config(), ["body_wall.1", "body_wall.2"])
                (root / name / "aero_summary.json").write_text(json.dumps(aero))
                rows.append({"case": name, "status": "complete", "aero_status": "complete",
                    "drag_per_unit_span_N_m": drag, "Cd": None, "peak_temperature_K": temp,
                    "max_heat_flux_W_m2": 100, "boundary_layer_used": True,
                    "thermal_loading_mode": "fixed_heat_flux"})
            created = create_trade_outputs(root, rows, root / "out")
            self.assertTrue(Path(created["peak_temperature_K_vs_drag_per_unit_span_N_m"]).is_file())
            self.assertNotIn("Cd_by_geometry", created)
            report = json.loads(Path(created["nose_trade_screening"]).read_text())
            self.assertEqual(report["pareto_cases"], ["A", "B"])
            rows[1]["boundary_layer_used"] = False
            created = create_trade_outputs(root, rows, root / "different")
            self.assertEqual(len(created), 1)
            report = json.loads(Path(created["nose_trade_screening"]).read_text())
            self.assertEqual(report["status"], "incomparable_conditions")


if __name__ == "__main__":
    unittest.main()
