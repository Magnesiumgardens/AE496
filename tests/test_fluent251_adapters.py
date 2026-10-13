"""Regressions for the actual values and failures observed on Windows Fluent251."""
import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from pipeline.aero import aerodynamic_summary, report_scalar, setting_scalar, validate_saved_conditions
from pipeline.cfd import export_saved_case
from pipeline.solve import _temperature_summary
from test_nose_trade import config


class Report251:
    def __init__(self, kind):
        object.__setattr__(self, "report_output_type", SimpleNamespace(
            allowed_values=lambda: [kind.title() + " Coefficient", kind.title() + " Force"]))
        object.__setattr__(self, "average_over", SimpleNamespace(is_active=lambda: False))

    def __setattr__(self, name, value):
        if name == "report_type":
            raise RuntimeError("report-type is inactive on Fluent251 typed force reports")
        if name == "average_over":
            raise RuntimeError("averaging inactive")
        object.__setattr__(self, name, value)


class ExistingReports(dict):
    def get_object_names(self):return list(self)
    def delete(self, **kwargs):raise RuntimeError("report deletion must not be needed for export")
    def create(self, **kwargs):raise RuntimeError("existing saved report must be reused")


class Fluent251Adapters(unittest.TestCase):
    def test_saved_reports_reused_and_named_force_enum_and_value_index_decoded(self):
        drag, lift = Report251("drag"), Report251("lift")
        calls = []
        def compute(report_defs):
            calls.append(report_defs)
            values = {"nose-study-drag": 309.64991, "nose-study-lift": -322.61433}
            # Multiple requested reports arrive in separate dictionaries.
            return [{name: [values[name], 0]} for name in report_defs]
        definitions = SimpleNamespace(
            drag=ExistingReports({"nose-study-drag": drag}),
            lift=ExistingReports({"nose-study-lift": lift}),
            compute=compute)
        solver = SimpleNamespace(settings=SimpleNamespace(
            setup=SimpleNamespace(reference_values=SimpleNamespace(),
                general=SimpleNamespace(solver=SimpleNamespace(
                    two_dim_space=SimpleNamespace(get_state=lambda: "planar")))),
            solution=SimpleNamespace(report_definitions=definitions)))
        result = aerodynamic_summary(solver, config(), ["body_wall"])
        self.assertEqual(result["drag_per_unit_span_N_m"], 309.64991)
        self.assertEqual(result["lift_per_unit_span_N_m"], -322.61433)
        self.assertEqual(calls, [["nose-study-drag"], ["nose-study-lift"]])
        self.assertEqual(drag.report_output_type, "Drag Force")
        self.assertEqual(lift.report_output_type, "Lift Force")
        self.assertIs(definitions.drag["nose-study-drag"], drag)

    def test_temperature_reports_are_also_queried_individually(self):
        values = {"auto-wall-temp-min": 290, "auto-wall-temp-avg": 300, "auto-wall-temp-max": 310}
        calls = []
        def compute(report_defs):
            calls.append(report_defs)
            return [{name: [values[name], 0]} for name in report_defs]
        solver = SimpleNamespace(settings=SimpleNamespace(solution=SimpleNamespace(
            report_definitions=SimpleNamespace(surface=ExistingReports(), compute=compute))))
        self.assertEqual(_temperature_summary(solver, ["body_wall"]), {
            "wall_temperature_min_k": 290, "wall_temperature_avg_k": 300, "wall_temperature_max_k": 310})
        self.assertEqual(calls, [[name] for name in values])

    def test_report_value_not_iteration_and_invalid_layouts_fail(self):
        self.assertEqual(report_scalar({"temp": [300.0, 0]}, "temp"), 300)
        self.assertEqual(report_scalar({"drag": [-12.0, 1161]}, "drag"), -12)
        for value in ([8, 2.5], [8, -1], [8, True], [8, 0, 1], [float("nan"), 0]):
            with self.assertRaises(ValueError):report_scalar({"drag": value}, "drag")
        with self.assertRaises(ValueError):report_scalar([{}, {"drag": [1161]}], "drag")

    def test_numeric_value_groups_and_profile_rejection(self):
        self.assertEqual(setting_scalar({"option": "value", "value": 8.0}), 8)
        self.assertEqual(setting_scalar({"option": "constant", "value": -1}), -1)
        for value in ({"option": "udf", "value": 8}, {"option": "profile", "value": 8},
                      {"option": "value"}, {"option": "value", "value": float("inf")}, True):
            with self.assertRaises(ValueError):setting_scalar(value)

    def test_full_boundary_validation_accepts_observed_dicts_and_still_detects_mismatch(self):
        def scalar(value):return SimpleNamespace(get_state=lambda: {"option": "value", "value": value})
        def plain(value):return SimpleNamespace(get_state=lambda: value)
        cfg = config(direction=(-1, 0))
        cfg["physics"] = {"viscous_model": "laminar", "air": {}}
        cfg["thermal"] = {"mode": "adiabatic"}
        far = SimpleNamespace(momentum=SimpleNamespace(
            mach_number=scalar(8), gauge_pressure=scalar(1090),
            flow_direction=plain([{"option": "value", "value": -1}, {"option": "value", "value": 0}])),
            thermal=SimpleNamespace(temperature=scalar(226.5)))
        farfields = ExistingReports({"farfield": far})
        walls = ExistingReports({"body_wall": SimpleNamespace(thermal=SimpleNamespace(
            thermal_condition=plain("Heat Flux"), heat_flux=scalar(0)))})
        solver = SimpleNamespace(settings=SimpleNamespace(setup=SimpleNamespace(
            general=SimpleNamespace(operating_conditions=SimpleNamespace(operating_pressure=plain(0))),
            boundary_conditions=SimpleNamespace(pressure_far_field=farfields, wall=walls),
            models=SimpleNamespace(viscous=SimpleNamespace(model=plain("laminar")),
                energy=SimpleNamespace(enabled=plain(True)), radiation=SimpleNamespace(model=plain("none"))),
            materials=SimpleNamespace(fluid={"air": SimpleNamespace(
                density=SimpleNamespace(option=plain("ideal-gas")),
                viscosity=SimpleNamespace(option=plain("sutherland")))}))))
        validate_saved_conditions(solver, cfg)
        cfg["flight"]["mach"] = 7
        with self.assertRaisesRegex(ValueError, "rerun CFD"):validate_saved_conditions(solver, cfg)

    def test_real_pinned_client_boundary_schema_is_grouped(self):
        from ansys.fluent.core.generated.solver import settings_251
        from ansys.fluent.core.solver.flobject import Group
        for cls in (settings_251.mach_number, settings_251.gauge_pressure, settings_251.temperature_1):
            self.assertTrue(issubclass(cls, Group))
            self.assertIn("option", cls.child_names)
            self.assertIn("value", cls.child_names)
        self.assertEqual(settings_251.delete_1.argument_names, ["name_list"])

    def test_saved_export_retains_case_data_and_completed_heating_on_force_failure(self):
        for fails in (False, True):
            with self.subTest(force_failure=fails), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cfg = config()
                cfg["project"] = {"name": "blunt_1", "work_dir": directory}
                cfg["thermal"] = {"mode": "adiabatic"}
                for name in ("04_final.cas.h5", "04_final.dat.h5"):
                    (root / name).write_bytes(b"existing simulation")
                events = []
                # There are deliberately no initialization, iteration or write APIs.
                solver = SimpleNamespace(settings=SimpleNamespace(
                    file=SimpleNamespace(read_case_data=lambda **args: events.append(("read", args))),
                    setup=SimpleNamespace(boundary_conditions=SimpleNamespace(
                        wall=ExistingReports({"body_wall": None})))),
                    exit=lambda: events.append(("exit", {})))
                def heating(session, settings, walls):
                    self.assertEqual(walls, ["body_wall"])
                    path = root / "temperature/wall_temperatures.csv"
                    path.parent.mkdir()
                    path.write_text("saved temperatures\n")
                    return {"status": "complete", "wall_temperatures_csv": str(path)}
                def forces(*args):
                    self.assertTrue((root / "temperature/wall_temperatures.csv").is_file())
                    if fails:raise RuntimeError("force export failed")
                    return {"status": "complete", "drag_per_unit_span_N_m": 12.0}
                with patch("pipeline.cfd.launch_solver", return_value=solver), \
                     patch("pipeline.cfd.validate_saved_conditions"), \
                     patch("pipeline.cfd.export_temperatures", side_effect=heating), \
                     patch("pipeline.cfd.aerodynamic_summary", side_effect=forces):
                    if fails:
                        with self.assertRaisesRegex(RuntimeError, "force export failed"):
                            export_saved_case(cfg)
                    else:
                        self.assertEqual(export_saved_case(cfg)["aerodynamics"]["drag_per_unit_span_N_m"], 12)
                for name in ("04_final.cas.h5", "04_final.dat.h5"):
                    self.assertEqual((root / name).read_bytes(), b"existing simulation")
                self.assertEqual([kind for kind, _ in events], ["read", "exit"])
                self.assertEqual((root / "temperature/wall_temperatures.csv").read_text(), "saved temperatures\n")
                self.assertEqual(json.loads((root / "aero_summary.json").read_text())["status"],
                                 "failed" if fails else "complete")


if __name__ == "__main__":unittest.main()
