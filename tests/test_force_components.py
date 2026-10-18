"""Whole-body signed dimensional force splits; no invented viscous contribution."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from pipeline.aero import (
    aerodynamic_summary, configure_force_reports, parse_force_components,
    report_iteration,
)
from test_nose_trade import config, fluent


ZONES = ["body_wall.1", "body_wall.2"]


def force_table(zones, pressure, viscous, total=None):
    total = pressure + viscous if total is None else total
    lines = [
        "                 Forces [N]                      Coefficients",
        "Zone             Pressure Viscous Total          Pressure Viscous Total",
    ]
    for zone in zones:
        # Coefficients are deliberate decoys. They must not be exported as N/m.
        lines.append(f"{zone} {pressure / len(zones)} {viscous / len(zones)} "
                     f"{total / len(zones)} 111 222 333")
    lines.append(f"Net {pressure} {viscous} {total} 444 555 999")
    return "\n".join(lines)


class NativeForceCommand:
    option = SimpleNamespace(allowed_values=lambda: ["forces", "moments"])

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["option"] == "forces"
        assert kwargs["wall_zones"] == ZONES
        assert kwargs["write_to_file"] is True
        assert kwargs["append_data"] is False
        vector = kwargs["direction_vector"]
        pressure = 10 * vector[0] + 2 * vector[1]
        viscous = 2 * vector[0] + vector[1]
        Path(kwargs["file_name"]).write_text(force_table(ZONES, pressure, viscous))


def split_solver():
    solver = fluent()
    native = NativeForceCommand()
    solver.settings.results = SimpleNamespace(report=SimpleNamespace(forces=native))
    return solver, native


class ForceComponentTests(unittest.TestCase):
    def test_signed_whole_body_components_project_freestream_and_sum_to_total(self):
        for direction, expected in (((1, 0), (10, 2, 12)),
                                    ((0, 2), (2, 1, 3)),
                                    ((-1, 0), (-10, -2, -12))):
            with self.subTest(direction=direction), tempfile.TemporaryDirectory() as directory:
                cfg = config(direction=direction, width=0.1)
                cfg["project"] = {"work_dir": directory}
                solver, native = split_solver()
                result = aerodynamic_summary(solver, cfg, ZONES)
                self.assertEqual(result["component_status"], "complete")
                self.assertEqual(result["pressure_drag_per_unit_span_N_m"], expected[0])
                self.assertEqual(result["viscous_drag_per_unit_span_N_m"], expected[1])
                self.assertEqual(result["drag_per_unit_span_N_m"], expected[2])
                self.assertEqual(result["total_drag_per_unit_span_N_m"], expected[2])
                self.assertEqual(result["drag_component_balance"]["status"], "passed")
                self.assertAlmostEqual(result["Cd"], result["Cd_pressure"] + result["Cd_viscous"])
                self.assertEqual(native.calls[0]["direction_vector"], result["drag_direction"])
                self.assertEqual(result["body_wall_zones"], ZONES)
                self.assertTrue(Path(result["component_report_file"]).is_file())

    def test_no_reference_area_inference_and_no_fabricated_unavailable_split(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config()
            cfg["project"] = {"work_dir": directory}
            result = aerodynamic_summary(fluent(), cfg, ZONES)
            self.assertEqual(result["drag_per_unit_span_N_m"], 12)
            self.assertEqual(result["component_status"], "unavailable")
            self.assertTrue(result["component_error"])
            for key in ("Cd", "Cd_pressure", "Cd_viscous",
                        "pressure_drag_per_unit_span_N_m", "viscous_drag_per_unit_span_N_m"):
                self.assertIsNone(result[key])

    def test_independent_total_disagreement_cannot_look_like_a_valid_split(self):
        with tempfile.TemporaryDirectory() as directory:
            solver, native = split_solver()
            def wrong_total(**kwargs):
                Path(kwargs["file_name"]).write_text(force_table(ZONES, 20, 4))
            solver.settings.results.report.forces = wrong_total
            cfg = config()
            cfg["project"] = {"work_dir": directory}
            result = aerodynamic_summary(solver, cfg, ZONES)
            self.assertEqual(result["component_status"], "unavailable")
            self.assertIn("differs", result["component_error"])
            self.assertEqual(result["drag_per_unit_span_N_m"], 12)
            self.assertIsNone(result["pressure_drag_per_unit_span_N_m"])

    def test_a_failed_native_write_cannot_reuse_an_old_force_table(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aero_forces_drag.txt"
            path.write_text(force_table(ZONES, 10, 2))
            solver, native = split_solver()
            solver.settings.results.report.forces = lambda **kwargs: None
            cfg = config()
            cfg["project"] = {"work_dir": directory}
            result = aerodynamic_summary(solver, cfg, ZONES)
            self.assertEqual(result["component_status"], "unavailable")
            self.assertFalse(path.exists())

    def test_shared_monitor_definitions_use_identical_dimensional_reference(self):
        solver = fluent()
        cfg = config(direction=(-3, -4), width=0.2)
        cfg["aero"]["reference_length_m"] = 0.3
        names, state = configure_force_reports(solver, cfg, ZONES)
        refs = solver.settings.setup.reference_values
        self.assertEqual(names, {"drag": "nose-study-drag", "lift": "nose-study-lift"})
        self.assertEqual(state["drag_direction"], [-0.6, -0.8, 0])
        self.assertEqual(refs.depth, 1)
        self.assertEqual(refs.area, 0.2)
        self.assertEqual(refs.length, 0.3)
        self.assertEqual(refs.density, state["density_kg_m3"])
        self.assertEqual(refs.velocity, state["velocity_m_s"])

    def test_table_ignores_coefficients_without_double_counting_net(self):
        self.assertEqual(parse_force_components(force_table(ZONES, 10, 2), ZONES), (10, 2, 12))
        without_net = "\n".join(force_table(ZONES, -10, -2).splitlines()[:-1])
        self.assertEqual(parse_force_components(without_net, ZONES), (-10, -2, -12))
        # A true zero viscous contribution is accepted only when present in the table.
        self.assertEqual(parse_force_components(force_table(ZONES, 10, 0), ZONES), (10, 0, 10))

    def test_missing_units_zones_bad_balance_and_unknown_tables_rejected(self):
        good = force_table(ZONES, 10, 2)
        bad = [good.replace("Forces [N]", "Coefficients"),
               good.replace("body_wall.2", "farfield"),
               good.replace("Pressure Viscous Total", "total pressure viscous"),
               good.replace("Net 10 2 12", "Net 10 2 13"),
               force_table(ZONES, 10, 2, total=13),
               good + "\nNet 10 2 12 0 0 0"]
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_force_components(text, ZONES)

    def test_supported_251_command_schema_has_no_guessed_component_controls(self):
        from ansys.fluent.core.generated.solver import settings_251
        self.assertIn("forces", settings_251.report_2.command_names)
        self.assertTrue({"option", "wall_zones", "direction_vector", "write_to_file",
                         "file_name", "append_data"}.issubset(settings_251.forces_1.argument_names))
        self.assertNotIn("force_type", settings_251.drag_child.child_names)

    def test_report_iteration_uses_only_actual_metadata(self):
        self.assertEqual(report_iteration({"drag": [12.5, 1161]}, "drag"), 1161)
        self.assertEqual(report_iteration([{"drag": [12.5, 1161]}], "drag"), 1161)
        self.assertEqual(report_iteration(({"drag": [12.5]}, {"drag": [1200]}), "drag"), 1200)
        self.assertIsNone(report_iteration({"drag": [1200]}, "drag"))
        self.assertIsNone(report_iteration({"other": [12.5, 100]}, "drag"))
        for iteration in (True, -1, 1.5, float("nan")):
            with self.subTest(iteration=iteration), self.assertRaises(ValueError):
                report_iteration({"drag": [12.5, iteration]}, "drag")


if __name__ == "__main__":
    unittest.main()
