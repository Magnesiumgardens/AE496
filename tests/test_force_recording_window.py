"""A converged checkpoint must still record enough force samples on continuation."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pipeline.iteration_history import iterate_with_histories
from test_force_history import RAW_NAME, config, context, final_solver, native_file, rows


class ResidualEquation:
    def __init__(self):
        self.value = True
        self.monitor = True
        self.absolute_criteria = .001

    @property
    def check_convergence(self):
        return SimpleNamespace(get_state=lambda: self.value)

    @check_convergence.setter
    def check_convergence(self, value):
        self.value = value


class ForceRecordingWindowTests(unittest.TestCase):
    def test_already_converged_case_records_requested_window_and_restores_stop_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            cfg = config(work)
            cfg["solver"] = {"max_iterations": 200, "temperature_history": {"enabled": False}}
            cfg["aero"]["force_history"]["require_stable"] = True
            original = copy.deepcopy(cfg)
            equations = {name: ResidualEquation() for name in ("continuity", "x-velocity", "y-velocity", "energy")}
            class Equations(dict):
                def get_object_names(self):return list(self)
            equations = Equations(equations)
            raw = work / RAW_NAME
            solver = final_solver(0)
            counts = []
            def iterate(iter_count):
                actual_count = 2 if any(eq.value for eq in equations.values()) else iter_count
                counts.append(actual_count)
                native_file(raw, rows([100.] * actual_count, start=2649, step=1))
            solution = solver.settings.solution
            solution.monitor = SimpleNamespace(residual=SimpleNamespace(equations=equations))
            solution.run_calculation = SimpleNamespace(iterate=iterate)
            with patch("pipeline.force_history.configure_history", return_value=context(cfg, raw)):
                result = iterate_with_histories(solver, cfg, ["body_wall"], force_recording_window=True)
            self.assertEqual(counts, [200])
            self.assertEqual(result["force_history"]["force_stability"]["status"], "passed")
            self.assertEqual(result["force_history"]["native_sample_count"], 200)
            self.assertTrue(all(eq.value for eq in equations.values()))
            self.assertTrue(all(eq.monitor and eq.absolute_criteria == .001 for eq in equations.values()))
            self.assertEqual(cfg, original)
            self.assertTrue(result["force_history"]["recording_window"]["residual_thresholds_unchanged"])

    def test_failed_native_iteration_restores_flags_without_claiming_results(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(Path(directory))
            cfg["solver"] = {"max_iterations": 200, "temperature_history": {"enabled": False}}
            cfg["aero"]["force_history"]["require_stable"] = True
            class Equations(dict):
                def get_object_names(self):return list(self)
            equations = Equations(continuity=ResidualEquation(), energy=ResidualEquation())
            solver = final_solver(0)
            solver.settings.solution.monitor = SimpleNamespace(residual=SimpleNamespace(equations=equations))
            def iterate(**kwargs):
                self.assertFalse(any(eq.value for eq in equations.values()))
                raise RuntimeError("native iteration failed")
            solver.settings.solution.run_calculation = SimpleNamespace(iterate=iterate)
            with patch("pipeline.force_history.configure_history", return_value=context(cfg, Path(directory) / RAW_NAME)):
                with self.assertRaisesRegex(RuntimeError, "native iteration failed"):
                    iterate_with_histories(solver, cfg, ["body_wall"], force_recording_window=True)
            self.assertTrue(all(eq.value for eq in equations.values()))
