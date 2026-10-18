"""Both iteration monitors share Fluent's original steady solve call."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pipeline.iteration_history import iterate_with_histories


class CoupledMonitorTests(unittest.TestCase):
    def run_monitors(self, *, temperature_error=None, solve_error=None):
        calls = []
        cfg = {"solver": {"max_iterations": 4800}}
        def iterate(iter_count):
            calls.append(("iterate", iter_count))
            if solve_error:raise solve_error
        solver = SimpleNamespace(settings=SimpleNamespace(solution=SimpleNamespace(
            run_calculation=SimpleNamespace(iterate=iterate))))
        def configure_force(*args):
            calls.append(("configure_force",));return {"status": "configured"}
        def configure_temperature(*args):
            calls.append(("configure_temperature",))
            if temperature_error:raise temperature_error
            return SimpleNamespace(name="temperature.out"), object()
        def finish_force(*args):
            calls.append(("finish_force",))
            return {"status": "complete", "force_stability": {"status": "passed"}}
        with patch("pipeline.force_history.configure_history", side_effect=configure_force), \
             patch("pipeline.iteration_history.configure_history", side_effect=configure_temperature), \
             patch("pipeline.iteration_history._finish_temperature_history", return_value={"status": "complete"}), \
             patch("pipeline.force_history.finish_history", side_effect=finish_force):
            result = iterate_with_histories(solver, cfg, ["body_wall"])
        return result, calls

    def test_monitors_are_configured_before_one_native_iterate_call(self):
        result, calls = self.run_monitors()
        self.assertEqual(calls, [("configure_force",), ("configure_temperature",),
                                 ("iterate", 4800), ("finish_force",)])
        self.assertEqual(result["temperature_history"]["status"], "complete")
        self.assertEqual(result["force_history"]["force_stability"]["status"], "passed")

    def test_temperature_adapter_failure_does_not_cancel_force_recording(self):
        result, calls = self.run_monitors(temperature_error=AttributeError("unsupported temperature report"))
        self.assertEqual(result["temperature_history"]["status"], "unavailable")
        self.assertEqual(result["force_history"]["status"], "complete")
        self.assertEqual([call for call in calls if call[0] == "iterate"], [("iterate", 4800)])

    def test_a_native_solve_failure_is_not_hidden_by_monitor_export(self):
        with self.assertRaisesRegex(RuntimeError, "native iterate failed"):
            self.run_monitors(solve_error=RuntimeError("native iterate failed"))
