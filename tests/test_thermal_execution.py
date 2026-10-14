"""Exercise Mechanical execution, progress and failure handling without an engine."""
from contextlib import ExitStack, nullcontext, redirect_stdout
import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from pipeline.thermal import run_thermal_analysis
from pipeline.thermal_loading import ThermalAnalysisError
from pipeline.utils import file_sha256

ROOT = Path(__file__).resolve().parents[1]


class MechanicalCase:
    """A saved CFD package sufficient for the actual separate solid runner."""

    def __init__(self, root, launch_timeout=42):
        self.work = root / "runs/nose"
        self.work.mkdir(parents=True)
        cfd = yaml.safe_load((ROOT / "config.yaml").read_text())
        cfd["project"] = {"name": "nose", "work_dir": str(self.work)}
        cfd["geometry"]["input_file"] = str(root / "nose.x_t")
        self.cfd_path = self.work / "_batch/config.yaml"
        self.cfd_path.parent.mkdir()
        self.cfd_path.write_text(yaml.safe_dump(cfd))
        self.thermal = yaml.safe_load((ROOT / "thermal_analysis.yaml").read_text())
        self.thermal["mapdl_start_timeout_s"] = launch_timeout
        self.thermal["transient"].update(duration_s=2., initial_time_step_s=1.,
                                         min_time_step_s=.1, max_time_step_s=2.)
        self.thermal["visualization"]["enabled"] = False
        self.thermal_path = root / "thermal.yaml"
        self.write_thermal()
        self.saved = {}
        for name in ("04_final.cas.h5", "04_final.dat.h5"):
            path = self.work / name
            path.write_bytes(b"Retained radiating-wall CFD " + name.encode())
            self.saved[name] = file_sha256(path)
        temperature = self.work / "temperature"
        temperature.mkdir()
        boundary = temperature / "wall_temperature_nodes.csv"
        with boundary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["surface_order_index", "x_m", "y_m", "wall_temperature_K"])
            writer.writeheader()
            for index, (x, y, temperature_k) in enumerate(((0, 0, 1000), (1, 0, 1200), (0, 1, 1400))):
                writer.writerow(dict(surface_order_index=index, x_m=x, y_m=y, wall_temperature_K=temperature_k))
        metadata = {"source_sha256": {key: self.saved[name] for key, name in
                    (("final_case", "04_final.cas.h5"), ("final_data", "04_final.dat.h5"))},
                    "conditions": {"thermal": cfd["thermal"]},
                    "wall_nodes": {"csv_sha256": file_sha256(boundary)}}
        (boundary.parent / "temperature_summary.json").write_text(json.dumps(metadata))

    def write_thermal(self):
        self.thermal_path.write_text(yaml.safe_dump(self.thermal))

    def run(self, launcher):
        # Source freshness is covered by the provenance tests. Only external
        # client startup is replaced here; solve_thermal and save_results run.
        with ExitStack() as stack:
            stack.enter_context(patch("pipeline.gas_reporting.temperature_complete", return_value=True))
            stack.enter_context(patch("pipeline.solution_quality.require_usable_solution",
                                      return_value={"status": "residual_screen_passed"}))
            launch = stack.enter_context(patch("ansys.mapdl.core.launch_mapdl", side_effect=launcher))
            result = run_thermal_analysis(cfd_config=self.cfd_path, thermal_config=self.thermal_path)
        return result, launch


class ResultReader:
    def __init__(self, engine):
        self.engine = engine

    def opened(self):
        if not self.engine.result_open:
            raise AssertionError("Result file must be opened with SET,LAST before metadata queries")

    @property
    def nsets(self):
        self.opened()
        self.engine.calls.append(("nsets",))
        return 2

    @property
    def time_values(self):
        self.opened()
        self.engine.calls.append(("time_values",))
        return [1., 2.]

    def nodal_temperature(self):
        self.engine.calls.append(("nodal_temperature", self.engine.result_index))
        # Corner and midside temperatures are fixed; only the interior warms.
        return np.array([1000., 1200., 1400., 1100., 1300., 1200.,
                         500. if self.engine.result_index == 1 else 800.])


class NativeMechanical:
    """Small native adapter with constraints on streaming and result opening."""

    def __init__(self, solve_error=False):
        self.calls = []
        self.point = self.line = 0
        self.result_open = False
        self.result_index = None
        self.solve_error = solve_error
        self.mesh = SimpleNamespace(nnum=np.arange(1, 8), enum=np.array([1]),
            nodes=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [.5, 0, 0],
                            [.5, .5, 0], [0, .5, 0], [.2, .2, 0]], dtype=float))
        self.post_processing = ResultReader(self)
        self.non_interactive = nullcontext()

    def __getattr__(self, name):
        def command(*args, **kwargs):
            self.calls.append((name, *args))
        return command

    def k(self, *args):
        self.point += 1
        return self.point

    def l(self, *args):
        self.line += 1
        return self.line

    def al(self, *args):
        return 1

    def amesh(self, area, *, verbose):
        if verbose is not True:
            raise AssertionError("Meshing must stream native output")
        self.calls.append(("amesh", area, verbose))

    def nsll(self, *args):
        self.mesh.nnum = np.arange(1, 7)

    def allsel(self):
        self.mesh.nnum = np.arange(1, 8)

    def solve(self, *, verbose):
        if verbose is not True:
            raise AssertionError("Solve must stream native output")
        self.calls.append(("solve", verbose))
        return "*** ERROR *** thermal solver failed" if self.solve_error else "Solution complete"

    def set(self, *args):
        self.calls.append(("set", *args))
        if args == ("LAST",):
            self.result_open = True
        else:
            self.result_index = args[1]


class ThermalExecutionTests(unittest.TestCase):
    def test_streamed_execution_opens_results_and_retains_saved_cfd(self):
        with tempfile.TemporaryDirectory() as directory:
            case = MechanicalCase(Path(directory))
            engine = NativeMechanical()
            console = io.StringIO()

            def launch(**options):
                # The last flushed progress line identifies startup if the
                # external launch stalls before returning an engine.
                self.assertIn("Launching Mechanical APDL", console.getvalue())
                self.assertIn("connection timeout=42 s", console.getvalue())
                return engine

            with redirect_stdout(console):
                result, launcher = case.run(launch)
            output = case.work / "thermal_analysis"
            options = launcher.call_args.kwargs
            self.assertEqual(options["start_timeout"], 42.)
            self.assertEqual(options["mode"], "grpc")
            self.assertEqual(Path(options["run_location"]), output)
            self.assertEqual(Path(options["log_apdl"]), output / "mapdl_commands.log")
            self.assertEqual(Path(options["mapdl_output"]), output / "mapdl_console.out")
            self.assertIn(("amesh", 1, True), engine.calls)
            self.assertIn(("solve", True), engine.calls)
            self.assertLess(engine.calls.index(("set", "LAST")), engine.calls.index(("nsets",)))
            self.assertEqual([call for call in engine.calls if call[0] == "nodal_temperature"],
                             [("nodal_temperature", 1), ("nodal_temperature", 2)])
            self.assertIn(("time", 2.), engine.calls)
            self.assertIn(("tunif", 300.), engine.calls)
            self.assertIn(("mp", "KXX", 1, 170.), engine.calls)
            self.assertIn(("exit",), engine.calls)
            for progress in ("Mechanical APDL connected", "Meshing solid", "Mesh complete",
                             "Prescribed CFD temperatures", "streaming MAPDL output",
                             "Reading 2 thermal result sets", "Read result set 2/2",
                             "Saving thermal histories", "Generating thermal plots", "Complete:"):
                self.assertIn(progress, console.getvalue())
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["thermal_mesh"]["prescribed_boundary_nodes"], 6)
            self.assertEqual(result["final_temperature"]["minimum_K"], 800.)
            with (output / "temperature_history.csv").open(newline="") as handle:
                history = list(csv.DictReader(handle))
            self.assertEqual([float(row["time_s"]) for row in history], [0., 1., 2.])
            for name, digest in case.saved.items():
                self.assertEqual(file_sha256(case.work / name), digest)

    def test_missing_launch_timeout_uses_a_bounded_default(self):
        with tempfile.TemporaryDirectory() as directory:
            case = MechanicalCase(Path(directory))
            case.thermal.pop("mapdl_start_timeout_s")
            case.write_thermal()
            with redirect_stdout(io.StringIO()):
                _, launcher = case.run(lambda **kwargs: NativeMechanical())
            self.assertEqual(launcher.call_args.kwargs["start_timeout"], 120.)

    def test_invalid_launch_timeout_is_rejected_before_starting_mechanical(self):
        with tempfile.TemporaryDirectory() as directory:
            case = MechanicalCase(Path(directory))
            for value in (None, 0, -1, "unbounded", float("nan"), float("inf")):
                with self.subTest(value=value):
                    case.thermal["mapdl_start_timeout_s"] = value
                    case.write_thermal()
                    with patch("ansys.mapdl.core.launch_mapdl") as launcher, redirect_stdout(io.StringIO()):
                        with self.assertRaisesRegex(ThermalAnalysisError, "mapdl_start_timeout_s"):
                            run_thermal_analysis(cfd_config=case.cfd_path, thermal_config=case.thermal_path)
                    launcher.assert_not_called()

    def test_launch_failure_identifies_startup_and_native_logs_without_discarding_cfd(self):
        with tempfile.TemporaryDirectory() as directory:
            case = MechanicalCase(Path(directory))
            console = io.StringIO()

            def launch(**kwargs):
                raise TimeoutError("License server did not respond")

            with redirect_stdout(console):
                with self.assertRaisesRegex(ThermalAnalysisError, "MAPDL launch failed.*License server") as failure:
                    case.run(launch)
            self.assertIn(str(case.work / "thermal_analysis"), str(failure.exception))
            self.assertIn("Launching Mechanical APDL", console.getvalue())
            self.assertNotIn("Mechanical APDL connected", console.getvalue())
            self.assertFalse((case.work / "thermal_analysis/thermal_summary.json").exists())
            for name, digest in case.saved.items():
                self.assertEqual(file_sha256(case.work / name), digest)

    def test_native_solve_error_closes_engine_and_never_claims_complete_results(self):
        with tempfile.TemporaryDirectory() as directory:
            case = MechanicalCase(Path(directory))
            engine = NativeMechanical(solve_error=True)
            with redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ThermalAnalysisError, "MAPDL solve failed"):
                    case.run(lambda **kwargs: engine)
            self.assertIn(("solve", True), engine.calls)
            self.assertIn(("exit",), engine.calls)
            self.assertNotIn(("nsets",), engine.calls)
            self.assertFalse((case.work / "thermal_analysis/thermal_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
