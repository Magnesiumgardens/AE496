"""Engine-free integration of RE CFD, cold calibration and transient surface loads.

Native solver execution is replaced by deterministic fields.  Export matching,
source hashes, orchestration resume, and the saved Mechanical result contract
are exercised through the application functions.
"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from pipeline.aero import aerodynamic_summary
from pipeline.config import load_config
from pipeline.study import BatchError, run_case
from pipeline.temperature_export import export_temperatures, temperature_conditions, temperature_paths
from pipeline.thermal import run_thermal_analysis
from pipeline.utils import atomic_json, file_sha256
from test_adiabatic_study import field_solver
from test_nose_trade import fluent


ROOT = Path(__file__).resolve().parents[1]
GOOD_LOG = ("iter continuity x-velocity y-velocity energy time/iter\n"
            "1161 9.9e-4 1.2e-4 6.0e-5 1.1e-4 0:00:00 0\n")


def solved_field_solver():
    """One square mesh, solved wall temperatures well above the cold state."""
    solver, _ = field_solver()
    read = solver.fields.field_data.get_field_data

    def fields(request):
        result = read(request)
        if getattr(request, "field_name", None) == "temperature":
            return {name: np.asarray(value) * 0 + np.array([1800., 2000., 2200., 1900.])
                    for name, value in result.items()}
        return result

    solver.fields.field_data.get_field_data = fields
    return solver


class AerothermalWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.work = self.root / "runs/nose"
        self.work.mkdir(parents=True)
        self.cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
        self.cfg["project"].update(name="nose", work_dir=str(self.work))
        cad = self.root / "nose.x_t"
        cad.write_text("single shared CAD fixture")
        self.cfg["geometry"]["input_file"] = str(cad)
        self.cfg["runtime"]["meshing_release_delay_s"] = 0
        self.cfg["calibration"] = dict(enabled=True, cold_wall_temperature_k=300.,
                                       heat_flux_into_wall_sign=-1)
        self.path = self.work / "_batch/config.yaml"
        self.path.parent.mkdir()
        self.write_config()
        settings = yaml.safe_load((ROOT / "thermal_analysis.yaml").read_text())
        settings["loading"] = {"mode": "aerodynamic_surface_load"}
        settings["include_radiation"] = True
        settings["radiation_temperature_k"] = self.cfg["thermal"]["external_radiation_temperature_k"]
        settings["visualization"]["enabled"] = False
        self.thermal = self.root / "thermal.yaml"
        self.thermal.write_text(yaml.safe_dump(settings))
        self.calls = []

    def write_config(self):
        self.path.write_text(yaml.safe_dump(self.cfg))

    def pair_hashes(self, work=None):
        work = work or self.work
        return {key: file_sha256(work / name) for key, name in
                (("final_case", "04_final.cas.h5"), ("final_data", "04_final.dat.h5"))}

    def study(self, **kwargs):
        return run_case(self.path, self.thermal, _runner=self.runner, **kwargs)

    def runner(self, command, log_path, **kwargs):
        cfg = load_config(command[command.index("--config") + 1])
        entry = Path(command[2]).name
        self.calls.append(entry)
        self.assertEqual(Path(cfg["project"]["work_dir"]), self.work)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(GOOD_LOG)
        if entry == "run_all.py":
            (self.work / "02_mesh.msh.h5").write_bytes(b"unchanged shared CFD mesh")
        elif entry == "run_solver_only.py":
            for name in ("04_final.cas.h5", "04_final.dat.h5"):
                (self.work / name).write_bytes((name + str(cfg["thermal"])).encode())
            atomic_json(self.work / "summary.json", dict(
                thermal_mode="radiative_equilibrium", source_conditions=temperature_conditions(cfg),
                source_geometry_sha256=file_sha256(cfg["geometry"]["input_file"]),
                source_mesh_sha256=file_sha256(self.work / "02_mesh.msh.h5"),
                body_wall_zones=["body_wall"], wall_thermal_setup={"body_wall": {
                    "thermal_condition": "Radiation", "emissivity": .9,
                    "external_radiation_temperature_k": 3}}))
        elif entry == "export_temperature_only.py":
            result = export_temperatures(solved_field_solver(), cfg, ["body_wall"])
            result.update(source_sha256=self.pair_hashes(),
                          wall_csv_sha256=file_sha256(temperature_paths(cfg)[0]))
            atomic_json(temperature_paths(cfg)[1], result)
            from pipeline.aerothermal_calibration import export_re_calibration
            export_re_calibration(solved_field_solver(), cfg, ["body_wall"])
        elif entry == "run_calibration.py":
            self.calibrate(cfg)
        elif entry == "export_aero_only.py":
            result = aerodynamic_summary(fluent(), cfg, ["body_wall.1", "body_wall.2"])
            result["source_sha256"] = self.pair_hashes()
            atomic_json(self.work / "aero_summary.json", result)
        elif entry == "run_thermal_analysis.py":
            thermal_config = command[command.index("--thermal-config") + 1]

            def solve(engine, load, settings, size):
                from pipeline.aerothermal_loading import prepare_boundary_quadrature
                self.assertEqual(load.mode, "aerodynamic_surface_load")
                self.assertTrue(settings["include_rad"])
                count = len(load.xy)
                midpoints = (load.xy + np.roll(load.xy, -1, axis=0)) / 2
                xy = np.vstack([load.xy, midpoints, load.xy.mean(axis=0)])
                xyz = np.c_[xy, np.zeros(len(xy))]
                ids = np.arange(1, len(xy) + 1)
                selected = SimpleNamespace(mesh=SimpleNamespace(nnum=ids[:2 * count]),
                                           lsel=lambda *a: None, nsll=lambda *a: None,
                                           allsel=lambda: None)
                prepare_boundary_quadrature(selected, range(1, count + 1), load, ids, xyz, size)
                # Synthetic engine outputs deliberately warm every surface point
                # gradually; none is constrained to the 1800--2200 K CFD field.
                values = [np.full(len(ids), 400.), np.full(len(ids), 500.)]
                return ids, xyz, values, [2., settings["duration"]], 4

            with patch("ansys.mapdl.core.launch_mapdl", return_value=SimpleNamespace(exit=lambda: None)), \
                    patch("pipeline.thermal.solve_thermal", side_effect=solve):
                run_thermal_analysis(cfd_config=self.path, thermal_config=thermal_config)
        else:
            self.fail("Unexpected workflow entry point " + entry)

    def calibrate(self, cfg, *, failed=False):
        from pipeline.aerothermal_calibration import (build_surface_package, calibration_cfg,
                                                     calibration_paths, export_cold_calibration,
                                                     export_re_calibration)
        paths = calibration_paths(cfg)
        cold = calibration_cfg(cfg)
        work = paths["cold_work"]
        work.mkdir(parents=True, exist_ok=True)
        for name in ("04_final.cas.h5", "04_final.dat.h5"):
            (work / name).write_bytes((name + str(cold["thermal"])).encode())
        atomic_json(work / "summary.json", dict(
            thermal_mode="fixed_temperature", source_conditions=temperature_conditions(cold),
            source_geometry_sha256=file_sha256(cfg["geometry"]["input_file"]),
            source_mesh_sha256=file_sha256(self.work / "02_mesh.msh.h5"),
            numerical_quality=dict(status="failed" if failed else "residual_screen_passed",
                                   source_sha256=self.pair_hashes(work))))
        if failed:
            raise BatchError("Cold calibration did not converge; diagnostic case/data retained")
        export_re_calibration(solved_field_solver(), cfg, ["body_wall"])
        solver, _ = field_solver(flux=-5e6)
        read = solver.fields.field_data.get_field_data

        def cold_fields(request):
            result = read(request)
            if getattr(request, "field_name", None) == "temperature":
                return {name: np.full(len(value), cold["thermal"]["wall_temperature_k"])
                        for name, value in result.items()}
            return result

        solver.fields.field_data.get_field_data = cold_fields
        export_cold_calibration(solver, cfg, ["body_wall"])
        return build_surface_package(cfg)

    def test_one_primary_re_solution_and_one_cold_calibration_feed_solved_surface_history(self):
        result = self.study()
        self.assertEqual(self.calls, ["run_all.py", "run_solver_only.py", "export_temperature_only.py",
                                     "export_aero_only.py", "run_calibration.py", "run_thermal_analysis.py"])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(self.calls.count("run_solver_only.py"), 1)
        self.assertEqual(self.calls.count("run_calibration.py"), 1)
        aero = json.loads((self.work / "aero_summary.json").read_text())
        self.assertEqual(aero["source_sha256"], self.pair_hashes())
        from pipeline.aerothermal_calibration import calibration_paths
        paths = calibration_paths(load_config(self.path))
        self.assertNotEqual(self.pair_hashes(), self.pair_hashes(paths["cold_work"]))
        summary = json.loads((self.work / "thermal_analysis/thermal_summary.json").read_text())
        self.assertEqual(summary["model_version"], "6.0-aerodynamic-surface-load")
        self.assertEqual(summary["loading_mode"], "aerodynamic_surface_load")
        self.assertEqual(summary["analysis"]["duration_s"], 200)
        self.assertIsNotNone(summary["net_heat_load_per_unit_span_J_m"])
        import csv
        with (self.work / "thermal_analysis/wall_temperature_history.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(rows)
        self.assertNotIn("prescribed_surface_temperature_K", rows[0])
        self.assertEqual({float(row["wall_temperature_K"]) for row in rows if float(row["time_s"]) == 0}, {300.})
        self.assertEqual({float(row["wall_temperature_K"]) for row in rows if float(row["time_s"]) == 2}, {400.})
        self.calls.clear()
        result = self.study()
        self.assertEqual(self.calls, [])
        self.assertEqual(set(result["stages"].values()), {"reused"})

    def test_calibration_disabled_is_rejected_before_any_solver_stage(self):
        self.cfg["calibration"]["enabled"] = False
        self.write_config()
        with self.assertRaisesRegex((BatchError, ValueError), "calibration"):
            self.study()
        self.assertEqual(self.calls, [])

    def test_200_to_400_seconds_only_reruns_mechanical(self):
        self.study()
        self.calls.clear()
        preserved = {name: file_sha256(self.work / name) for name in (
            "02_mesh.msh.h5", "04_final.cas.h5", "04_final.dat.h5", "aero_summary.json",
            "thermal_bc/surface_load_nodes.csv", "thermal_bc/surface_load_summary.json")}
        settings = yaml.safe_load(self.thermal.read_text())
        settings["transient"]["duration_s"] = 400
        self.thermal.write_text(yaml.safe_dump(settings))
        self.study()
        self.assertEqual(self.calls, ["run_thermal_analysis.py"])
        for name, digest in preserved.items():
            self.assertEqual(file_sha256(self.work / name), digest)
        summary = json.loads((self.work / "thermal_analysis/thermal_summary.json").read_text())
        self.assertEqual(summary["analysis"]["duration_s"], 400)

    def test_cold_wall_temperature_change_leaves_primary_re_and_aero_unchanged(self):
        self.study()
        self.calls.clear()
        saved_re = self.pair_hashes()
        saved_aero = file_sha256(self.work / "aero_summary.json")
        self.cfg["calibration"]["cold_wall_temperature_k"] = 350
        self.write_config()
        self.study()
        self.assertEqual(self.calls, ["run_calibration.py", "run_thermal_analysis.py"])
        self.assertEqual(self.pair_hashes(), saved_re)
        self.assertEqual(file_sha256(self.work / "aero_summary.json"), saved_aero)

    def test_failed_cold_calibration_blocks_thermal_without_destroying_primary_re(self):
        def failed(command, log_path, **kwargs):
            if Path(command[2]).name == "run_calibration.py":
                self.calls.append("run_calibration.py")
                cfg = load_config(command[command.index("--config") + 1])
                return self.calibrate(cfg, failed=True)
            return self.runner(command, log_path, **kwargs)

        with self.assertRaisesRegex(BatchError, "Cold calibration did not converge"):
            run_case(self.path, self.thermal, _runner=failed)
        self.assertNotIn("run_thermal_analysis.py", self.calls)
        self.assertTrue((self.work / "04_final.cas.h5").is_file())
        self.assertTrue((self.work / "04_final.dat.h5").is_file())
        self.assertTrue((self.work / "cfd/cold_calibration/04_final.dat.h5").is_file())
        self.assertFalse((self.work / "thermal_analysis").exists())
        state = json.loads((self.work / "_batch/stage_state.json").read_text())
        self.assertEqual(state["stages"]["solver"]["status"], "complete")
        self.assertEqual(state["stages"]["calibration"]["status"], "failed")

    def test_modified_calibration_source_is_rejected_and_rebuilt_before_thermal(self):
        self.study()
        self.calls.clear()
        from pipeline.aerothermal_calibration import verify_surface_package
        from pipeline.thermal_loading import ThermalAnalysisError
        (self.work / "cfd/cold_calibration/04_final.dat.h5").write_bytes(b"unrelated CFD data")
        with self.assertRaises(ThermalAnalysisError):
            verify_surface_package(load_config(self.path))
        saved_re = self.pair_hashes()
        self.study()
        self.assertEqual(self.calls, ["run_calibration.py", "run_thermal_analysis.py"])
        self.assertEqual(self.pair_hashes(), saved_re)


if __name__ == "__main__":
    unittest.main()
