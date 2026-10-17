"""Exercise the calibration entrypoint across native-process/export boundaries."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

import run_calibration
from pipeline.aerothermal_calibration import (
    build_surface_package, calibration_cfg, calibration_paths,
    verify_surface_package,
)
from pipeline.cfd import export_saved_case
from pipeline.config import ConfigError, load_config, validate_config
from pipeline.temperature_export import temperature_conditions
from pipeline.thermal_loading import ThermalAnalysisError
from pipeline.utils import atomic_json, file_sha256
from test_aerothermal_calibration import calibration_case

ROOT = Path(__file__).resolve().parents[1]


def runner_case(work):
    fixture = calibration_case(work)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    for section, values in fixture.items():
        if isinstance(values, dict):
            cfg.setdefault(section, {}).update(values)
        else:
            cfg[section] = values
    cfg["calibration"]["enabled"] = True
    path = work / "_batch/config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    cfg = load_config(path)
    paths = calibration_paths(cfg)
    for state, effective in (("re", cfg), ("cold", calibration_cfg(cfg))):
        metadata = json.loads(paths[f"{state}_summary"].read_text())
        metadata["conditions"] = temperature_conditions(effective)
        atomic_json(paths[f"{state}_summary"], metadata)
    return cfg


class CalibrationRunnerTests(unittest.TestCase):
    def _hooks(self, cfg, *, unstable=False, wrong_mesh=False):
        """The real runner assesses a transcript and builds/verifies real CSVs."""
        paths = calibration_paths(cfg)
        calls = []

        def native_solve(cold, mesh):
            calls.append(("solve", copy.deepcopy(cold), Path(mesh)))
            self.assertEqual(cold["_calibration_kind"], "cold_wall")
            self.assertEqual(cold["thermal"], dict(mode="fixed_temperature", wall_temperature_k=300.))
            self.assertEqual(cold["flight"], cfg["flight"])
            self.assertEqual(Path(mesh), paths["root"] / "02_mesh.msh.h5")
            for filename in ("04_final.cas.h5", "04_final.dat.h5"):
                (paths["cold_work"] / filename).write_bytes(("fresh calibrated cold " + filename).encode())
            summary = dict(thermal_mode="fixed_temperature", source_conditions=temperature_conditions(cold),
                source_geometry_sha256=file_sha256(cfg["geometry"]["input_file"]),
                source_mesh_sha256="wrong mesh" if wrong_mesh else file_sha256(mesh))
            atomic_json(paths["cold_work"] / "summary.json", summary)
            return summary

        def subprocess_stage(command, log, **kwargs):
            calls.append(("stage", list(command), Path(log)))
            self.assertEqual(command[-1], "--cold-solve")
            self.assertEqual(command[-2], cfg["_config_path"])
            with patch("sys.argv", command[2:]), patch("pipeline.solve.solve_case", side_effect=native_solve), \
                 patch("builtins.print"):
                self.assertEqual(run_calibration.main(), 0)
            log = Path(log)
            log.parent.mkdir(parents=True, exist_ok=True)
            residual = "2.0" if unstable else "1e-5"
            log.write_text("iter continuity x-velocity y-velocity energy time/iter\n"
                           f"100 {residual} 1e-5 1e-5 1e-5 0:00:00 0\n", encoding="utf-8")

        def export(effective, thermal=True, aero=True, **kwargs):
            state = "cold" if effective.get("_calibration_kind") == "cold_wall" else "re"
            calls.append(("export", state, copy.deepcopy(effective), thermal, aero))
            self.assertFalse(thermal)
            self.assertFalse(aero)
            metadata = json.loads(paths[f"{state}_summary"].read_text())
            metadata["conditions"] = temperature_conditions(effective)
            metadata["source_sha256"] = {key: file_sha256(paths[f"{state}_work"] / filename)
                for key, filename in (("final_case", "04_final.cas.h5"), ("final_data", "04_final.dat.h5"))}
            for key in ("faces", "nodes"):
                metadata[f"{key}_csv_sha256"] = file_sha256(paths[f"{state}_{key}"])
            atomic_json(paths[f"{state}_summary"], metadata)
            return {"calibration": metadata}

        return calls, subprocess_stage, export

    def test_cached_re_export_and_primary_pair_are_preserved_during_fresh_cold_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = runner_case(Path(directory) / "nose")
            paths = calibration_paths(cfg)
            original = {name: file_sha256(paths["root"] / name) for name in ("04_final.cas.h5", "04_final.dat.h5")}
            calls, stage, export = self._hooks(cfg)
            with patch("pipeline.study.run_stage", side_effect=stage), \
                 patch("pipeline.cfd.export_saved_case", side_effect=export), \
                 patch("pipeline.solution_quality.save_diagnostics") as diagnostics:
                result = run_calibration.run_calibration(cfg)
            self.assertEqual(result["status"], "complete")
            self.assertEqual([call[1] for call in calls if call[0] == "export"], ["cold"])
            self.assertEqual({name: file_sha256(paths["root"] / name) for name in original}, original)
            cold_file = load_config(paths["cold_work"] / "_batch/config.yaml")
            self.assertEqual(cold_file["_study_work_dir"], str(paths["root"]))
            self.assertEqual(cold_file["thermal"]["mode"], "fixed_temperature")
            diagnostics.assert_called_once()
            self.assertEqual(verify_surface_package(cfg)[1]["source_sha256"], result["source_sha256"])

    def test_stale_re_export_is_refreshed_before_the_native_cold_solve(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = runner_case(Path(directory) / "nose")
            paths = calibration_paths(cfg)
            metadata = json.loads(paths["re_summary"].read_text())
            metadata["nodes_csv_sha256"] = "stale export"
            atomic_json(paths["re_summary"], metadata)
            calls, stage, export = self._hooks(cfg)
            with patch("pipeline.study.run_stage", side_effect=stage), \
                 patch("pipeline.cfd.export_saved_case", side_effect=export), \
                 patch("pipeline.solution_quality.save_diagnostics"):
                run_calibration.run_calibration(cfg)
            self.assertEqual([(call[0], call[1] if call[0] == "export" else None) for call in calls],
                             [("export", "re"), ("stage", None), ("solve", None), ("export", "cold")])

    def test_unconverged_cold_pair_is_retained_and_never_exported_or_used_as_a_load(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = runner_case(Path(directory) / "nose")
            paths = calibration_paths(cfg)
            build_surface_package(cfg)
            calls, stage, export = self._hooks(cfg, unstable=True)
            with patch("pipeline.study.run_stage", side_effect=stage), \
                 patch("pipeline.cfd.export_saved_case", side_effect=export), \
                 patch("pipeline.solution_quality.save_diagnostics") as diagnostics:
                with self.assertRaisesRegex(ValueError, "did not pass the residual screen; diagnostic pair retained"):
                    run_calibration.run_calibration(cfg)
            self.assertFalse(any(call[0] == "export" for call in calls))
            self.assertIn(b"fresh calibrated cold", (paths["cold_work"] / "04_final.dat.h5").read_bytes())
            summary = json.loads((paths["cold_work"] / "summary.json").read_text())
            self.assertEqual(summary["numerical_quality"]["status"], "failed")
            self.assertEqual(summary["numerical_quality"]["source_sha256"]["final_data"],
                             file_sha256(paths["cold_work"] / "04_final.dat.h5"))
            diagnostics.assert_called_once()
            with self.assertRaises(ThermalAnalysisError): verify_surface_package(cfg)

    def test_cold_solver_using_a_different_mesh_cannot_build_a_surface_package(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = runner_case(Path(directory) / "nose")
            calls, stage, export = self._hooks(cfg, wrong_mesh=True)
            with patch("pipeline.study.run_stage", side_effect=stage), \
                 patch("pipeline.cfd.export_saved_case", side_effect=export), \
                 patch("pipeline.solution_quality.save_diagnostics"):
                with self.assertRaisesRegex(ThermalAnalysisError, "unchanged common study mesh"):
                    run_calibration.run_calibration(cfg)
            self.assertTrue(calibration_paths(cfg)["cold_work"].joinpath("04_final.dat.h5").is_file())

    def test_live_cold_export_loads_only_its_case_and_passes_verified_internal_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = runner_case(Path(directory) / "nose")
            cold = calibration_cfg(cfg)
            read_paths, verified, endpoint_args, exits = [], [], [], []
            engine = SimpleNamespace(settings=SimpleNamespace(
                file=SimpleNamespace(read_case_data=lambda **kwargs: read_paths.append(kwargs["file_name"])),
                setup=SimpleNamespace(boundary_conditions=SimpleNamespace(
                    wall=SimpleNamespace(get_object_names=lambda: ["body_wall", "other_wall", "body_wall.2"])))),
                exit=lambda: exits.append(True))
            def endpoint(solver, effective, walls):
                endpoint_args.append((solver, effective, walls))
                return {"status": "complete"}
            with patch("pipeline.cfd.launch_solver", return_value=engine), \
                 patch("pipeline.cfd.validate_saved_conditions", side_effect=lambda solver, effective: verified.append(effective)), \
                 patch("pipeline.endpoint_export.export_endpoint", side_effect=endpoint), \
                 patch("pipeline.cfd.export_temperatures") as temperatures, \
                 patch("pipeline.cfd.aerodynamic_summary") as forces:
                result = export_saved_case(cold, thermal=False, aero=False)
            self.assertEqual(result, {"calibration": {"status": "complete"}})
            self.assertEqual(read_paths, [str(calibration_paths(cfg)["cold_work"] / "04_final.cas.h5")])
            self.assertEqual(verified, [cold])
            self.assertEqual(endpoint_args[0][2], ["body_wall", "body_wall.2"])
            self.assertEqual(endpoint_args[0][1]["thermal"], dict(mode="fixed_temperature", wall_temperature_k=300.))
            self.assertEqual(exits, [True])
            temperatures.assert_not_called()
            forces.assert_not_called()

    def test_missing_cold_quality_rejects_export_before_launching_fluent(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = runner_case(Path(directory) / "nose")
            cold = calibration_cfg(cfg)
            summary_path = calibration_paths(cfg)["cold_work"] / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary.pop("numerical_quality")
            atomic_json(summary_path, summary)
            with patch("pipeline.cfd.launch_solver") as launch:
                with self.assertRaisesRegex(ValueError, "matching passed residual screen"):
                    export_saved_case(cold, thermal=False, aero=False)
            launch.assert_not_called()

    def test_fixed_temperature_requires_explicit_internal_calibration_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = runner_case(Path(directory) / "nose")
            cold = calibration_cfg(cfg)
            validate_config(cold)
            cold.pop("_calibration_kind")
            with self.assertRaisesRegex(ConfigError, "single-solve workflow requires"):
                validate_config(cold)


if __name__ == "__main__": unittest.main()
