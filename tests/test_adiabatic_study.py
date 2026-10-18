import copy
import csv
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from pipeline.config import ConfigError, load_config
from pipeline.gas_reporting import create_gas_reports, temperature_complete
from pipeline.migration import adiabatic_config, migrate_config
from pipeline.solve import _set_wall_thermal, _iterate_with_temperature_history
from pipeline.temperature_export import export_temperatures, temperature_paths
from pipeline.utils import file_sha256
from scripts.update_project import update_project
from test_nose_trade import fluent
from pipeline.aero import aerodynamic_summary


ROOT = Path(__file__).resolve().parents[1]


def config(work):
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    # These regressions exercise the earlier direct-adiabatic path explicitly;
    # the application template now selects one radiative-equilibrium solve.
    cfg.pop("workflow", None)
    cfg["aero"].pop("force_history", None)
    cfg.pop("calibration", None)
    cfg.pop("thermal_export", None)
    cfg["thermal"] = {"mode": "adiabatic"}
    cfg["solver"].update(initialization="freestream", courant_number=.1,
                         flow_scheme="first-order-upwind",
                         residual_thresholds={"continuity":1e-3, "x-velocity":1e-3,
                                              "y-velocity":1e-3, "energy":1e-6})
    cfg["project"] = dict(name=work.name, work_dir=str(work))
    cfg["geometry"]["input_file"] = str(work.parent / "nose.x_t")
    return cfg


def field_solver(flux=0):
    from ansys.fluent.core.fields.field_data_interfaces import SurfaceFieldDataRequest
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    wall = SimpleNamespace(vertices=vertices, connectivity=np.array([2,0,1,2,1,2,2,2,3,2,3,0]),
        face_centroids=np.array([[.5,0,0],[1,.5,0],[.5,1,0],[0,.5,0]]))
    calls = []
    def read(request):
        calls.append(request)
        zone = request.surfaces[0]
        if isinstance(request, SurfaceFieldDataRequest):return {zone: wall}
        if request.field_name == "heat-flux":return {zone: np.full(4, flux)}
        if request.node_value:return {zone: np.array([220., 1000., 2100., 500.])}
        return {zone: np.array([2000., 2200., 2100., 2300.]) - (0 if request.boundary_value else 100)}
    solver = SimpleNamespace(fields=SimpleNamespace(field_data=SimpleNamespace(get_field_data=read,
        scalar_fields=SimpleNamespace(allowed_values=lambda: ["temperature", "heat-flux"]))),
        settings=SimpleNamespace(setup=SimpleNamespace(boundary_conditions=SimpleNamespace(
            interior=SimpleNamespace(get_object_names=lambda: ["interior:fluid"])))))
    return solver, calls


class AdiabaticPhysicsTests(unittest.TestCase):
    def test_native_history_preserves_single_solve_and_actual_early_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(Path(directory));cfg["solver"]["max_iterations"] = 100
            calls = []
            raw = Path(directory)/"solver_temperature_history.out"
            report = SimpleNamespace(active=True)
            def iterate(iter_count):
                calls.append(iter_count)
                raw.write_text('"Iteration" "auto-wall-temp-max" "auto-wall-temp-min" "auto-wall-temp-avg"\n25 2500 1000 2000\n32 2600 1100 2100\n')
            solver = SimpleNamespace(settings=SimpleNamespace(solution=SimpleNamespace(
                run_calculation=SimpleNamespace(iterate=iterate))))
            with patch("pipeline.iteration_history.configure_history",return_value=(raw,report)):
                result = _iterate_with_temperature_history(solver,cfg,["body_wall"])
            self.assertEqual(calls,[100])
            with (Path(directory)/result["csv"]).open() as handle:rows=list(csv.DictReader(handle))
            self.assertEqual([int(row["iteration"]) for row in rows],[25,32])
            self.assertEqual(float(rows[-1]["wall_temperature_max_k"]),2600)
            self.assertFalse(report.active)
            with patch("pipeline.iteration_history.configure_history",return_value=(raw,report)), \
                 patch("pipeline.iteration_history.atomic_csv",side_effect=OSError("CSV is locked")):
                result = _iterate_with_temperature_history(solver,cfg,["body_wall"])
            self.assertEqual(result["status"],"unavailable")
            self.assertIn("locked",result["error"])

    def test_missing_iteration_counter_retains_original_single_solve(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(Path(directory));cfg["solver"]["max_iterations"] = 100
            calls=[]
            solver = SimpleNamespace(rp_vars=SimpleNamespace(allowed_values=lambda:["number-of-iterations"]),
                settings=SimpleNamespace(solution=SimpleNamespace(run_calculation=SimpleNamespace(
                    iterate=lambda iter_count:calls.append(iter_count)))))
            result = _iterate_with_temperature_history(solver,cfg,["body_wall"])
            self.assertEqual(result["status"],"unavailable")
            self.assertEqual(calls,[100])
            self.assertFalse((Path(directory)/"solver_temperature_history.csv").exists())

    def test_wall_sets_zero_flux_without_accessing_temperature(self):
        flux = SimpleNamespace()
        wall = SimpleNamespace(thermal=SimpleNamespace(heat_flux=flux))
        state = _set_wall_thermal(wall, {"mode": "adiabatic"})
        self.assertEqual(wall.thermal.thermal_condition, "Heat Flux")
        self.assertEqual((flux.option, flux.value), ("value", 0.0))
        self.assertEqual(state["heat_flux_W_m2"], 0)
        self.assertFalse(hasattr(wall.thermal, "temperature"))
        from ansys.fluent.core.generated.solver import settings_251
        self.assertEqual(settings_251.heat_flux_1.child_names[:2], ["option", "value"])

    def test_config_no_longer_requires_wall_temperature_and_rejects_isothermal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = config(root)
            path = root / "case.yaml"
            path.write_text(yaml.safe_dump(cfg))
            self.assertEqual(load_config(path)["thermal"], {"mode": "adiabatic"})
            for thermal in ({"mode": "fixed_temperature", "wall_temperature_k": 300},
                            {"mode": "adiabatic", "wall_temperature_k": 300}):
                cfg["thermal"] = thermal;path.write_text(yaml.safe_dump(cfg))
                with self.assertRaises(ConfigError):load_config(path)

    def test_export_distinguishes_wall_fluid_values_and_requests_only_temperature(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = config(root)
            solver, calls = field_solver()
            result = export_temperatures(solver, cfg, ["body_wall"])
            with temperature_paths(cfg)[0].open(newline="") as handle:rows = list(csv.DictReader(handle))
            self.assertEqual(result["face_count"], 4)
            self.assertEqual(result["wall_temperature"]["length_weighted_mean_K"], 2150)
            self.assertEqual(result["near_wall_fluid_temperature"]["length_weighted_mean_K"], 2050)
            self.assertEqual(result["wall_nodes"]["point_count"], 4)
            self.assertTrue(all(getattr(r,"field_name","temperature") == "temperature" for r in calls))
            self.assertNotIn("wall_heat_flux_fluent_W_m2",rows[0])
            self.assertTrue(all(float(r["wall_temperature_K"]) - float(r["near_wall_fluid_temperature_K"]) == 100 for r in rows))
            requests = [r for r in calls if getattr(r, "field_name", None) == "temperature" and not r.node_value]
            self.assertEqual([r.boundary_value for r in requests], [True, False])
            self.assertEqual(result["fluid_nodes"]["point_count"], 4)
            self.assertTrue((temperature_paths(cfg)[0].parent / "fluid_temperature_nodes.csv").is_file())

    def test_migration_preserves_mesh_and_marks_old_solver_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);path = root / "runs/nose/_batch/config.yaml"
            path.parent.mkdir(parents=True)
            cfg = config(path.parent.parent)
            cfg["thermal"] = {"mode": "fixed_temperature", "wall_temperature_k": 300, "emissivity": .9}
            cfg["thermal_export"] = {"enabled": True, "heat_flux_into_wall_sign": -1}
            path.write_text(yaml.safe_dump(cfg))
            state_path = path.parent / "stage_state.json"
            state_path.write_text(json.dumps({"stages": {"mesh": {"status": "complete", "fingerprint": "same"},
                "solver": {"status": "complete"}, "thermal": {"status": "complete"}}}))
            self.assertTrue(migrate_config(path))
            self.assertFalse(migrate_config(path))
            new = load_config(path)
            self.assertEqual(new["mesh"], cfg["mesh"])
            self.assertEqual(new["flight"], cfg["flight"])
            self.assertEqual(new["workflow"]["strategy"], "single_solve")
            self.assertEqual(new["thermal"], {"mode":"radiative_equilibrium", "wall_temperature_k":300,
                                              "emissivity":.9, "external_radiation_temperature_k":3})
            self.assertNotIn("thermal_export", new)
            state = json.loads(state_path.read_text())["stages"]
            self.assertEqual(state["mesh"], {"status": "complete", "fingerprint": "same"})
            self.assertEqual(state["solver"]["status"], "stale")
            self.assertEqual(state["thermal"]["status"], "stale")
            self.assertEqual(len(list(path.parent.glob("config_before_single_solve*"))), 1)

    def test_solid_thermal_runner_requires_h_before_launch(self):
        from pipeline.thermal import run_thermal_analysis
        from pipeline.thermal_loading import ThermalAnalysisError
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml";path.write_text(yaml.safe_dump(config(Path(directory))))
            thermal = yaml.safe_load((ROOT / "thermal_analysis.yaml").read_text())
            thermal["loading"] = {"mode":"convection", "heat_transfer_coefficient_w_m2k":None}
            thermal["include_radiation"] = False
            thermal_path = Path(directory) / "thermal.yaml"
            thermal_path.write_text(yaml.safe_dump(thermal))
            with self.assertRaisesRegex(ThermalAnalysisError, "heat_transfer_coefficient"):
                run_thermal_analysis(cfd_config=path, thermal_config=thermal_path)


class GasReportTests(unittest.TestCase):
    def test_two_nose_batch_completes_all_stages_and_offline_comparisons(self):
        from pipeline.study import main, stage_files
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);geometry = root / "geometry";geometry.mkdir()
            for name in ("blunt", "sharp"):(geometry / (name + ".x_t")).write_text("CAD fixture")
            base = config(root / "template");base["runtime"]["meshing_release_delay_s"] = 0
            (root / "config.yaml").write_text(yaml.safe_dump(base))
            (root / "batch.yaml").write_text(yaml.safe_dump(dict(base_config="config.yaml",
                geometry_directory="geometry", runs_directory="runs", comparison_outputs=True)))
            calls=[]
            def runner(command, log_path, **kwargs):
                cfg = load_config(command[command.index("--config") + 1])
                work = Path(cfg["project"]["work_dir"]);entry = Path(command[2]).name
                calls.append((cfg["project"]["name"], entry));work.mkdir(parents=True, exist_ok=True)
                if entry == "run_all.py":(work / "02_mesh.msh.h5").write_bytes(b"mesh fixture")
                elif entry == "run_solver_only.py":
                    for path in stage_files(work,cfg)["solver"]:path.write_bytes(b"adiabatic solution fixture")
                elif entry == "export_temperature_only.py":
                    result = export_temperatures(field_solver()[0], cfg, ["body_wall"])
                    csv_path, summary_path = temperature_paths(cfg)
                    result["wall_csv_sha256"] = file_sha256(csv_path)
                    result["source_sha256"] = {"final_case":file_sha256(work/"04_final.cas.h5"),
                                               "final_data":file_sha256(work/"04_final.dat.h5")}
                    summary_path.write_text(json.dumps(result))
                elif entry == "export_aero_only.py":
                    (work / "aero_summary.json").write_text(json.dumps(aerodynamic_summary(
                        fluent(),cfg,["body_wall.1","body_wall.2"])))
                else:self.fail(f"Unexpected stage: {entry}")
            with patch("pipeline.study.run_stage", side_effect=runner):
                self.assertEqual(main(root / "batch.yaml"),0)
            self.assertEqual(len(calls),8)
            manifest = json.loads((root / "runs/comparison/adiabatic_report.json").read_text())
            self.assertTrue(manifest["comparable_conditions"])
            self.assertTrue((root / "runs/comparison/near_wall_fluid_temperature_K_comparison.png").is_file())
            self.assertFalse(any((root / "runs" / name / "thermal_analysis").exists() for name in ("blunt","sharp")))

    def save_case(self, work):
        cfg = config(work)
        (work / "_batch").mkdir(parents=True)
        (work / "_batch/config.yaml").write_text(yaml.safe_dump(cfg))
        for name in ("04_final.cas.h5", "04_final.dat.h5"):(work / name).write_bytes(b"adiabatic solution")
        result = export_temperatures(field_solver()[0], cfg, ["body_wall"])
        csv_path, summary_path = temperature_paths(cfg)
        result["source_sha256"] = {"final_case":file_sha256(work/"04_final.cas.h5"), "final_data":file_sha256(work/"04_final.dat.h5")}
        result["wall_csv_sha256"] = file_sha256(csv_path)
        summary_path.write_text(json.dumps(result))
        aero = aerodynamic_summary(fluent(), cfg, ["body_wall.1", "body_wall.2"])
        (work / "aero_summary.json").write_text(json.dumps(aero))
        return cfg

    def test_reports_include_temperature_drag_and_exclude_old_isothermal_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.save_case(root / "blunt")
            old = root / "isothermal/_batch";old.mkdir(parents=True)
            old_cfg = copy.deepcopy(cfg);old_cfg["thermal"] = {"mode": "fixed_temperature", "wall_temperature_k": 300}
            (old / "config.yaml").write_text(yaml.safe_dump(old_cfg))
            with patch("pipeline.cfd.launch_solver", side_effect=AssertionError("Reports must not launch Fluent")):
                rows, created = create_gas_reports(root, root / "comparison")
            self.assertEqual([r["case"] for r in rows], ["blunt"])
            self.assertEqual(rows[0]["maximum_near_wall_fluid_temperature_K"], 2200)
            self.assertEqual(rows[0]["status"], "complete")
            self.assertTrue(Path(created["drag_vs_temperature"]).is_file())
            self.assertTrue((root / "blunt/reports/temperature_profile.png").is_file())
            self.assertTrue((root / "blunt/reports/fluid_temperature_nodes.png").is_file())
            self.assertTrue(temperature_complete(cfg))
            (root / "blunt/04_final.dat.h5").write_bytes(b"changed solution")
            self.assertFalse(temperature_complete(cfg))


class FullUpdateTests(unittest.TestCase):
    def test_git_install_migrates_configs_and_keeps_existing_run_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);source, target = root / "source", root / "target"
            source.mkdir();target.mkdir()
            (source / "study.py").write_text("# new application\n")
            (source / "config.yaml").write_text((ROOT / "config.yaml").read_text())
            (source / "batch.yaml").write_text((ROOT / "batch.yaml").read_text())
            (source / "thermal_analysis.yaml").write_text((ROOT / "thermal_analysis.yaml").read_text())
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "add", "."], check=True)
            subprocess.run(["git", "-C", str(source), "-c", "user.name=Study Test", "-c", "user.email=study@example.invalid", "commit", "-qm", "fixture"], check=True)
            (target / "study.py").write_text("# existing application\n")
            old = config(target / "runs/nose");old["flight"]["mach"] = 7
            old["thermal"] = {"mode": "fixed_temperature", "wall_temperature_k": 300}
            (target / "config.yaml").write_text(yaml.safe_dump(old))
            case = target / "runs/nose"; (case / "_batch").mkdir(parents=True)
            (case / "_batch/config.yaml").write_text(yaml.safe_dump(old))
            (case / "_batch/stage_state.json").write_text(json.dumps({"stages":{"mesh":{"status":"complete"},"solver":{"status":"complete"}}}))
            (case / "02_mesh.msh.h5").write_bytes(b"keep this mesh")
            (case / "04_final.dat.h5").write_bytes(b"keep previous solution")
            thermal = yaml.safe_load((ROOT / "thermal_analysis.yaml").read_text())
            thermal["loading"] = {"mode":"convection", "heat_transfer_coefficient_w_m2k":123}
            thermal["transient"]["duration_s"] = 180
            (target / "thermal_analysis.yaml").write_text(yaml.safe_dump(thermal))
            backup = update_project(target, source)
            installed = yaml.safe_load((target / "thermal_analysis.yaml").read_text())
            self.assertEqual(installed["loading"]["mode"], "aerodynamic_surface_load")
            self.assertTrue(installed["include_radiation"])
            self.assertNotIn("heat_transfer_coefficient_w_m2k", installed["loading"])
            self.assertEqual(installed["transient"]["duration_s"], 180)
            self.assertEqual(load_config(case / "_batch/config.yaml")["thermal"],
                             {"mode":"radiative_equilibrium", "wall_temperature_k":300,
                              "emissivity":.9, "external_radiation_temperature_k":3})
            self.assertEqual(load_config(case / "_batch/config.yaml")["flight"]["mach"], 7)
            self.assertEqual((case / "02_mesh.msh.h5").read_bytes(), b"keep this mesh")
            self.assertEqual((case / "04_final.dat.h5").read_bytes(), b"keep previous solution")
            self.assertEqual((backup / "study.py").read_text(), "# existing application\n")
            self.assertEqual(json.loads((case / "_batch/stage_state.json").read_text())["stages"]["solver"]["status"], "stale")


if __name__ == "__main__":unittest.main()
