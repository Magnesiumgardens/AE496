"""Exercise stage orchestration using deterministic subprocess substitutes."""
from pathlib import Path
import json
import tempfile
import unittest

import yaml

from pipeline.aero import MODEL_VERSION, reference_state
from pipeline.config import load_config
from pipeline.study import (BatchError, REPO_ROOT, run_case, stage_files,
                            stage_fingerprint, write_case_config)


class StudyResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.work = self.root / "runs/nose"
        self.geometry = self.root / "nose.x_t"
        self.geometry.write_text("original geometry")
        self.config = self.root / "case.yaml"
        self.cfg = yaml.safe_load((REPO_ROOT / "tests/fixtures/prescribed_reference/config.yaml").read_text())
        self.cfg.pop("workflow", None);self.cfg.pop("thermal_export", None)
        self.cfg["thermal"] = {"mode":"adiabatic"}
        self.cfg["solver"].update(initialization="freestream",courant_number=.1,
                                 flow_scheme="first-order-upwind",residual_thresholds={"energy":1e-6})
        self.cfg["runtime"]["meshing_release_delay_s"] = 0
        self.cfg["project"] = dict(name="nose", work_dir=str(self.work))
        self.cfg["geometry"]["input_file"] = str(self.geometry)
        self.thermal = self.root / "thermal.yaml"
        self.thermal_cfg = dict(loading=dict(mode="fixed_heat_flux"),
            material=dict(conductivity_w_mk=170, density_kg_m3=19250, specific_heat_j_kgk=130),
            transient=dict(duration_s=400))
        self.calls = []
        self.write_configs()

    def tearDown(self):
        self.temp.cleanup()

    def write_configs(self):
        self.config.write_text(yaml.safe_dump(self.cfg))
        self.thermal.write_text(yaml.safe_dump(self.thermal_cfg))

    def runner(self, command, log_path, **_):
        names = {"run_all.py": "mesh", "run_solver_only.py": "solver",
                 "export_temperature_only.py": "temperature_export", "export_aero_only.py": "aero",
                 "run_thermal_analysis.py": "disabled_thermal"}
        stage = names[Path(command[2]).name]
        self.calls.append(stage)
        cfg = load_config(self.config)
        for path in stage_files(self.work, cfg)[stage]:
            path.parent.mkdir(parents=True, exist_ok=True)
            if stage == "aero":
                from pipeline.force_history import assess_saved_force_history
                path.write_text(json.dumps(dict(status="complete", model_version=MODEL_VERSION,
                    body_wall_zones=[cfg["geometry"]["body_wall_name"]],
                    force_stability=assess_saved_force_history(cfg), **reference_state(cfg))))
            elif stage == "temperature_export":
                if path.suffix == ".json":
                    from pipeline.utils import file_sha256
                    from pipeline.temperature_export import temperature_conditions
                    csv_path = stage_files(self.work, cfg)[stage][0]
                    path.write_text(json.dumps(dict(status="complete", thermal_mode="adiabatic", face_count=3,
                        temperature_export_signature=cfg.get("temperature_export", {}),
                        conditions=temperature_conditions(cfg),
                        zero_flux_tolerance_W_m2=1, wall_csv_sha256=file_sha256(csv_path),
                        source_sha256={"final_case":file_sha256(self.work/"04_final.cas.h5"),
                                       "final_data":file_sha256(self.work/"04_final.dat.h5")})))
                else:
                    path.write_text("surface_order_index,x_m,y_m,segment_length_m,wall_temperature_K,near_wall_fluid_temperature_K,wall_heat_flux_fluent_W_m2\n0,0,0,1,2000,1900,0\n1,1,0,1,2200,2100,0\n2,0,1,1,2000,1900,0\n")
            elif path.suffix == ".json":
                path.write_text(json.dumps(dict(status="complete", loading_mode="fixed_heat_flux")))
            else:
                path.write_text(f"{stage} result {len(self.calls)}")

    def run_case(self, **kwargs):
        return run_case(self.config, self.thermal, _runner=self.runner, **kwargs)

    def test_completed_mesh_is_not_rebuilt_after_later_stage_failure(self):
        self.run_case(stop_after="mesh")
        def failing_solver(*a, **kw):
            raise BatchError("solver startup failed")
        with self.assertRaisesRegex(BatchError, "solver startup"):
            run_case(self.config, self.thermal, stop_after="solver", _runner=failing_solver)
        self.calls.clear()
        self.run_case(stop_after="solver")
        self.assertEqual(self.calls, ["solver"])

    def test_known_mesh_without_state_can_resume_solver_without_remeshing(self):
        self.work.mkdir(parents=True)
        (self.work / "02_mesh.msh.h5").write_text("existing mesh")
        result = self.run_case(stop_after="solver")
        self.assertEqual(self.calls, ["solver"])
        self.assertEqual(result["stages"]["mesh"], "adopted_unverified")

    def test_legacy_flight_change_preserves_existing_mesh(self):
        generated = self.work / "_batch/config.yaml"
        write_case_config(self.cfg, self.geometry, "nose", self.work, generated)
        (self.work / "02_mesh.msh.h5").write_text("existing mesh")
        self.cfg["flight"]["mach"] = 7
        write_case_config(self.cfg, self.geometry, "nose", self.work, generated)
        self.config = generated
        result = self.run_case(stop_after="solver")
        self.assertEqual(self.calls, ["solver"])
        self.assertEqual(result["stages"]["mesh"], "adopted_unverified")

    def test_regenerating_missing_mesh_never_adopts_old_solver_results(self):
        self.work.mkdir(parents=True)
        for name in ("04_final.cas.h5", "04_final.dat.h5"):
            (self.work / name).write_text("old solver with unknown mesh")
        self.run_case(stop_after="solver")
        self.assertEqual(self.calls, ["mesh", "solver"])
        retained = list((self.work / "_history").glob("*/04_final.dat.h5"))
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0].read_text(), "old solver with unknown mesh")

    def test_changed_physics_reruns_solver_and_retains_previous_results(self):
        self.run_case()
        original = (self.work / "04_final.dat.h5").read_text()
        self.calls.clear()
        self.cfg["flight"]["mach"] = 7
        self.write_configs()
        self.run_case()
        self.assertEqual(self.calls, ["solver", "temperature_export", "aero"])
        old = list((self.work / "_history").glob("*/04_final.dat.h5"))
        self.assertEqual(len(old), 1)
        self.assertEqual(old[0].read_text(), original)

    def test_changed_reference_width_does_not_rerun_thermal(self):
        self.run_case()
        original = (self.work / "temperature/temperature_summary.json").read_bytes()
        self.calls.clear()
        self.cfg["aero"]["reference_area_per_unit_span_m"] = 0.1
        self.write_configs()
        self.run_case()
        self.assertEqual(self.calls, ["aero"])
        self.assertEqual((self.work / "temperature/temperature_summary.json").read_bytes(), original)

    def test_changed_temperature_export_settings_do_not_rerun_cfd(self):
        self.run_case()
        self.calls.clear()
        self.cfg["temperature_export"]["topology_tolerance_m"] = 5e-10
        self.write_configs()
        self.run_case()
        self.assertEqual(self.calls, ["temperature_export"])

    def test_changing_h_reuses_cfd_and_reruns_only_mechanical(self):
        self.thermal_cfg = yaml.safe_load((REPO_ROOT / "thermal_analysis.yaml").read_text())
        self.thermal_cfg["loading"] = {"mode":"convection", "heat_transfer_coefficient_w_m2k":None}
        self.thermal_cfg["include_radiation"] = False
        self.thermal_cfg["loading"]["heat_transfer_coefficient_w_m2k"] = 42
        self.write_configs()
        def runner(command,log_path,**kwargs):
            if Path(command[2]).name == "run_thermal_analysis.py":
                self.calls.append("thermal")
                for path in stage_files(self.work,load_config(self.config))["thermal"]:
                    path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_text(json.dumps({"status":"complete","loading_mode":"convection"}) if path.suffix == ".json" else "solid fixture")
            else:
                self.runner(command,log_path,**kwargs)
                if Path(command[2]).name == "export_temperature_only.py":
                    (self.work/"temperature/wall_temperature_nodes.csv").write_text("nodal fixture")
        run_case(self.config,self.thermal,stop_after="thermal",_runner=runner)
        self.calls.clear()
        self.thermal_cfg["loading"]["heat_transfer_coefficient_w_m2k"] = 43
        self.write_configs()
        run_case(self.config,self.thermal,stop_after="thermal",_runner=runner)
        self.assertEqual(self.calls,["thermal"])

    def test_failed_residual_screen_stops_all_downstream_stages(self):
        def runner(command,log_path,**kwargs):
            self.runner(command,log_path,**kwargs)
            if Path(command[2]).name == "run_solver_only.py":
                log_path.parent.mkdir(parents=True,exist_ok=True)
                log_path.write_text("iter continuity x-velocity y-velocity energy time/iter\n1200 2.8 1.8 2.1 3.8 0:00:00 0\n")
        with self.assertRaisesRegex(BatchError,"did not converge"):
            run_case(self.config,self.thermal,_runner=runner)
        self.assertEqual(self.calls,["mesh","solver"])
        self.assertTrue((self.work/"04_final.dat.h5").is_file())
        self.assertFalse((self.work/"temperature/wall_temperatures.csv").exists())

    def test_changed_geometry_bytes_invalidate_all_results(self):
        self.run_case()
        self.calls.clear()
        self.geometry.write_text("revised geometry")
        self.run_case()
        self.assertEqual(self.calls, ["mesh", "solver", "temperature_export", "aero"])

    def test_modified_cached_output_is_not_accepted(self):
        self.run_case(stop_after="mesh")
        self.calls.clear()
        (self.work / "02_mesh.msh.h5").write_text("different mesh content")
        self.run_case(stop_after="mesh")
        self.assertEqual(self.calls, ["mesh"])

    def test_corrupt_legacy_boundary_reruns_export_only(self):
        self.run_case(stop_after="temperature_export")
        old_mesh = (self.work / "02_mesh.msh.h5").read_bytes()
        old_solver = (self.work / "04_final.dat.h5").read_bytes()
        (self.work / "_batch/stage_state.json").unlink()
        boundary = self.work / "temperature/wall_temperatures.csv"
        boundary.write_text("corrupt nonempty boundary file")
        self.calls.clear()
        self.run_case(stop_after="temperature_export")
        self.assertEqual(self.calls, ["solver", "temperature_export"])
        self.assertEqual((self.work / "02_mesh.msh.h5").read_bytes(), old_mesh)
        self.assertNotEqual((self.work / "04_final.dat.h5").read_bytes(), old_solver)
        retained = list((self.work / "_history").glob("*/temperature/wall_temperatures.csv"))
        self.assertEqual(retained[0].read_text(), "corrupt nonempty boundary file")

    def test_executed_export_with_nonfinite_flux_fails(self):
        def corrupt_runner(command, log_path, **kwargs):
            self.runner(command, log_path, **kwargs)
            if Path(command[2]).name == "export_temperature_only.py":
                (self.work / "temperature/wall_temperatures.csv").write_text(
                    "surface_order_index,x_m,y_m,heat_flux_into_wall_W_m2\n0,0,0,nan\n1,1,0,20\n2,0,1,30\n")
        with self.assertRaisesRegex(BatchError, "Invalid temperature_export"):
            run_case(self.config, self.thermal, stop_after="temperature_export", _runner=corrupt_runner)
        state = json.loads((self.work / "_batch/stage_state.json").read_text())
        self.assertEqual(state["stages"]["temperature_export"]["status"], "failed")

    def test_successful_process_without_required_outputs_is_failure(self):
        with self.assertRaisesRegex(BatchError, "required output"):
            run_case(self.config, self.thermal, stop_after="mesh", _runner=lambda *a, **kw: None)
        state = json.loads((self.work / "_batch/stage_state.json").read_text())
        self.assertEqual(state["stages"]["mesh"]["status"], "failed")

    def test_stage_fingerprints_depend_on_relevant_inputs(self):
        self.run_case()
        cfg = load_config(self.config)
        files = stage_files(self.work, cfg)
        first = stage_fingerprint("mesh", cfg, self.thermal_cfg, files)
        cfg["flight"]["mach"] = 9
        self.assertEqual(first, stage_fingerprint("mesh", cfg, self.thermal_cfg, files))
        cfg["mesh"]["global_max_size"] *= 2
        self.assertNotEqual(first, stage_fingerprint("mesh", cfg, self.thermal_cfg, files))


if __name__ == "__main__":
    unittest.main()
