"""Single-solve migration preserves user inputs and never relabels endpoint data."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml

from pipeline.config import ConfigError, load_config, validate_config
from pipeline.migration import (migrate_config, single_solve_batch_config,
    single_solve_config, single_solve_state, single_solve_thermal_config)
from scripts.update_project import update_project


ROOT = Path(__file__).resolve().parents[1]


def template():
    return yaml.safe_load((ROOT / "config.yaml").read_text())


def endpoint_template():
    return yaml.safe_load((ROOT / "tests/fixtures/two_endpoint_config.yaml").read_text())


class SingleConfigurationTests(unittest.TestCase):
    def test_default_is_one_radiating_wall_with_explicit_solid_prescription(self):
        cfg = validate_config(template())
        self.assertEqual(cfg["workflow"], {"strategy": "single_solve"})
        self.assertEqual(cfg["thermal"], dict(mode="radiative_equilibrium", wall_temperature_k=300,
            emissivity=.9, external_radiation_temperature_k=3))
        self.assertEqual(cfg["flight"]["flow_direction"], [-1, 0])
        self.assertEqual(cfg["runtime"]["geometry_launch_mode"], "spaceclaim")
        self.assertIsNone(cfg["solver"]["courant_number"])
        self.assertIsNone(cfg["solver"]["flow_scheme"])
        self.assertEqual(set(cfg["solver"]["residual_thresholds"].values()), {1e-3})
        self.assertNotIn("thermal_export", cfg)
        thermal = yaml.safe_load((ROOT / "thermal_analysis.yaml").read_text())
        self.assertEqual(thermal["loading"], {"mode": "prescribed_temperature"})
        self.assertFalse(thermal["include_radiation"])
        self.assertEqual(thermal["transient"]["duration_s"], 200)

    def test_invalid_radiation_inputs_and_incompatible_single_boundaries_fail(self):
        cases = [("emissivity", value) for value in (0, -1, 1.1, float("nan"), float("inf"), None)]
        cases += [("external_radiation_temperature_k", value) for value in (0, -3, float("nan"), None)]
        cases += [("wall_temperature_k", 0), ("mode", "adiabatic"), ("mode", "fixed_temperature")]
        for name, value in cases:
            with self.subTest(name=name, value=value):
                cfg = template(); cfg["thermal"][name] = value
                with self.assertRaises(ConfigError): validate_config(cfg)
        cfg = template(); cfg["thermal"]["emissivity"] = 1
        self.assertEqual(validate_config(cfg)["thermal"]["emissivity"], 1)

    def test_conversion_keeps_successful_solver_geometry_flow_and_gui_inputs(self):
        cfg = endpoint_template()
        cfg["solver"].update(initialization="freestream", courant_number=.3,
            flow_scheme="first-order-upwind", max_iterations=2400,
            residual_thresholds={"energy":1e-6})
        cfg["flight"]["mach"] = 7.8
        cfg["geometry"]["input_file"] = "my nose.x_t"
        cfg["mesh"]["body_min_size"] = .00012
        cfg["runtime"]["fluent_gui"] = True
        cfg["reporting"]["nose_direction"] = [0, 1]
        cfg["workflow"]["adiabatic_solver"] = {"courant_number": .8}
        before = copy.deepcopy(cfg)
        result = single_solve_config(cfg)
        self.assertEqual(cfg, before)
        for name in ("solver", "geometry", "flight", "mesh", "runtime", "physics", "aero", "reporting", "project"):
            self.assertEqual(result[name], before[name])
        self.assertEqual(result["workflow"], {"strategy": "single_solve"})
        self.assertNotIn("thermal_export", result)
        self.assertEqual(single_solve_config(result), result)
        validate_config(result)

    def test_primary_endpoint_numerical_overrides_survive_single_solve_migration(self):
        cfg=endpoint_template()
        cfg['workflow']['cold_wall_solver']={'courant_number':.2,'flow_scheme':'first-order-upwind',
                                             'max_iterations':2400}
        cfg['workflow']['adiabatic_solver']={'courant_number':.8}
        result=single_solve_config(cfg)
        self.assertEqual(result['solver']['courant_number'],.2)
        self.assertEqual(result['solver']['flow_scheme'],'first-order-upwind')
        self.assertEqual(result['solver']['max_iterations'],2400)
        self.assertEqual(result['workflow'],{'strategy':'single_solve'})
        self.assertEqual(single_solve_config(result),result)
        validate_config(result)

    def test_existing_radiation_inputs_are_retained_and_missing_history_is_optional(self):
        cfg = template()
        cfg["thermal"].update(emissivity=.75, external_radiation_temperature_k=270,
                              wall_temperature_k=340, user_review_note="Custom radiation environment")
        cfg["solver"]["temperature_history"]["enabled"] = False
        self.assertEqual(single_solve_config(cfg), cfg)

    def test_thermal_conversion_keeps_exposure_material_and_visualization(self):
        old = yaml.safe_load((ROOT / "tests/fixtures/endpoint_thermal_analysis.yaml").read_text())
        old["transient"]["duration_s"] = 175
        old["material"]["conductivity_w_mk"] = 145
        old["visualization"]["make_gif"] = True
        old["loading"]["heat_transfer_coefficient_w_m2k"] = 500
        previous = copy.deepcopy(old)
        result = single_solve_thermal_config(old,
            yaml.safe_load((ROOT / "thermal_analysis.yaml").read_text()))
        self.assertEqual(old, previous)
        for name in ("transient", "material", "visualization"):
            self.assertEqual(result[name], old[name])
        self.assertEqual(result["loading"], {"mode": "prescribed_temperature"})
        self.assertFalse(result["include_radiation"])
        self.assertEqual(single_solve_thermal_config(result), result)

    def test_batch_removes_endpoint_stage_dependencies_and_keeps_owned_timeouts(self):
        cfg = dict(run_thermal=True, run_adiabatic_reference=True, stop_on_error=True,
            timeouts={"cold_wall":{"hard_s":3000}, "adiabatic":{"hard_s":1000},
                "adiabatic_export":{"hard_s":600}, "solver":{"hard_s":3500},
                "thermal":{"hard_s":7000}, "mesh":{"hard_s":9000}})
        result = single_solve_batch_config(cfg)
        self.assertEqual(result["timeouts"], dict(solver={"hard_s":3500},
            temperature_export={"hard_s":600}, thermal={"hard_s":7000}, mesh={"hard_s":9000}))
        self.assertNotIn("run_adiabatic_reference", result)
        self.assertTrue(result["stop_on_error"])
        self.assertEqual(single_solve_batch_config(result), result)

    def test_state_preserves_mesh_record_without_adopting_successful_endpoints(self):
        mesh = dict(status="complete", fingerprint="mesh", outputs={"02_mesh.msh.h5":"sha"})
        state = dict(stages=dict(mesh=mesh, cold_wall={"status":"complete"},
                                adiabatic={"status":"complete"}, solver={"status":"complete"}))
        before = copy.deepcopy(state)
        result = single_solve_state(state)
        self.assertEqual(state, before)
        self.assertEqual(result["stages"]["mesh"], mesh)
        self.assertTrue(all(record["status"] == "stale" for name, record in result["stages"].items() if name != "mesh"))

    def test_explicit_migration_retains_results_and_backs_up_physical_model(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "runs/nose"
            path = work / "_batch/config.yaml"; path.parent.mkdir(parents=True)
            path.write_text(yaml.safe_dump(endpoint_template()))
            original = path.read_bytes()
            (work / "cfd/cold_wall").mkdir(parents=True)
            pair = work / "cfd/cold_wall/04_final.dat.h5"; pair.write_bytes(b"old endpoint")
            (path.parent / "stage_state.json").write_text(json.dumps(dict(stages=dict(mesh={"fingerprint":"mesh"}))))
            self.assertTrue(migrate_config(path)); self.assertFalse(migrate_config(path))
            self.assertEqual(load_config(path)["workflow"]["strategy"], "single_solve")
            self.assertEqual(pair.read_bytes(), b"old endpoint")
            backups = list(path.parent.glob("config_before_single_solve_*.yaml"))
            self.assertEqual(len(backups), 1); self.assertEqual(backups[0].read_bytes(), original)


class SingleInstallerTests(unittest.TestCase):
    def test_git_installer_preserves_data_and_user_configuration_across_repeated_update(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source", Path(directory) / "target"
            source.mkdir(); target.mkdir()
            for name in ("config.yaml", "thermal_analysis.yaml", "batch.yaml"):
                (source / name).write_bytes((ROOT / name).read_bytes())
            (source / "study.py").write_text("# new code\n")
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "add", "."], check=True)
            subprocess.run(["git", "-C", str(source), "-c", "user.name=Single Test",
                "-c", "user.email=single@example.invalid", "commit", "-qm", "fixture"], check=True)
            (target / "study.py").write_text("# existing code\n")
            old = endpoint_template(); old["solver"]["courant_number"] = .3
            old["runtime"]["fluent_gui"] = True
            case = target / "runs/nose"; (case / "_batch").mkdir(parents=True)
            for path in (target / "config.yaml", case / "_batch/config.yaml"):
                path.write_text(yaml.safe_dump(old))
            mesh_record = {"status":"complete", "fingerprint":"old-mesh"}
            (case / "_batch/stage_state.json").write_text(json.dumps(dict(stages=dict(mesh=mesh_record, cold_wall={"status":"complete"}))))
            artifacts = {name: b"retain " + name.encode() for name in (
                "02_mesh.msh.h5", "04_final.dat.h5", "cfd/cold_wall/04_final.cas.h5", "cfd/adiabatic/04_final.dat.h5")}
            for name, data in artifacts.items():
                path = case / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
            thermal = yaml.safe_load((ROOT / "tests/fixtures/endpoint_thermal_analysis.yaml").read_text())
            thermal["transient"]["duration_s"] = 150
            thermal["material"]["conductivity_w_mk"] = 140
            thermal["visualization"]["make_gif"] = True
            (target / "thermal_analysis.yaml").write_text(yaml.safe_dump(thermal))
            backup = update_project(target, source)
            current = load_config(case / "_batch/config.yaml")
            self.assertEqual(current["workflow"], {"strategy":"single_solve"})
            self.assertEqual(current["solver"]["courant_number"], .3)
            self.assertTrue(current["runtime"]["fluent_gui"])
            self.assertEqual(current["flight"]["flow_direction"], [-1, 0])
            installed = yaml.safe_load((target / "thermal_analysis.yaml").read_text())
            self.assertEqual(installed["loading"], {"mode":"prescribed_temperature"})
            self.assertFalse(installed["include_radiation"])
            self.assertEqual(installed["transient"]["duration_s"], 150)
            self.assertEqual(installed["material"]["conductivity_w_mk"], 140)
            self.assertTrue(installed["visualization"]["make_gif"])
            self.assertEqual(json.loads((case / "_batch/stage_state.json").read_text())["stages"]["mesh"], mesh_record)
            for name, data in artifacts.items(): self.assertEqual((case / name).read_bytes(), data)
            self.assertTrue((backup / "thermal_analysis.yaml").is_file())
            existing_bytes = (target / "thermal_analysis.yaml").read_bytes()
            update_project(target, source)
            self.assertEqual((target / "thermal_analysis.yaml").read_bytes(), existing_bytes)
            self.assertEqual(load_config(case / "_batch/config.yaml")["solver"]["courant_number"], .3)


if __name__ == "__main__":
    unittest.main()
