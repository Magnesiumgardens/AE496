"""Restore endpoint calibration without losing local simulation inputs."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml

from pipeline.config import ConfigError, load_config, validate_config
from pipeline.migration import (adiabatic_config, endpoint_config, endpoint_state,
                                endpoint_thermal_config, migrate_config)
from scripts.update_project import update_project


ROOT = Path(__file__).resolve().parents[1]


def template():
    return yaml.safe_load((ROOT / "tests/fixtures/two_endpoint_config.yaml").read_text())


def old_adiabatic():
    cfg = template()
    cfg.pop("workflow")
    cfg = adiabatic_config(cfg)
    cfg["solver"].update(initialization="freestream", courant_number=.3,
                         flow_scheme="first-order-upwind", residual_thresholds={"energy": 1e-6})
    return cfg


class EndpointConfigurationTests(unittest.TestCase):
    def test_root_restores_cold_wall_hybrid_and_native_numerics(self):
        cfg = validate_config(template())
        self.assertEqual(cfg["workflow"], dict(strategy="two_endpoint", adiabatic_reference_iterations=600,
                                              min_delta_temperature_k=25, coordinate_tolerance_m=1e-9))
        self.assertEqual(cfg["thermal"], dict(mode="fixed_temperature", wall_temperature_k=300))
        self.assertEqual(cfg["flight"]["flow_direction"], [-1., 0.])
        self.assertEqual(cfg["solver"]["initialization"], "hybrid")
        self.assertIsNone(cfg["solver"]["courant_number"])
        self.assertIsNone(cfg["solver"]["flow_scheme"])
        self.assertEqual(set(cfg["solver"]["residual_thresholds"].values()), {1e-3})
        thermal = yaml.safe_load((ROOT / "tests/fixtures/endpoint_thermal_analysis.yaml").read_text())
        self.assertEqual(thermal["loading"]["mode"], "convective_from_cfd")
        self.assertEqual(thermal["transient"]["duration_s"], 200)
        self.assertNotIn("heat_transfer_coefficient_w_m2k", thermal["loading"])

    def test_endpoint_validation_rejects_wrong_primary_boundary_or_invalid_reference(self):
        bad = [dict(thermal={"mode": "adiabatic"}),
               dict(thermal={"mode": "fixed_temperature", "wall_temperature_k": 0}),
               dict(workflow={"strategy": "two_endpoint", "adiabatic_reference_iterations": 600.5}),
               dict(workflow={"strategy": "unrecognized"})]
        for update in bad:
            with self.subTest(update=update):
                cfg = template();cfg.update(update)
                with self.assertRaises(ConfigError):validate_config(cfg)
        cfg = template();cfg["thermal_export"]["heat_flux_into_wall_sign"] = 0
        with self.assertRaises(ConfigError):validate_config(cfg)

    def test_endpoint_overrides_are_validated_before_starting_fluent(self):
        for name, overrides in (("cold_wall_solver", {"courant_number":0}),
                                ("adiabatic_solver", {"max_iterations":1.5}),
                                ("adiabatic_solver", {"flow_scheme":""}),
                                ("adiabatic_solver", {"residual_thresholds":{"energy":float("nan")}})):
            with self.subTest(name=name, overrides=overrides):
                cfg = template();cfg["workflow"][name] = overrides
                with self.assertRaises(ConfigError):validate_config(cfg)
        for name in ("min_delta_temperature_k", "coordinate_tolerance_m"):
            cfg = template();cfg["workflow"][name] = 0
            with self.assertRaises(ConfigError):validate_config(cfg)

    def test_internal_adiabatic_configuration_roundtrips_without_changing_primary_model(self):
        from pipeline.endpoint_workflow import effective_cfg
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);cfg = template()
            cfg["project"]["work_dir"] = str(root / "nose")
            effective = effective_cfg(cfg, "adiabatic")
            path = root / "endpoint.yaml";path.write_text(yaml.safe_dump(effective))
            loaded = load_config(path)
            self.assertEqual(loaded["thermal"], {"mode":"adiabatic"})
            self.assertEqual(loaded["_study_work_dir"], str(root / "nose"))
            self.assertEqual(cfg["thermal"]["mode"], "fixed_temperature")
            loaded["_endpoint_kind"] = "cold_wall"
            with self.assertRaises(ConfigError):validate_config(loaded)

    def test_temperature_exports_cannot_escape_case_or_archive_the_case_itself(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);cfg = template()
            cfg["project"]["work_dir"] = str(root / "nose")
            path = root / "case.yaml"
            for output in ("../outside", str(root / "outside"), "."):
                with self.subTest(output=output):
                    cfg["temperature_export"]["output_directory"] = output
                    path.write_text(yaml.safe_dump(cfg))
                    with self.assertRaisesRegex(ConfigError, "inside the selected case"):load_config(path)
            cfg["temperature_export"]["output_directory"] = str(root / "nose/temperature")
            path.write_text(yaml.safe_dump(cfg))
            self.assertEqual(load_config(path)["temperature_export"]["output_directory"], str(root / "nose/temperature"))

    def test_direct_adiabatic_cases_remain_readable_without_strategy(self):
        cfg = old_adiabatic()
        self.assertEqual(validate_config(cfg)["thermal"], {"mode": "adiabatic"})
        cfg["thermal"] = dict(mode="fixed_temperature", wall_temperature_k=300)
        with self.assertRaises(ConfigError):validate_config(cfg)

    def test_migration_preserves_physical_inputs_and_replaces_ad_startup(self):
        cfg = old_adiabatic()
        cfg["flight"]["mach"] = 7.5
        cfg["geometry"]["input_file"] = "customCAD.x_t"
        cfg["mesh"]["body_min_size"] = .00012
        cfg["solver"]["max_iterations"] = 2400
        before = copy.deepcopy(cfg)
        result = endpoint_config(cfg)
        self.assertEqual(cfg, before)
        for section in ("flight", "geometry", "mesh", "physics", "project", "runtime"):
            self.assertEqual(result[section], cfg[section])
        self.assertEqual(result["solver"]["max_iterations"], 2400)
        self.assertEqual(result["solver"]["initialization"], "hybrid")
        self.assertIsNone(result["solver"]["courant_number"])
        self.assertEqual(result["thermal_export"]["heat_flux_into_wall_sign"], -1)
        self.assertEqual(endpoint_config(result), result)

    def test_already_endpoint_update_preserves_deliberate_numeric_and_sign_choices(self):
        cfg = template()
        cfg["solver"].update(initialization="freestream", courant_number=.4, flow_scheme="second-order-upwind")
        cfg["solver"]["residual_thresholds"]["energy"] = 1e-5
        cfg["thermal"]["wall_temperature_k"] = 350
        cfg["thermal_export"]["heat_flux_into_wall_sign"] = 1
        self.assertEqual(endpoint_config(cfg), cfg)

    def test_thermal_migration_preserves_duration_and_material_and_uses_derived_h(self):
        cfg = yaml.safe_load((ROOT / "thermal_analysis.yaml").read_text())
        cfg["loading"] = dict(mode="convection", heat_transfer_coefficient_w_m2k=500)
        cfg["transient"]["duration_s"] = 175
        cfg["material"]["conductivity_w_mk"] = 140
        cfg["include_radiation"] = False
        result = endpoint_thermal_config(cfg)
        self.assertEqual(result["transient"], cfg["transient"])
        self.assertEqual(result["material"], cfg["material"])
        self.assertFalse(result["include_radiation"])
        self.assertEqual(result["loading"], dict(mode="convective_from_cfd", min_delta_temperature_k=25))

    def test_migrate_backs_up_config_and_retains_mesh_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs/nose/_batch/config.yaml"
            path.parent.mkdir(parents=True)
            path.write_text(yaml.safe_dump(old_adiabatic()))
            original = path.read_bytes()
            mesh = dict(status="complete", fingerprint="mesh-hash", provenance="verified")
            state = dict(stages=dict(mesh=mesh, solver=dict(status="complete"), thermal=dict(status="complete")))
            state_path = path.parent / "stage_state.json"
            state_path.write_text(json.dumps(state))
            self.assertTrue(migrate_config(path))
            self.assertFalse(migrate_config(path))
            restored = json.loads(state_path.read_text())
            self.assertEqual(restored["stages"]["mesh"], mesh)
            self.assertTrue(all(record["status"] == "stale" for stage, record in restored["stages"].items() if stage != "mesh"))
            backups = list(path.parent.glob("config_before_single_solve_*.yaml"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)


class EndpointInstallerTests(unittest.TestCase):
    def test_authenticated_git_install_preserves_artifacts_and_custom_thermal_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source", Path(directory) / "target"
            source.mkdir();target.mkdir()
            for name in ("config.yaml", "thermal_analysis.yaml", "batch.yaml"):
                (source / name).write_bytes((ROOT / name).read_bytes())
            (source / "study.py").write_text("# application fixture\n")
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "add", "."], check=True)
            subprocess.run(["git", "-C", str(source), "-c", "user.name=Endpoint Test",
                            "-c", "user.email=endpoint@example.invalid", "commit", "-qm", "fixture"], check=True)
            (target / "study.py").write_text("# previous application\n")
            case = target / "runs/nose"
            (case / "_batch").mkdir(parents=True)
            old = old_adiabatic();old["flight"]["mach"] = 7.5
            old["mesh"]["body_min_size"] = .00012
            for path in (target / "config.yaml", case / "_batch/config.yaml"):
                path.write_text(yaml.safe_dump(old))
            mesh = dict(status="complete", fingerprint="retained-mesh")
            (case / "_batch/stage_state.json").write_text(json.dumps(dict(stages=dict(mesh=mesh, solver=dict(status="complete")))))
            artifacts = {name: b"existing " + name.encode() for name in ("02_mesh.msh.h5", "04_final.cas.h5", "04_final.dat.h5")}
            for name, data in artifacts.items():(case / name).write_bytes(data)
            thermal = yaml.safe_load((ROOT / "thermal_analysis.yaml").read_text())
            thermal["loading"] = dict(mode="convection", heat_transfer_coefficient_w_m2k=None)
            thermal["transient"]["duration_s"] = 175
            thermal["material"]["conductivity_w_mk"] = 140
            (target / "thermal_analysis.yaml").write_text(yaml.safe_dump(thermal))
            backup = update_project(target, source)
            migrated = yaml.safe_load((case / "_batch/config.yaml").read_text())
            self.assertEqual(migrated["workflow"]["strategy"], "single_solve")
            self.assertEqual(migrated["flight"]["mach"], 7.5)
            self.assertEqual(migrated["mesh"]["body_min_size"], .00012)
            restored_thermal = yaml.safe_load((target / "thermal_analysis.yaml").read_text())
            self.assertEqual(restored_thermal["transient"]["duration_s"], 175)
            self.assertEqual(restored_thermal["material"]["conductivity_w_mk"], 140)
            self.assertEqual(restored_thermal["loading"]["mode"], "prescribed_temperature")
            self.assertNotIn("heat_transfer_coefficient_w_m2k", restored_thermal["loading"])
            self.assertEqual(json.loads((case / "_batch/stage_state.json").read_text())["stages"]["mesh"], mesh)
            for name, data in artifacts.items():self.assertEqual((case / name).read_bytes(), data)
            self.assertTrue((backup / "thermal_analysis.yaml").is_file())
            # A later code-only update must preserve newly chosen endpoint inputs.
            migrated["solver"]["courant_number"] = .6
            (case / "_batch/config.yaml").write_text(yaml.safe_dump(migrated))
            thermal_bytes = (target / "thermal_analysis.yaml").read_bytes()
            update_project(target, source)
            self.assertEqual((target / "thermal_analysis.yaml").read_bytes(), thermal_bytes)
            self.assertEqual(yaml.safe_load((case / "_batch/config.yaml").read_text())["solver"]["courant_number"], .6)


if __name__ == "__main__":
    unittest.main()
