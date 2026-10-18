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
FORCE_HISTORY_DEFAULTS = dict(enabled=True, sample_every_iterations=25,
    recent_samples=5, min_samples=5, relative_tolerance=.01,
    absolute_drag_tolerance_N_m=1e-6, require_stable=True)


def template():
    return yaml.safe_load((ROOT / "config.yaml").read_text())


def endpoint_template():
    return yaml.safe_load((ROOT / "tests/fixtures/two_endpoint_config.yaml").read_text())


def previous_single_template():
    """Exact solver controls shipped before the stabilization correction."""
    cfg = template()
    cfg["solver"].update(initialization="hybrid", courant_number=None,
                         flow_scheme=None, max_iterations=1200)
    return cfg


def expected_aero(previous):
    """Monitoring defaults augment aerodynamic inputs without choosing a width."""
    result = copy.deepcopy(previous)
    result["force_history"] = {**FORCE_HISTORY_DEFAULTS, **result.get("force_history", {})}
    return result


class SingleConfigurationTests(unittest.TestCase):
    def test_default_is_one_radiating_wall_with_calibrated_solved_surface_loading(self):
        cfg = validate_config(template())
        self.assertEqual(cfg["workflow"], {"strategy": "single_solve"})
        self.assertEqual(cfg["thermal"], dict(mode="radiative_equilibrium", wall_temperature_k=300,
            emissivity=.9, external_radiation_temperature_k=3))
        self.assertEqual(cfg["flight"]["flow_direction"], [-1, 0])
        self.assertEqual(cfg["runtime"]["geometry_launch_mode"], "spaceclaim")
        self.assertEqual(cfg["solver"]["initialization"], "freestream")
        self.assertEqual(cfg["solver"]["courant_number"], .3)
        self.assertEqual(cfg["solver"]["flow_scheme"], "first-order-upwind")
        self.assertEqual(cfg["solver"]["max_iterations"], 4800)
        self.assertEqual(set(cfg["solver"]["residual_thresholds"].values()), {1e-3})
        self.assertNotIn("thermal_export", cfg)
        self.assertTrue(cfg["calibration"]["enabled"])
        self.assertEqual(cfg["calibration"]["cold_wall_temperature_k"], 300)
        self.assertEqual(cfg["calibration"]["heat_flux_into_wall_sign"], -1)
        self.assertEqual(cfg["aero"]["force_history"], FORCE_HISTORY_DEFAULTS)
        self.assertIsNone(cfg["aero"]["reference_area_per_unit_span_m"])
        thermal = yaml.safe_load((ROOT / "thermal_analysis.yaml").read_text())
        self.assertEqual(thermal["loading"], {"mode": "aerodynamic_surface_load"})
        self.assertTrue(thermal["include_radiation"])
        self.assertEqual(thermal["initial_temperature_k"], 300)
        self.assertNotIn("emissivity", thermal)
        self.assertNotIn("radiation_temperature_k", thermal)
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
        for name in ("solver", "geometry", "flight", "mesh", "runtime", "physics", "reporting", "project"):
            self.assertEqual(result[name], before[name])
        self.assertEqual(result["aero"], expected_aero(before["aero"]))
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

    def test_generated_hybrid_native_profile_is_repaired_for_both_strategies(self):
        for previous in (previous_single_template(), endpoint_template()):
            with self.subTest(strategy=previous["workflow"]["strategy"]):
                before = copy.deepcopy(previous)
                result = single_solve_config(previous)
                self.assertEqual(previous, before)
                self.assertEqual(result["solver"]["initialization"], "freestream")
                self.assertEqual(result["solver"]["courant_number"], .3)
                self.assertEqual(result["solver"]["flow_scheme"], "first-order-upwind")
                self.assertEqual(result["solver"]["max_iterations"], 4800)
                for name in ("geometry", "mesh", "flight", "runtime", "physics"):
                    self.assertEqual(result[name], before[name])
                self.assertEqual(result["aero"], expected_aero(before["aero"]))
                if before["thermal"]["mode"] == "radiative_equilibrium":
                    self.assertEqual(result["thermal"], before["thermal"])
                self.assertEqual(single_solve_config(result), result)
                validate_config(result)

    def test_missing_solver_controls_and_legacy_hybrid_flag_use_stable_defaults(self):
        for controls in ({}, {"hybrid_initialize": True},
                         {"initialization": "hybrid", "courant_number": None},
                         {"hybrid_initialize": True, "flow_scheme": None}):
            with self.subTest(controls=controls):
                cfg = previous_single_template()
                cfg["solver"] = copy.deepcopy(controls)
                result = single_solve_config(cfg)
                self.assertEqual(result["solver"]["initialization"], "freestream")
                self.assertEqual(result["solver"]["courant_number"], .3)
                self.assertEqual(result["solver"]["flow_scheme"], "first-order-upwind")
                self.assertEqual(result["solver"]["max_iterations"], 4800)
                self.assertNotIn("hybrid_initialize", result["solver"])
                self.assertEqual(single_solve_config(result), result)

    def test_profile_repair_keeps_explicit_iteration_budget_and_radiation_environment(self):
        cfg = previous_single_template()
        cfg["solver"]["max_iterations"] = 6400
        cfg["thermal"].update(emissivity=.7, external_radiation_temperature_k=250,
                              wall_temperature_k=340)
        before = copy.deepcopy(cfg)
        result = single_solve_config(cfg)
        self.assertEqual(result["solver"]["max_iterations"], 6400)
        self.assertEqual(result["solver"]["initialization"], "freestream")
        self.assertEqual(result["solver"]["courant_number"], .3)
        self.assertEqual(result["thermal"], before["thermal"])
        self.assertEqual(result["flight"], before["flight"])
        self.assertEqual(result["mesh"], before["mesh"])

    def test_explicit_numerical_overrides_are_not_reset_by_profile_repair(self):
        examples = (
            {"courant_number": .2},
            {"flow_scheme": "first-order-upwind"},
            {"initialization": "freestream"},
            {"initialization": "freestream", "courant_number": None, "flow_scheme": None},
            {"courant_number": 1.0, "flow_scheme": "second-order-upwind", "max_iterations": 2400},
        )
        for override in examples:
            with self.subTest(override=override):
                cfg = previous_single_template()
                cfg["solver"].update(override)
                result = single_solve_config(cfg)
                for key in ("initialization", "courant_number", "flow_scheme", "max_iterations"):
                    self.assertEqual(result["solver"][key], cfg["solver"][key])
                self.assertEqual(single_solve_config(result), result)

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
        self.assertEqual(result["loading"], {"mode": "aerodynamic_surface_load"})
        self.assertTrue(result["include_radiation"])
        self.assertNotIn("emissivity", result)
        self.assertNotIn("radiation_temperature_k", result)
        self.assertEqual(single_solve_thermal_config(result), result)

    def test_prescribed_loading_migration_keeps_initial_state_400_s_and_material(self):
        old = yaml.safe_load((ROOT / "tests/fixtures/prescribed_reference/thermal_analysis.yaml").read_text())
        old["initial_temperature_k"] = 300
        old["transient"]["duration_s"] = 400
        old["material"].update(conductivity_w_mk=145, density_kg_m3=19100,
                               specific_heat_j_kgk=135)
        old["emissivity"] = .6
        old["radiation_temperature_k"] = 290
        before = copy.deepcopy(old)
        migrated = single_solve_thermal_config(old,
            yaml.safe_load((ROOT / "thermal_analysis.yaml").read_text()))
        self.assertEqual(old, before)
        for name in ("initial_temperature_k", "transient", "material", "visualization"):
            self.assertEqual(migrated[name], before[name])
        self.assertEqual(migrated["loading"], {"mode": "aerodynamic_surface_load"})
        self.assertTrue(migrated["include_radiation"])
        # Radiation is defined once by the CFD configuration, not stale TA defaults.
        self.assertNotIn("emissivity", migrated)
        self.assertNotIn("radiation_temperature_k", migrated)
        self.assertEqual(single_solve_thermal_config(migrated), migrated)

    def test_adding_calibration_and_force_monitor_preserves_current_re_primary_stage_records(self):
        old = yaml.safe_load((ROOT / "tests/fixtures/prescribed_reference/config.yaml").read_text())
        migrated = single_solve_config(old)
        self.assertTrue(migrated["calibration"]["enabled"])
        records = {name: dict(status="complete", fingerprint=f"{name}-fingerprint",
                             outputs={f"{name}-output": f"{name}-sha"})
                   for name in ("mesh", "solver", "temperature_export", "aero", "thermal")}
        state = dict(version=4, strategy="single_solve", stages=records)
        before = copy.deepcopy(state)
        result = single_solve_state(state, old, migrated)
        self.assertEqual(state, before)
        for name in ("mesh", "solver", "temperature_export"):
            self.assertEqual(result["stages"][name], records[name])
        for name in ("aero", "calibration", "thermal"):
            self.assertEqual(result["stages"][name]["status"], "stale")
        self.assertEqual(single_solve_config(migrated), migrated)

    def test_force_monitor_only_migration_preserves_all_thermal_and_primary_cfd_records(self):
        old = template()
        old["aero"].pop("force_history", None)
        old["calibration"]["cold_wall_temperature_k"] = 325.
        before = copy.deepcopy(old)
        migrated = single_solve_config(old)
        self.assertEqual(old, before)
        for name in ("geometry", "mesh", "flight", "physics", "thermal", "solver", "calibration"):
            self.assertEqual(migrated[name], old[name])
        self.assertEqual(migrated["aero"], expected_aero(old["aero"]))
        self.assertIsNone(migrated["aero"]["reference_area_per_unit_span_m"])
        records = {name: dict(status="complete", fingerprint=f"{name}-fingerprint",
                             output_sha256={f"{name}-output": f"{name}-sha"})
                   for name in ("mesh", "solver", "temperature_export", "aero", "calibration", "thermal")}
        state = dict(version=4, strategy="single_solve", stages=records)
        migrated_state = single_solve_state(state, old, migrated)
        for name in ("mesh", "solver", "temperature_export", "calibration", "thermal"):
            self.assertEqual(migrated_state["stages"][name], records[name])
        self.assertEqual(migrated_state["stages"]["aero"]["status"], "stale")
        self.assertEqual(state["stages"]["aero"]["status"], "complete")

    def test_monitor_migration_keeps_explicit_sampling_and_stability_overrides(self):
        old = template()
        old["aero"]["force_history"] = dict(enabled=True, sample_every_iterations=10,
            recent_samples=8, min_samples=10, relative_tolerance=.005,
            absolute_drag_tolerance_N_m=1e-5, require_stable=False)
        migrated = single_solve_config(old)
        self.assertEqual(migrated["aero"], old["aero"])
        self.assertIsNone(migrated["aero"]["reference_area_per_unit_span_m"])
        self.assertEqual(single_solve_config(migrated), migrated)

    def test_changed_re_wall_or_solver_still_invalidates_primary_stage_records(self):
        old = template()
        for section, field, value in (("thermal", "emissivity", .7),
                                      ("solver", "courant_number", .2)):
            with self.subTest(section=section):
                changed = copy.deepcopy(old); changed[section][field] = value
                state = dict(stages={name: dict(status="complete", fingerprint=name)
                             for name in ("mesh", "solver", "temperature_export", "aero", "thermal")})
                result = single_solve_state(state, old, changed)
                self.assertEqual(result["stages"]["mesh"], state["stages"]["mesh"])
                for name in ("solver", "temperature_export", "aero", "calibration", "thermal"):
                    self.assertEqual(result["stages"][name]["status"], "stale")

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
    def test_monitor_only_git_update_retains_thermal_settings_and_results_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source", Path(directory) / "target"
            source.mkdir(); target.mkdir()
            for name in ("config.yaml", "thermal_analysis.yaml", "batch.yaml"):
                (source / name).write_bytes((ROOT / name).read_bytes())
            (source / "study.py").write_text("# current source fixture\n")
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "add", "."], check=True)
            subprocess.run(["git", "-C", str(source), "-c", "user.name=Force Monitor Test",
                "-c", "user.email=monitor@example.invalid", "commit", "-qm", "fixture"], check=True)
            (target / "study.py").write_text("# previous source\n")
            previous = template(); previous["aero"].pop("force_history", None)
            previous["calibration"]["cold_wall_temperature_k"] = 325.
            case = target / "runs/blunt_1"; (case / "_batch").mkdir(parents=True)
            for path in (target / "config.yaml", case / "_batch/config.yaml"):
                path.write_text(yaml.safe_dump(previous))
            thermal = yaml.safe_load((ROOT / "thermal_analysis.yaml").read_text())
            thermal["transient"]["duration_s"] = 400
            thermal["material"]["conductivity_w_mk"] = 145
            # Exact bytes ensure this aero update does not rewrite the load model,
            # material table, exposure or a user-reviewed comment.
            thermal_bytes = ("# User-reviewed thermal settings\n" + yaml.safe_dump(thermal)).encode()
            (target / "thermal_analysis.yaml").write_bytes(thermal_bytes)
            records = {name: dict(status="complete", fingerprint=f"{name}-fingerprint")
                for name in ("mesh", "solver", "temperature_export", "aero", "calibration", "thermal")}
            state_path = case / "_batch/stage_state.json"
            state_path.write_text(json.dumps(dict(version=4, strategy="single_solve", stages=records)))
            artifacts = {name: ("existing " + name).encode() for name in (
                "02_mesh.msh.h5", "04_final.cas.h5", "04_final.dat.h5", "aero_summary.json",
                "cfd/cold_calibration/04_final.dat.h5", "thermal_bc/surface_load_nodes.csv",
                "thermal_analysis/wall_temperature_history.csv")}
            for name, data in artifacts.items():
                path = case / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)

            update_project(target, source)

            self.assertEqual((target / "thermal_analysis.yaml").read_bytes(), thermal_bytes)
            installed = yaml.safe_load((case / "_batch/config.yaml").read_bytes())
            self.assertEqual(installed["aero"]["force_history"], FORCE_HISTORY_DEFAULTS)
            self.assertIsNone(installed["aero"]["reference_area_per_unit_span_m"])
            for name in ("thermal", "solver", "calibration", "mesh", "flight", "geometry"):
                self.assertEqual(installed[name], previous[name])
            state = json.loads(state_path.read_text())
            for name in ("mesh", "solver", "temperature_export", "calibration", "thermal"):
                self.assertEqual(state["stages"][name], records[name])
            self.assertEqual(state["stages"]["aero"]["status"], "stale")
            for name, data in artifacts.items(): self.assertEqual((case / name).read_bytes(), data)

    def test_git_installer_adds_calibration_without_reclassifying_current_re_results(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source", Path(directory) / "target"
            source.mkdir(); target.mkdir()
            for name in ("config.yaml", "thermal_analysis.yaml", "batch.yaml"):
                (source / name).write_bytes((ROOT / name).read_bytes())
            (source / "study.py").write_text("# current application fixture\n")
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "add", "."], check=True)
            subprocess.run(["git", "-C", str(source), "-c", "user.name=Calibration Test",
                "-c", "user.email=calibration@example.invalid", "commit", "-qm", "fixture"], check=True)
            (target / "study.py").write_text("# previous application\n")
            old = yaml.safe_load((ROOT / "tests/fixtures/prescribed_reference/config.yaml").read_text())
            old["thermal"].update(emissivity=.72, external_radiation_temperature_k=275)
            case = target / "runs/nose"; (case / "_batch").mkdir(parents=True)
            for path in (target / "config.yaml", case / "_batch/config.yaml"):
                path.write_text(yaml.safe_dump(old))
            thermal = yaml.safe_load((ROOT / "tests/fixtures/prescribed_reference/thermal_analysis.yaml").read_text())
            thermal["transient"]["duration_s"] = 400
            thermal["initial_temperature_k"] = 300
            thermal["material"].update(conductivity_w_mk=142, density_kg_m3=19100,
                                      specific_heat_j_kgk=136)
            (target / "thermal_analysis.yaml").write_text(yaml.safe_dump(thermal))
            records = {name: dict(status="complete", fingerprint=f"{name}-fingerprint",
                                 outputs={f"{name}-output": f"{name}-sha"})
                       for name in ("mesh", "solver", "temperature_export", "aero", "thermal")}
            state_path = case / "_batch/stage_state.json"
            state_path.write_text(json.dumps(dict(version=4, strategy="single_solve", stages=records)))
            artifacts = {name: b"existing RE output " + name.encode() for name in (
                "02_mesh.msh.h5", "04_final.cas.h5", "04_final.dat.h5", "aero_summary.json",
                "temperature/wall_temperatures.csv", "thermal_analysis/wall_temperature_history.csv")}
            for name, data in artifacts.items():
                path = case / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)

            update_project(target, source)

            installed = yaml.safe_load((case / "_batch/config.yaml").read_bytes())
            self.assertTrue(installed["calibration"]["enabled"])
            for name in ("geometry", "mesh", "flight", "physics", "thermal", "solver"):
                self.assertEqual(installed[name], old[name])
            self.assertEqual(installed["aero"], expected_aero(old["aero"]))
            state = json.loads(state_path.read_text())
            for name in ("mesh", "solver", "temperature_export"):
                self.assertEqual(state["stages"][name], records[name])
            self.assertEqual(state["stages"]["aero"]["status"], "stale")
            self.assertEqual(state["stages"]["calibration"]["status"], "stale")
            self.assertEqual(state["stages"]["thermal"]["status"], "stale")
            installed_thermal = yaml.safe_load((target / "thermal_analysis.yaml").read_bytes())
            for name in ("initial_temperature_k", "transient", "material"):
                self.assertEqual(installed_thermal[name], thermal[name])
            self.assertEqual(installed_thermal["loading"], {"mode": "aerodynamic_surface_load"})
            for name, data in artifacts.items():self.assertEqual((case / name).read_bytes(), data)
            retained_state = state_path.read_bytes()
            update_project(target, source)
            self.assertEqual(state_path.read_bytes(), retained_state)

    def test_git_installer_repairs_generated_root_and_case_profile_without_touching_mesh_or_results(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source", Path(directory) / "target"
            source.mkdir(); target.mkdir()
            for name in ("config.yaml", "thermal_analysis.yaml", "batch.yaml"):
                (source / name).write_bytes((ROOT / name).read_bytes())
            (source / "study.py").write_text("# replacement source\n")
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "add", "."], check=True)
            subprocess.run(["git", "-C", str(source), "-c", "user.name=Stable Test",
                "-c", "user.email=stable@example.invalid", "commit", "-qm", "fixture"], check=True)
            (target / "study.py").write_text("# existing source\n")
            cfg = previous_single_template()
            cfg["thermal"].update(emissivity=.85, external_radiation_temperature_k=275)
            case = target / "runs/blunt_1"; (case / "_batch").mkdir(parents=True)
            cfg_paths = (target / "config.yaml", case / "_batch/config.yaml")
            old_bytes = yaml.safe_dump(cfg).encode()
            for path in cfg_paths: path.write_bytes(old_bytes)
            mesh_record = {"status": "complete", "fingerprint": "verified-mesh",
                           "outputs": {"02_mesh.msh.h5": "mesh-sha"}}
            (case / "_batch/stage_state.json").write_text(json.dumps(dict(version=4,
                strategy="single_solve", stages=dict(mesh=mesh_record,
                    solver={"status": "complete", "fingerprint": "old-native-profile"}))))
            artifacts = {name: b"retain original " + name.encode() for name in (
                "02_mesh.msh.h5", "04_final.cas.h5", "04_final.dat.h5",
                "temperature/wall_temperatures.csv", "thermal_results/temperature_history.csv")}
            for name, data in artifacts.items():
                path = case / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
            thermal_bytes = (ROOT / "thermal_analysis.yaml").read_bytes()
            (target / "thermal_analysis.yaml").write_bytes(thermal_bytes)
            backup = update_project(target, source)
            for path in cfg_paths:
                installed = yaml.safe_load(path.read_bytes())
                self.assertEqual(installed["solver"]["initialization"], "freestream")
                self.assertEqual(installed["solver"]["courant_number"], .3)
                self.assertEqual(installed["solver"]["flow_scheme"], "first-order-upwind")
                self.assertEqual(installed["solver"]["max_iterations"], 4800)
                for name in ("geometry", "mesh", "flight", "runtime", "physics", "thermal"):
                    self.assertEqual(installed[name], cfg[name])
                self.assertEqual(installed["aero"], expected_aero(cfg["aero"]))
                self.assertEqual((backup / path.relative_to(target)).read_bytes(), old_bytes)
            state = json.loads((case / "_batch/stage_state.json").read_text())
            self.assertEqual(state["stages"]["mesh"], mesh_record)
            self.assertEqual(state["stages"]["solver"]["status"], "stale")
            self.assertEqual((target / "thermal_analysis.yaml").read_bytes(), thermal_bytes)
            self.assertEqual(yaml.safe_load(thermal_bytes)["transient"]["duration_s"], 200)
            for name, data in artifacts.items(): self.assertEqual((case / name).read_bytes(), data)
            updated_bytes = {path: path.read_bytes() for path in cfg_paths}
            updated_state = (case / "_batch/stage_state.json").read_bytes()
            update_project(target, source)
            for path, data in updated_bytes.items(): self.assertEqual(path.read_bytes(), data)
            self.assertEqual((case / "_batch/stage_state.json").read_bytes(), updated_state)
            for name, data in artifacts.items(): self.assertEqual((case / name).read_bytes(), data)

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
            self.assertEqual(installed["loading"], {"mode":"aerodynamic_surface_load"})
            self.assertTrue(installed["include_radiation"])
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
