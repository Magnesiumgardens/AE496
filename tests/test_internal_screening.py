"""Internal screening contracts; no licensed Ansys engine is simulated here."""
import copy
import json
import shutil
from pathlib import Path
import tempfile
import sys
from types import ModuleType
import unittest
from unittest.mock import patch, Mock

import numpy as np
import yaml

from pipeline.internal_screening import (MODEL_VERSION, compare_internal, probe_screening,
                                         screening_settings, _signature, native_launch_options)
from pipeline.aerothermal_loading import integrated_powers
from pipeline.thermal_energy import prepare_solid_energy
from pipeline.thermal import solve_thermal, transient_schedule
from pipeline.thermal_loading import ThermalAnalysisError
from pipeline.thermal_materials import from_material_config, apply_material
from pipeline.utils import atomic_json, file_sha256
from test_aerothermal_thermal import load_fixture, quadratic_boundary, settings
from test_thermal_radiation import SurfaceMeshAdapter

ROOT = Path(__file__).resolve().parents[1]


class InternalRunnerTests(unittest.TestCase):
    def test_native_execution_defaults_to_shared_memory_and_accepts_single_thread_override(self):
        self.assertEqual(native_launch_options({"processors":4}),
                         ("smp", {"nproc":4, "additional_switches":"-smp"}))
        self.assertEqual(native_launch_options({"processors":4}, processors=1),
                         ("smp", {"nproc":1, "additional_switches":"-smp"}))
        self.assertEqual(native_launch_options({"mapdl_parallel_mode":"dmp", "processors":2}),
                         ("dmp", {"nproc":2, "additional_switches":"-dis"}))
        for count in (True, 0, -1, 1.5, float("inf")):
            with self.subTest(count=count), self.assertRaises(ThermalAnalysisError):
                native_launch_options({}, processors=count)
        with self.assertRaises(ThermalAnalysisError):
            native_launch_options({"mapdl_parallel_mode":"unknown"})

    def test_reference_input_requires_both_geometric_and_material_choices(self):
        path = ROOT / "internal_screening.yaml"
        with self.assertRaisesRegex(ThermalAnalysisError, "probe"):
            screening_settings(path)
        with self.assertRaisesRegex(ThermalAnalysisError, "material_axes"):
            screening_settings(path, depth_mode="axial_tip")
        _, _, law, config, _ = screening_settings(path, depth_mode="axial_tip", material_axes=["x", "z"])
        self.assertEqual(config["duration"], 200.)
        self.assertEqual(config["depth_m"], .01)
        self.assertEqual(config["limit_K"], 1800.)
        self.assertEqual(law.density, 1600.)
        self.assertLess(law.conductivity(1000, "z"), law.conductivity(1000, "x"))

    def test_crossing_is_not_interpolated_from_changing_hottest_probe(self):
        time = [0., 100., 200.]
        # The hottest probe changes at t=100. The max envelope would predict
        # 120s, while neither probe crosses before 166 2/3s.
        values = np.array([[300., 300.], [1700., 1000.], [1200., 2200.]])
        result, each = probe_screening(time, values, [{"probe_id":"a"}, {"probe_id":"b"}], 1800.)
        self.assertAlmostEqual(result["time_to_limit_s"], 100 + 800 / 12)
        self.assertEqual(result["critical_probe_id"], "b")
        self.assertEqual(each[0]["status"], "not_reached_within_duration")
        self.assertEqual(result["peak_temperature_K"], 2200.)

    def test_duration_override_changes_the_physical_exposure_without_editing_config(self):
        path = ROOT / "internal_screening.yaml"
        original = path.read_bytes()
        _,_,_,settings,_ = screening_settings(path,depth_mode="axial_tip",material_axes=["x","z"],duration_s=170.)
        self.assertEqual(settings["duration"],170.)
        self.assertEqual(path.read_bytes(),original)
        with self.assertRaises(ThermalAnalysisError):
            screening_settings(path,depth_mode="axial_tip",material_axes=["x","z"],duration_s=0.)

    def test_native_tabulated_material_retains_convection_radiation_and_free_solid_temperature(self):
        load = load_fixture([[0., 0.], [1., 0.], [0., 1.]])
        numbers, xyz = quadratic_boundary(load)
        engine = SurfaceMeshAdapter(numbers, xyz, lines=(1, 2, 3))
        _, _, config = settings(duration=200.)
        config["material_axes_2d"] = ["x", "z"]
        load.material_law = from_material_config(yaml.safe_load((ROOT / "materials/reference_carbon_carbon_2d.yaml").read_text()))
        solve_thermal(engine, load, config, .1, save_nodal_temperatures_only=True,
                      model_database_file="internal_model")
        calls = engine.calls
        self.assertTrue(any(n == "mpdata" and a[:2] == ("KXX", 1) for n, a, _ in calls))
        self.assertTrue(any(n == "mpdata" and a[:2] == ("KYY", 1) and a[3] == 3.89 for n, a, _ in calls))
        self.assertTrue(any(n == "mpdata" and a[:2] == ("C", 1) for n, a, _ in calls))
        self.assertTrue(any(n == "sfl" and a[1] == "CONV" for n, a, _ in calls))
        self.assertEqual([a for n, a, _ in calls if n == "d"], [(7, "TEMP", 3.)])
        self.assertEqual(load.solid_triangles.shape, (1, 6))
        self.assertEqual([a for n,a,k in calls if n == "outres"],[("ALL","NONE"),("NSOL","ALL")])
        save_index = next(i for i,(n,a,k) in enumerate(calls) if n == "save")
        self.assertEqual(calls[save_index][1],("internal_model","db"))
        self.assertLess(save_index, next(i for i,(n,a,k) in enumerate(calls) if n == "solve"))

    def test_startup_refinement_continues_one_state_and_reads_both_native_load_steps(self):
        load = load_fixture([[0., 0.], [1., 0.], [0., 1.]])
        numbers, xyz = quadratic_boundary(load)
        class MultiStepAdapter(SurfaceMeshAdapter):
            def __getattr__(self, name):
                if name != "set":return super().__getattr__(name)
                def select(*args, **kwargs):
                    self.calls.append((name, args, kwargs))
                    if args != ("LAST",):
                        if "nset" not in kwargs:raise AssertionError("Results must use global indices across load steps")
                        self.active_set = kwargs["nset"]
                return select
        engine = MultiStepAdapter(numbers, xyz, lines=(1,2,3))
        engine.post_processing.nsets = 4
        engine.post_processing.time_values = [.05,5.,5.05,200.]
        engine.post_processing.nodal_temperature = lambda:np.full(6,[310.,400.,410.,500.][engine.active_set-1])
        _, _, config = settings(duration=200.)
        config.update(startup_duration=5., startup_dtmax=.05, dtmin=.000001, dt0=.001, dtmax=.5)
        _, _, temperatures, times, _ = solve_thermal(engine,load,config,.1)
        self.assertEqual(times,[.05,5.,5.05,200.])
        self.assertEqual([float(v[0]) for v in temperatures],[310.,400.,410.,500.])
        self.assertEqual([a for n,a,k in engine.calls if n == "time"],[(5.,),(200.,)])
        self.assertEqual([a for n,a,k in engine.calls if n == "deltim"],[(.001,.000001,.05),(.05,.000001,.5)])
        self.assertEqual(sum(n == "solve" for n,a,k in engine.calls),2)
        self.assertEqual(sum(n == "tunif" for n,a,k in engine.calls),1)
        self.assertEqual(sum(n == "antype" for n,a,k in engine.calls),1)
        self.assertEqual(sum(n == "sftran" for n,a,k in engine.calls),1)
        self.assertEqual([a for n,a,k in engine.calls if n == "d"],[(7,"TEMP",3.)])

    def test_short_exposure_has_no_duplicate_or_zero_length_phase(self):
        config = dict(duration=2., dt0=.001, dtmin=.000001, dtmax=.5,
                      startup_duration=5., startup_dtmax=.05)
        self.assertEqual(transient_schedule(config),[(2.,.001,.05)])
        config["startup_duration"] = 0.
        self.assertEqual(transient_schedule(config),[(2.,.001,.5)])
        config["startup_duration"] = .0000001
        with self.assertRaisesRegex(ThermalAnalysisError,"Startup phase duration"):
            transient_schedule(config)

    def test_different_noses_can_share_physics_signature_without_sharing_geometry_or_mesh(self):
        calibration = dict(conditions=dict(flight={"mach":8}, physics={"viscous_model":"laminar"},
            thermal={"mode":"radiative_equilibrium"}, mesh={"size":1}, geometry={"name":"a"}),
            radiation={"emissivity":.45}, thermal_scope={"rear_boundary":"insulated"})
        other = copy.deepcopy(calibration)
        other["conditions"].update(mesh={"size":2}, geometry={"name":"b"})
        self.assertEqual(_signature(calibration, {}, {}), _signature(other, {}, {}))
        other["conditions"]["flight"]["mach"] = 9
        self.assertNotEqual(_signature(calibration, {}, {}), _signature(other, {}, {}))

    def test_report_only_rejects_stale_material_and_discovers_case_without_old_thermal_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); runs = root / "runs"; case = runs / "mk3"
            (case / "_batch").mkdir(parents=True)
            (case / "_batch/config.yaml").write_text("{}")
            output = case / "internal_analysis"; output.mkdir()
            current = dict(model_version=MODEL_VERSION, material_file_sha256="current", settings={},
                           comparison_signature={"same":"physics"}, source_geometry_sha256="CAD")
            atomic_json(output / "internal_summary.json", dict(status="complete", **{**current, "material_file_sha256":"old"}))
            comparison = runs / "comparison/internal_heating"; comparison.mkdir(parents=True)
            stale_plot = comparison / "internal_temperature_history_comparison.png"; stale_plot.write_bytes(b"stale")
            with patch("pipeline.internal_screening._case_inputs", return_value=(None,None,None,None,None,current)), \
                 patch("pipeline.internal_screening.run_internal_case") as native:
                result = compare_internal(runs, ROOT / "internal_screening.yaml", report_only=True,
                                          depth_mode="axial_tip", material_axes=["x","z"])
            native.assert_not_called()
            self.assertEqual(result["status"], "unavailable")
            self.assertIn("material", result["excluded"][0]["reason"])
            self.assertFalse(stale_plot.exists())

    def test_selected_cad_must_be_the_calibrated_geometry_before_native_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); runs = root / "runs"; runs.mkdir()
            cad = root / "geometry_batch"; cad.mkdir(); (cad / "mk4.x_t").write_bytes(b"new shape")
            current = dict(source_geometry_sha256="old calibrated shape", comparison_signature={})
            with patch("pipeline.internal_screening._case_inputs", return_value=(None,None,None,None,None,current)), \
                 patch("pipeline.internal_screening.run_internal_case") as native:
                result = compare_internal(runs, ROOT / "internal_screening.yaml",
                                          depth_mode="axial_tip", material_axes=["x","z"])
            native.assert_not_called()
            self.assertIn("CAD differs", result["excluded"][0]["reason"])

    def test_saved_physical_time_pipeline_and_report_regeneration_without_native_run(self):
        # A scalar implicit enthalpy balance supplies a transparent uniform-field
        # fixture. This tests result plumbing and conservation, not native FEA.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); case = root / "runs/mk3"; (case / "_batch").mkdir(parents=True)
            (case / "_batch/config.yaml").write_text("{}")
            boundary = root / "source.csv"; boundary.write_text("calibration fixture")
            config_path = root / "screen.yaml"
            options = yaml.safe_load((ROOT / "internal_screening.yaml").read_text())
            options.update(material_file=str(ROOT / "materials/reference_carbon_carbon_2d.yaml"),
                depth_mode="axial_tip", material_axes_2d=["x","z"], duration_s=.2,
                initial_time_step_s=.05, max_time_step_s=.1, startup_duration_s=0.)
            config_path.write_text(yaml.safe_dump(options))
            opts, material, law, config, material_path = screening_settings(config_path)
            config.update(emissivity=.9, trad=3., nose_direction=[1.,0.])
            snapshot = dict(model_version=MODEL_VERSION, settings=config, material=material,
                material_file_sha256=file_sha256(material_path), screening_config_sha256=file_sha256(config_path),
                boundary_csv_sha256=file_sha256(boundary), source_sha256={"fixture":"source"},
                source_geometry_sha256="fixture", comparison_signature={"fixture":"common assumptions"})
            cfg = {"project":{"work_dir":str(case)}, "reporting":{"nose_direction":[1.,0.]}}
            def fixture_solve(engine, load, settings, mesh_size, *, save_nodal_temperatures_only, model_database_file):
                self.assertTrue(save_nodal_temperatures_only)
                self.assertEqual(model_database_file,"internal_model")
                numbers, xyz = quadratic_boundary(load)
                load.solid_triangles = numbers[None, :]
                load.solid_energy_quadrature = prepare_solid_energy(numbers, xyz, load.solid_triangles)
                load.native_material = apply_material(engine, law, settings["material_axes_2d"])
                load.radiation_boundary = {"status":"test transport"}
                values, previous = [], settings["initial_t"]
                for dt in (.1, .1):
                    low, high = previous, 1500.
                    for _ in range(70):
                        current = (low + high) / 2
                        mass = settings["rho"] * load.solid_energy_quadrature.area_m2
                        energy = mass * (law.enthalpy_change(current) - law.enthalpy_change(previous))
                        power = integrated_powers(load, np.full(6,current), settings)[2]
                        if energy > dt * power:high = current
                        else:low = current
                    previous = (low + high) / 2
                    values.append(np.full(6, previous))
                return numbers, xyz, values, [.1,.2], 1
            native = ModuleType("ansys.mapdl.core"); native.launch_mapdl = Mock(return_value=Mock())
            inputs = (cfg, opts, law, material_path, boundary, snapshot)
            output = case / "internal_analysis"; output.mkdir()
            stale_result = output / "internal_nose.rth"; stale_result.write_bytes(b"old combined result")
            stale_model = output / "internal_nose.db"; stale_model.write_bytes(b"old model")
            with patch.dict(sys.modules, {"ansys.mapdl.core":native}), \
                 patch("pipeline.internal_screening._case_inputs", return_value=inputs), \
                 patch("pipeline.internal_screening.read_load", side_effect=lambda _:load_fixture([[0.,-.1],[.1,0.],[0.,.1]])), \
                 patch("pipeline.internal_screening.solve_thermal", side_effect=fixture_solve):
                first = compare_internal(root / "runs", config_path, processors=1)
            self.assertEqual(first["status"], "complete")
            native.launch_mapdl.assert_called_once()
            launched = native.launch_mapdl.call_args.kwargs
            self.assertEqual(launched["nproc"],1)
            self.assertEqual(launched["additional_switches"],"-smp")
            run_directory = Path(launched["run_location"])
            self.assertEqual(run_directory.parent,output / "native_runs")
            self.assertTrue(run_directory.is_dir())
            self.assertEqual(stale_result.read_bytes(),b"old combined result")
            self.assertEqual(stale_model.read_bytes(),b"old model")
            receipt = json.loads((case / "internal_analysis/internal_summary.json").read_text())
            self.assertEqual(receipt["native_runtime"]["processors"],1)
            self.assertEqual(receipt["native_runtime"]["native_directory"],str(run_directory))
            self.assertEqual(receipt["energy_balance"]["status"], "passed")
            self.assertEqual(receipt["screening"]["status"], "not_reached_within_duration")
            self.assertAlmostEqual(receipt["probe_definitions"][0]["x_m"], .09)
            with patch("pipeline.internal_screening._case_inputs", return_value=inputs), \
                 patch("pipeline.internal_screening.run_internal_case") as rerun:
                second = compare_internal(root / "runs", config_path, report_only=True)
            rerun.assert_not_called()
            self.assertEqual(second["rows"], first["rows"])
            partner = root / "runs/mk4"
            (partner / "_batch").mkdir(parents=True)
            (partner / "_batch/config.yaml").write_text("{}")
            shutil.copytree(output, partner / "internal_analysis")
            other = copy.deepcopy(receipt)
            other.update(case="mk4", screening={**receipt["screening"], "design":"mk4"})
            atomic_json(partner / "internal_analysis/internal_summary.json", other)
            unchanged = {p:p.read_bytes() for p in (output / "internal_summary.json", partner / "internal_analysis/internal_summary.json")}
            def per_case_inputs(path, *args):
                return ({"project":{"work_dir":str(Path(path).parent.parent)}}, *inputs[1:])
            with patch("pipeline.internal_screening._case_inputs", side_effect=per_case_inputs), \
                 patch("pipeline.internal_screening.run_internal_case") as rerun:
                swapped = compare_internal(root / "runs",config_path,report_only=True,swap_labels=["mk3","mk4"])
            rerun.assert_not_called()
            self.assertEqual(swapped["case_count"],2)
            self.assertEqual([(r["source_case"],r["design"]) for r in swapped["rows"]],[("mk3","mk4"),("mk4","mk3")])
            self.assertEqual(swapped["display_name_overrides"],{"mk3":"mk4","mk4":"mk3"})
            for p,data in unchanged.items():self.assertEqual(p.read_bytes(),data)
            shutil.rmtree(partner)
            for name in ("internal_temperature_history_comparison.png", "peak_internal_temperature_comparison.png",
                         "time_to_internal_limit_comparison.png"):
                self.assertGreater((root / "runs/comparison/internal_heating" / name).stat().st_size, 1000)

            # A server abort and a failed cleanup must retain the original cause,
            # use a new native folder, and never promote stale saved outputs.
            native.launch_mapdl.return_value.exit.side_effect = RuntimeError("connection refused during cleanup")
            with patch.dict(sys.modules, {"ansys.mapdl.core":native}), \
                 patch("pipeline.internal_screening._case_inputs", return_value=inputs), \
                 patch("pipeline.internal_screening.read_load", side_effect=lambda _:load_fixture([[0.,-.1],[.1,0.],[0.,.1]])), \
                 patch("pipeline.internal_screening.solve_thermal", side_effect=RuntimeError("SIG$ABORT")):
                aborted = compare_internal(root / "runs",config_path,processors=1)
            self.assertEqual(aborted["case_count"],0)
            self.assertEqual(aborted["excluded"][0]["reason"],"SIG$ABORT")
            failed = json.loads((output / "internal_summary.json").read_text())
            self.assertEqual(failed["status"],"failed")
            self.assertEqual(failed["error"],"SIG$ABORT")
            self.assertNotEqual(failed["native_runtime"]["native_directory"],str(run_directory))
            self.assertEqual(stale_result.read_bytes(),b"old combined result")


if __name__ == "__main__":unittest.main()
