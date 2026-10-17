"""Explicit equilibrium-T Dirichlet loading and all FE boundary-node mapping."""
import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from pipeline.convection_loading import validate_settings
from pipeline.prescribed_loading import PrescribedTemperatureLoad, interpolate_boundary_temperature, read_load
from pipeline.thermal import apply_prescribed_temperature, apply_surface_loads, run_thermal_analysis, save_results
from pipeline.thermal_loading import ThermalAnalysisError
from pipeline.utils import file_sha256

ROOT = Path(__file__).resolve().parents[1]
PRESCRIBED_FIXTURES = ROOT / "tests/fixtures/prescribed_reference"


def configurations(work):
    # This module deliberately tests the retained prescribed-temperature mode,
    # rather than inheriting whichever loading model the new template selects.
    cfd = yaml.safe_load((PRESCRIBED_FIXTURES / "config.yaml").read_text())
    cfd["project"] = {"name": work.name, "work_dir": str(work)}
    cfd["geometry"]["input_file"] = str(work.parent / "nose.x_t")
    cfd["workflow"] = {"strategy": "single_solve"}
    cfd["thermal"] = {"mode": "radiative_equilibrium", "emissivity": .9,
                      "external_radiation_temperature_k": 3., "wall_temperature_k": 300.}
    thermal = yaml.safe_load((PRESCRIBED_FIXTURES / "thermal_analysis.yaml").read_text())
    thermal["loading"] = {"mode": "prescribed_temperature"}
    thermal["include_radiation"] = False
    thermal["initial_temperature_k"] = 300.
    thermal["transient"].update(duration_s=2., initial_time_step_s=1., min_time_step_s=.1, max_time_step_s=2.)
    thermal["visualization"]["enabled"] = False
    return cfd, thermal


def node_csv(path, xy, temperatures):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["surface_order_index", "x_m", "y_m", "wall_temperature_K"])
        writer.writeheader()
        writer.writerows({"surface_order_index": i, "x_m": p[0], "y_m": p[1], "wall_temperature_K": t}
                        for i, (p, t) in enumerate(zip(xy, temperatures)))


class PrescribedBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.xy = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        self.load = PrescribedTemperatureLoad(self.xy, np.array([1000., 2000., 3000., 4000.]))

    def test_nodes_at_corners_and_mid_edges_receive_the_interpolated_temperature(self):
        # Four corner nodes, four midside nodes, one interior node. NSLL selects
        # precisely the boundary-associated nodes, not the interior node.
        xyz = np.c_[np.vstack([self.xy, [[.5, 0], [1, .5], [.5, 1], [0, .5], [.5, .5]]]), np.zeros(9)]
        calls = []
        mesh = SimpleNamespace(nnum=np.arange(1, 10))
        def select_nodes(*args):
            calls.append(("NSLL", *args)); mesh.nnum = np.arange(1, 9)
        engine = SimpleNamespace(mesh=mesh, lsel=lambda *args: calls.append(("LSEL", *args)),
                                 nsll=select_nodes, allsel=lambda: setattr(mesh, "nnum", np.arange(1, 10)),
                                 d=lambda *args: calls.append(("D", *args)))
        selected, values = apply_prescribed_temperature(engine, [11, 12, 13, 14], self.load,
                                                        np.arange(1, 10), xyz, .1)
        np.testing.assert_equal(selected, np.arange(1, 9))
        np.testing.assert_allclose(values, [1000, 2000, 3000, 4000, 1500, 2500, 3500, 2500])
        self.assertIn(("NSLL", "S", 1), calls)
        self.assertEqual([call[1] for call in calls if call[0] == "D"], list(range(1, 9)))
        self.assertEqual(self.load.constrained_node_count, 8)

    def test_selected_interior_node_is_rejected_rather_than_prescribed(self):
        with self.assertRaisesRegex(ThermalAnalysisError, "does not lie"):
            interpolate_boundary_temperature(self.load, [[.5, .5]], 1e-8)

    def test_radiation_or_convection_is_not_applied_to_prescribed_boundary(self):
        engine = SimpleNamespace(sfl=lambda *args: self.fail("No surface heat-transfer loads may be applied"))
        apply_surface_loads(engine, [1, 2, 3, 4], self.load, {"include_rad": False})
        with self.assertRaises(ThermalAnalysisError):
            apply_surface_loads(engine, [1, 2, 3, 4], self.load, {"include_rad": True})
        cfd, thermal = configurations(Path("/tmp/nose"))
        settings = validate_settings(cfd, thermal)
        self.assertEqual(settings["loading_mode"], "prescribed_temperature")
        self.assertNotIn("h", settings)
        thermal["include_radiation"] = True
        with self.assertRaisesRegex(ThermalAnalysisError, "Disable include_radiation"):
            validate_settings(cfd, thermal)
        thermal["include_radiation"] = False
        thermal["loading"]["mode"] = "convection"
        with self.assertRaises(ThermalAnalysisError):
            validate_settings(cfd, thermal)

    def test_wall_temperatures_stay_attached_to_vertices_when_orientation_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wall.csv"
            node_csv(path, self.xy[::-1], self.load.surface_temperature[::-1])
            load = read_load(path)
            self.assertEqual(dict(zip(map(tuple, load.xy), load.surface_temperature)),
                             dict(zip(map(tuple, self.xy), self.load.surface_temperature)))
            node_csv(path, self.xy, [1000., 2000., float("nan"), 4000.])
            with self.assertRaises(ThermalAnalysisError): read_load(path)

    def test_saved_history_distinguishes_initial_state_from_prescribed_surface(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            cfd, thermal = configurations(output)
            settings = validate_settings(cfd, thermal)
            xyz = np.c_[np.vstack([self.xy, [[.5, .5]]]), np.zeros(5)]
            results = [np.r_[self.load.surface_temperature, 500], np.r_[self.load.surface_temperature, 800]]
            history, _, _, heat = save_results(output, self.load, np.arange(1, 6), xyz,
                                               results, [1., 2.], settings, .1)
            self.assertIsNone(heat)
            self.assertEqual([row["time_s"] for row in history], [0., 1., 2.])
            self.assertEqual(history[0]["maximum_K"], 300.)
            self.assertEqual(history[1]["maximum_K"], 4000.)
            with (output / "wall_temperature_history.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(all(row["boundary_load_state"] == "initial_before_application" for row in rows[:4]))
            self.assertTrue(all(row["boundary_load_state"] == "prescribed_surface" for row in rows[4:]))
            self.assertFalse(any("flux" in key for key in rows[0]))
            with np.load(output / "thermal_snapshots.npz") as arrays:
                self.assertIn("boundary_prescribed_temperature_K", arrays)
                self.assertNotIn("boundary_flux_W_m2", arrays)
                self.assertNotIn("boundary_gas_temperature_K", arrays)
            bad = copy.deepcopy(results); bad[0][0] = 301.
            with self.assertRaisesRegex(ThermalAnalysisError, "do not match"):
                save_results(output, self.load, np.arange(1, 6), xyz, bad, [1., 2.], settings, .1)


class PrescribedRunnerTests(unittest.TestCase):
    def test_separate_mechanical_runner_records_temperature_provenance_without_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); work = root / "runs/nose"; work.mkdir(parents=True)
            cfg, thermal = configurations(work)
            path = work / "_batch/config.yaml"; path.parent.mkdir(); path.write_text(yaml.safe_dump(cfg))
            thermal_path = root / "thermal.yaml"; thermal_path.write_text(yaml.safe_dump(thermal))
            temperature = work / "temperature"; temperature.mkdir()
            xy = np.array([[0., 0.], [1., 0.], [0., 1.]])
            boundary = temperature / "wall_temperature_nodes.csv"
            node_csv(boundary, xy, [1000., 1200., 1400.])
            for name in ("04_final.cas.h5", "04_final.dat.h5"):
                (work / name).write_bytes(b"Radiating CFD fixture")
            source = {key: file_sha256(work / name) for key, name in
                      (("final_case", "04_final.cas.h5"), ("final_data", "04_final.dat.h5"))}
            metadata = {"source_sha256": source, "conditions": {"thermal": cfg["thermal"]},
                        "wall_nodes": {"csv_sha256": file_sha256(boundary)}}
            summary_path = temperature / "temperature_summary.json"
            summary_path.write_text(json.dumps(metadata))
            def solve(engine, load, settings, mesh_size):
                self.assertEqual(load.mode, "prescribed_temperature")
                self.assertFalse(settings["include_rad"])
                self.assertNotIn("h", settings)
                return np.arange(1, 5), np.c_[np.vstack([load.xy, [[.2, .2]]]), np.zeros(4)], \
                    [np.r_[load.surface_temperature, 500.], np.r_[load.surface_temperature, 800.]], [1., 2.], 1
            with patch("pipeline.gas_reporting.temperature_complete", return_value=True), \
                 patch("pipeline.solution_quality.require_usable_solution", return_value={"status": "residual_screen_passed"}), \
                 patch("ansys.mapdl.core.launch_mapdl", return_value=SimpleNamespace(exit=lambda: None)), \
                 patch("pipeline.thermal.solve_thermal", side_effect=solve):
                result = run_thermal_analysis(cfd_config=path, thermal_config=thermal_path)
            self.assertEqual(result["model_version"], "5.0-prescribed-temperature")
            self.assertEqual(result["loading_mode"], "prescribed_temperature")
            self.assertEqual(result["source_sha256"], source)
            self.assertEqual(result["source_temperature_summary_sha256"], file_sha256(summary_path))
            self.assertEqual(result["boundary_csv_sha256"], file_sha256(boundary))
            self.assertEqual(result["temperature_conditions"], metadata["conditions"])
            self.assertIsNone(result["net_heat_load_per_unit_span_J_m"])
            self.assertNotIn("convection_package_sha256", result)
            self.assertEqual(result["analysis"]["duration_s"], 2.)
            self.assertEqual(result["visualization_settings"]["loading_mode"], "prescribed_temperature")


if __name__ == "__main__": unittest.main()
