"""Solved surface temperatures, load/energy reports and calibration freshness."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pipeline.mechanical_reporting import mechanical_complete
from pipeline.reporting import (create_comparison_outputs, create_thermal_visualizations,
                                read_csv, write_csv, _surface_load_history_outputs)
from pipeline.thermal_loading import ThermalAnalysisError
from pipeline.utils import atomic_json, file_sha256


def saved_aerothermal(work):
    work.mkdir(parents=True, exist_ok=True)
    (work / "_batch").mkdir(exist_ok=True)
    (work / "_batch/config.yaml").write_text("reporting: {nose_direction: [1, 0]}\n")
    output = work / "thermal_analysis"; output.mkdir()
    xy = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
    wall, flux, energy, global_energy = [], [], [], []
    for time in (0., 100., 200.):
        for i, (x, y) in enumerate(xy):
            base = dict(time_s=time, surface_order_index=i, x_m=x, y_m=y)
            q = dict(convective_heat_flux_into_solid_W_m2=1000-time+10*i,
                     radiative_heat_flux_out_W_m2=10+.45*time,
                     net_heat_flux_into_solid_W_m2=990-1.45*time+10*i)
            cumulative = dict(cumulative_convective_energy_J_m2=(1000+10*i)*time-.5*time**2,
                              cumulative_radiative_energy_J_m2=10*time+.225*time**2,
                              cumulative_net_energy_J_m2=(990+10*i)*time-.725*time**2)
            wall.append(dict(base, wall_temperature_K=300+time*(1+.01*i), **q, **cumulative))
            flux.append(dict(base, **q)); energy.append(dict(base, **cumulative))
        global_energy.append(dict(time_s=time,
            convective_power_per_unit_span_W_m=4*(1015-time),
            radiative_power_per_unit_span_W_m=4*(10+.45*time),
            net_power_per_unit_span_W_m=4*(1005-1.45*time),
            cumulative_convective_energy_per_unit_span_J_m=4*(1015*time-.5*time**2),
            cumulative_radiative_energy_per_unit_span_J_m=4*(10*time+.225*time**2),
            cumulative_net_energy_per_unit_span_J_m=4*(1005*time-.725*time**2)))
    write_csv(output / "wall_temperature_history.csv", wall)
    write_csv(output / "surface_heat_flux_history.csv", flux)
    write_csv(output / "surface_energy_history.csv", energy)
    write_csv(output / "global_energy_history.csv", global_energy)
    write_csv(output / "wall_temperature_final.csv", wall[-len(xy):])
    write_csv(output / "temperature_history.csv", [dict(time_s=t, minimum_K=300+t/2,
        mean_K=300+t*.8, maximum_K=300+t*1.03) for t in (0,100,200)])
    write_csv(output / "thermal_nodes_final.csv", [dict(node_id=i+1,x_m=x,y_m=y,temperature_K=500+i)
                                                for i,(x,y) in enumerate(xy)])
    np.savez_compressed(output / "thermal_snapshots.npz", boundary_xy_m=np.asarray(xy),
        coordinates_m=np.c_[xy,np.zeros(len(xy))],time_s=np.array([0.,100.,200.]),
        temperatures_K=np.array([[300+t*(1+.01*i) for i in range(len(xy))] for t in (0.,100.,200.)]))
    summary = dict(status="complete", model_version="6.0-aerodynamic-surface-load",
                   loading_mode="aerodynamic_surface_load", material={"name":"Tungsten"},
                   analysis={"duration_s":200,"initial_temperature_k":300},
                   settings={"loading_mode":"aerodynamic_surface_load","duration":200},
                   final_temperature={"maximum_K":506,"mean_K":460},
                   result_sha256={path.name:file_sha256(path) for path in output.iterdir() if path.suffix in {".csv",".npz"}})
    atomic_json(output / "thermal_summary.json", summary)
    return summary


class AerothermalReportTests(unittest.TestCase):
    def test_solved_wall_histories_and_load_energy_reports_are_regenerated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("blunt", "sharp"):saved_aerothermal(root / name)
            from pipeline import reporting
            original_save, titles, drawstyles = reporting._save, [], []
            def recording_save(fig, path, plt, dpi=160):
                if Path(path).name == "wall_temperature_history_comparison.png":
                    titles.append(fig.axes[0].get_title())
                    drawstyles.extend(line.get_drawstyle() for line in fig.axes[0].lines)
                return original_save(fig, path, plt, dpi)
            with patch.object(reporting, "_save", side_effect=recording_save):
                outputs = create_comparison_outputs(root, [], root / "comparison")
            manifest = json.loads(Path(outputs["report_manifest"]).read_text())
            self.assertEqual(manifest["prescribed_surface_cases"], [])
            self.assertEqual(manifest["aerodynamic_surface_load_cases"], ["blunt", "sharp"])
            self.assertTrue(all("Solved" in title for title in titles))
            self.assertEqual(set(drawstyles), {"default"})
            histories = read_csv(outputs["wall_temperature_history_comparison"])
            values = [float(row["max_wall_temperature_K"]) for row in histories if row["design"] == "blunt"]
            self.assertEqual(values, [300., 403., 506.])
            for name in ("surface_heat_flux_history_csv", "surface_energy_history_csv", "global_energy_history_csv",
                         "surface_heat_flux_history_png", "surface_energy_history_png", "global_surface_power_history_png",
                         "global_surface_energy_history_png", "surface_heat_flux_max_mean_history_png",
                         "surface_heat_load_max_mean_history_png", "final_surface_heat_flux_normalized_position_png",
                         "final_surface_heat_load_normalized_position_png", "final_surface_heat_flux_maps_png",
                         "final_surface_heat_load_maps_png", "max_temperature_comparison_csv", "max_temperature_comparison_png",
                         "peak_heat_flux_comparison_csv", "peak_heat_flux_comparison_png",
                         "total_heat_load_comparison_csv", "total_heat_load_comparison_png"):
                self.assertTrue(Path(outputs[name]).is_file(), name)
            raw = read_csv(outputs["surface_energy_history_csv"])
            self.assertEqual(len(raw), 24)
            self.assertIn("cumulative_net_energy_J_m2", raw[0])
            global_rows = read_csv(outputs["global_energy_history_csv"])
            last = [row for row in global_rows if row["design"] == "blunt"][-1]
            self.assertEqual(float(last["cumulative_net_energy_per_unit_span_J_m"]), 688000.)
            for name in ("blunt", "sharp"):
                self.assertTrue((root / name / "reports/final_wall_heat_flux_map.png").is_file())
                self.assertTrue((root / "comparison/cases" / name / "surface_heat_flux_history.csv").is_file())
                for filename in ("final_surface_heat_flux_position.png", "final_surface_heat_load_position.png",
                                 "surface_heat_flux_max_mean_history.png", "surface_heat_load_max_mean_history.png",
                                 "peak_heat_flux_comparison.png", "total_heat_load_comparison.png"):
                    self.assertTrue((root / name / "reports" / filename).is_file())
            self.assertNotIn("h", outputs)

    def test_new_mode_visualization_cannot_label_calibration_flux_as_applied_cold_wall_load(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); viz = output / "visualization"; viz.mkdir()
            for name in ("applied_heat_flux_map.png", "applied_heat_flux_position.png"):(viz/name).write_bytes(b"old")
            xy = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
            xyz = np.c_[np.vstack([xy,[.5,.5]]),np.zeros(5)]
            result = create_thermal_visualizations(output,xy,np.ones(4)*1e6,xyz,
                [dict(time_s=200,minimum_K=450,mean_K=480,maximum_K=500)],
                [dict(time_s=200,temperature_K=np.array([500,490,480,470,450]))],
                dict(loading_mode="aerodynamic_surface_load",make_gif=False))
            self.assertTrue(Path(result["final_temperature_field_png"]).is_file())
            self.assertFalse(any("applied_heat_flux" in key for key in result))
            self.assertFalse((viz/"applied_heat_flux_map.png").exists())

    def test_report_freshness_requires_all_calibration_sources_and_new_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); work = root / "nose"; summary = saved_aerothermal(work)
            thermal = root / "thermal.yaml"; thermal.write_text("loading: {mode: aerodynamic_surface_load}\n")
            nodes = root / "surface_load_nodes.csv"; nodes.write_text("verified boundary package")
            package = root / "surface_load_summary.json"; package.write_text("verified metadata")
            metadata = dict(source_sha256={"re_case":"a","re_data":"b","cold_case":"c","cold_data":"d"},
                            conditions={"thermal":{"mode":"radiative_equilibrium"}})
            summary.update(thermal_config_path=str(thermal),thermal_config_sha256=file_sha256(thermal),
                           boundary_csv_sha256=file_sha256(nodes),source_sha256=metadata["source_sha256"],
                           surface_package_sha256=file_sha256(package),surface_load_conditions=metadata["conditions"])
            path = work / "thermal_analysis/thermal_summary.json"; atomic_json(path,summary)
            cfg = {"project":{"work_dir":str(work)},"thermal":{"mode":"radiative_equilibrium"},
                   "workflow":{"strategy":"single_solve"}}
            with patch("pipeline.aerothermal_calibration.verify_surface_package",return_value=(nodes,metadata)) as verify, \
                 patch("pipeline.aerothermal_calibration.calibration_paths",return_value={"summary":package}):
                self.assertTrue(mechanical_complete(work,cfg)); verify.assert_called_once_with(cfg)
                summary["source_sha256"] = dict(summary["source_sha256"],cold_data="changed")
                atomic_json(path,summary); self.assertFalse(mechanical_complete(work,cfg))
                summary["source_sha256"] = metadata["source_sha256"]
                del summary["result_sha256"]["surface_energy_history.csv"]
                atomic_json(path,summary); self.assertFalse(mechanical_complete(work,cfg))
            with patch("pipeline.aerothermal_calibration.verify_surface_package",side_effect=ThermalAnalysisError("changed CFD")):
                self.assertFalse(mechanical_complete(work,cfg))

    def test_nonfinite_load_history_is_rejected_instead_of_plotted(self):
        with tempfile.TemporaryDirectory() as directory:
            records = dict(time_s=10,surface_order_index=0,convective_heat_flux_into_solid_W_m2=float("nan"),
                           radiative_heat_flux_out_W_m2=0,net_heat_flux_into_solid_W_m2=1)
            with self.assertRaisesRegex(ValueError,"nonfinite"):
                _surface_load_history_outputs({"nose":{"flux":[records]}},Path(directory))

    def test_real_calibration_verification_rejects_changed_sources_and_snapshot_arrays(self):
        from pipeline.aerothermal_calibration import build_surface_package, calibration_paths
        from test_aerothermal_calibration import calibration_case
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); work = root / "nose"
            cfg = calibration_case(work); metadata = build_surface_package(cfg)
            summary = saved_aerothermal(work); paths = calibration_paths(cfg)
            thermal = root / "thermal.yaml"; thermal.write_text("loading: {mode: aerodynamic_surface_load}\n")
            summary.update(thermal_config_path=str(thermal),thermal_config_sha256=file_sha256(thermal),
                boundary_csv_sha256=file_sha256(paths["nodes"]),source_sha256=metadata["source_sha256"],
                surface_package_sha256=file_sha256(paths["summary"]),surface_load_conditions=metadata["conditions"])
            atomic_json(work / "thermal_analysis/thermal_summary.json",summary)
            self.assertTrue(mechanical_complete(work,cfg))
            for source in (paths["cold_work"] / "04_final.dat.h5", Path(cfg["geometry"]["input_file"]),
                           work / "thermal_analysis/thermal_snapshots.npz"):
                original = source.read_bytes(); source.write_bytes(b"changed source")
                self.assertFalse(mechanical_complete(work,cfg))
                source.write_bytes(original)
                self.assertTrue(mechanical_complete(work,cfg))


if __name__ == "__main__":unittest.main()
