"""RE/cold affine gas-heating calibration, geometry matching and provenance."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pipeline.aerothermal_calibration import (
    STEFAN_BOLTZMANN, build_surface_package, calibration_cfg, calibration_paths,
    calibration_settings, derive_affine_heating, export_cold_calibration, export_re_calibration,
    verify_calibration_export, verify_surface_package,
)
from pipeline.reporting import read_csv, write_csv
from pipeline.temperature_export import temperature_conditions
from pipeline.thermal_loading import ThermalAnalysisError
from pipeline.utils import atomic_json, file_sha256


def calibration_case(work):
    """Small valid shared-mesh fixture usable by orchestration tests."""
    work.mkdir(parents=True, exist_ok=True)
    cad = work / "nose.x_t"
    cad.write_bytes(b"one CAD shape")
    cfg = dict(project=dict(name=work.name, work_dir=str(work)),
        geometry=dict(input_file=str(cad), body_wall_name="body_wall", input_kind="prepared_fluid_surface"),
        flight=dict(mach=8., static_pressure_pa=1090., static_temperature_k=226.5, flow_direction=[-1., 0.]),
        physics=dict(viscous_model="laminar"), mesh=dict(length_unit="m"),
        workflow=dict(strategy="single_solve"),
        thermal=dict(mode="radiative_equilibrium", wall_temperature_k=300., emissivity=.9, external_radiation_temperature_k=3.),
        solver=dict(max_iterations=4800, initialization="freestream", courant_number=.3, flow_scheme="first-order-upwind"),
        calibration=dict(cold_wall_temperature_k=300., heat_flux_into_wall_sign=-1))
    (work / "02_mesh.msh.h5").write_bytes(b"one common mesh")
    points = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    teq = np.array([1200., 1300., 1400., 1500.])
    qeq = .9 * STEFAN_BOLTZMANN * (teq**4 - 3.**4)
    qcold = qeq + np.array([100., 120., 140., 160.]) * (teq - 300.)
    for state, effective in (("re", cfg), ("cold", calibration_cfg(cfg))):
        paths = calibration_paths(cfg)
        endpoint = paths[f"{state}_work"]
        endpoint.mkdir(parents=True, exist_ok=True)
        for filename in ("04_final.cas.h5", "04_final.dat.h5"):
            (endpoint / filename).write_bytes((state + filename).encode())
        pair = {key: file_sha256(endpoint / filename) for key, filename in (
            ("final_case", "04_final.cas.h5"), ("final_data", "04_final.dat.h5"))}
        atomic_json(endpoint / "summary.json", dict(thermal_mode=effective["thermal"]["mode"],
            numerical_quality=dict(status="residual_screen_passed", source_sha256=pair),
            source_geometry_sha256=file_sha256(cad), source_mesh_sha256=file_sha256(work / "02_mesh.msh.h5")))
        faces, nodes = [], []
        for index, point in enumerate(points):
            end = points[(index + 1) % len(points)]
            temperature = 300. if state == "cold" else float(teq[index])
            nodes.append(dict(surface_order_index=index, x_m=float(point[0]), y_m=float(point[1]), z_m=0., wall_temperature_K=temperature))
            face = dict(surface_order_index=index, zone_name="body_wall", face_index=index,
                x_m=float((point[0] + end[0]) / 2), y_m=float((point[1] + end[1]) / 2), z_m=0.,
                segment_length_m=1., start_x_m=float(point[0]), start_y_m=float(point[1]),
                end_x_m=float(end[0]), end_y_m=float(end[1]), wall_temperature_K=temperature)
            if state == "cold":
                face.update(raw_heat_flux_W_m2=float(-qcold[index]), heat_flux_into_wall_W_m2=float(qcold[index]))
            faces.append(face)
        write_csv(paths[f"{state}_nodes"], nodes)
        write_csv(paths[f"{state}_faces"], faces)
        fields = dict(temperature="temperature")
        if state == "cold": fields.update(heat_flux="heat-flux", heat_flux_into_wall_sign=-1)
        atomic_json(paths[f"{state}_summary"], dict(status="complete", thermal_mode=effective["thermal"]["mode"],
            conditions=temperature_conditions(effective), source_sha256=pair,
            mesh_sha256=file_sha256(work / "02_mesh.msh.h5"),
            faces_csv_sha256=file_sha256(paths[f"{state}_faces"]),
            nodes_csv_sha256=file_sha256(paths[f"{state}_nodes"]), fields=fields))
    return cfg


def refresh_export(cfg, state):
    paths = calibration_paths(cfg)
    metadata = json.loads(paths[f"{state}_summary"].read_text())
    for key in ("faces", "nodes"):
        metadata[f"{key}_csv_sha256"] = file_sha256(paths[f"{state}_{key}"])
    atomic_json(paths[f"{state}_summary"], metadata)


class AerothermalCalibrationTests(unittest.TestCase):
    def test_two_anchors_identify_decreasing_gas_heating_and_equilibrium(self):
        tc, teq, eps, trad = 300., np.array([1200., 1600.]), .9, 3.
        qeq = eps * STEFAN_BOLTZMANN * (teq**4 - trad**4)
        qcold = qeq + np.array([100., 250.]) * (teq - tc)
        h, source, actual_qeq = derive_affine_heating(tc, teq, qcold, eps, trad)
        np.testing.assert_allclose(h, [100., 250.])
        np.testing.assert_allclose(source - h * tc, qcold)
        np.testing.assert_allclose(source - h * teq, actual_qeq)
        np.testing.assert_allclose(actual_qeq - eps * STEFAN_BOLTZMANN * (teq**4 - trad**4), 0.)
        self.assertTrue(np.all(source - h * (tc + 100) < qcold))
        with self.assertRaisesRegex(ThermalAnalysisError, "positive aerodynamic conductance"):
            derive_affine_heating(tc, teq, -qcold, eps, trad)
        with self.assertRaisesRegex(ThermalAnalysisError, "T_eq - T_cold"):
            derive_affine_heating(tc, [310., 1600.], qcold, eps, trad)
        with self.assertRaisesRegex(ThermalAnalysisError, "positive aerodynamic conductance"):
            derive_affine_heating(tc, teq, qeq, eps, trad)

    def test_auxiliary_cfg_preserves_mesh_flight_solver_and_primary_model(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = calibration_case(Path(directory) / "nose")
            before = copy.deepcopy(cfg)
            cold = calibration_cfg(cfg)
            self.assertEqual(cfg, before)
            self.assertEqual(cold["workflow"]["strategy"], "single_solve")
            self.assertEqual(cold["_calibration_kind"], "cold_wall")
            self.assertEqual(cold["project"]["work_dir"], str(Path(cfg["project"]["work_dir"]) / "cfd/cold_calibration"))
            self.assertEqual(cold["thermal"], dict(mode="fixed_temperature", wall_temperature_k=300.))
            self.assertEqual(cold["flight"], cfg["flight"])
            self.assertEqual(cold["mesh"], cfg["mesh"])
            self.assertEqual(cold["solver"]["courant_number"], .3)
            self.assertFalse(cold["solver"]["temperature_history"]["enabled"])

    def test_re_export_requests_temperature_only_not_ambiguous_generic_flux(self):
        from test_adiabatic_study import field_solver
        with tempfile.TemporaryDirectory() as directory:
            cfg = calibration_case(Path(directory) / "nose")
            solver, requests = field_solver()
            metadata = export_re_calibration(solver, cfg, ["body_wall"])
            scalars = [request for request in requests if hasattr(request, "field_name")]
            self.assertEqual([request.field_name for request in scalars], ["temperature", "temperature"])
            self.assertEqual(metadata["fields"], dict(temperature="temperature"))

    def test_cold_export_preserves_signed_native_flux_as_calibration_anchor(self):
        from test_adiabatic_study import field_solver
        with tempfile.TemporaryDirectory() as directory:
            cfg = calibration_case(Path(directory) / "nose")
            solver, requests = field_solver(flux=-12500.)
            metadata = export_cold_calibration(solver, cfg, ["body_wall"])
            rows = read_csv(calibration_paths(cfg)["cold_faces"])
            np.testing.assert_allclose([float(row["raw_heat_flux_W_m2"]) for row in rows], -12500.)
            np.testing.assert_allclose([float(row["heat_flux_into_wall_W_m2"]) for row in rows], 12500.)
            self.assertEqual(metadata["thermal_mode"], "fixed_temperature")
            self.assertEqual(metadata["fields"]["heat_flux_into_wall_sign"], -1)
            self.assertEqual([request.field_name for request in requests if hasattr(request, "field_name")],
                             ["temperature", "temperature", "heat-flux"])

    def test_matched_face_calibration_projects_conductance_and_source_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = calibration_case(Path(directory) / "nose")
            metadata = build_surface_package(cfg)
            paths = calibration_paths(cfg)
            faces = read_csv(paths["faces"])
            nodes = read_csv(paths["nodes"])
            hface = np.array([float(row["aerodynamic_conductance_W_m2K"]) for row in faces])
            sourceface = np.array([float(row["aerodynamic_source_W_m2"]) for row in faces])
            np.testing.assert_allclose(hface, [100., 120., 140., 160.])
            np.testing.assert_allclose([float(row["aerodynamic_conductance_W_m2K"]) for row in nodes],
                                       (hface + np.roll(hface, 1)) / 2)
            np.testing.assert_allclose([float(row["aerodynamic_source_W_m2"]) for row in nodes],
                                       (sourceface + np.roll(sourceface, 1)) / 2)
            self.assertEqual(metadata["derivation"]["re_convective_flux_source"], "radiative_equilibrium_balance")
            self.assertEqual(set(metadata["source_sha256"]), {"re_case", "re_data", "cold_case", "cold_data"})
            self.assertGreater(metadata["diagnostics"]["maximum_projected_RE_nodal_net_flux_residual_W_m2"], 0.)
            self.assertEqual(verify_surface_package(cfg)[0], paths["nodes"])

    def test_parallel_native_indices_and_loop_rotation_do_not_change_physical_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = calibration_case(Path(directory) / "nose")
            build_surface_package(cfg)
            paths = calibration_paths(cfg)
            before = read_csv(paths["nodes"])
            for kind in ("faces", "nodes"):
                rows = read_csv(paths[f"cold_{kind}"])
                rows = rows[1:] + rows[:1]
                for index, row in enumerate(rows):
                    row["surface_order_index"] = index
                    if kind == "faces": row["face_index"] = 100 - index
                write_csv(paths[f"cold_{kind}"], rows)
            refresh_export(cfg, "cold")
            build_surface_package(cfg)
            self.assertEqual(read_csv(paths["nodes"]), before)

    def test_wrong_geometry_and_cold_wall_temperature_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = calibration_case(Path(directory) / "nose")
            paths = calibration_paths(cfg)
            rows = read_csv(paths["cold_faces"])
            rows[0]["x_m"] = .55
            write_csv(paths["cold_faces"], rows)
            refresh_export(cfg, "cold")
            with self.assertRaisesRegex(ThermalAnalysisError, "geometry do not match"):
                build_surface_package(cfg)
            rows[0]["x_m"] = .5
            rows[0]["wall_temperature_K"] = 350.
            write_csv(paths["cold_faces"], rows)
            refresh_export(cfg, "cold")
            with self.assertRaisesRegex(ThermalAnalysisError, "prescribed wall temperature"):
                build_surface_package(cfg)

    def test_changed_cad_mesh_pair_quality_and_sign_cannot_reuse_package(self):
        changes = ("cad", "mesh", "cold_data", "quality", "sign", "nodes")
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                cfg = calibration_case(Path(directory) / "nose")
                build_surface_package(cfg)
                paths = calibration_paths(cfg)
                if change == "cad": Path(cfg["geometry"]["input_file"]).write_bytes(b"changed CAD")
                elif change == "mesh": (paths["root"] / "02_mesh.msh.h5").write_bytes(b"changed mesh")
                elif change == "cold_data": (paths["cold_work"] / "04_final.dat.h5").write_bytes(b"changed cold checkpoint")
                elif change == "sign": cfg["calibration"]["heat_flux_into_wall_sign"] = 1
                elif change == "nodes": paths["nodes"].write_text(paths["nodes"].read_text() + "\n")
                else:
                    summary_path = paths["cold_work"] / "summary.json"
                    summary = json.loads(summary_path.read_text())
                    summary["numerical_quality"]["status"] = "failed"
                    atomic_json(summary_path, summary)
                with self.assertRaises(ThermalAnalysisError): verify_surface_package(cfg)

    def test_insufficient_or_inverted_temperature_and_flux_anchors_not_clipped(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = calibration_case(Path(directory) / "nose")
            paths = calibration_paths(cfg)
            rows = read_csv(paths["cold_faces"])
            rows[0]["heat_flux_into_wall_W_m2"] = -float(rows[0]["heat_flux_into_wall_W_m2"])
            write_csv(paths["cold_faces"], rows)
            refresh_export(cfg, "cold")
            with self.assertRaisesRegex(ThermalAnalysisError, "No sign flip or clipping"):
                build_surface_package(cfg)
            cfg["calibration"]["heat_flux_into_wall_sign"] = True
            with self.assertRaisesRegex(ThermalAnalysisError, "-1 or \\+1"):
                calibration_settings(cfg)

    def test_endpoint_export_requires_actual_common_mesh_and_passed_matching_solution(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = calibration_case(Path(directory) / "nose")
            self.assertEqual(verify_calibration_export(cfg, "re")["status"], "complete")
            paths = calibration_paths(cfg)
            summary_path = paths["cold_work"] / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["source_mesh_sha256"] = "a different mesh hash"
            atomic_json(summary_path, summary)
            with self.assertRaisesRegex(ThermalAnalysisError, "unchanged common study mesh"):
                verify_calibration_export(cfg, "cold")


if __name__ == "__main__": unittest.main()
