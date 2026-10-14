from pathlib import Path
from types import SimpleNamespace
import copy
import csv
import tempfile
import unittest

import numpy as np
import yaml

from pipeline.config import ConfigError, load_config
from pipeline.thermal_export import export_thermal_bc_package, order_segments, unpack_faces


class ConfigTests(unittest.TestCase):
    @staticmethod
    def direct_adiabatic_config():
        data = yaml.safe_load(Path("config.yaml").read_text())
        data.pop("workflow", None);data.pop("thermal_export", None)
        data["thermal"] = {"mode":"adiabatic"}
        data["solver"].update(initialization="freestream",courant_number=.1,
                             flow_scheme="first-order-upwind",residual_thresholds={"energy":1e-6})
        return data

    def test_relative_paths_are_case_config_relative_and_direction_is_normalized(self):
        data = self.direct_adiabatic_config()
        data["flight"]["flow_direction"] = [0, -2]
        data["project"]["work_dir"] = "../case"
        data["geometry"]["input_file"] = "../shape.x_t"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "settings").mkdir()
            config = root / "settings" / "case.yaml"
            config.write_text(yaml.safe_dump(data))
            cfg = load_config(config)
            self.assertEqual(Path(cfg["project"]["work_dir"]), root / "case")
            self.assertEqual(Path(cfg["geometry"]["input_file"]), root / "shape.x_t")
            self.assertEqual(cfg["flight"]["flow_direction"], [0, -1])

    def test_directory_is_not_a_config_and_invalid_physics_fail_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ConfigError):load_config(directory)
            data = self.direct_adiabatic_config()
            config = Path(directory) / "case.yaml"
            for field, value in (("max_iterations", 1.5), ("max_iterations", float("nan"))):
                data["solver"][field] = value
                config.write_text(yaml.safe_dump(data))
                with self.assertRaises(ConfigError):load_config(config)


class ExportTests(unittest.TestCase):
    @staticmethod
    def segments(offset=0):
        points = [np.array([offset + x, y, 0]) for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))]
        return [{"start": points[i], "end": points[(i + 1) % 4], "id": i} for i in range(4)]

    def test_connectivity_parsing_and_topology_preserve_faces(self):
        faces = unpack_faces(np.array([2, 0, 1, 2, 1, 2]))
        self.assertEqual([list(f) for f in faces], [[0, 1], [1, 2]])
        data = self.segments()
        data[2]["start"], data[2]["end"] = data[2]["end"], data[2]["start"]
        ordered = order_segments([data[i] for i in (2, 0, 3, 1)])
        self.assertEqual({row["id"] for row in ordered}, {0, 1, 2, 3})
        with self.assertRaises(ValueError):unpack_faces([2, 0])
        with self.assertRaises(ValueError):order_segments(data + self.segments(offset=3))
        with self.assertRaises(ValueError):order_segments(data, tolerance=0)

    def test_export_uses_face_values_and_explicit_sign_with_atomic_csv(self):
        from ansys.fluent.core.fields.field_data_interfaces import SurfaceFieldDataRequest
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        surface = SimpleNamespace(vertices=vertices,
            connectivity=np.array([2, 0, 1, 2, 1, 2, 2, 2, 3, 2, 3, 0]),
            face_centroids=np.array([[0.5, 0, 0], [1, 0.5, 0], [0.5, 1, 0], [0, 0.5, 0]]))
        def read(request):
            self.assertEqual(request.surfaces, ["body_wall.1"])
            if isinstance(request, SurfaceFieldDataRequest):return {"body_wall.1": surface}
            self.assertFalse(request.node_value)
            return {"body_wall.1": np.array([-1, -2, -3, -4])}
        solver = SimpleNamespace(fields=SimpleNamespace(
            field_data=SimpleNamespace(get_field_data=read,
                scalar_fields=SimpleNamespace(allowed_values=lambda: ["heat-flux"]))))
        with tempfile.TemporaryDirectory() as directory:
            cfg = {"project": {"work_dir": directory}, "geometry": {},
                "thermal": {"wall_temperature_k": 300},
                "thermal_export": {"heat_flux_into_wall_sign": -1}}
            result = export_thermal_bc_package(solver, cfg, ["body_wall.1"])
            with open(result["body_wall_faces_csv"], newline="") as f: rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4)
            self.assertEqual(sum(float(r["heat_flux_into_wall_W_m2"]) for r in rows), 10)
            self.assertEqual(result["source_field"], "heat-flux")
            self.assertEqual(list(Path(directory).rglob("*.tmp")), [])
            with self.assertRaises(ValueError):export_thermal_bc_package(solver, cfg, ["farfield"])


if __name__ == "__main__":unittest.main()
