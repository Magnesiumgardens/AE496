"""Compatibility entrypoint: solve the mesh belonging to the selected config."""
import argparse
from pipeline.config import load_config
from pipeline.solve import solve_case
from pipeline.utils import work_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    mesh = work_paths(cfg)["mesh"]
    if not mesh.is_file() or mesh.stat().st_size == 0:
        parser.error(f"A nonempty mesh is required: {mesh}")
    print(f"[solver-only] Using existing mesh: {mesh}")
    print(solve_case(cfg, mesh))


if __name__ == "__main__":
    main()
