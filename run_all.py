"""Compatibility entrypoint: prepare one fluid surface, mesh it and run CFD."""
import argparse
import time

from pipeline.config import load_config
from pipeline.geometry import prepare_geometry
from pipeline.mesh import build_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--stop-after", choices=("geometry", "mesh", "solve"), default="solve")
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(f"Project: {cfg['project']['name']}")
    print(f"Work directory: {cfg['project']['work_dir']}")
    geometry = prepare_geometry(cfg)
    if args.stop_after == "geometry":return
    mesh = build_mesh(cfg, geometry)
    if args.stop_after == "mesh":return
    delay = float(cfg.get("runtime", {}).get("meshing_release_delay_s", 10))
    if delay > 0:
        print(f"[pipeline] Waiting {delay:g} s for meshing process release...")
        time.sleep(delay)
    from pipeline.solve import solve_case
    print(solve_case(cfg, mesh))


if __name__ == "__main__":
    main()
