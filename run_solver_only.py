"""Compatibility entrypoint: solve the mesh belonging to the selected config."""
import argparse
from pipeline.config import load_config
from pipeline.solve import solve_case
from pipeline.utils import work_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--continue-saved", action="store_true")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.get("workflow",{}).get("strategy") == "two_endpoint" and not cfg.get("_endpoint_kind"):
        if args.continue_saved:parser.error("Use study.py continue --config <case.yaml> --endpoint cold_wall|adiabatic")
        from pipeline.study import run_case
        print(run_case(args.config,stop_after="cold_wall"))
        return
    if args.continue_saved:
        if not args.iterations or args.iterations <= 0 or not args.output_dir:parser.error("Continuation needs positive --iterations and --output-dir")
        from pipeline.continuation import continue_solver
        print(continue_solver(cfg,args.iterations,args.output_dir))
        return
    mesh = work_paths(cfg)["mesh"]
    if not mesh.is_file() or mesh.stat().st_size == 0:
        parser.error(f"A nonempty mesh is required: {mesh}")
    print(f"[solver-only] Using existing mesh: {mesh}")
    print(solve_case(cfg, mesh))


if __name__ == "__main__":
    main()
