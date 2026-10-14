"""Install the complete study through Git; preserve run data and migrate configs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import yaml

# The initial helper runs from a freshly fetched source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.migration import endpoint_config, endpoint_state, endpoint_thermal_config, endpoint_batch_config


def git(repository, *args):
    return subprocess.check_output(["git", "-C", str(repository), *args])


def update_project(root, source, revision=None):
    root, source = Path(root).resolve(), Path(source).resolve()
    if not (root / "study.py").is_file():raise ValueError("Select the AE496 folder containing study.py")
    revision = revision or git(source, "rev-parse", "HEAD").decode().strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):raise ValueError("Expected a full Git commit SHA")
    tracked = git(source, "ls-tree", "-r", "--name-only", revision).decode().splitlines()
    selected = [name for name in tracked if
        (name.endswith(".py") and ("/" not in name or name.startswith(("pipeline/", "scripts/", "tests/"))))
        or name in ("README.md", "NOSE_TRADE_README.md", "requirements.txt", "thermal_analysis.yaml")
        or (name.startswith("docs/") and name.endswith(".md"))
        or name.startswith("examples/reference_results/")]
    payload = {name: git(source, "show", f"{revision}:{name}") for name in selected}
    for name, data in payload.items():
        if name.endswith(".py"):compile(data, name, "exec")
    config_paths = [root / "config.yaml", *sorted((root / "runs").glob("*/_batch/config.yaml"))]
    for path in config_paths:
        old = yaml.safe_load(path.read_bytes()) if path.is_file() else yaml.safe_load(git(source, "show", f"{revision}:config.yaml"))
        new = endpoint_config(old)
        if old != new or not path.is_file():
            payload[str(path.relative_to(root))] = yaml.safe_dump(new, sort_keys=False).encode()
            state_path = path.parent / "stage_state.json"
            if path.parent.name == "_batch" and state_path.is_file():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                payload[str(state_path.relative_to(root))] = (json.dumps(endpoint_state(state), indent=2) + "\n").encode()
    batch_path = root / "batch.yaml"
    batch = yaml.safe_load(batch_path.read_bytes()) if batch_path.is_file() else yaml.safe_load(git(source, "show", f"{revision}:batch.yaml"))
    batch = endpoint_batch_config(batch)
    payload["batch.yaml"] = yaml.safe_dump(batch, sort_keys=False).encode()
    thermal_path = root / "thermal_analysis.yaml"
    if thermal_path.is_file():
        previous = yaml.safe_load(thermal_path.read_bytes()) or {}
        if (previous.get("loading", {}) or {}).get("mode") == "convective_from_cfd":
            # Existing endpoint exposure/material values are inputs, not code.
            payload.pop("thermal_analysis.yaml", None)
        else:
            new_thermal = endpoint_thermal_config(previous, yaml.safe_load(payload["thermal_analysis.yaml"]))
            payload["thermal_analysis.yaml"] = yaml.safe_dump(new_thermal, sort_keys=False).encode()
    originals = {}
    for name in payload:
        path = root / name
        path.resolve().relative_to(root)
        originals[name] = path.read_bytes() if path.is_file() else None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = root / "logs/source_backups" / stamp
    for name, data in originals.items():
        if data is not None:
            target = backup / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    backup.mkdir(parents=True, exist_ok=True)
    (backup / "update.json").write_text(json.dumps(dict(revision=revision, files=list(payload)), indent=2), encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="study-update-", dir=root) as directory:
        staged = Path(directory)
        for name, data in payload.items():
            target = staged / name;target.parent.mkdir(parents=True, exist_ok=True);target.write_bytes(data)
        replaced = []
        try:
            for name in payload:
                target = root / name;target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged / name, target);replaced.append(name)
        except Exception:
            for name in reversed(replaced):
                if originals[name] is None:(root / name).unlink(missing_ok=True)
                else:(root / name).write_bytes(originals[name])
            raise
    print(f"[update] Installed complete study from {revision}")
    print(f"[update] Previous code/configs saved: {backup}")
    print("[update] Two-endpoint CFD calibration restored. Existing meshes and simulation results are retained.")
    print("[update] Migrated cases reuse their mesh, then run cold-wall and warm-started adiabatic CFD.")
    print("[update] Mechanical uses the derived spatial h; material choices and exposure durations are retained.")
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-repo", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--revision")
    args = parser.parse_args()
    try:update_project(args.root, args.git_repo, args.revision)
    except Exception as exc:parser.exit(1, f"[update] Failed: {exc}\n")
    return 0


if __name__ == "__main__":raise SystemExit(main())
