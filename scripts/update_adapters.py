"""Install the consolidated Fluent adapters without changing configs or results."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.request import urlopen


FILES = (
    "pipeline/aero.py",
    "pipeline/cfd.py",
    "pipeline/thermal_export.py",
    "pipeline/mesh.py",
    "pipeline/solve.py",
)
REPOSITORY = "https://raw.githubusercontent.com/Magnesiumgardens/AE496"


def download(revision, relative):
    with urlopen(f"{REPOSITORY}/{revision}/{relative}", timeout=30) as response:
        return response.read()


def update(root, revision, *, fetch=download):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("Use the full 40-character Git commit revision")
    root = Path(root).resolve()
    if not (root / "study.py").is_file():
        raise ValueError("Run this from the AE496 project folder containing study.py")
    originals = {}
    for relative in FILES:
        target = root / relative
        target.resolve().relative_to(root)
        originals[relative] = target.read_bytes()

    # Download and compile the complete set before changing any source file.
    with tempfile.TemporaryDirectory(prefix="adapter-update-", dir=root) as directory:
        staged = Path(directory)
        for relative in FILES:
            data = fetch(revision, relative)
            compile(data, relative, "exec")
            path = staged / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = root / "logs" / "source_backups" / stamp
        for relative, data in originals.items():
            path = backup / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (backup / "update.json").write_text(
            json.dumps({"revision": revision, "files": list(FILES)}, indent=2) + "\n",
            encoding="utf-8",
        )
        replaced = []
        try:
            for relative in FILES:
                os.replace(staged / relative, root / relative)
                replaced.append(relative)
        except Exception:
            for relative in replaced:
                (root / relative).write_bytes(originals[relative])
            raise
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Full pinned Git commit revision")
    parser.add_argument("--root", default=".", help="AE496 project folder (default: current folder)")
    args = parser.parse_args()
    try:
        backup = update(args.root, args.revision)
    except Exception as exc:
        parser.exit(1, f"[update] Failed: {exc}\n")
    print(f"[update] Installed {len(FILES)} adapters from {args.revision}")
    print(f"[update] Previous source saved: {backup}")
    print("[update] Configurations, CAD, meshes and simulation results retained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
