"""Fetch the maintained Git branch and install it with source/config backups."""
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    source = root / ".runtime/source_checkout"
    source.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not source.exists():
            subprocess.run(["git", "clone", "--single-branch", "--branch", "codex/nose-trade-foundation",
                "https://github.com/Magnesiumgardens/AE496.git", str(source)], check=True)
        else:
            subprocess.run(["git", "-C", str(source), "pull", "--ff-only", "origin", "codex/nose-trade-foundation"], check=True)
        return subprocess.run([sys.executable, str(source / "scripts/update_project.py"),
            "--git-repo", str(source), "--root", str(root)]).returncode
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[update] Git update failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":raise SystemExit(main())
