#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
source scripts/activate_cloud.sh
python -m pip install --only-binary=:all: -r requirements.txt
python -m pip check
python -m unittest discover -s tests -v
python -c 'import ansys.fluent.core, ansys.geometry.core, ansys.mapdl.core; print("All three PyAnsys clients imported successfully")'
