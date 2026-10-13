# Migration from the previous project

Use the commands in [README.md](README.md). The code now lives in shared
`pipeline/` modules; the familiar `run_all.py`, `run_solver_only.py`,
`export_thermal_only.py`, `export_aero_only.py`, `run_thermal_analysis.py`,
`run_batch.py`, and `compare_nose_trade.py` are compatibility entry points.

Place original CAD inputs in `geometry_batch/`. Keep existing `runs/<case>/`
folders if you want to resume them. The case-specific `_batch/config.yaml`
remains the authoritative case/work-directory mapping. Existing outputs without
fingerprint metadata can be adopted, explicitly marked as unverified provenance.
Changed inputs preserve previous outputs under `_history/` before rerunning.

The new default thermal model is direct CFD heat flux with conduction and
radiation, as requested. Adiabatic-reference runs and endpoint requirements
have been removed. Historical CSVs containing convection/adiabatic columns are
still readable for reports, but their physics are not relabeled as new runs.

Review the new YAML against local settings before replacing it. CFD reference
wall temperature remains explicit; it does not hold the transient solid at that
temperature. Confirm freestream direction, heat-flux sign, reference width and
material parameters on a known case before starting a new batch.
