# Compare peak heat flux from saved results

From AE496 in PowerShell:

```powershell
.\.venv\Scripts\python.exe study.py update
if ($LASTEXITCODE -eq 0) {
    .\.venv\Scripts\python.exe study.py report-heat-flux --runs-dir ".\runs"
}
```

The dedicated command reads saved CSVs directly. It does not run Fluent,
Mechanical, meshing, calibration, or a batch. It needs Python's plotting/data
libraries but does not import or launch native Ansys clients. The normal
`study.py report --runs-dir runs` command also creates this comparison.

Outputs are in `runs/comparison/peak_heat_flux`:

- `peak_heat_flux_comparison.png`: three panels comparing all included noses'
  maximum incoming convection, outward radiation, and net heat flux in W/m².
- `peak_heat_flux_comparison.csv`: the same peaks, their physical times,
  surface sample indices, coordinates when available, source CSV/hash,
  actual saved duration, and validation status.
- `peak_heat_flux_report.json`: included groups, available run conditions,
  the interpretation of the peaks, and explicit exclusion reasons.

## Select the current geometry batch and compare mk3 with mk4

When `geometry_batch` exists beside `runs`, the report uses its `*.x_t` CAD
inventory to select saved case folders. Old run folders with no corresponding
CAD input do not enter this selection. It uses the same filename-to-case-name
mapping as batch execution, with case-insensitive matching on Windows and other
platforms. It never opens CAD in Ansys or generates a new mesh.

To select mk3/mk4 filenames explicitly:

```powershell
.\.venv\Scripts\python.exe study.py report-heat-flux --runs-dir ".\runs" --geometry-dir ".\geometry_batch" --cases "*mk3*" "*mk4*"
```

The case filters accept case-insensitive names or wildcard patterns. Use exact
filenames/stems instead when selecting a particular variant. `--geometry-glob`
can match the CAD pattern in `batch.yaml` if it differs from `*.x_t`.
The JSON report records the entire CAD inventory, selection, matched run names,
and any missing saved results. Available calibration CAD hashes are checked;
if a CAD file was changed after its saved calibration, the old result is
excluded rather than labeled as the new geometry. Archives without those
hashes are explicitly identified as filename-only matches.

The comparison includes spatial curves overlaying the selected noses:

- `peak_heat_flux_spatial_envelope.png`/`.csv`: maximum signed flux at each
  saved wall location over all saved times. Different locations can peak at
  different times; this envelope is not one physical snapshot.
- `initial_heat_flux_spatial_comparison.png`/`.csv`: first saved distribution,
  labeled with its actual physical time (normally 0 s).
- `final_heat_flux_spatial_comparison.png`/`.csv`: final saved distribution,
  labeled with its actual physical time (normally 200 s).

Each image separates incoming convection, outward radiation, and net flux.
The horizontal coordinate is normalized distance along each nose's included
surface, back-to-nose-to-back, with the nose at 0.5. Each nose uses its own
original points; no flux values are interpolated between mk3 and mk4. The CSVs
also retain physical distance and original x/y coordinates. An insulated rear
closure is not drawn as a connecting profile segment. Invalid/incomplete spatial
data are reported separately; a valid scalar peak bar can still be produced.

Different recorded settings or exposure durations get separate `conditions_*`
subfolders instead of a single mixed-condition chart. CSV-only archives with
unknown conditions are separated from receipt-backed results. Use `--output-dir`
to select another report folder.

For each nose, the preferred input is
`thermal_analysis/surface_heat_flux_history.csv`. If that file is absent, the
command can use `thermal_analysis/wall_temperature_history.csv` only when it
already contains these three flux columns:

- `convective_heat_flux_into_solid_W_m2`
- `radiative_heat_flux_out_W_m2`
- `net_heat_flux_into_solid_W_m2`

Temperature-only histories do not identify heat flux. Cold-wall/RE calibration
anchors are not silently substituted for a transient heat-flux history.

Each peak is the **maximum signed local sample across all saved physical times
and included wall locations**, including time zero when saved. Components can
peak at different times or positions; their separate maxima do not necessarily
subtract to the maximum net flux. Negative net cooling remains negative. The
quantity is neither total power nor accumulated heat load. Unrecorded peaks
between times or wall samples cannot be recovered from these files.

The outgoing `thermal_segment_active` flag does not exclude a shared corner:
that vertex can still belong to an included forward edge. An explicit
`thermal_active=0` sample is omitted. The existing thermal rear-cut selection
and all source files remain unchanged.

The report checks declared CSV hashes and any available failed thermal/energy
receipts. Known failed/running results or mismatched hashes are excluded with
printed reasons. Archives without complete receipts can still be inspected,
but their CSVs say `saved_csv_only` or `saved_receipt_only`; this is not a new
CFD convergence, mesh-independence, or complete Mechanical validation. The
normal validated solid reports retain their existing acceptance checks.
