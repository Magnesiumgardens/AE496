# Automatically copy results to a shared OneDrive folder

Open the shared results folder in your browser and choose **Add shortcut to My
files** or **Sync**. Wait for that folder to appear in File Explorer under the
OneDrive account with access. The web URL itself is not a Windows copy target.
The OneDrive desktop client must be running and signed in to upload the files.

From AE496, update the project and start the Windows helper:

```powershell
.\.venv\Scripts\python.exe study.py update
if ($LASTEXITCODE -eq 0) {
    powershell.exe -NoProfile -STA -ExecutionPolicy Bypass -File ".\scripts\sync_onedrive.ps1" -Watch
}
```

In the folder picker, select the locally synced **results** folder itself.
The helper copies plots, CSVs and JSON summaries from `runs`, preserving each
nose's folder and the comparison folder, then repeats every 60 seconds. New
versions replace existing files with the same relative path. It does not
delete destination-only files or edit teammates' unrelated files. `_history`
backups and temporary/lock files are excluded. Existing source files are never
moved or deleted. Stop the helper with Ctrl+C; keep the window open while you
want automatic copying.

Omit `-Watch` for a single copy, suitable after a report command. Override the
frequency with `-IntervalSeconds 120`. For unattended launches, supply the
known local folder path to `-Destination 'C:\...\Shared Project\results'` to
skip the picker. The default `-Mode Reports` excludes native CFD/MAPDL binaries;
`-Mode All` also copies those larger run files. Logs are appended to
`logs/onedrive_sync/sync.log`.

Robocopy success codes 0 through 7 are treated as success. Failed or locked
files are reported and retried in the next watch cycle. A successful local copy
means files reached the synced folder; the OneDrive desktop client's status
shows whether their cloud upload has completed. This script does not connect
directly to SharePoint, change sharing permissions, regenerate figures, or run
Ansys. Run report regeneration as usual to produce updated graphs.
