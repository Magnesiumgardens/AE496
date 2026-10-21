<#
.SYNOPSIS
Copy current AE496 results into a locally synced OneDrive shared folder.
.EXAMPLE
.\scripts\sync_onedrive.ps1 -Destination 'C:\Users\mattk\OneDrive - University\Shared Project\Results' -Watch
#>
[CmdletBinding()]
param(
    [string]$Destination,
    [string]$Source,
    [ValidateSet('Reports', 'All')]
    [string]$Mode = 'Reports',
    [switch]$Watch,
    [ValidateRange(10, 86400)]
    [int]$IntervalSeconds = 60
)

$ErrorActionPreference = 'Stop'
# Resolve script-relative defaults after parameter binding. Windows PowerShell
# can evaluate parameter defaults before PSScriptRoot has been populated.
$scriptDirectory = $PSScriptRoot
if (-not $scriptDirectory -and $PSCommandPath) {
    $scriptDirectory = [System.IO.Path]::GetDirectoryName($PSCommandPath)
}
if (-not $scriptDirectory -and $MyInvocation.MyCommand.Path) {
    $scriptDirectory = [System.IO.Path]::GetDirectoryName($MyInvocation.MyCommand.Path)
}
if (-not $scriptDirectory) {
    throw 'Run the saved sync_onedrive.ps1 file with powershell.exe -File, rather than pasting its contents.'
}
if (-not $Source) { $Source = Join-Path $scriptDirectory '..\runs' }
if (-not (Get-Command robocopy.exe -ErrorAction SilentlyContinue)) {
    throw 'This script requires Windows robocopy.exe.'
}
if ($Destination -match '^https?://') {
    throw 'Use the shared folder path from File Explorer. Add the shared folder to OneDrive and sync it locally first.'
}
if (-not $Destination) {
    Add-Type -AssemblyName System.Windows.Forms
    $picker = New-Object System.Windows.Forms.FolderBrowserDialog
    $picker.Description = 'Select the locally synced OneDrive shared results folder (add its shortcut in OneDrive first).'
    $picker.ShowNewFolderButton = $false
    if ($env:OneDriveCommercial) { $picker.SelectedPath = $env:OneDriveCommercial }
    elseif ($env:OneDrive) { $picker.SelectedPath = $env:OneDrive }
    try {
        if ($picker.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
            Write-Host 'No destination selected; nothing copied.'
            exit 0
        }
        $Destination = $picker.SelectedPath
    } finally {
        $picker.Dispose()
    }
}
$sourcePath = (Resolve-Path -LiteralPath $Source).ProviderPath
if (-not (Test-Path -LiteralPath $sourcePath -PathType Container)) {
    throw "Source must be a results folder: $sourcePath"
}
$destinationPath = [System.IO.Path]::GetFullPath($Destination)
$sourcePrefix = $sourcePath.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
$destinationPrefix = $destinationPath.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
if ($sourcePrefix.StartsWith($destinationPrefix, [System.StringComparison]::OrdinalIgnoreCase) -or
    $destinationPrefix.StartsWith($sourcePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Source and destination must be separate folders; neither may contain the other.'
}
[System.IO.Directory]::CreateDirectory($destinationPath) | Out-Null
$logDirectory = Join-Path $scriptDirectory '..\logs\onedrive_sync'
[System.IO.Directory]::CreateDirectory($logDirectory) | Out-Null
$logPath = Join-Path $logDirectory 'sync.log'
$filePatterns = if ($Mode -eq 'Reports') {
    @('*.csv', '*.json', '*.png', '*.gif', '*.svg', '*.pdf', '*.html')
} else {
    @('*')
}
# Copy changed files in either timestamp direction. No /XO: regenerated files
# replace their previous copies even if the destination has a newer timestamp.
# No /MIR, /PURGE, /MOVE or /MOV: shared-folder content is never deleted.
$copyArguments = @($sourcePath, $destinationPath) + $filePatterns + @(
    '/E', '/COPY:DAT', '/DCOPY:T', '/R:2', '/W:2', '/XJ', '/NP', '/NFL', '/NDL',
    '/XD', '_history', '/XF', '*.lock', '*.lck', '*.tmp',
    "/LOG+:$logPath", '/TEE'
)

Write-Host "Source:      $sourcePath"
Write-Host "Destination: $destinationPath"
Write-Host "Mode: $Mode; existing files with matching relative paths are replaced. Other shared files are retained."
if ($Watch) { Write-Host "Repeating every $IntervalSeconds seconds. Press Ctrl+C to stop." }
do {
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Copying results..."
    & robocopy.exe @copyArguments
    $copyExitCode = $LASTEXITCODE
    # Robocopy codes 0--7 describe successful copies / differences, not errors.
    if ($copyExitCode -ge 8) {
        if (-not $Watch) { throw "Copy failed (robocopy $copyExitCode). See $logPath" }
        Write-Warning "Some files could not be copied (robocopy $copyExitCode); retrying next cycle. See $logPath"
    } else {
        Write-Host 'Local copy complete. OneDrive uploads these files through its desktop sync client.'
    }
    if ($Watch) { Start-Sleep -Seconds $IntervalSeconds }
} while ($Watch)
exit 0
