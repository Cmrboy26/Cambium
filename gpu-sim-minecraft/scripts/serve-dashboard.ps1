<#
.SYNOPSIS
Host the read-only simulator training dashboard over loopback HTTP.
.DESCRIPTION
Shows live and saved runs without starting the legacy backend. Relative paths
resolve from the repository root. Leave the terminal open; Ctrl+C stops it.
.EXAMPLE
.\scripts\serve-dashboard.ps1
.EXAMPLE
.\scripts\serve-dashboard.ps1 -RunsDirectory experiments -Port 8768
.EXAMPLE
.\scripts\serve-dashboard.ps1 -Run @('experiments/close-duel', 'experiments/wide-duel') -DryRun
#>
[CmdletBinding()]
param(
    [string] $RunsDirectory = 'simulator/training/runs',
    [string[]] $Run = @(),
    [ValidateRange(1, 65535)][int] $Port = 8767,
    [string] $Python,
    [switch] $DryRun
)

. (Join-Path $PSScriptRoot '_common.ps1')

$command = @('-B', '-m', 'simulator.training.dashboard',
    '--runs-dir', (Resolve-RepositoryPath $RunsDirectory), '--port', "$Port")
foreach ($runDirectory in $Run) {
    $command += @('--run', (Resolve-RepositoryPath $runDirectory))
}
Write-Host "Dashboard: http://127.0.0.1:$Port"
Write-Host 'Leave this terminal open; Ctrl+C stops the dashboard.'
Invoke-WorkflowCommand (Get-WorkflowPython $Python) $command -DryRun:$DryRun