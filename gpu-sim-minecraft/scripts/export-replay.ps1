<#
.SYNOPSIS
Export a scripted simulator duel, not a trained-policy replay.
.DESCRIPTION
Requires a built runner. Use train.ps1 -ReplayEvery for learned-policy samples.
ToMinecraft resolves the instance via an explicit path, MCSIM_MINECRAFT_DIRECTORY,
or unambiguous standard launcher detection. DryRun never exports a replay.
.EXAMPLE
.\scripts\export-replay.ps1 -Environment simulator/environments/duel-full-gear.json -ToMinecraft
.EXAMPLE
.\scripts\export-replay.ps1 -Device cuda -BuildDirectory simulator/build-cuda -DryRun
#>
[CmdletBinding()]
param(
    [ValidateSet('cpu', 'cuda')][string] $Device = 'cpu',
    [ValidateRange(1, 100000)][int] $Ticks = 300,
    [string] $BuildDirectory,
    [string] $Runner,
    [string] $Environment,
    [string] $Output,
    [string] $MinecraftDirectory,
    [switch] $ToMinecraft,
    [switch] $DryRun
)

. (Join-Path $PSScriptRoot '_common.ps1')

if ($Output -and ($ToMinecraft -or $MinecraftDirectory)) { throw 'Choose -Output OR -ToMinecraft/-MinecraftDirectory, not both.' }
if (-not $BuildDirectory) { $BuildDirectory = if ($Device -eq 'cuda') { 'simulator/build-cuda' } else { 'simulator/build' } }
if (-not $Runner) {
    $Runner = Get-BuildExecutable $BuildDirectory 'minecraft_sim_runner'
}
$command = @('--duel', '--ticks', "$Ticks")
if ($Device -eq 'cuda') { $command += '--cuda' }
if ($Environment) { $command += @('--environment', (Resolve-RepositoryPath $Environment)) }
if ($ToMinecraft -or $MinecraftDirectory) {
    $folder = Get-MinecraftReplayDirectory $MinecraftDirectory
    $Output = Join-Path $folder ("duel-scripted-{0}-{1}.jsonl" -f (Get-Date -Format 'yyyyMMdd-HHmmss-fff'), ([Guid]::NewGuid().ToString('N').Substring(0, 8)))
    if (-not $DryRun) { New-Item -ItemType Directory -Path $folder -Force | Out-Null }
}
$command += '--replay'
if ($Output) { $command += Resolve-RepositoryPath $Output }
Write-Host 'This exports a scripted simulator duel, NOT a trained-policy sample. Use train.ps1 -ReplayEvery N for policy samples.'
Invoke-WorkflowCommand (Resolve-RepositoryPath $Runner) $command -DryRun:$DryRun