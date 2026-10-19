<#
.SYNOPSIS
Start or resume standalone CPU/CUDA training, optionally saving replay samples.
.DESCRIPTION
Run from any directory; relative paths are resolved from this repository.
ReplayEvery counts completed duels, not optimizer updates. Without a Minecraft
destination, samples stay in the run's replays folder. Use DryRun to preview.
.EXAMPLE
.\scripts\train.ps1 -Plan simulator/training/presets/shield-axe-curriculum.json -ReplayEvery 100
.EXAMPLE
.\scripts\train.ps1 -ReplayEvery 100 -ReplayToMinecraft -MinecraftDirectory 'D:\Games\MyInstance\minecraft'
.EXAMPLE
.\scripts\train.ps1 -Resume latest -Updates 100
#>
[CmdletBinding()]
param(
    [ValidateSet('cpu', 'cuda')][string] $Device = 'cpu',
    [ValidateRange(0, 65536)][int] $Environments = 0,
    [ValidateRange(1, 2147483647)][int] $Updates = 100,
    [ValidateRange(1, 2147483647)][int] $RolloutSteps = 128,
    [ValidateRange(1, 256)][int] $Threads = 4,
    [ValidateRange(0, 2147483647)][int] $ReplayEvery = 0,
    [ValidateRange(1, 2147483647)][int] $SaveEvery = 10,
    [string] $ReplayDirectory,
    [string] $MinecraftDirectory,
    [string] $Plan,
    [string] $Initialize,
    [string] $Library,
    [string] $Output,
    [string] $Resume,
    [string] $RunsDirectory = 'simulator/training/runs',
    [string] $Python,
    [string[]] $ExtraArguments = @(),
    [switch] $ReplayToMinecraft,
    [switch] $DryRun
)

. (Join-Path $PSScriptRoot '_common.ps1')

if (($ReplayDirectory -or $ReplayToMinecraft -or $MinecraftDirectory) -and $ReplayEvery -eq 0) {
    throw 'Replay output options require -ReplayEvery N greater than zero.'
}
if ($ReplayDirectory -and ($ReplayToMinecraft -or $MinecraftDirectory)) {
    throw 'Choose -ReplayDirectory OR -ReplayToMinecraft/-MinecraftDirectory, not both.'
}
if ($Initialize -and $Resume) { throw 'Choose -Initialize or -Resume, not both.' }
$command = @('-B', '-m', 'simulator.training.train', '--device', $Device,
    '--updates', "$Updates", '--threads', "$Threads", '--save-every', "$SaveEvery")
if ($Environments -gt 0) { $command += @('--envs', "$Environments") }
if ($PSBoundParameters.ContainsKey('RolloutSteps')) { $command += @('--rollout-steps', "$RolloutSteps") }
if ($Resume) {
    $checkpoint = Get-WorkflowCheckpoint $Resume $RunsDirectory
    $command += @('--resume', $checkpoint)
    if (-not $Output) { $Output = Split-Path -Parent $checkpoint }
}
if ($Plan) { $command += @('--plan', (Resolve-RepositoryPath $Plan)) }
if ($Initialize) { $command += @('--initialize', (Resolve-RepositoryPath $Initialize)) }
if ($Library) { $command += @('--library', (Resolve-RepositoryPath $Library)) }
if ($Output) { $command += @('--output', (Resolve-RepositoryPath $Output)) }
if ($ReplayEvery -gt 0) {
    $command += @('--replay-every', "$ReplayEvery")
    if ($ReplayToMinecraft -or $MinecraftDirectory) { $ReplayDirectory = Get-MinecraftReplayDirectory $MinecraftDirectory }
    if ($ReplayDirectory) { $command += @('--replay-dir', (Resolve-RepositoryPath $ReplayDirectory)) }
}
$command += $ExtraArguments
Invoke-WorkflowCommand (Get-WorkflowPython $Python) $command -DryRun:$DryRun