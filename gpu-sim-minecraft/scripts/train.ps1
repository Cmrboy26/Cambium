<#
.SYNOPSIS
Start or resume standalone CPU/CUDA training, optionally saving replay samples.
.DESCRIPTION
Run from any directory; relative paths are resolved from this repository.
ReplayEvery counts completed duels, not optimizer updates. Without a Minecraft
destination, samples stay in the run's replays folder. Use DryRun to preview.
.EXAMPLE
.\scripts\train.ps1 -Environment simulator/environments/duel-full-gear.json -ReplayEvery 100
.EXAMPLE
.\scripts\train.ps1 -ReplayEvery 100 -ReplayToMinecraft -MinecraftDirectory 'D:\Games\MyInstance\minecraft'
.EXAMPLE
.\scripts\train.ps1 -Resume latest -Updates 100
#>
[CmdletBinding()]
param(
    [ValidateSet('cpu', 'cuda')][string] $Device = 'cpu',
    [ValidateSet('v1', 'v2')][string] $Schema = 'v1',
    [ValidateRange(0, 65536)][int] $Environments = 0,
    [ValidateRange(1, 2147483647)][int] $Updates = 100,
    [ValidateRange(1, 2147483647)][int] $RolloutSteps = 128,
    [ValidateRange(1, 256)][int] $Threads = 4,
    [ValidateRange(0, 2147483647)][int] $ReplayEvery = 0,
    [ValidateRange(1, 2147483647)][int] $SaveEvery = 10,
    [string] $ReplayDirectory,
    [string] $MinecraftDirectory,
    [string] $Environment,
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
if ($Plan -and $Environment) { throw 'Choose -Plan or -Environment, not both.' }
if ($Initialize -and $Resume) { throw 'Choose -Initialize or -Resume, not both.' }
if ($Initialize -and -not $Plan) { throw '-Initialize requires -Plan.' }
$command = @('-B', '-m', 'simulator.training.train', '--device', $Device,
    '--updates', "$Updates", '--rollout-steps', "$RolloutSteps", '--threads', "$Threads", '--save-every', "$SaveEvery")
if ($Environments -gt 0) { $command += @('--envs', "$Environments") }
if ($Resume) {
    $checkpoint = Get-WorkflowCheckpoint $Resume $RunsDirectory
    $command += @('--resume', $checkpoint)
    if (-not $Output) { $Output = Split-Path -Parent $checkpoint }
    $configuration = Join-Path (Split-Path -Parent $checkpoint) 'config.json'
    if (Test-Path -LiteralPath $configuration -PathType Leaf) {
        $saved = Get-Content -LiteralPath $configuration -Raw | ConvertFrom-Json
        $savedSchemaId = $saved.metadata.schema.PSObject.Properties['id']
        if (-not $PSBoundParameters.ContainsKey('Schema') -and $savedSchemaId -and $savedSchemaId.Value -eq 'minecraft_sim_duel_31_v2') {
            $Schema = 'v2'
        }
        $command += @('--hidden-dim', "$($saved.metadata.hidden_dim)")
        $savedPlan = $saved.PSObject.Properties['plan']
        if ($savedPlan -and $savedPlan.Value) {
            if (-not $PSBoundParameters.ContainsKey('Environments')) {
                $command += @('--envs', "$($saved.options.envs)")
            }
            if (-not $PSBoundParameters.ContainsKey('RolloutSteps')) {
                $command += @('--rollout-steps', "$($saved.options.rollout_steps)")
            }
            if ($null -ne $saved.options.max_ticks) {
                $command += @('--max-ticks', "$($saved.options.max_ticks)")
            }
        } else {
            $command += @('--max-ticks', "$($saved.metadata.max_ticks)")
        }
    } else {
        Write-Warning 'No adjacent config.json; supply original hidden dimension/max ticks through -ExtraArguments if needed. The environment is restored from the checkpoint; task compatibility is validated.'
    }
}
$command += @('--schema', $Schema)
if ($Environment) { $command += @('--environment', (Resolve-RepositoryPath $Environment)) }
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