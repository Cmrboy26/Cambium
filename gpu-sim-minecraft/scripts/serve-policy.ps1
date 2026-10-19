<#
.SYNOPSIS
Serve a trusted standalone checkpoint for live /simduel inference.
.DESCRIPTION
Defaults to CPU inference and prompts for a run, newest checkpoint first, with
environment names and completed updates. Use -Checkpoint latest to skip selection.
Leave the service running; Ctrl+C stops it. No native simulator library is needed.
.EXAMPLE
.\scripts\serve-policy.ps1
.EXAMPLE
.\scripts\serve-policy.ps1 -List
.EXAMPLE
.\scripts\serve-policy.ps1 -Checkpoint latest -DryRun
.EXAMPLE
.\scripts\serve-policy.ps1 -RunsDirectory experiments -Port 8767
#>
[CmdletBinding()]
param(
    [string] $Checkpoint = 'latest',
    [string] $RunsDirectory = 'simulator/training/runs',
    [ValidateSet('cpu', 'cuda')][string] $Device = 'cpu',
    [ValidateRange(1, 65535)][int] $Port = 8766,
    [ValidateRange(1, 256)][int] $Threads = 1,
    [ValidateRange(0, 2147483647)][int] $Seed = 1,
    [string] $Python,
    [switch] $Deterministic,
    [switch] $List,
    [switch] $DryRun
)

. (Join-Path $PSScriptRoot '_common.ps1')

if ($List -or -not $PSBoundParameters.ContainsKey('Checkpoint')) {
    $checkpoints = @(Get-RunCheckpoints $RunsDirectory)
    if ($checkpoints.Count -eq 0) {
        Write-Host "No checkpoints in $(Resolve-RepositoryPath $RunsDirectory). Start training with scripts/train.ps1."
        return
    }
    $runs = @(for ($index = 0; $index -lt $checkpoints.Count; $index++) {
        $checkpointFile = $checkpoints[$index]
        $environment = 'Unknown'
        $updates = 'Unknown'
        $configPath = Join-Path $checkpointFile.DirectoryName 'config.json'
        if (Test-Path -LiteralPath $configPath -PathType Leaf) {
            try {
                $configuration = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
                if ($configuration.metadata.environment.name) {
                    $environment = $configuration.metadata.environment.name
                }
            } catch {
                Write-Warning "Could not read environment for $($checkpointFile.Directory.Name): $_"
            }
        }
        $metricsPath = Join-Path $checkpointFile.DirectoryName 'metrics.jsonl'
        if (Test-Path -LiteralPath $metricsPath -PathType Leaf) {
            try {
                $progress = Get-Content -LiteralPath $metricsPath -Tail 1 | ConvertFrom-Json
                if ($null -ne $progress) { $updates = $progress.update }
            } catch {
                Write-Warning "Could not read updates for $($checkpointFile.Directory.Name): $_"
            }
        }
        [PSCustomObject]@{
            Number = $index + 1
            Run = $checkpointFile.Directory.Name
            Environment = $environment
            Updates = $updates
            LastSaved = $checkpointFile.LastWriteTime
            Checkpoint = $checkpointFile.FullName
        }
    })
    $runs | Select-Object Number, Run, Environment, Updates, LastSaved | Format-Table -AutoSize -Wrap | Out-Host
    if ($List) { return }

    while ($true) {
        $choice = (Read-Host 'Select a run number (or Q to cancel)').Trim()
        if ($choice -eq 'q') { return }
        $number = 0
        if ([int]::TryParse($choice, [ref] $number) -and $number -ge 1 -and $number -le $runs.Count) {
            $Checkpoint = $runs[$number - 1].Checkpoint
            break
        }
        Write-Host "Enter a number from 1 to $($runs.Count), or Q to cancel."
    }
}

$selected = Get-WorkflowCheckpoint $Checkpoint $RunsDirectory
Write-Host "Checkpoint: $selected"
Write-Host "Minecraft: /simduel arena $Port OR /simduel pos1, /simduel pos2, /simduel world $Port"
Write-Host 'Leave this terminal open; Ctrl+C stops inference. Restart to load newer weights.'
$command = @('-B', '-m', 'simulator.training.serve', $selected, '--device', $Device,
    '--port', "$Port", '--threads', "$Threads", '--seed', "$Seed")
if ($Deterministic) { $command += '--deterministic' }
Invoke-WorkflowCommand (Get-WorkflowPython $Python) $command -DryRun:$DryRun