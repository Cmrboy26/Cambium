<#
.SYNOPSIS
Configure the in-game Rosie launcher, or sync artifacts and serve a local policy.
.EXAMPLE
.\scripts\rosie-play.ps1 -Setup -MinecraftDirectory 'C:\Games\Minecraft' -SshHost 2
.EXAMPLE
.\scripts\rosie-play.ps1 -Action replay -Run rosie-123
.EXAMPLE
.\scripts\rosie-play.ps1 -Action duel -Run rosie-123 -Offline
.EXAMPLE
.\scripts\rosie-play.ps1 -Connect
#>
[CmdletBinding()]
param(
    [switch] $Setup,
    [switch] $Connect,
    [ValidateSet('runs', 'replays', 'replay', 'sync', 'duel')][string] $Action = 'runs',
    [string] $Run = 'latest',
    [string] $Replay = 'latest',
    [ValidateSet('policy', 'latest', 'all')][string] $SyncScope = 'policy',
    [ValidateRange(1, 4)][int] $Downloads = 3,
    [string] $MinecraftDirectory,
    [string] $SshHost = $env:ROSIE_SSH_JUMP,
    [string] $SshUser = $env:ROSIE_SSH_USER,
    [string] $RemoteRunsDirectory = '~/minecraftlol/Cambium/gpu-sim-minecraft/simulator/training/runs',
    [string] $Python,
    [switch] $Offline,
    [switch] $DryRun
)

. (Join-Path $PSScriptRoot '_common.ps1')

if ($Setup -and $Connect) { throw 'Choose -Setup or -Connect, not both.' }
if ($Connect -and $Offline) { throw '-Connect requires an online SSH connection.' }
$game = Get-MinecraftGameDirectory $MinecraftDirectory
$selectedPython = Get-WorkflowPython $Python
$command = @('-u', '-B', '-m', 'simulator.training.rosie')
if ($Setup) {
    if (-not $SshHost) {
        if ($DryRun) { throw 'Specify -SshHost for setup with -DryRun.' }
        $SshHost = (Read-Host 'Assigned Rosie management node (1-4), hostname or SSH alias').Trim()
    }
    if ($SshHost -match '^[1-4]$') { $SshHost = "dh-mgmt$SshHost.hpc.msoe.edu" }
    elseif ($SshHost -match '^dh-mgmt[1-4]$') { $SshHost += '.hpc.msoe.edu' }
    if (-not $SshUser) {
        $SshUser = if ($env:USERNAME) { $env:USERNAME } else { $env:USER }
        if (-not $DryRun) {
            $answer = (Read-Host "Rosie SSH username [$SshUser]").Trim()
            if ($answer) { $SshUser = $answer }
        }
    }
    if (-not $SshUser) { throw 'Specify -SshUser for your Rosie account.' }
    if ($SshUser -notmatch '@') { $SshUser += '@ad.msoe.edu' }
    $command += @('setup', '--minecraft-directory', $game, '--host', $SshHost,
        '--user', $SshUser, '--runs-directory', $RemoteRunsDirectory)
} elseif ($Connect) {
    $command += @('connect', '--minecraft-directory', $game)
} else {
    $command += @($Action, '--minecraft-directory', $game, '--run', $Run, '--replay', $Replay)
    if ($Action -eq 'sync') { $command += @('--scope', $SyncScope, '--workers', $Downloads) }
    if ($Offline) { $command += '--offline' }
}

Invoke-WorkflowCommand $selectedPython $command -DryRun:$DryRun
if ($Setup -and -not $DryRun) {
    Write-Host 'Password login: run .\scripts\rosie-play.ps1 -Connect in this terminal and leave it open.'
    Write-Host "Optional key/agent login check: ssh -l `"$SshUser`" $SshHost"
    Write-Host 'In Minecraft: /simrosie runs, /simrosie replay <run>, /simrosie duel <run>'
    Write-Host 'Enter SSH passwords and host-key confirmations only in the terminal, never Minecraft chat.'
}