<#
.SYNOPSIS
Run the native and Python regression checks, with optional rebuilding/mod checks.
.DESCRIPTION
Build rebuilds an existing CMake configuration; it does not configure a new one.
Scope selects a focused slice. Policy includes native contract fixtures.
All, Policy and Mod also check generated catalogue freshness with Python.
Mod runs the Fabric mod checks with Java 25. CUDA requires a capable machine.
.EXAMPLE
.\scripts\check.ps1 -Build
.EXAMPLE
.\scripts\check.ps1 -Mod -DryRun
#>
[CmdletBinding()]
param(
    [ValidateSet('All', 'Native', 'Training', 'Policy', 'Rewards', 'Dashboard', 'Mod')][string] $Scope = 'All',
    [string] $BuildDirectory = 'simulator/build',
    [string] $Python,
    [switch] $Build,
    [switch] $SkipNative,
    [switch] $SkipPython,
    [switch] $Mod,
    [switch] $DryRun
)

. (Join-Path $PSScriptRoot '_common.ps1')

if ($SkipNative -and $Build) { throw '-Build cannot be combined with -SkipNative.' }
if ($Build -and $Scope -in @('Dashboard', 'Mod')) { throw '-Build requires a native or training scope.' }
$runNative = -not $SkipNative -and $Scope -in @('All', 'Native', 'Policy')
$runPython = -not $SkipPython -and $Scope -in @('All', 'Training', 'Policy', 'Rewards', 'Dashboard')
$runMod = $Mod -or $Scope -eq 'Mod'
if (-not ($runNative -or $runPython -or $runMod)) { throw 'No checks selected.' }
if ($Scope -in @('All', 'Policy', 'Mod') -or $Mod) {
    Invoke-WorkflowCommand (Get-WorkflowPython $Python) @('-B', (Join-Path $RepositoryRoot 'scripts/generate-contract.py'), '--check') -DryRun:$DryRun
}
$buildPath = Resolve-RepositoryPath $BuildDirectory
if ($Build) {
    $buildArguments = @('--build', $buildPath, '--config', 'Release', '--parallel', '1')
    if ($Scope -in @('Training', 'Policy', 'Rewards')) { $buildArguments += @('--target', 'minecraft_sim_training') }
    if ($Scope -eq 'Policy') { $buildArguments += 'minecraft_sim_checks' }
    Invoke-WorkflowCommand (Get-WorkflowTool 'cmake') $buildArguments -DryRun:$DryRun
}
if ($runNative) {
    $nativeArguments = @('--test-dir', $buildPath, '-C', 'Release', '--output-on-failure', '--no-tests=error')
    if ($Scope -eq 'Policy') { $nativeArguments += @('-R', '^shared_core_checks$') }
    Invoke-WorkflowCommand (Get-WorkflowTool 'ctest') $nativeArguments -DryRun:$DryRun
}
if ($runPython) {
    $testArguments = @('-B', '-m', 'unittest')
    if ($Scope -ne 'Dashboard') { $testArguments += 'simulator.training.test_training' }
    if ($Scope -in @('All', 'Dashboard')) { $testArguments += 'simulator.training.test_dashboard' }
    if ($Scope -eq 'Policy') { $testArguments += @('-k', 'contract', '-k', 'configured', '-k', 'policy') }
    if ($Scope -eq 'Rewards') { $testArguments += @('-k', 'reward', '-k', 'absorption') }
    Invoke-WorkflowCommand (Get-WorkflowPython $Python) ($testArguments + '-v') -DryRun:$DryRun
}
if ($runMod) {
    $modTask = if ($Scope -eq 'Mod') { 'check' } else { 'build' }
    & (Join-Path $PSScriptRoot 'build-mod.ps1') -Task $modTask -DryRun:$DryRun
}