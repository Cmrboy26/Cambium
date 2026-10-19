<#
.SYNOPSIS
Set up the Python environment and configure, build and test the CPU simulator.
.DESCRIPTION
Creates or reuses the repository .venv, installs training requirements, then
configures a CPU-only CMake build and runs native and Python checks.
#>
[CmdletBinding()]
param(
    [string] $BuildDirectory = 'simulator/build',
    [string] $Python,
    [switch] $DryRun
)

. (Join-Path $PSScriptRoot '_common.ps1')

$venv = Resolve-RepositoryPath '.venv'
$venvPython = Join-Path $venv 'Scripts/python.exe'
$bootstrapPython = Get-WorkflowPython $Python
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    Invoke-WorkflowCommand $bootstrapPython @('-m', 'venv', $venv) -DryRun:$DryRun
}

Invoke-WorkflowCommand $venvPython @('-m', 'pip', 'install', '-r',
    (Resolve-RepositoryPath 'simulator/training/requirements.txt')) -DryRun:$DryRun

$buildPath = Resolve-RepositoryPath $BuildDirectory
$configure = @('-S', (Join-Path $RepositoryRoot 'simulator'), '-B', $buildPath,
    '-DMCSIM_ENABLE_CUDA=OFF', '-DBUILD_TESTING=ON', '-DCMAKE_BUILD_TYPE=Release')
Invoke-WorkflowCommand (Get-WorkflowTool 'cmake') $configure -DryRun:$DryRun
Invoke-WorkflowCommand (Get-WorkflowTool 'cmake') @('--build', $buildPath,
    '--config', 'Release', '--parallel') -DryRun:$DryRun
Invoke-WorkflowCommand (Get-WorkflowTool 'ctest') @('--test-dir', $buildPath,
    '-C', 'Release', '--output-on-failure', '--no-tests=error') -DryRun:$DryRun
Invoke-WorkflowCommand $venvPython @('-B', '-m', 'unittest',
    'simulator.training.test_training', '-v') -DryRun:$DryRun