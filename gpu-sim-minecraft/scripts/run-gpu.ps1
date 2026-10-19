<#
.SYNOPSIS
Build and validate CUDA physics, optionally measuring physics throughput.
.DESCRIPTION
Requires NVIDIA drivers, a CUDA Toolkit and a supported host compiler. The default
architecture is native (the build machine's visible GPU); supply Architecture for
cross-compilation. CMake selects its generator unless Generator is supplied.
All execution modes require a non-skipped device parity test; DryRun previews only.
.EXAMPLE
.\scripts\run-gpu.ps1 -Mode All -DryRun
.EXAMPLE
.\scripts\run-gpu.ps1 -Generator 'Visual Studio 17 2022'
.EXAMPLE
.\scripts\run-gpu.ps1 -Architecture 80 -BuildDirectory simulator/build-a100
#>
[CmdletBinding()]
param(
    [ValidateSet('Validate', 'Benchmark', 'All')]
    [string] $Mode = 'Validate',
    [string] $BuildDirectory = 'simulator/build-cuda',
    [ValidatePattern('^(native|[0-9]{2,3}(;[0-9]{2,3})*)$')]
    [string] $Architecture = 'native',
    [string] $Generator,
    [ValidateRange(1, 1073741823)]
    [int[]] $Environments = @(8192),
    [ValidateRange(1, 2147483647)]
    [int] $Launches = 5000,
    [ValidateRange(1, 2147483647)]
    [int] $TicksPerLaunch = 1,
    [ValidateRange(0, 2147483647)]
    [int] $WarmupLaunches = 10,
    [switch] $SkipBuild,
    [switch] $DryRun
)

. (Join-Path $PSScriptRoot '_common.ps1')

function Confirm-CudaTests {
    param([string] $Report)

    [xml] $results = Get-Content -LiteralPath $Report -Raw
    $cuda = $results.SelectNodes("//testcase[@name='shared_core_cuda_checks']")
    if ($cuda.Count -ne 1) {
        throw 'CUDA parity test is absent. Configure with MCSIM_ENABLE_CUDA=ON and BUILD_TESTING=ON.'
    }
    $test = $cuda.Item(0)
    if ($test.SelectSingleNode('skipped') -or $test.SelectSingleNode('failure') -or
        $test.SelectSingleNode('error') -or $test.GetAttribute('status') -eq 'notrun') {
        throw 'CUDA parity test did not pass on a device (skipped/failed). Check the NVIDIA driver and device visibility.'
    }
}

$buildPath = Resolve-RepositoryPath $BuildDirectory
$report = Join-Path $buildPath 'gpu-tests.xml'

Push-Location $RepositoryRoot
try {
    if (-not $DryRun) {
        foreach ($tool in @('cmake', 'ctest', 'nvidia-smi')) {
            if (-not (Get-Command (Get-WorkflowTool $tool) -ErrorAction SilentlyContinue)) {
                throw "Missing $tool. See scripts/README.md for the Windows CUDA prerequisites."
            }
        }
        if (-not $SkipBuild -and -not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
            throw 'Missing nvcc. Install the CUDA Toolkit and open a terminal with a supported host compiler configured.'
        }
    }
    Invoke-WorkflowCommand (Get-WorkflowTool 'nvidia-smi') @() -DryRun:$DryRun
    if (-not $SkipBuild) {
        Invoke-WorkflowCommand (Get-WorkflowTool 'nvcc') @('--version') -DryRun:$DryRun
        $configure = @('-S', (Join-Path $RepositoryRoot 'simulator'), '-B', $buildPath,
            '-DMCSIM_ENABLE_CUDA=ON', '-DBUILD_TESTING=ON', '-DCMAKE_BUILD_TYPE=Release', "-DCMAKE_CUDA_ARCHITECTURES=$Architecture")
        if ($Generator) {
            $configure += @('-G', $Generator)
            if ($Generator -like 'Visual Studio*') { $configure += @('-A', 'x64') }
        }
        Invoke-WorkflowCommand (Get-WorkflowTool 'cmake') $configure -DryRun:$DryRun
        Invoke-WorkflowCommand (Get-WorkflowTool 'cmake') @('--build', $buildPath, '--config', 'Release', '--parallel') -DryRun:$DryRun
    }
    Invoke-WorkflowCommand (Get-WorkflowTool 'ctest') @('--test-dir', $buildPath, '-C', 'Release', '--output-on-failure', '--output-junit', $report) -DryRun:$DryRun
    if (-not $DryRun) {
        Confirm-CudaTests $report
        Write-Host "CUDA test passed on device. Test report: $report"
    }
    if ($Mode -in @('Validate', 'All')) {
        Invoke-WorkflowCommand (Get-BuildExecutable $buildPath 'minecraft_sim_runner') @('--duel', '--cuda', '--ticks', '80') -DryRun:$DryRun
    }
    if ($Mode -in @('Benchmark', 'All')) {
        foreach ($count in $Environments) {
            Invoke-WorkflowCommand (Get-BuildExecutable $buildPath 'minecraft_sim_benchmark') @('--environments', "$count",
                '--launches', "$Launches", '--ticks-per-launch', "$TicksPerLaunch", '--warmup-launches', "$WarmupLaunches") -DryRun:$DryRun
        }
    }
    if ($DryRun) {
        Write-Host 'Preview only: no build, device checks, tests or benchmarks were executed.'
    }
} finally {
    Pop-Location
}