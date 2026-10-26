<#
.SYNOPSIS
Build or check the Fabric mod; install only when explicitly requested.
.DESCRIPTION
Generates the shared policy catalogue, then requires JDK 25 or newer. Python
overrides the standard interpreter discovery. JavaHome overrides the cached
project JDK, JAVA_HOME and PATH discovery. Installation requires Minecraft to
be closed; it replaces the same-named mod JAR but does not remove differently
named older versions.
.EXAMPLE
.\scripts\build-mod.ps1
.EXAMPLE
.\scripts\build-mod.ps1 -Install -MinecraftDirectory 'D:\Games\MyInstance\minecraft' -DryRun
.EXAMPLE
.\scripts\build-mod.ps1 -JavaHome 'C:\Program Files\Java\jdk-25' -Task check
#>
[CmdletBinding()]
param(
    [ValidateSet('build', 'check', 'compileJava', 'replayAnalysisChecks')][string] $Task = 'build',
    [string] $Python,
    [string] $JavaHome,
    [string] $MinecraftDirectory,
    [switch] $Install,
    [switch] $DryRun
)

. (Join-Path $PSScriptRoot '_common.ps1')

if ($Install -and $Task -ne 'build') { throw '-Install requires -Task build.' }
if ($MinecraftDirectory -and -not $Install) { throw '-MinecraftDirectory requires -Install.' }
if (-not $JavaHome) {
    $portable = Join-Path $RepositoryRoot 'cambiummod/.gradle/java25'
    if (Test-Path -LiteralPath $portable -PathType Container) {
        $homes = @(Get-ChildItem -LiteralPath $portable -Directory | Where-Object {
            Test-Path -LiteralPath (Join-Path $_.FullName 'bin/java.exe') -PathType Leaf
        } | Sort-Object Name -Descending)
        if ($homes.Count -gt 0) { $JavaHome = $homes[0].FullName }
    }
    if (-not $JavaHome) { $JavaHome = $env:JAVA_HOME }
    if (-not $JavaHome) {
        $javaCommand = Get-Command java -ErrorAction SilentlyContinue
        if ($javaCommand) { $JavaHome = Split-Path -Parent (Split-Path -Parent $javaCommand.Source) }
    }
}
if (-not $JavaHome) { throw 'JDK 25 required. Specify -JavaHome or configure JAVA_HOME; no Java installation is modified by this script.' }
$selectedJavaHome = Resolve-RepositoryPath $JavaHome
$previousJavaHome = $env:JAVA_HOME
$destination = $null
if ($Install) { $destination = Join-Path (Get-MinecraftGameDirectory $MinecraftDirectory) 'mods' }
try {
    if (-not $DryRun) {
        $java = Join-Path $selectedJavaHome 'bin/java.exe'
        if (-not (Test-Path -LiteralPath $java -PathType Leaf)) { throw "Java executable missing under $selectedJavaHome" }
        $version = (& $java '--version') -join "`n"
        if ($LASTEXITCODE -ne 0 -or $version -notmatch '(?m)^(?:openjdk|java) (\d+)' -or [int] $Matches[1] -lt 25) {
            throw "JDK 25 or newer required. Selected Java: $selectedJavaHome"
        }
    }
    $env:JAVA_HOME = $selectedJavaHome
    Write-Host "JAVA_HOME=$selectedJavaHome (restored on exit)"
    Invoke-WorkflowCommand (Get-WorkflowPython $Python) @('-B',
        (Join-Path $RepositoryRoot 'scripts/generate-contract.py')) -DryRun:$DryRun
    Invoke-WorkflowCommand (Join-Path $RepositoryRoot 'cambiummod/gradlew.bat') @('-p',
        (Join-Path $RepositoryRoot 'cambiummod'), $Task, '--console=plain', '--no-daemon') -DryRun:$DryRun
} finally {
    $env:JAVA_HOME = $previousJavaHome
}
if ($Install) {
    $jar = Join-Path $RepositoryRoot 'cambiummod/build/libs/cambiumfabric-1.0.0.jar'
    Write-Host "Install: $jar -> $destination (close Minecraft first; replaces the same-named mod JAR)"
    if (-not $DryRun) {
        if (-not (Test-Path -LiteralPath $jar -PathType Leaf)) { throw "Built JAR missing: $jar" }
        New-Item -ItemType Directory -Path $destination -Force | Out-Null
        Copy-Item -LiteralPath $jar -Destination $destination -Force
    }
}