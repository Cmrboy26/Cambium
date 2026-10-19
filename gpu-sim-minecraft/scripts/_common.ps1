Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$RepositoryRoot = Split-Path -Parent $PSScriptRoot

function Resolve-RepositoryPath {
    param([Parameter(Mandatory = $true)][string] $Path)

    if ([IO.Path]::IsPathRooted($Path)) {
        return [IO.Path]::GetFullPath($Path)
    }
    return [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $Path))
}

function Get-WorkflowPython {
    param([string] $Python)

    if ($Python) {
        if (-not [IO.Path]::IsPathRooted($Python) -and $Python -notmatch '[/\\]') {
            return Get-WorkflowTool $Python
        }
        return Resolve-RepositoryPath $Python
    }
    $candidates = @((Join-Path $RepositoryRoot '.venv/Scripts/python.exe'), (Join-Path $RepositoryRoot '.venv/bin/python'))
    if ($env:VIRTUAL_ENV) {
        $candidates += Join-Path $env:VIRTUAL_ENV 'Scripts/python.exe'
        $candidates += Join-Path $env:VIRTUAL_ENV 'bin/python'
    }
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    foreach ($name in @('python', 'python3')) {
        $found = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue
        if ($found) { return $found.Source }
    }
    return 'python'
}

function Get-WorkflowTool {
    param([string] $Name)

    $found = Get-Command $Name -ErrorAction SilentlyContinue
    if ($found) { return $found.Source }
    if ($Name -in @('cmake', 'ctest') -and $env:ProgramFiles) {
        $installed = Join-Path $env:ProgramFiles "CMake/bin/$Name.exe"
        if (Test-Path -LiteralPath $installed -PathType Leaf) { return $installed }
    }
    return $Name
}

function Get-RunCheckpoints {
    param([string] $RunsDirectory = 'simulator/training/runs')

    $folder = Resolve-RepositoryPath $RunsDirectory
    if (Test-Path -LiteralPath $folder -PathType Container) {
        Get-ChildItem -LiteralPath $folder -Filter 'latest.pt' -File -Recurse |
            Sort-Object -Property @{ Expression = 'LastWriteTimeUtc'; Descending = $true }, FullName
    }
}

function Get-WorkflowCheckpoint {
    param([string] $Checkpoint = 'latest', [string] $RunsDirectory = 'simulator/training/runs')

    if (-not $Checkpoint -or $Checkpoint -eq 'latest') {
        $found = @(Get-RunCheckpoints $RunsDirectory)
        if ($found.Count -eq 0) {
            throw "No latest.pt checkpoint in $(Resolve-RepositoryPath $RunsDirectory). Run scripts/train.ps1 first or supply -Checkpoint with a real file path."
        }
        return $found[0].FullName
    }
    $resolved = Resolve-RepositoryPath $Checkpoint
    if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
        throw "Checkpoint not found: $resolved. Use scripts/serve-policy.ps1 -List to see available runs."
    }
    return $resolved
}

function Get-MinecraftGameDirectories {
    $candidates = @()
    if ($env:APPDATA) { $candidates += Join-Path $env:APPDATA '.minecraft' }
    foreach ($base in @($env:APPDATA, $env:LOCALAPPDATA) | Where-Object { $_ } | Select-Object -Unique) {
        foreach ($launcher in @('PrismLauncher', 'MultiMC')) {
            $instances = Join-Path $base "$launcher/instances"
            if (Test-Path -LiteralPath $instances -PathType Container) {
                foreach ($instance in Get-ChildItem -LiteralPath $instances -Directory) {
                    $candidates += Join-Path $instance.FullName 'minecraft'
                    $candidates += Join-Path $instance.FullName '.minecraft'
                }
            }
        }
    }
    $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Container } |
        ForEach-Object { [IO.Path]::GetFullPath($_) } | Sort-Object -Unique
}

function Get-MinecraftGameDirectory {
    param([string] $MinecraftDirectory)

    $selected = $MinecraftDirectory
    if (-not $selected) { $selected = $env:MCSIM_MINECRAFT_DIRECTORY }
    if (-not $selected) {
        $found = @(Get-MinecraftGameDirectories)
        if ($found.Count -eq 1) {
            $selected = $found[0]
        } elseif ($found.Count -gt 1) {
            throw ("Multiple Minecraft game folders found:`n  " + ($found -join "`n  ") +
                "`nChoose -MinecraftDirectory <game-folder> or set MCSIM_MINECRAFT_DIRECTORY. Portable/custom launchers can use either option too.")
        } else {
            throw 'No Minecraft game folder detected. Specify -MinecraftDirectory <game-folder> or set MCSIM_MINECRAFT_DIRECTORY. Select the instance folder containing mods, not the mods folder itself.'
        }
    }
    $game = Resolve-RepositoryPath $selected
    if ((Split-Path -Leaf $game) -in @('mods', 'sim-replays')) {
        throw 'Select the Minecraft game folder containing mods and sim-replays, not either subfolder.'
    }
    if (-not (Test-Path -LiteralPath $game -PathType Container)) {
        throw "Minecraft game directory not found: $game. Specify -MinecraftDirectory for your actual instance."
    }
    Write-Host "Minecraft game directory: $game"
    return $game
}

function Get-MinecraftReplayDirectory {
    param([string] $MinecraftDirectory)

    $game = Get-MinecraftGameDirectory $MinecraftDirectory
    return Join-Path $game 'sim-replays'
}

function Invoke-WorkflowCommand {
    param([string] $Executable, [string[]] $CommandArguments, [switch] $DryRun)

    $display = @($Executable) + $CommandArguments | ForEach-Object { '"' + $_ + '"' }
    Write-Host ($(if ($DryRun) { 'DRY RUN> ' } else { '> ' }) + ($display -join ' '))
    if ($DryRun) {
        return
    }
    if (-not (Get-Command $Executable -ErrorAction SilentlyContinue)) {
        throw "Executable not found: $Executable. Set up the repository .venv/toolchain or select the executable explicitly. See scripts/README.md."
    }
    Push-Location $RepositoryRoot
    try {
        & $Executable @CommandArguments
        if ($LASTEXITCODE -ne 0) {
            throw "$Executable failed with exit code $LASTEXITCODE. Stopping before subsequent steps."
        }
    } finally {
        Pop-Location
    }
}

function Get-BuildExecutable {
    param([string] $BuildDirectory, [string] $Name)

    $buildPath = Resolve-RepositoryPath $BuildDirectory
    $filename = if ($env:OS -eq 'Windows_NT') { "$Name.exe" } else { $Name }
    $candidates = @((Join-Path $buildPath "Release/$filename"), (Join-Path $buildPath $filename))
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    return $candidates[0]
}