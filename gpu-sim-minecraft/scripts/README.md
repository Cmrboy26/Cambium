# Workflow Scripts

Windows PowerShell 5.1+ helpers. Run these commands from the repository root.
Every launcher supports `-DryRun` to preview its commands.

## First-Time Setup

Install Git, Python 3.10+, CMake 3.24+ and a C++20 toolchain, such as Visual
Studio 2022 Build Tools with the C++ workload. Open a compiler-enabled terminal:

```powershell
.\scripts\setup.ps1
.\scripts\train.ps1 -Updates 1 -Environments 2 -RolloutSteps 16
```

Setup creates `.venv`, installs Python requirements, builds the CPU simulator
and runs checks. It does not install external toolchains. Recreate generated
environments and build caches on each machine.

GPU training also requires an NVIDIA GPU, compatible CUDA Toolkit/compiler and
CUDA-enabled PyTorch. See the [GPU setup](AGENTS.md#run-gpu).

## Common Workflows

| Task | Command |
| --- | --- |
| Train on CPU | `.\scripts\train.ps1` |
| Train on CUDA | `.\scripts\train.ps1 -Device cuda -Environments 8192` |
| Train sword/axe/shield controls | `.\scripts\train.ps1 -Schema v2` |
| Train a randomized aiming/combat curriculum | `.\scripts\train.ps1 -Plan simulator/training/presets/gear-curriculum.json -Updates 1800` |
| Resume newest checkpoint | `.\scripts\train.ps1 -Resume latest` |
| Open training charts | `.\scripts\serve-dashboard.ps1` |
| Choose and serve a saved policy | `.\scripts\serve-policy.ps1` |
| Configure in-game Rosie replay/duel downloads | `.\scripts\rosie-play.ps1 -Setup` |
| Build/check native and Python code | `.\scripts\check.ps1 -Build` |
| Also check the mod | `.\scripts\check.ps1 -Build -Mod` |
| Check policy/contract changes | `.\scripts\check.ps1 -Scope Policy -Build -Mod` |
| Check only the dashboard | `.\scripts\check.ps1 -Scope Dashboard` |
| Validate and benchmark CUDA physics | `.\scripts\run-gpu.ps1 -Mode All` |

Dashboard: **http://127.0.0.1:8767**. Inference uses port **8766** by default.
Leave their terminals running; Ctrl+C stops them. Choose another `-Port` if busy.
`latest` means most recently modified, not best-performing. Only load trusted
checkpoints. CUDA requests never silently fall back to CPU.

## Minecraft Folder

Install JDK 25 and create a Fabric Minecraft **26.3** instance with matching
Fabric API. Select its game folder containing `mods`, with Minecraft closed:

```powershell
$env:MCSIM_MINECRAFT_DIRECTORY = Read-Host 'Minecraft game folder (containing mods)'
.\scripts\build-mod.ps1 -Install -DryRun
.\scripts\build-mod.ps1 -Install
```

Alternatively pass `-MinecraftDirectory`. A single standard installation can be
detected automatically; multiple candidates require a choice. No launcher or
instance name is assumed. Remove differently named older Cambium JARs yourself.

## Replays And Live Duels

```powershell
.\scripts\train.ps1 -ReplayEvery 100 -ReplayToMinecraft
.\scripts\export-replay.ps1 -ToMinecraft
.\scripts\serve-policy.ps1 -List
.\scripts\serve-policy.ps1
```

Training samples a learned-policy duel every 100 finished rounds, not updates.
`export-replay.ps1` instead exports a scripted duel. Without a destination,
training samples stay in the run's `replays` folder.

In private singleplayer, use `/simreplay load latest`, then `/simreplay restart`
for playback, or `/simduel arena` with inference running for live play.
See the [mod guide](../cambiummod/README.md) for controls and world-safety limits.

## Development

[AGENTS.md](AGENTS.md) contains every script's parameter/default table, path
resolution rules, GPU runbook and troubleshooting. Use
`Get-Help .\scripts\train.ps1 -Detailed` for launcher help.