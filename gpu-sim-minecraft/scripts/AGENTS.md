# Workflow Scripts: Development And Reference

Follow the repository [development guide](../AGENTS.md). Commands below run from
the repository root. Human-facing setup belongs in [README.md](README.md).

## Launcher Invariants

- Target Windows PowerShell 5.1+. Windows launchers are not validated on other
  operating systems; use direct [native](../simulator/AGENTS.md#linux-build) and
  [Python](../simulator/training/AGENTS.md#setup) commands on Linux.
- Resolve relative paths from the repository, including absolute invocation from
  another working directory. Support paths with spaces.
- [_common.ps1](_common.ps1) contains internal path/command helpers, not a launcher.
- Scripts do not install external Python, Java, CUDA or compiler toolchains,
  change execution policy, elevate privileges, update drivers or fetch updates.
- Every launcher supports `-DryRun`: print resolved commands without training,
  inference, builds, tests, exports, folder creation or copies. Input paths and
  checkpoint selection are still validated.
- Nonzero native exit codes stop execution. Restore working directories and
  temporary `JAVA_HOME` changes. All switches default off.
- Auto Python means explicit `-Python` (command name or executable path), then
  repository `.venv`, active virtual environment, then `python`/`python3` on PATH.
  Print the selected interpreter.
- Use separate build trees for different generators/toolchains. Never delete
  unrelated builds or reuse another machine's CMake/Gradle/venv caches.

## First-Time Setup

Prerequisites: Git, Python 3.10+ (3.12 is suitable), CMake 3.24+, a C++20 compiler
and build driver. Visual Studio 2022 Build Tools with the C++ workload is one
Windows option; [MinGW details](../simulator/AGENTS.md#windows-build) are separate.

`./scripts/setup.ps1` creates/reuses `.venv`, installs training requirements,
configures/builds the CPU simulator and runs native/Python checks. `-Python`
selects the interpreter used to create `.venv`; `-BuildDirectory` selects another
tree, such as `simulator/build-cpu`. This is CPU-only setup. Install the CUDA
toolchain separately and use `run-gpu.ps1` for device validation.

A first training check:

```powershell
.\scripts\train.ps1 -Environment simulator/environments/duel-full-gear.json `
  -Updates 1 -Environments 2 -RolloutSteps 16
```

Omit the size overrides for longer training. Runs contain configuration, metrics
and `latest.pt`; `-ReplayEvery 100` adds samples under the run unless redirected.
Game workflows additionally require JDK 25 and a Fabric Minecraft 26.3 instance
with matching Fabric API. Build/install does not create that instance or API.

## Minecraft Folder

Select the game folder containing `mods`, not the launcher, instance parent,
`mods` itself or a world save. Check its Minecraft version separately.

```powershell
$env:MCSIM_MINECRAFT_DIRECTORY = Read-Host 'Minecraft game folder (containing mods)'
.\scripts\build-mod.ps1 -Install -DryRun
```

Selection order: explicit `-MinecraftDirectory`, then
`MCSIM_MINECRAFT_DIRECTORY`, then a single detected folder in standard Windows
`.minecraft`, PrismLauncher or MultiMC roaming/local application-data locations.
Detection accepts arbitrary instance names and both `minecraft`/`.minecraft`
subfolders; it detects folders, not versions. Multiple candidates are an error
listing choices. Missing paths never fall back silently. Portable/custom
launchers need an explicit path or the environment variable.

The environment variable above lasts for the terminal session; use OS user
environment settings to persist it. Print the selected folder. Installation
writes only `mods`; replay export writes `sim-replays`. `-ReplayDirectory` can
select any destination without detecting Minecraft.

## Train

| Parameter | Accepted value / purpose | Default |
| --- | --- | --- |
| `-Device` | `cpu`, `cuda` | `cpu` |
| `-Schema` | `v1`, `v2`; embedded environment policy wins | `v1`; saved v2 restored on resume when available |
| `-Environments` | Integer 0..65536; parallel worlds | `0`: auto, 16 CPU / 8192 CUDA |
| `-Updates` | Positive integer; additional PPO updates | `100` |
| `-RolloutSteps` | Positive integer; ticks per rollout | `128` |
| `-Threads` | Integer 1..256; CPU threads | `4` |
| `-ReplayEvery` | Nonnegative integer; completed duels between samples | `0`: disabled |
| `-SaveEvery` | Positive integer; updates between checkpoint saves | `10` |
| `-Environment` | Arena/experiment JSON path | Default duel; checkpoint definition on resume |
| `-Plan` | Research training-plan JSON; exclusive with `-Environment` | Unset: shared self-play |
| `-Initialize` | Compatible learner weights for a new `-Plan`; exclusive with `-Resume` | Unset |
| `-Library` | Native training DLL/SO | Discover from native builds |
| `-Output` | Run directory | Timestamped run; checkpoint folder on resume |
| `-Resume` | `latest` or checkpoint path | Unset: new run |
| `-RunsDirectory` | Search root for `-Resume latest` | `simulator/training/runs` |
| `-ReplayDirectory` | Policy sample directory | Run's `replays` folder |
| `-ReplayToMinecraft` | Send samples to game's `sim-replays` | Off |
| `-MinecraftDirectory` | Game folder; also selects replay output | Auto Minecraft when requested |
| `-Python` | Interpreter command or executable path | Auto Python |
| `-ExtraArguments` | String array of Python trainer flags | Empty array |
| `-DryRun` | Preview only | Off |

CUDA requires CUDA PyTorch and a CUDA-built training library; never fall back to
CPU silently. Build the library first. Embedded environment `policy` descriptors
override presets and derive custom dimensions. See the
[experiment contract](../docs/simulation_contract.md).

Resume appends to the selected run unless `-Output` redirects it. Adjacent
`config.json`, when present, restores legacy preset, hidden size and round length;
other options use wrapper defaults unless supplied. Without `-Environment`, the
embedded task is reconstructed even if source JSON was deleted/edited. Explicit
incompatible task/layout overrides fail. `latest` is newest-modified, not best.

Replays require `-ReplayEvery > 0`. Choose either `-ReplayDirectory` or a Minecraft
destination, not both. The interval counts completed duels, including timeouts,
not PPO updates. Each export is a separate complete evaluation duel, not the
current rollout or an optimizer update.

```powershell
.\scripts\train.ps1 -Schema v2
.\scripts\train.ps1 -Plan simulator/training/presets/gear-curriculum.json
.\scripts\train.ps1 -Environment simulator/environments/research-duel.json -ReplayEvery 100 -ReplayToMinecraft
.\scripts\train.ps1 -Resume latest -Updates 100
.\scripts\train.ps1 -Updates 1 -Environments 2 -RolloutSteps 16 `
  -ExtraArguments @('--epochs', '1', '--minibatch-size', '64')
```

Research plans add scripted/checkpoint opponent mixtures, fixed controls,
reset-time sampling and curriculum stages. See the
[plan reference](../simulator/training/AGENTS.md#research-training-plans).
Research resume restores the embedded plan without the original source files;
the wrapper recovers world count/rollout length unless explicitly supplied.
Completed curricula stop before the invocation's update cap when appropriate.

## Serve Policy

| Parameter | Accepted value / purpose | Default |
| --- | --- | --- |
| `-Checkpoint` | `latest` or trusted standalone checkpoint; bypass chooser | Prompt for run, newest checkpoint first |
| `-RunsDirectory` | Checkpoint search root | `simulator/training/runs` |
| `-Device` | `cpu`, `cuda` | `cpu`, even for GPU-trained weights |
| `-Port` | Integer 1..65535; loopback port | `8766` |
| `-Threads` | Integer 1..256 | `1` |
| `-Seed` | Nonnegative integer | `1` |
| `-Python` | Interpreter command or path | Auto Python |
| `-Deterministic` | Per-head argmax instead of sampling | Off |
| `-List` | List runs without loading weights, then exit | Off |
| `-DryRun` | Preview only | Off |

Without `-Checkpoint`, enter a run number or `Q` to cancel. Only runs with
`latest.pt` appear. List output includes environments, completed updates and
checkpoint save times. Logged updates may be ahead of saved weights; absent
metadata is `Unknown`. Print the selected checkpoint before starting.

The service binds only `127.0.0.1`, requires no native simulator DLL, and loads
trusted standalone PPO weights once. Restart to load newer weights; Ctrl+C stops
it. CUDA inference requires CUDA PyTorch. V1/v2/custom descriptors negotiate
layouts/loadouts; unsupported capabilities fail before spawning.

In private singleplayer use `/simduel arena`, or mark `/simduel pos1` and
`/simduel pos2` then `/simduel world`. Marks capture feet position/look; explicit
coordinates such as `/simduel pos1 100.5 64 100.5` work. World mode does not build
terrain. Match nondefault ports with `/simduel arena 8768` or `world 8768`.
Pause/resume/reset/status/stop and participation details are in the
[mod guide](../cambiummod/AGENTS.md#live-policy-duels).

## Rosie Sync And Play

`rosie-play.ps1` configures the in-game `/simrosie` helper or runs the same
artifact workflow from a terminal. It uses existing game-folder/Python selection
rules and repository-relative invocation. No Git Bash, native simulator or local
CUDA is needed; SSH must be on PATH and local duels need the training Python
requirements. [Authentication, caching and protocol](../rosie/AGENTS.md#local-artifact-launcher).

| Parameter | Purpose / default |
| --- | --- |
| `-Setup` | Save game configuration; prompt for missing host/account |
| `-Connect` | Authenticate in the local terminal; keep one SSH connection open for Minecraft; incompatible with `-Setup`/`-Offline` |
| `-Action` | `runs` (default), `replays`, `replay`, `sync`, `duel` |
| `-Run` | Run basename or `latest` (newest saved, not best) |
| `-Replay` | Replay basename or `latest` |
| `-MinecraftDirectory` | Instance game folder; normal selection rules |
| `-SshHost` | Setup only; management host, SSH alias or 1..4; `ROSIE_SSH_JUMP` |
| `-SshUser` | Setup only; Rosie account, `ROSIE_SSH_USER` or prompted local username |
| `-RemoteRunsDirectory` | Setup only; default `~/minecraftlol/Cambium/gpu-sim-minecraft/simulator/training/runs` |
| `-Python` | Local interpreter; setup records it for Minecraft |
| `-Offline` | Use cache without SSH |
| `-DryRun` | Preview only; no config, transfer or process; setup needs a host |

Terminal `replay` prints the downloaded path; it does not issue Minecraft commands.
Password users first run `-Connect`, enter SSH credentials only in that terminal,
and leave it open. Minecraft and other launcher terminals reuse its tunnel;
Ctrl+C disconnects. Existing key/agent users can omit this step.
Terminal `duel` stays running and prints its automatically selected port; use
`/simduel arena <port>`, then `/simduel join`. Ctrl+C stops inference. In-game
commands perform these handoffs automatically. `sync` downloads every replay
in the selected run, so use `replay` for on-demand viewing of just one sample.

## Serve Dashboard

| Parameter | Accepted value / purpose | Default |
| --- | --- | --- |
| `-RunsDirectory` | Parent of run directories | `simulator/training/runs` |
| `-Run` | String array of additional individual runs | Empty array |
| `-Port` | Integer 1..65535; loopback port | `8767` |
| `-Python` | Interpreter command or path | Auto Python |
| `-DryRun` | Preview only | Off |

Requires Python 3.10+ only, not PyTorch/native library or the legacy backend.
Open `http://127.0.0.1:8767`; Ctrl+C stops it. Choose another port if occupied.

```powershell
.\scripts\serve-dashboard.ps1 -RunsDirectory experiments -Port 8768
.\scripts\serve-dashboard.ps1 -Run @('experiments/close-duel', 'experiments/wide-duel')
```

See [dashboard internals](../simulator/training/AGENTS.md#training-dashboard).

## Build Mod

| Parameter | Accepted value / purpose | Default |
| --- | --- | --- |
| `-Task` | `build`, `check`, `compileJava`, `replayAnalysisChecks` | `build` |
| `-JavaHome` | JDK 25+ directory | Cached project JDK, then `JAVA_HOME`, then PATH |
| `-MinecraftDirectory` | Game folder; requires `-Install` | Auto Minecraft when installing |
| `-Install` | Copy installable JAR into `mods` | Off: build only |
| `-DryRun` | Preview build/copy | Off |

Cached Java lives in `cambiummod/.gradle/java25`; fresh clones do not contain it.
The helper never downloads Java or changes the system installation. Default
output is `cambiummod/build/libs/cambiumfabric-1.0.0.jar`, not `-sources.jar`.
Close Minecraft before installing. `-Install` requires `-Task build`; it replaces
only the same-named JAR and does not remove differently named older versions.
Build-only mode never changes the game installation.

## Replay And Analysis

Export is a scripted duel, not a learned policy; use training's `-ReplayEvery`
for policy samples. An already-built runner is required.

| Parameter | Accepted value / purpose | Default |
| --- | --- | --- |
| `-Device` | `cpu`, `cuda` | `cpu` |
| `-Ticks` | Integer 1..100000; maximum duration | `300` |
| `-BuildDirectory` | Runner build tree | CPU `simulator/build`; CUDA `simulator/build-cuda` |
| `-Runner` | Explicit executable | Discover in build's `Release` or root |
| `-Environment` | Arena/experiment JSON | Default duel |
| `-Output` | JSONL file; refuse overwrite | Unique timestamped file in repository root |
| `-ToMinecraft` | Use game's `sim-replays` | Off |
| `-MinecraftDirectory` | Game folder; also selects replay output | Auto Minecraft when requested |
| `-DryRun` | Preview only | Off |

Choose `-Output` or a Minecraft destination, not both. Lethal duels can finish
before `-Ticks`. CUDA always passes its flag; no CPU fallback is requested.
In Minecraft, `/simreplay load latest` then `/simreplay analyze` writes reports
under `sim-replays/analysis`. Share summary and detailed JSONL together. Analysis
compares recorded simulator results with vanilla input playback, does not run
CUDA inside Minecraft, and does not certify general physics parity.

## Checks

| Parameter | Accepted value / purpose | Default |
| --- | --- | --- |
| `-Scope` | `All`, `Native`, `Training`, `Policy`, `Rewards`, `Dashboard`, `Mod` | `All` |
| `-BuildDirectory` | Configured native tree | `simulator/build` |
| `-Python` | Interpreter command or path | Auto Python |
| `-Build` | Rebuild existing native configuration | Off |
| `-SkipNative` | Skip CTest and native rebuild | Off |
| `-SkipPython` | Skip Python unit tests, not catalogue freshness | Off |
| `-Mod` | Also build/check Fabric mod with Java 25 | Off |
| `-DryRun` | Preview only | Off |

Default checks run catalogue freshness, native CTest and Python training/replay/
inference/dashboard tests. `-Build` does not configure a new tree and cannot combine with
`-SkipNative`. At least one check group must remain enabled. Python tests still
need a built training library when native tests are skipped. CUDA and Java bridge
checks are opt-in in the [training suite](../simulator/training/AGENTS.md#verification-and-limits).

`Native` selects CTest; `Training` selects the full training module; `Policy`
selects native shared-core/contract fixtures and policy/configuration Python
tests; `Rewards` selects reward/absorption Python tests; `Dashboard` needs only
stdlib Python; `Mod` runs Java checks. All/Policy/Mod (and `-Mod`) also require
Python for catalogue freshness. `-Build` rebuilds everything for All/Native,
training library plus checks for Policy, and only the training library for
Training/Rewards. It is invalid for Dashboard/Mod scopes. `-Mod` adds the existing
full mod build/check to other scopes; Scope Mod needs no native tree.

The [validation matrix](../simulator/training/AGENTS.md#validation-matrix)
maps changes to commands. `python scripts/generate-contract.py` refreshes shared
C++/Java tables; `--check` fails on drift and never writes. Root CI uses direct
Linux commands with CPU PyTorch/JDK 25, and does not run CUDA or Minecraft.

## Run GPU

Prerequisites: a supported NVIDIA GPU/current driver (`nvidia-smi` lists it),
CUDA Toolkit with C++20 support (`nvcc --version`), compatible Visual Studio 2022
Build Tools/MSVC v143/Desktop C++/Windows SDK, CMake 3.24+ and Git. Use Developer
PowerShell for VS 2022. The driver capability shown by `nvidia-smi` is not proof
of an installed toolkit; the full Visual Studio IDE is not required.

| Parameter | Accepted value / purpose | Default |
| --- | --- | --- |
| `-Mode` | `Validate` (duel), `Benchmark`, `All` | `Validate` |
| `-BuildDirectory` | CUDA build tree | `simulator/build-cuda` |
| `-Architecture` | `native` or 2/3-digit targets, e.g. `80`, `86`, `'80;86'` | `native` |
| `-Generator` | Installed CMake generator | CMake default / existing cache |
| `-Environments` | Integer array, each 1..1073741823; benchmark sizes | `8192` |
| `-Launches` | Positive integer; timed launches | `5000` |
| `-TicksPerLaunch` | Positive integer | `1` |
| `-WarmupLaunches` | Nonnegative integer | `10` |
| `-SkipBuild` | Reuse build; still test/check device | Off |
| `-DryRun` | Preview without CUDA | Off |

Default flow: check GPU visibility, configure for visible GPU, build Release,
run all tests, require `shared_core_cuda_checks` to pass rather than skip, then
run the synthetic CUDA duel. All modes enforce device parity; `-SkipBuild`
skips only configure/compile and requires a current successful build. The CTest
report `gpu-tests.xml` in the build tree is replaced on later runs.

```powershell
.\scripts\run-gpu.ps1 -Mode All -Environments 4096,8192,16384,65536
.\scripts\run-gpu.ps1 -Mode Benchmark -Environments 8192
.\scripts\run-gpu.ps1 -SkipBuild
.\scripts\run-gpu.ps1 -DryRun -Mode All
.\scripts\run-gpu.ps1 -Generator 'Visual Studio 17 2022'
.\scripts\run-gpu.ps1 -Architecture 80 -BuildDirectory simulator/build-a100
```

`native` requires GPU visibility at configure time; numeric targets allow
cross-compilation (80=A100, 86=RTX 3060 Ti). Existing `build-cuda-3060` folders
remain usable via `-BuildDirectory`; select their library explicitly for training
if needed. Changing generators requires a fresh directory. The batch-size array
is benchmark-only; validation uses one two-player world. Benchmarks use movement
controls, not learned combat or end-to-end PPO. See
[manual CUDA commands](../simulator/AGENTS.md#cuda-throughput-benchmark).

## Troubleshooting

Use `Get-Help ./scripts/train.ps1 -Detailed`, `-DryRun`, and
`./scripts/serve-policy.ps1 -List` to inspect selection without starting work.

| Problem | Action |
| --- | --- |
| Several game folders | Set `MCSIM_MINECRAFT_DIRECTORY` or `-MinecraftDirectory`. |
| Missing Python module | Install requirements with the printed interpreter; override with `-Python`. |
| Missing native library | Configure/build targets or select `train.ps1 -Library`. |
| No checkpoint | Complete/save an update or choose `-RunsDirectory`/`-Checkpoint`. |
| Occupied inference port | Stop old service or use `-Port 8768` and `/simduel arena 8768`. |
| Wrong Java | Select JDK 25+ with `-JavaHome`. |
| Generator mismatch | Use a fresh build tree; do not remove unrelated builds. |
| Replay absent | Interval counts finished duels, not updates; check output folder. |

Follow the machine's execution policy rather than changing managed policy.
Synthetic duels are not captured fights; see the
[real-game validation plan](../docs/environment_interface_next_steps.md#real-game-melee-validation).