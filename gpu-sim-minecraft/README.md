# gpu-sim-minecraft

A Minecraft Java **26.3** physics simulator and reinforcement-learning workspace.
Train two-player combat policies on CPU or CUDA, then evaluate them in Minecraft
with the Cambium Fabric mod. This is a focused simulation, not a server replacement.

## Get Started

On Windows, follow the [first-time setup guide](scripts/README.md#first-time-setup)
from the repository root. It installs Python requirements, builds the CPU
simulator and runs checks:

```powershell
.\scripts\setup.ps1
.\scripts\train.ps1 -Updates 1 -Environments 2 -RolloutSteps 16
```

For local Linux CPU training, use the [Linux native build](simulator/README.md#linux-build)
and the short [training smoke test](simulator/training/README.md#linux-cpu-smoke-test).
For GPU jobs on Rosie, follow the [Rosie setup and submission guide](rosie/README.md).
For manual native builds and CUDA checks, see the [simulator guide](simulator/README.md).

On a configured Windows CUDA machine, train with:

```powershell
.\scripts\train.ps1 -Device cuda -Environments 8192
```

Runs, metrics, checkpoints and replay samples are saved under
`simulator/training/runs`. Resume the most recently modified checkpoint with
`.\scripts\train.ps1 -Resume latest`.

## Find Your Workflow

| Guide | What it covers |
| --- | --- |
| [Scripts](scripts/README.md) | Setup, training, dashboard, inference and replay commands. |
| [Simulator](simulator/README.md) | Native builds, checks and benchmarks. |
| [Training](simulator/training/README.md) | Python setup, experiments and checkpoints. |
| [Training setups](simulator/setups/README.md) | Author, validate and train experiment definitions. |
| [Rosie](rosie/README.md) | Linux/WSL setup, GPU job submission, artifacts and dashboard. |
| [Minecraft mod](cambiummod/README.md) | Install the mod, record sessions and run live duels. |
| [Backend](backend/README.md) | Shared policy model and PPO constants. |

Experiment with terrain, loadouts and policy inputs using
[research-duel.json](simulator/environments/research-duel.json) and the
[scenario contract](docs/simulation_contract.md).

## Development

Read [AGENTS.md](AGENTS.md) for architecture, development rules and component
implementation guides. Minecraft behavior is grounded in the
[simulation specification](docs/minecraft_simulation_specs.md) and matching
26.3 source.

Use the [extension recipes and validation matrix](simulator/training/AGENTS.md#extension-recipes)
for new observations/actions/rewards. Root CI runs native/Python/Java fixtures
and the HTTP bridge; GPU and in-game validation remain explicit.

The simulator covers a growing subset of movement and combat. Passing regression
checks does not establish full Minecraft parity, and CUDA physics throughput is
not end-to-end training throughput. Use a private, disposable singleplayer world
for mod experiments; only load checkpoints you trust.