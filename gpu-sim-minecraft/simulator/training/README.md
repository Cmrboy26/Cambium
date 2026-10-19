# Batched Simulator PPO

Train one shared policy controlling both players in independent simulated duels.
No Minecraft, RCON or legacy backend server is needed for training.

## Setup

Requires Python 3.10+ and the [native simulator](../README.md). From the repository
root on Windows, the quickest setup is:

```powershell
.\scripts\setup.ps1
```

For Linux or manual Python/library setup, see [AGENTS.md](AGENTS.md#setup).
CUDA needs a CUDA-built simulator and CUDA-enabled PyTorch; see
[CUDA setup](AGENTS.md#cuda-mode). Recreate `.venv` and builds on each machine.

## Train

```powershell
.\scripts\train.ps1
.\scripts\train.ps1 -Schema v2
.\scripts\train.ps1 -Device cuda -Environments 8192
.\scripts\train.ps1 -Environment simulator/environments/research-duel.json
```

Default training uses CPU, 16 worlds and 100 updates. V1 is sword-only; V2 adds
sword/axe selection and shields. Customize terrain, equipment, observations,
controls and reward overrides with the [scenario contract](../../docs/simulation_contract.md).

## Checkpoints

Runs under `simulator/training/runs` contain configuration, metrics and `latest.pt`.
Checkpoints save every ten updates and on normal completion.

```powershell
.\scripts\train.ps1 -Resume latest -Updates 100
```

This adds 100 updates and starts fresh episodes. `latest` means newest-modified,
not strongest. Resume must preserve the task, policy layout and reward settings.

## Training Dashboard

In another terminal:

```powershell
.\scripts\serve-dashboard.ps1
```

Open **http://127.0.0.1:8767** for live/saved reward, loss and throughput charts.
It needs only Python, not PyTorch or a native library. Ctrl+C stops the server;
use `-Port` if occupied. Missing historical reward breakdowns remain blank.

For a concise text or JSON summary of a saved or live run:

```powershell
python -B -m simulator.training.analyze_metrics
python -B -m simulator.training.analyze_metrics "simulator/training/runs/<run-id>" --windows 10 50
python -B -m simulator.training.analyze_metrics "simulator/training/runs/<run-id>" --json
```

The summary includes recent PPO trends, duel endings, timeouts and normalized
reward contributions. Shared self-play reward is not a win rate; use a fixed
opponent evaluation to compare policy strength.

## Evaluate In Minecraft

```powershell
.\scripts\train.ps1 -ReplayEvery 100 -ReplayToMinecraft
.\scripts\serve-policy.ps1
```

Replay intervals count completed duels, not updates; each export is a separate
policy sample. Without a destination, samples stay in the run's `replays` folder.
Follow the [mod guide](../../cambiummod/README.md) for installation, game-folder
selection, playback and live duels. Inference defaults to loopback port 8766.

## Development

[AGENTS.md](AGENTS.md) holds observation/action schemas, reward semantics,
checkpoint compatibility, dashboard internals, replay contracts and test commands.
CUDA physics still uses host-staged observations/rewards; physics benchmarks
are not full training throughput, and training success does not prove game parity.