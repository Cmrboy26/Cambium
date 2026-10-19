# Batched Simulator PPO

Train shared self-play or a learner against scripted/checkpoint opponents.
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
.\scripts\train.ps1 -Plan simulator/training/presets/gear-curriculum.json
```

Default training uses CPU, 16 worlds and 100 updates. V1 is sword-only; V2 adds
sword/axe selection and shields. Customize terrain, equipment, observations,
controls and reward overrides with the [scenario contract](../../docs/simulation_contract.md).
The [research preset](presets/gear-curriculum.json) adds aiming/combat stages,
random spawns, optional shields/axes and held-out evaluation. See the
[training-plan reference](AGENTS.md#research-training-plans) for custom bots,
checkpoint pools, sampling frequencies and advancement rules.

## Checkpoints

Runs under `simulator/training/runs` contain configuration, metrics and `latest.pt`.
Checkpoints save every ten updates and on normal completion.

```powershell
.\scripts\train.ps1 -Resume latest -Updates 100
```

This adds up to 100 updates and starts fresh episodes; a completed curriculum
stops early. Research resumes restore the embedded plan, world assignments and
stage progress. `latest` means newest-modified, not strongest. Use `-Initialize`
with a new `-Plan` to transfer compatible weights without resuming the old task.

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