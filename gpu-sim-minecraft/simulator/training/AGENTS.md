# Simulator Training: Development And Reference

Follow the [root](../../AGENTS.md) and [native simulator](../AGENTS.md) guidance.
Keep human-facing startup commands in [README.md](README.md).

## Training Architecture

Standalone training reuses backend `ActorCritic`/PPO loss without FastAPI,
Minecraft, RCON or legacy timers. Two players/world share one policy; worlds are
independent. Both decisions use the same pre-tick observation before ordered
physics updates. Hold weights fixed through a rollout, then synchronously update
PPO minibatches. No opponent pool/frozen-opponent schedule or distributed/multi-GPU
optimization is implemented.

The native library exposes a small C ABI through `ctypes`, without pybind11 or
Python development headers. CPU worlds use OpenMP when available, sequential
execution otherwise. CUDA training still stages controls/state through host
memory; observations/rays/rewards run on CPU. Do not present physics benchmark
rates as full PPO rates. Collection time includes enabled replay sampling.

## Setup

From the repository root, with Python 3.10+ and a configured
[C++20 build](../AGENTS.md#windows-build):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r simulator/training/requirements.txt
cmake --build simulator/build --target minecraft_sim_training --config Release --parallel
```

Use the configured build tree if different. Recreate venv/build caches per
machine; they are not portable. MinGW stages dependencies beside the DLL.
Linux: `python3 -m venv .venv`, then
`.venv/bin/python -m pip install -r simulator/training/requirements.txt`.
Discovery searches `simulator/build*`, including VS Release/Debug; `--library`
avoids ambiguity.

## CPU Mode

```powershell
.\.venv\Scripts\python.exe -B -m simulator.training.train --device cpu --updates 100
```

Defaults: 16 worlds, CPU physics/policy, four CPU/OpenMP threads. A short check:
`--updates 1 --rollout-steps 16 --envs 4`; `--threads` controls parallelism.

## CUDA Mode

Install CUDA-enabled PyTorch using the
[official selector](https://pytorch.org/get-started/locally/) appropriate to the
driver. A CPU wheel cannot train CUDA policies. Simulator compilation also needs
a C++20 CUDA Toolkit and supported host compiler.

```powershell
.\scripts\run-gpu.ps1
.\.venv\Scripts\python.exe -B -m simulator.training.train --device cuda --envs 8192 `
  --library simulator/build-cuda/Release/minecraft_sim_training.dll
```

Default is 8192 worlds and CUDA physics/inference/PPO. Validate with
`--envs 256 --updates 1` before scaling. Never silently fall back to CPU.

Linux/A100, after venv/NumPy/CUDA PyTorch setup:

```sh
cmake -S simulator -B simulator/build-cuda -DMCSIM_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release
cmake --build simulator/build-cuda --parallel
ctest --test-dir simulator/build-cuda --output-on-failure
python -B -m simulator.training.train --device cuda --envs 8192 \
  --library simulator/build-cuda/minecraft_sim_training.so
```

## Task And Controls

Custom experiments use [the contract](../../docs/simulation_contract.md) and
[research-duel.json](../environments/research-duel.json). Embedded `policy`
descriptors select/order named observations/categorical controls; per-player
`loadout` defines supported Minecraft-ID weapons, armor/shields. Dimensions derive
from descriptors, not new preset IDs. Replay/live consume the resolved contract.

`--environment` selects an arena (default [duel.json](../environments/duel.json)).
Parse/cache during native setup. Resets reuse spawns/immutable terrain; CUDA
terrain stays resident. Seed RNG per world/player/episode.

- One decision advances one physics tick; never real-time sleeps.
- V1 default: movement(9), jump(2), sprint(2), attack(2), yaw(9), pitch(5).
  Movement order: neutral, forward, forward-right, right, backward-right,
  backward, backward-left, left, forward-left. Not legacy movement indices.
- Yaw deltas: `[-20,-10,-5,-2,0,2,5,10,20]`; pitch: `[-5,-2,0,2,5]` degrees.
  Attack is one tick's press; sneak=false; no v1 item/hotbar controls.
- The 22 observations retain old encoder feature groups, not stale caches.
  Eight horizontal world-yaw rays (0,45,...315) start at pose eyes and normalize
  to 50 blocks. Opponent offsets/aim use feet-relative positions. `SCHEMA` in
  [environment.py](environment.py) is the full versioned layout.
- Death, X/Z arena exit or falling below `bounds.min.y - 4` terminates. Exit is
  a task loss, not physics damage.
- Top-level environment `max_ticks` supplies episode length; `--max-ticks`
  overrides. Fallback 300 ticks=15 simulated seconds. Timeout is truncation:
  bootstrap final observation in GAE without carrying trace into reset.
- V1 returns observations `[worlds,2,22]`, rewards `[worlds,2]`, separate
  termination/truncation `[worlds]`. `reset(mask)` resets complete selected
  worlds, not single players. Return finals before reset; stepping finished
  worlds is an error. Rollout boundaries are not episode ends.

### Reward Semantics

Load default [reward_weights.json](reward_weights.json) from this component at
environment startup. An environment's top-level `reward_weights` object overrides
individual values; omitted values inherit defaults, explicit zero disables a
term. Record resolved settings and `reward_semantics` in checkpoint metadata.
Changed weights or semantics require a new run, not incompatible resume.
Rebuild older native libraries for reward ABI 2; ABI 1 is rejected explicitly.

- Damage rewards are `doing_damage * health_removed / 20` for attackers and
  `taking_damage * health_removed / 20` for targets. Weights specify points per
  full 20-HP health bar, not per hit. Use actual health removed after armor,
  absorption and damage immunity, capped by health remaining before the hit.
  Misses, blocks and absorption-only hits earn no damage reward and do not
  satisfy the episode's damage requirement. Critical/sweep mechanics remain,
  but have no separate reward multiplier: their actual damage is counted once.
- `blocked_with_shield` defaults 5, rewards actual non-axe blocks only. Axe hits,
  including disables, earn no block bonus. Held shields alone earn nothing;
  blocked hits earn neither damage reward nor damage penalty.
- Apply terminal win/loss and timeout/no-damage penalties once. Simultaneous
  losers both lose, not win. Shaping uses `good_aim`, `proximity`, `pitch_control`,
  `aim_hold` and `extreme_pitch_penalty` as per-player episode budgets, not
  per-tick rewards. Pay `weight * score / max_ticks` per tick, with scores 0..1.
  `aim_hold: 2` pays at most +2 over a full episode, whether its limit is 100 or
  1000 ticks. Negative weights bound total penalties. Early termination leaves
  the remainder unpaid. CLI episode-limit overrides use the same normalization.
  Damage/block/win/loss/timeout rewards are not divided by episode length.

Environment override example, with other defaults inherited:

```json
{
  "reward_weights": {
    "aim_hold": 2.0,
    "pitch_control": 0.0
  }
}
```

### Sword/Axe/Shield V2

`./scripts/train.ps1 -Schema v2`, or
`python -B -m simulator.training.train --schema v2`, opts in.
`minecraft_sim_duel_31_v2` preserves first 22 observations/six heads, appending
`weapon` (0=diamond sword,1=diamond axe) and `shield` (0=release,1=hold use).
Players start unarmored, sword selected, offhand shield. Existing arenas work;
cached definition records `unarmored_sword_axe_shield`. No durability/inventory/
enchantments are added.

| Index | Feature |
| --- | --- |
| 22,23 | Own/opponent weapon is axe |
| 24,25 | Own/opponent using shield, including raise delay |
| 26 | Own shield disabled by axe |
| 27 | Opponent disabled, inferred from observed axe-disable event/countdown |
| 28 | Own disable cooldown / 100, clamped to 1 |
| 29 | Own attack charge, clamped 0..1 |
| 30 | Own raise progress / 5, clamped to 1; zero when not using |

Shapes: observations `[worlds,2,31]`, actions `[worlds,2,8]`. Never expose exact
opponent cooldown/charge/raise progress. Lowered is not disabled. Disable lasts
100 ticks; first post-hit observation is 99 after normal decrement. Same weapon
preserves charge, switching resets. Attack/use are independent: no auto-lowering
or bypass. Both heads execute in shared physics.

V1 remains default/unchanged. Match schema on resume; v1/v2 are not interchangeable.
Wrapper restores schema from adjacent config unless explicitly overridden.
Older libraries must fail with a rebuild message, not accept incorrect strides.
V2 supports periodic replays/live duels: store semantic controls, not head indices.
`/info` advertises descriptors/fingerprint for v1/v2/custom layouts.

## Checkpoints

Timestamped `simulator/training/runs/` directories contain `config.json`,
`metrics.jsonl`, rolling `latest.pt`. `--output` overrides. Save every ten updates
and normal completion; `--save-every` changes cadence. Nonempty output requires
`--resume`. Wrapper: `./scripts/train.ps1 -Resume latest -Updates 100`.

Resume restores policy, optimizer, counters and PyTorch RNG, but starts fresh
episodes, not an exact simulator snapshot. `--updates` means additional updates.
Match observation/action schema, environment, rewards, hidden size and episode
limit; device/world count may change. Without explicit `--environment`, recover
the embedded definition even if source moved/changed; explicit overrides must
match. Reject legacy eight-movement-head checkpoints; existing backend callers
still default to eight movement categories and otherwise remain unchanged.

## Training Dashboard

Launcher `./scripts/serve-dashboard.ps1`; direct
`python -B -m simulator.training.dashboard`. Loopback-only default
`http://127.0.0.1:8767`, Python 3.10+ only, no torch/native imports/legacy backend.
Bundled Chart.js 4.5.1/MIT supports offline use.

- List live/saved runs under default runs root in descending natural name order
  (newest timestamp/job names first); refresh every three seconds. Training may
  start before/after server, with no trainer flags.
- `--runs-dir` selects a parent; repeat `--run` for individual extra directories.
  Use `--port` if occupied; Ctrl+C stops server.
- Environment identity, source, episode limit and arena/policy appear above the
  metrics; expandable settings include loadouts and resolved reward weights.
- Charts: mean reward/player step, total/update, stacked reward types, separate
  policy loss, value/total losses, entropy, player-transition throughput,
  terminated/timeouts and collection/optimization duration. Each chart has a
  value/derivative mode. Derivatives are differences of EMA-smoothed values
  divided by update-number differences, computed before windowing; missing
  values break the series. Reward derivatives retain signed stacking.
  Window/EMA/pause controls are local. Counts use all updates, not chart window.
  Connected means readable, not training.
- Time estimates use the last 20 timed updates' mean collection + optimization
  duration multiplied by remaining updates, independent of chart window/EMA.
  Recorded training time sums these durations for the current session, not wall
  time/checkpoint overhead. New `config.json` records `training.start_update`
  and `training.target_update`; these do not change checkpoint metadata.
  Legacy fresh runs use `options.updates`; legacy resumed runs with unknown
  session boundaries leave remaining time and session duration unavailable.
- Read only `config.json` and newline-complete `metrics.jsonl`; never load, alter,
  save or serve checkpoints. Wait for partial final lines, warn on malformed
  records, replace superseded logged updates after older-checkpoint resume.
- Old runs estimate total from mean*sample count when consecutive counters
  permit. Updated trainer logs exact totals/types: damage dealt/taken, wins,
  deaths, timeouts, no-damage, shields, good aim, proximity, pitch control,
  aim hold, extreme pitch. Sweep/crit scale damage, not independent events.
- Missing historical type data stays blank, not zero. Otherwise-compatible old
  libraries log total reward with a missing-breakdown notice. To enable future
  breakdowns, rebuild `minecraft_sim_training`, then restart/resume updated
  trainer. Stop training before rebuilding a loaded Windows DLL. Use the actual
  CUDA build directory for CUDA runs.
- Counters exclude replay evaluation and include per-update contributions from
  episodes crossing update boundaries. No per-duel returns/opponent win rates.
- Tests: `python -B -m unittest simulator.training.test_dashboard -v`, also in
  `scripts/check.ps1`. Chart math, run ordering and timing checks use Node.js
  when installed; HTTP/store checks need only Python's standard library.

## Periodic Minecraft Replays

`--replay-every N` counts completed training duels in any world, including
timeouts, not updates/ticks. At each threshold the current policy plays a complete
separate sample on selected CPU/CUDA. Stochastic like training, excluded from PPO
experience, preserving training RNG. No export before N completions; multiple
crossings in one batch step each export.

```powershell
.\scripts\train.ps1 -Environment simulator/environments/duel-full-gear.json `
  -ReplayEvery 100 -ReplayToMinecraft
```

Choose game folder via explicit directory, environment variable or unambiguous
detection ([selection rules](../../scripts/AGENTS.md#minecraft-folder)). Other
destinations: wrapper `-ReplayDirectory`, direct `--replay-dir`; fallback run's
`replays`. Omit interval/set zero to disable. Create sample environment once,
reset between samples, never reload terrain per sample. Larger N limits added
wall time/disk cost with many worlds.

Unique envelope-3 JSONL includes arena/frame zero, every transition through
death/exit/timeout, resolved equipment and semantic controls independent of head
order. Header records completed-round threshold and already-completed updates.
Publish complete files only, never intentionally overwrite. Limits: 100000
transitions, 64 MiB. Round counters persist in checkpoints; old ones start at zero.

Viewer: `/simreplay load` (newest), then `/simreplay restart`. Both input/absolute
modes work; quoted absolute paths load files elsewhere.

## Live Minecraft Inference

V1/v2/custom checkpoint contracts supply dimensions/loadouts and frozen-food/no-
regeneration conditions in arena/world modes. Install updated
[mod](../../cambiummod/AGENTS.md#live-policy-duels) and serve a trusted standalone
checkpoint on the same machine:

```powershell
.\scripts\serve-policy.ps1 -List
.\scripts\serve-policy.ps1
```

Weights load once into evaluation-only policy. Default CPU works for CPU/GPU-
trained weights without native library. Optional CUDA needs supported PyTorch.
Default sampling stochastic; `-Deterministic` per-head argmax, `-Seed` sampling.
Loopback-only port 8766; `-Port` overrides. Direct Python flags use lowercase
`--device`, `--deterministic`, `--seed`, `--port`. Restart for newer checkpoints;
legacy formats fail explicitly.

In private singleplayer `/simduel arena` reconstructs embedded arena. Or mark
`pos1`/`pos2` at feet positions/look (absolute coordinates allowed), then `world`
uses existing terrain. Minecraft computes observations/applies actions; no native
stepping/PPO. Mod owns pause/resume/reset/status/stop and round resets. This is a
bounded server-player adapter, not arbitrary-terrain transfer/network-client parity.

## Verification And Limits

Default `./scripts/check.ps1 -Build` runs native and Python checks. Add `-Mod`
for the mod build/checks. The root CI workflow also runs the Java/Python bridge;
CUDA and live-world behavior are not covered by hosted CPU CI.

```powershell
.\.venv\Scripts\python.exe -B -m unittest simulator.training.test_training -v
```

GPU: build both backends (a CUDA library also supports CPU), then:

```powershell
$env:MCSIM_TEST_CUDA = '1'
.\.venv\Scripts\python.exe -B -m unittest simulator.training.test_training -v
Remove-Item Env:MCSIM_TEST_CUDA
```

CPU tests: isolation/selective reset, invalid controls, cached maps, terminal/
timeout rewards, exits, GAE, actual parameter updates/resume. Replay: complete
episodes, thresholds/cadence, unchanged experience/RNG. Inference: contracts,
finite observations, action ranges, fingerprints, loopback HTTP. Custom: head
order/count, three slots, armor, portable resume/equipped replay. CUDA opt-in
compares observations/rewards/resets; run on every target GPU.

Real Java/Python/native integration: set `MCSIM_LIVE_JAVA_CHECK=1` and Java 25
`JAVA_HOME` before Python suite. Gradle `check` also tests Java action/observation
ordering/stale response rejection. Live in-game behavior is not automated. A CPU
pass does not exercise CUDA; successful training does not prove combat fidelity.
Real-game conformance captures remain required.

## Extension Recipes

Start with configuration: existing feature names, head choices/order, terrain,
loadouts and weights need experiment JSON, not another preset or mod code.
Keep architecture and supported-slice decisions in the
[contract specification](../../docs/simulation_contract.md#catalogue-and-fixtures).

### New Observation

1. Define what is observable, its sampling phase, normalization and bounds.
  Avoid privileged opponent state; inspect the controlling Minecraft source.
2. Append its name to the shared catalogue and regenerate tables. Existing
  native feature indices and legacy preset prefixes must not change order.
3. Implement native encoding in `apps/observation.hpp` (base math) or the
  equipment catalogue assembly in `apps/training.cpp`; implement live state
  extraction in `LivePolicy` and pure math in `ObservationEncoder`.
4. Extend shared fixtures with independently calculated expected values and
  edge cases. Shared base math has native/Java consumers; descriptor projection
  and rejection cases also run in Python. Add native/live equipment-specific
  tests when needed; a new name alone is not an implementation.
5. Run Policy and Mod scopes, then rebuild/install the mod for live checks.
  A changed input layout needs a new policy/run unless explicitly migrated.

### New Action Or Equipment

1. If it is an existing primitive/slot, change JSON only. For a new primitive,
  define held versus one-shot behavior, neutral choice and application phase.
2. Update catalogue/value kind, regenerate, then implement `PolicyLayout::decode`
  and `SimulationContract.decode`; implement actual mechanics in native core
  and the real Minecraft adapter, never in the network or observation encoder.
3. Keep replay controls semantic. Extend serialization/readers together if
  replay needs a new control; do not record categorical head indices.
4. Test reordered heads, invalid choices, loadout bounds, CPU/CUDA and live
  application/reset. `ActorCritic` derives configured head sizes, so existing
  primitives do not need a fixed network rewrite.

### Reward Changes

1. Defaults belong in `reward_weights.json`; experiments override by name.
  Define event versus per-episode-budget units and terminal/reset behavior.
2. Keep native evaluation/totals, Python ordering/counts and dashboard labels
  synchronized. Update the native buffer descriptor when counts change.
3. Test one event and episode budget, zero weights, partial health/absorption,
  terminal/timeout and selective reset as applicable.
4. Bump `REWARD_SEMANTICS` for changed meaning, not a weight-only override.
  Reject incompatible training resumes; do not break inference-only weights
  by rewriting legacy preset identities.

### Compatibility Checklist

- Preserve v1/v2 schema dictionaries and meanings; descriptor order/scales and
  choices are checkpoint inputs, not incidental formatting.
- Classify the change: experiment config, additive primitive, semantic change,
  raw C ABI/buffer change, replay envelope or HTTP protocol change. Change the
  relevant identity/version and rejection tests rather than silently defaulting.
- Keep native/Java decoders, reset phase and replay controls aligned. Update
  capabilities/rules and reject unsupported conditions before spawning/stepping.
- Native `mcsim_build_info` and `mcsim_batch_info` are startup diagnostics and
  buffer guards. Rebuild stale binaries. `native_build` in checkpoint metadata
  is provenance, excluded from resume compatibility; never require an old path.
- Inference `/info` exposes current `catalogue_hash` and optional `training_build`
  without loading a native library. Older checkpoints may lack provenance.

### Validation Matrix

Commands are from repository root, with an existing configured native build.

| Changed Slice | Focused Command | Additional Gate |
| --- | --- | --- |
| Native physics | `./scripts/check.ps1 -Scope Native -Build` | CUDA parity and source-derived/game captures |
| Python adapter/PPO/checkpoint | `./scripts/check.ps1 -Scope Training -Build` | GPU path if touched |
| Catalogue/observations/actions | `./scripts/check.ps1 -Scope Policy -Build -Mod` | HTTP bridge, then live play for adapter changes |
| Rewards | `./scripts/check.ps1 -Scope Rewards -Build` | Resume semantics; CUDA when relevant |
| Dashboard only | `./scripts/check.ps1 -Scope Dashboard` | Browser check for UI behavior |
| Mod only | `./scripts/check.ps1 -Scope Mod` | In-game replay/live checks |
| Cross-project/full PR | `./scripts/check.ps1 -Build -Mod` | Explicit bridge and target GPU gates |

For the HTTP bridge, set `MCSIM_LIVE_JAVA_CHECK=1` and JDK 25 `JAVA_HOME`, then
run `python -B -m unittest simulator.training.test_training.InferenceChecks.test_java_bridge -v`.
Use the selected venv interpreter on Windows. For CUDA, rebuild its native tree,
run `scripts/run-gpu.ps1` on that tree and set `MCSIM_TEST_CUDA=1` for Python
parity. Restore environment variables afterward. These are separate evidence
layers; shared fixtures do not validate Minecraft raycasts, equipment sampling,
world behavior or complete live parity.