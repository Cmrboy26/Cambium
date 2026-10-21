# Simulator Training: Development And Reference

Follow the [root](../../AGENTS.md) and [native simulator](../AGENTS.md) guidance.
Keep human-facing startup commands in [README.md](README.md).

## Training Architecture

Standalone training reuses backend `ActorCritic`/PPO loss without FastAPI,
Minecraft, RCON or legacy timers. Each world has one learner and one independently
selected scripted/checkpoint/self-play opponent; worlds are independent. Both decisions
use the same pre-tick observation before ordered
physics updates. Hold weights fixed through a rollout, then synchronously update
PPO minibatches. Only learner decisions enter PPO. Distributed/multi-GPU
optimization is not implemented. Historical shared-self-play code is isolated in
[compatibility_training.py](compatibility_training.py) for regression fixtures;
it is not the public training entry point. Common PPO/GAE lives in [ppo.py](ppo.py).

The native library exposes a small C ABI through `ctypes`, without pybind11 or
Python development headers. CPU worlds use OpenMP when available, sequential
execution otherwise. CUDA training still stages controls/state through host
memory; observations/rays/rewards run on CPU. Do not present physics benchmark
rates as full PPO rates. Collection time includes enabled replay sampling.

## Research Training Plans

Pass `--plan simulator/setups/curricula/shield-axe-curriculum.json` (PowerShell
`-Plan`), also the default for new runs. Plans are
`minecraft_sim_training_plan`, version 2; unknown fields, capabilities, controls,
rewards and items fail rather than silently defaulting. Version-1 authoring plans
must be rewritten; historical checkpoint inference/replay remains supported.
Paths inside authoring files are relative to the containing file's directory.
`--updates` is the maximum additional updates for this invocation; stage
budgets may complete the curriculum sooner. The plan's seed owns sampling and
learner initialization. The removed `--schema`, `--environment` and `--seed`
training flags have no implicit migration aliases.

### Plan And Episode Contracts

- Authoring lives centrally in [setups](../setups/), separating
  [arenas](../setups/arenas/) (bounds/blocks/rules),
  [environments](../setups/environments/) (initialization/distributions/task),
  [policies](../setups/policies/) (observation/action descriptors) and
  [curricula](../setups/curricula/) (learner/controllers/mixtures/stages/evaluation).
  The compiler produces one `minecraft_sim_experiment`, version 1, with no
  unresolved authoring references. Config uses `experiment`, not an authored plan
  copied alongside repeated environment defaults.
- Curricula are optional, editable experiment data, not test fixtures. Do not add
  per-preset assertions or read installed curriculum files in automated tests.
  Use test-owned plans for compiler, sampler, training and submission behavior;
  changing or emptying the curricula directory must not break those tests.
  For an explicit one-time check, run
  `python -m simulator.training.validate_plans [file-or-folder ...]` from the
  project root. With no paths, it recursively checks the curricula directory;
  an empty selection succeeds. It uses the compiler to validate JSON, references
  and semantics, reports each file and exits nonzero if any fail. It does not
  train, load a native library, submit jobs or enforce preset-specific settings,
  and is deliberately separate from the automated suite.
- All stages share one immutable policy layout. The equipment policy fixes 78 named catalogue
  features and eight heads with a stable weapon-slot count across all stages
  and evaluation scenes; later catalogue additions do not change it. Explicit
  descriptors permit smaller research layouts. It includes own slot contents,
  but only visible selected-weapon/shield/armor information for the opponent,
  not hidden inventory or the legacy opponent slot index.
- Arenas contain geometry only. Environment arena selection resolves to a
  weighted set of concrete episode templates internally. Environments are embedded
  into the resolved experiment/checkpoint; edited/deleted source files do not alter
  implicit resumes. Arena assignment is sampled once per world. Worlds sharing
  a scene use one native batch and immutable terrain. Changing only player
  state/loadouts never rebuilds or uploads terrain.
- Environment spawn sampling: `sample: world|episode`, optional `region` with `min/max`
  XYZ, `minimum_separation` (at least 0.6), `yaw` and `pitch`. Sampling uses
  256 independent rejection attempts for collision-free positions in the
  supported air/stone scene. On exhaustion, choose a random collision-free,
  separated region-corner pair and randomize player assignment; this fallback
  is not a uniform interior draw. If no valid corner pair exists either, fail
  explicitly with world, region and separation diagnostics. Normal successful
  draws and checkpointed component RNG ownership are unchanged. Without a
  region, preserve positions.
- Environment equipment sampling: `sample: world|episode`, `players: independent`, and
  `loadout`. Specified loadout fields replace the base field; unspecified ones
  inherit it. Weapon slots never shift; the compiler pads missing slots with air
  to preserve the nine-slot policy interface. Air means an empty hand/absent shield
  or armor. Each slot can be literal or a weighted choice. Duplicate non-air
  weapon IDs across slots are rejected; repeated air is valid.
  Armor can vary independently by slot using the same choice syntax.
- Weighted choices use `{"choices":[{"value":"minecraft:shield","weight":1},
  {"value":"minecraft:air","weight":1}]}`. Look values also accept
  `{"uniform":[-180,180]}`. Literal values do not draw RNG. Separate persistent
  streams own arena selection, spawns, equipment, opponent selection and player
  side selection. World samples cache across resets; changing a component's
  stage configuration invalidates only that component's cache.

An episode resolves concrete players/loadouts before native reset. ABI2 additive
`mcsim_reset_players` validates the entire selected batch before mutation,
retains per-world slots, and reuses terrain. `mcsim_scenario_for` returns the
actual initial scenario, not a sampling distribution. Ordinary native reset
restores the world's most recently assigned players. Initial scene templates
are validated when the sampler is constructed or reconfigured, not per episode;
sampled players still undergo loadout normalization and finite-value checks.
The reset hot path draws both candidate positions in one NumPy call and uses
scalar separation checks, retaining six uniform draws per rejection attempt,
the 256-attempt limit and the existing corner fallback. Player assembly copies
only mutable loadout containers before the final validating deep copy; returned
players cannot alias scene templates or cached equipment/spawns. Regression
checks compare seeded samples and checkpoint RNG state with the previous draw
pattern, including fallback and selective-reset behavior. Reset optimizations
do not require a native rebuild; already-running trainers must restart/resume
to load updated Python code. CPU reset microbenchmarks are not CUDA throughput
measurements.
Sampled resets reuse the observations returned by `mcsim_reset_players` rather
than immediately recomputing them; untouched arena groups are observed normally.
Initial scene templates
and live world state have distinct ownership: future block edits must be
per-world, restore on reset, and distinguish initial/protected blocks from
placed blocks. Placement/breaking are not implemented; do not accept settings
that pretend to enable them or allocate mutable terrain for unchanged worlds.

### Controllers And Constraints

`opponents` maps IDs to supported controller definitions:

```json
{
  "target": {"type":"scripted","controller":"stationary_target"},
  "fighter": {"type":"scripted","controller":"basic_melee","options":{"strafe":true}},
  "frozen": {"type":"checkpoint","path":"simulator/training/runs/baseline/latest.pt"},
  "policy": {"type":"self_play"},
  "past": {"type":"frozen_self_play","every_updates":100}
}
```

Checkpoint files are trusted input, loaded with `weights_only=True`; matching
semantic contracts and finite parameters are required. Their SHA256 identities
are embedded in the plan; changed opponents reject resume. Different legacy or
custom layouts require explicit adapters and are not implicitly remapped.
Checkpoints do not import their optimizer or curriculum into the opponent.

Self-play binds to the current learner model, with weights fixed during each
rollout and following subsequent PPO updates, including episodes spanning updates.
It acts from the opposing player's own observations and honors the current
stage/evaluation fixed controls. Only the designated learner's transitions enter
PPO. Sampling uses a separate, checkpointed RNG; the learner weights are saved
once, not duplicated in controller state. Optional `deterministic: true` selects
argmax actions; default inference samples actions. Evaluation and replay pools
bind to the same learner without changing training-pool RNG state.

`frozen_self_play` uses an independent frozen copy of the learner, initially at
update zero (after `--initialize` if supplied). Required `every_updates` counts
global completed PPO updates, not stage-local updates. After each matching update,
capture the new weights and the current stage's fixed controls. New episodes use
the latest snapshot; episodes already running keep their assigned version and
controls until reset. Inference has a separate checkpointed sampling RNG and
accepts the same optional `deterministic` flag as self-play/checkpoint opponents.
Only the latest snapshot and versions still assigned to episodes are retained;
obsolete versions are discarded, not accumulated into an opponent history.
The resume checkpoint embeds their weights, controls, assignments and RNG state.
Resuming still starts fresh episodes, all assigned the latest saved snapshot;
it does not recapture the resumed learner or reset the global capture schedule.
Evaluation uses the latest training snapshot with isolated RNG, even in suites
pinned to another stage. Replay uses the snapshot assigned to the sampled world;
its opponent provenance includes `snapshot_update`. Neither captures the learner
implicitly. No extra checkpoint files are written for these rolling opponents.

Use [controllers.py](controllers.py)'s `register_controller`, or an explicit
trusted `package.module:factory` controller name. Factories receive `contract`,
`count`, `seed`, `options`; see existing builtins for their exact signature.
Implement `act(observations, indices=None)`, `reset(mask)`, `state_dict`,
`load_state_dict`, and `describe`/`provenance` properties. Inputs are `[N,F]`,
actions `[N,H]`; indices address persistent slots within the full world count.
Reset masks always span the full count. State must be `weights_only`-safe.
Controllers use contract feature scales and semantic choices, not preset indices.
Module factories execute local Python: only configure trusted code.

Each world's opponent and learner side are sampled per episode and held fixed
until it ends. Only learner transitions reach PPO. Frozen controllers use local
sampling generators rather than consuming the learner's RNG. The built-in
fighter requires target distance/yaw/pitch-error features; shielding and explicit
axe/sword switching are opt-in options with validated prerequisites.

Stage `controls.fixed` maps head names to semantic values, e.g.
`{"attack":false,"movement":"neutral","weapon":0}`. Omitted heads remain learned.
Fixed heads bypass sampling and contribute neither log probability nor entropy;
optimization evaluates the same constrained distribution. Constraints are saved
with the rollout, checkpoint and replay; HTTP and checkpoint inference honor
them. Do not overwrite unconstrained samples after sampling.

### Stages, Evaluation And Checkpoints

Stages have unique names, an environment reference, a named opponent mixture,
optional controls, typed environment/task patches and advancement criteria.
Stage configuration resolves independently from the declared environment,
not the previous stage. Arrays replace, null disables optional sampling,
and partial spawn patches need not repeat an entire environment.
Environment task rewards are explicit: unspecified terms are zero. A stage's
task reward patch replaces the reward map, avoiding accidental inherited
no-damage/timeout penalties. Supported reward mechanics/units
remain those documented above/below; adding weights does not implement mechanics.

Advancement supports `{"updates":300}`, `{"episodes":1000}`,
`{"transitions":100000}`, or
`{"evaluation":{"suite":"held","metric":"win_rate","at_least":0.75}}`.
Compose with explicit `any`/`all` lists. Updates count completed PPO updates
within the stage; transitions count learner decisions, not opponent decisions.
Budgets are checked after a complete rollout/update, so episode/transition
budgets can overshoot by one rollout. Transitions reset active episodes without
combat losses and record `curriculum_boundary` in metrics. Completed plans stop;
extending them is a new experiment, not an incompatible resume.

Evaluations remain synchronous: after scheduled PPO updates, run separate seeded
worlds in model eval mode and isolated RNG. They never contribute training
transitions. Each suite requires `every_updates`, `episodes` and `seed`;
optional `worlds` selects parallel worlds, capped at its episode count. Omission
preserves the historical maximum of 16 worlds. No native rebuild is needed.

Stage-local `evaluations` run only while their stage is active. Intervals count
completed updates within that stage, including updates restored on resume.
Suites inherit the stage's fully resolved environment, opponents, controls,
sampling, rewards and episode limit. For example, inside an aim stage:

```json
"evaluations": {
  "aim_accuracy": {
    "every_updates": 25, "worlds": 64, "episodes": 64, "seed": 1729
  }
},
"advance": {
  "evaluation": {"suite": "aim_accuracy", "metric": "reward_mean", "at_least": 6}
}
```

Top-level suites use global update intervals. A `stage` reference pins their
conditions to that named stage, regardless of which stage is currently training:

```json
"evaluations": {
  "shield_axe_combat": {
    "stage": "shield_axe_combat",
    "every_updates": 50, "worlds": 64, "episodes": 64, "seed": 1729,
    "overrides": {"opponents": "shield_axe"}
  }
}
```

Inherited suites accept condition changes only inside `overrides`: `environment`,
`opponents`, `controls`, `task_override`, `spawns` and `equipment`. Controls merge
by head; `{"controls":{"fixed":{"attack":null}}}` releases one inherited
head, and `{"controls":{"fixed":null}}` releases all. A supplied task reward
map replaces all inherited weights; omitted terms are zero. Task fields and
sampling patches otherwise retain unspecified values; arrays replace and null
disables optional sampling. An environment override replaces scene/player and
sampling defaults with the new environment, but retains the inherited task,
controls and opponents unless separately overridden. Paths remain relative to
the training-plan file. Policy layout is never overridden by a suite.

Existing top-level suites with an explicit `environment`, opponent mixture and
direct condition fields remain supported, with default unrestricted controls;
they do not inherit a stage. Do not mix those direct fields with inherited
`stage`/`overrides` syntax. Suite names must be unique across the plan. Advancement
may reference a global suite or one belonging to the current stage, never a
different stage's local suite. Conditions are compiled into portable experiments;
resuming needs no authoring files. Existing resolved plans retain their hashes.

Suites report exact episode count, mean episode return and wins/episodes. Wins
require opponent death/exit with the learner still alive/in bounds; timeouts count
as non-wins. Evaluation `reward_mean` is per-episode return, unlike training's
per-decision `reward_mean`. Results retain `worlds`, `seconds`, `snapshot_update`
and training `stage`/`stage_index`, including when reused on later metric rows.
Advancement is checked after evaluation and before collecting the next rollout;
stage changes clear prior advancement results.

Metrics separate `collection_seconds`, optimization-only `update_seconds`,
`evaluation_seconds`, `evaluation_episodes` completed this update, and
`replay_seconds`. `iteration_seconds` includes all these phases and bookkeeping,
but excludes stage-boundary resets, logging and checkpoint writes. Training-only
`player_transitions_per_second` uses this inclusive duration; evaluation samples
never inflate the numerator. CUDA runs additionally report per-update PyTorch
allocator peaks, `cuda_peak_allocated_bytes` and `cuda_peak_reserved_bytes`;
these do not include the native simulator's allocations or total device memory.

Replay cadence counts completed training episodes.
Samples are separate complete duels using a concrete sampled scene, real pool
opponent, learner side and stage constraints, with semantic replay controls.
Evaluation/replay must leave learner RNG and subsequent training unchanged.

Research checkpoints retain the PPO envelope and embed the resolved experiment, stage/budget
progress, sampler caches/streams, pool/controller state, and opponent identities.
Resume without `--plan` restores the embedded plan. World count, rollout length,
episode-limit override and architecture must match. Resume starts fresh episodes,
not exact mid-episode trajectories: native world state is not serialized.
`--initialize CHECKPOINT --plan NEW_PLAN` transfers compatible learner weights
only, starting new optimizer/schedule/sampling state. Research checkpoints serve
through the same descriptor-based HTTP/mod bridge as existing policies.

Run `python -B -m unittest simulator.training.test_controllers
simulator.training.test_experiments simulator.training.test_research
simulator.training.test_deployment -v` after building the native library; Training
and All launcher/CI scopes include these checks. CUDA and JDK25/live gates remain
separate. For new primitives, follow the extension recipes below, including
independent native/live encoders and fixtures; do not add preset-specific PPO
branches.

## Shield/Axe Curriculum Preset

[shield-axe-curriculum.json](../setups/curricula/shield-axe-curriculum.json) is a 1,500-update
experiment: aiming 100, attack timing 200, pursuit 200, equipment combat 1,000.
Budgets are fixed updates, not evaluation thresholds. The run needs an invocation
cap of at least 1,500 updates to finish in one invocation.
Each stage sets `task_override.max_ticks` to cap individual episodes: aiming 100,
attack timing 300, pursuit 600, equipment combat 1,200 ticks. Evaluation retains
the environment's 1,200-tick cap. An explicit CLI `--max-ticks` overrides stage caps.
The preset owns its H100-oriented learner settings: hidden width 512, two PPO
epochs and minibatches of 16,384. General compiler defaults are 256/4/1,024;
explicit CLI overrides are saved into the resolved experiment.

All stages use the same flat enclosed arena, no armor, iron sword in slot 0,
and slot 1 containing an iron axe with probability 0.75 or air otherwise.
An offhand shield is present with probability 0.75. Both players independently
sample equipment once per world; ordinary resets and stage changes preserve it.
Spawns and look directions resample each episode. The attack stage samples
separations 2.0..approximately 2.61 blocks, but leaves movement learned so
knockback cannot permanently put the target out of reach. It fixes sprint,
jump, shield and weapon selection. The pursuit stage introduces moving/attacking
sword opponents; the final stage releases all constraints.

Final training opponent probabilities are 50% current learner policy, 5% sword
fighter, 15% shield fighter, 15% axe-switching fighter and 15% shield-and-axe
fighter. The first three stages remain scripted-only. These describe controller
behaviors, not guaranteed gear: absent-axe fighters fall back to their sword;
absent-shield fighters do not request shield use. Optional gear observations
gate builtin equipment choices; older contracts without those features retain
their existing behavior. Fixed-seed held-out combat evaluation runs 64 episodes
every 50 global updates with unrestricted learner controls and the original
scripted-only 10%/30%/30%/30% mix. It monitors progress without advancing stages
early, so its win rate does not include self-play wins.

[axe-shield-fighting.json](../setups/curricula/axe-shield-fighting.json) trains
one unrestricted 1,500-update combat stage, also with 50% current-policy
opponents. Its remaining episodes preserve the scripted shield/axe/combined
fighter ratio of 1:1:2, and its held-out evaluation remains scripted-only.
Opponent selection is sampled per episode, so these are probabilities rather
than exact per-rollout quotas. Resuming an older checkpoint preserves its
embedded mixtures; use a new run or `--initialize` with the edited plan to adopt
self-play instead of changing a resumed experiment.

This is a starting research budget, not a convergence claim or measured H100
runtime. No block edits, procedural terrain or checkpoint opponents are used.

## Pursuit/Melee Learning-Rate Comparison

[pursuit-melee-lr-comparison.json](../setups/curricula/v2/pursuit-melee-lr-comparison.json)
is a longer, scripted-only comparison with seed 42. Pursuit trains for 300
updates against stationary targets with at least four blocks of spawn
separation and a 600-tick episode cap. Melee then trains for 1,200 updates
against the sprinting/strafe-enabled sword fighter with a 1,200-tick cap.
Both stages keep movement, sprint, attack and look learned; jump is disabled,
weapon slot 0 is fixed, shields are disabled, and equipment sampling is removed.
Both players use the environment's fixed iron sword and no armor/offhand shield.
Reward weights inherit the environment unchanged; this is not a sparse-reward
or self-play experiment.

`melee_benchmark` pins the final stage's conditions across the whole run and
executes every 10 global updates (seed 1729). `pursuit_benchmark` executes every
10 pursuit updates (seed 2718). Each uses 128 episodes across 64 worlds and
reports win rate and mean episode return. Fixed update budgets never advance
early based on score; the global benchmark remains comparable across the stage
boundary even though training opponent/spawn/episode conditions change.

Select `v2/pursuit-melee-lr-comparison.json` in `bash rosie/submit.sh`, use a fresh
name such as `pursuit-lr`, and enter `multiple` for learning rate with `.0001`,
`.0003`, `.0006`, then an empty line. Keep world count, rollout length and other
settings identical. The wizard derives a 1,500-update invocation cap. Defaults
are hidden width 512, two PPO epochs and minibatches of 16,384. Runtime and
convergence are unmeasured; this first seed comparison does not establish a
statistically reliable best rate. Do not submit jobs from automated checks.

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
the reusable authoring files linked above. Separate policy
descriptors select/order named observations/categorical controls; per-player
`loadout` defines supported Minecraft-ID weapons, armor/shields. Dimensions derive
from descriptors, not new preset IDs. Replay/live consume the resolved contract.

The environment selects geometry and task settings; `--plan` selects the
experiment. Parse/cache during setup. Resets reuse immutable terrain; CUDA
terrain stays resident. Stage timeout changes update the native task limit
without rebuilding geometry. Future block edits must use separate mutable state.

Historical low-level presets below describe compatibility fixtures, not the
public authoring API:

- One decision advances one physics tick; never real-time sleeps.
- Historical V1: movement(9), jump(2), sprint(2), attack(2), yaw(9), pitch(5).
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

Authored environment `task.rewards` maps explicitly define enabled rewards;
omitted terms are zero. Compiler output carries every supported weight.
The low-level concrete fixture adapter still reads historical
[reward_weights.json](reward_weights.json) defaults when no explicit weights are
supplied; this is not the authoring contract. Record resolved settings and
`reward_semantics` in checkpoint metadata.
Changed weights or semantics require a new run, not incompatible resume.
Rebuild older native libraries for reward ABI 2; ABI 1 is rejected explicitly.

- Damage rewards are `doing_damage * health_removed / 20` for attackers and
  `taking_damage * health_removed / 20` for targets. Weights specify points per
  full 20-HP health bar, not per hit. Use actual health removed after armor,
  absorption and damage immunity, capped by health remaining before the hit.
  Misses, blocks and absorption-only hits earn no damage reward and do not
  satisfy the episode's damage requirement. Critical/sweep mechanics remain,
  but have no separate reward multiplier: their actual damage is counted once.
- `blocked_with_shield` rewards actual non-axe blocks only. Axe hits,
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

Authored task example, with other rewards disabled:

```json
{
  "task": {
    "max_ticks": 300,
    "rewards": {"aim_hold": 2.0}
  }
}
```

### Historical Sword/Axe/Shield V2

This is the historical fixed-layout artifact contract; new authoring uses
separate semantic policies rather than a public preset selector.
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

V1/V2 meanings remain unchanged for old artifact readers; they are not interchangeable.
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
Match the resolved experiment, world count, rollout length and smoke episode-limit
override; device may change. Without `--plan`, recover the embedded experiment
even if source moved/changed. Explicit incompatible overrides fail.
Reject legacy eight-movement-head checkpoints; existing backend callers
still default to eight movement categories and otherwise remain unchanged.

### Retained Policy Snapshots

Plans and stages optionally declare `snapshots`, a list of named save rules.
Omission (or an empty list) writes no retained snapshots; normal rolling
`latest.pt` saves and frozen-opponent resume state are unaffected. For example,
these fields belong at the plan root:

```json
{
  "snapshots": [{"name":"curriculum-final","on_completion":true}]
}
```

Inside an individual stage, use:

```json
{
  "snapshots": [
    {"name":"combat-100","at_updates":100},
    {"name":"combat-final","on_completion":true}
  ]
}
```

Each rule requires `name` and exactly one trigger: positive integer `at_updates`
or literal `on_completion: true`. Updates are stage-local inside a stage and
global at the plan root. Completion means reaching the stage's advancement rule,
or completing the entire curriculum at the root, not exhausting CLI `--updates`.
A stage that advances before an update trigger does not produce that save.
Names must be unique across the whole plan, begin with an ASCII letter/digit,
and otherwise contain only letters, digits, underscores or hyphens.

Triggered policies are atomically saved as `snapshots/<name>.pt` in the run
directory. They use the existing standalone PPO policy envelope and retain
resolved layout, actual scenario, capture-stage controls, global update and
stage-local update provenance. Capture happens after optimization and before
changing stages, including a rule that fires on the stage's final update.
Metrics list paths under `snapshots` only on updates that write them.
These are policy-only saves, not optimizer/training-resume checkpoints: use
`--initialize` to start a new run from one, or `type: checkpoint` plus its path
to reuse it as a frozen opponent. They are not automatically added to mixtures.
Use `latest.pt` for resume; its embedded plan restores pending save rules without
requiring the authored JSON. Resuming from an older checkpoint can replace a
same-named save when its trigger is reached again.

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
- The overview leads with experiment identity and a centered curriculum-stage
  browser, then seed/compute/policy/PPO settings, headline metrics and timing.
  Arrows or stage buttons browse conditions without changing training/global charts;
  left/right and Home/End keys also navigate. Current and selected stages are
  separate. Browsing survives refresh; changing runs resets to the current stage.
  Stage details group advancement/progress, environment, opponents and controls.
  Reward rows show the selected stage's configured signed weights, not observed
  returns. Active terms lead; explicit zeros are collapsed separately and absent
  maps stay unknown. Use resolved stage weights, then explicit task overrides,
  then recorded per-arena maps; never borrow deployment/global defaults.
  Reward swatches match the charts; signed magnitude bars normalize within each
  map. Opponent bars show mixture probabilities, not measured episode shares.
  Simple recorded budgets get a progress bar; composed/unknown budgets do not.
  Saved experiments, not current authoring files, describe a run. Boundaries
  select the next stage but retain the just-finished rollout's progress only
  under its own stage. Runs without stages retain the historical summary.
- Selected milestones graph each advancement condition, including nested ANY/ALL,
  over that stage's complete recorded history with a dashed threshold line.
  Each condition has its own scale; evaluation `reward_mean` is episode return,
  not training's per-step mean, and win rates use percentages. Missing samples
  remain pending, not zero. Stage-local updates label the horizontal axis;
  historical rows missing stage counters use explicitly labeled PPO updates.
  Snapshot updates deduplicate carried evaluation results and exclude other
  stages. Historical results without snapshot metadata use saved suite cadence,
  or consecutive-result changes when cadence is unavailable (identical later
  measurements cannot then be distinguished). Budget metrics use recorded
  stage progress. These raw charts are independent of global window/EMA controls;
  browsing/refresh preserves selection, reuses live charts and releases old ones.
- Section links lead to training results, curriculum/evaluation, world/replays,
  run configuration and recent updates. Charts group outcomes, optimization
  and runtime. Stage details/timeline and full configuration start collapsed;
  held-out evaluation results remain visible. The run path, loadouts, resolved
  reward weights and experiment JSON live in the configuration section.
- Resolved experiments show stage budgets, environment/arena identity, opponent
  mixtures, fixed controls and held-out evaluations. Boundary rows distinguish
  the just-finished stage from the upcoming stage; completion is explicit.
  The timeline and resolved JSON are read-only, not a configuration editor.
- Charts: mean reward/player step, total/update, stacked reward types, separate
  policy loss, value/total losses, entropy, player-transition throughput,
  terminated/timeouts and collection/optimization/evaluation/replay duration. Each chart has a
  value/derivative mode. Derivatives are differences of EMA-smoothed values
  divided by update-number differences, computed before windowing; missing
  values break the series. Reward derivatives retain signed stacking.
  Window/EMA/pause controls are local. Counts use all updates, not chart window.
  Connected means readable, not training.
- Time estimates use the last 20 timed updates' mean `iteration_seconds`, falling
  back to collection + optimization for historical rows, multiplied by remaining
  updates, independent of chart window/EMA. Evaluation and replay time are included
  for new rows. Recorded training time sums these durations for the current session,
  not wall time/checkpoint overhead. New `config.json` records `training.start_update`
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
  episodes crossing update boundaries. Training rewards are not a win rate;
  separately logged held-out suites report episode returns and wins/episodes.
- Opponent-pool `reward_types` contains only learner contributions, selected
  before episode resets/side changes through `mcsim_step_reward_totals`. Their
  sum matches learner `reward_total` within floating-point rounding. Older
  libraries emit an explicit rebuild notice rather than fabricated breakdowns.
  Rebuild native code and restart/resume the trainer to enable this logging;
  historical rows without per-type data cannot be reconstructed from totals.
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
.\scripts\train.ps1 -Plan simulator/setups/curricula/shield-axe-curriculum.json `
  -ReplayEvery 100 -ReplayToMinecraft
```

Choose game folder via explicit directory, environment variable or unambiguous
detection ([selection rules](../../scripts/AGENTS.md#minecraft-folder)). Other
destinations: wrapper `-ReplayDirectory`, direct `--replay-dir`; fallback run's
`replays`. Omit interval/set zero to disable. Sample export uses isolated worlds
and RNG so it cannot perturb subsequent training. Larger N limits added
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

### Independent Local Duels

`python -m simulator.training.serve --duel <file.json>` serves two independent
fighters using the existing protocol-1 `/info` and `/act` routes. The JSON format
is `minecraft_sim_duel`, version 1, with `policy` (resolved semantic descriptor),
`environment` (concrete v1 scene), and exactly two `fighters`. A trusted optional
`reference` checkpoint can supply the policy/environment instead. Unknown keys,
incompatible checkpoint layouts, unsupported controllers and invalid scripted
options fail before the service starts.

Each fighter is `{"type":"checkpoint","path":"...","name":"..."}` or
`{"type":"scripted","controller":"basic_melee","options":{},"name":"..."}`;
`stationary_target` is the other built-in script. Checkpoint paths are relative
to the duel file unless absolute. Each controller sees only its own observation
row; scripted state resets when the request session changes. Controllers keep
independent fixed controls/RNG, and `/info.fighters` carries readable provenance.
Live configs do not run self-play pools, curricula or arbitrary Python factories.
Legacy positional-checkpoint serving remains unchanged; choose it or `--duel`,
not both.

Rosie `sync` additionally caches safe `snapshots/*.pt` artifacts and writes game
`sim-rosie-cache/library.json`. It indexes checkpoint scenes and resolved stage
arena templates, preserving aliases by ID and entries from other runs. The mod's
[native panel](../../cambiummod/AGENTS.md#native-panel-and-local-library) writes
the local duel config. Rosie action `local --duel <file> --offline` starts CPU
inference on a free loopback port without constructing a Rosie client, reading
SSH configuration or requiring a native library. This still needs the local
training Python dependencies; GPU-trained checkpoint tensors load on CPU.

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

1. Task rewards belong in environment authoring files; omitted terms are zero.
  `reward_weights.json` is historical low-level fixture configuration only.
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