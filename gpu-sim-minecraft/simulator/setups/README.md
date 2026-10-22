# Create Training Setups

Use JSON to define a task, its policy and the sequence of training stages.
Start with the self-contained [example plan](curricula/example/melee.json).
Paths resolve relative to the JSON file containing them, not the working directory.

Create your own person-named subfolder inside each of the four folders below
for definitions you author; use the same name across all four. Keep intentionally
shared definitions directly at the category root, outside personal folders.
The `example/` folders are starter templates, not a place for personal experiments.
Copy their files into your folders, rename JSON `name` values and update references.

## Choose The Right File

| Folder / example | Owns | Format / version |
| --- | --- | --- |
| [arenas/example/flat-duel.json](arenas/example/flat-duel.json) | Geometry, bounds and physical rules | `minecraft_sim_arena` / 1 |
| [environments/example/sword-duel.json](environments/example/sword-duel.json) | Players, loadouts, spawn/gear sampling, limits and rewards | `minecraft_sim_environment` / 2 |
| [policies/example/melee.json](policies/example/melee.json) | Observations and available action choices | `minecraft_sim_policy` / 1 |
| [curricula/example/melee.json](curricula/example/melee.json) | Learner settings, opponents, stages and evaluation | `minecraft_sim_training_plan` / 2 |

Reuse shared definitions when appropriate; do not depend on another person's files.
One policy layout stays fixed throughout a plan; new plans should use version 2.

## Rewards

Set weights in the environment's `task.rewards`. Omitted terms are zero; use
negative weights for penalties. Contributions add together for each player.
An episode is one duel, ending in a loss or at `task.max_ticks`.

| Reward | What it measures | When / how much it pays |
| --- | --- | --- |
| `doing_damage` | Opponent HP actually removed after armor, absorption and immunity; no overkill | Per damaging hit: `weight * HP_removed / 20`, not the full weight per hit |
| `taking_damage` | Own HP actually removed by the opponent | Per damaging hit: `weight * HP_removed / 20`; normally negative |
| `won_duel` | Opponent dies or leaves the arena while you do not | Full weight once at duel end; no win on timeout or simultaneous losses |
| `death` | You die or leave the arena | Full weight once at duel end; both players receive it if both lose |
| `episode_timeout` | Tick limit reached without a loss | Full weight once for each player at timeout |
| `no_damage_penalty` | You dealt no actual health damage during a timed-out duel | Full weight once at timeout, in addition to `episode_timeout`; not applied on death/arena loss |
| `missed_attack` | An attempted swing targets no player, including swings into blocks | Full weight per miss; normally negative. No penalty for shield contact, absorption, damage immunity or suppressed attack inputs |
| `shield_disabled` | Your attack actually disables the opponent's active shield | Full weight to the attacker once per disable; selecting an axe or hitting an already-disabled shield earns nothing; no per-duel cap |
| `win_speed` | Win sooner within the tick limit | Once on a win: `weight * max(0, 1 - elapsed_ticks / max_ticks)`, added to `won_duel`; zero for losses, simultaneous losses and timeouts |
| `blocked_with_shield` | You actually block a non-axe hit | Full weight per block; repeated blocks can exceed the weight over a duel; holding a shield or blocking an axe pays nothing |
| `good_aim` | Visible opponent eye-to-eye aim: score decreases linearly from 1 at perfect aim to 0 at 30 degrees error | Episode budget, distributed per tick |
| `proximity` | 3D feet-to-feet distance: score 1 within 3 blocks, decreasing linearly to 0 at 10 blocks | Episode budget, distributed per tick |
| `pitch_control` | Level view: score decreases linearly from 1 at pitch 0 to 0 at pitch +/-90 degrees | Episode budget, distributed per tick |
| `aim_hold` | Score 1 while visible opponent aim error is strictly below 20 degrees, otherwise 0; no minimum hold time | Episode budget, distributed per tick |
| `extreme_pitch_penalty` | Score 1 when absolute pitch is strictly above 60 degrees, otherwise 0 | Episode budget, distributed per tick; normally negative |

Each budgeted term pays `weight * score / max_ticks` per tick, with score 0..1.
Its total magnitude is at most its weight's magnitude over a full duel, not that
amount every tick. Early endings leave the rest unpaid; no shaping pays on the
death/arena-loss tick. Aim error is the larger absolute yaw/pitch error; terrain
blocking the eye-to-eye line gives zero for both aim terms.

For example, `doing_damage: 10` gives +1 for removing one heart (2 HP), or +10
for removing all 20 HP across any number of hits, plus a separate win reward.
Misses, fully blocked hits and absorption-only hits give no damage reward and do
not prevent `no_damage_penalty`. Critical hits count their actual damage once.
The optional `missed_attack`, `shield_disabled` and `win_speed` weights default
to zero. To try them, add `"missed_attack": -0.1`, `"shield_disabled": 2` and
`"win_speed": 20` to your existing reward map. A win halfway through the limit
then earns +10 speed reward; this bonus never rewards losing quickly.
See [reward semantics](../training/AGENTS.md#reward-semantics) for the runtime contract.

## Choose Opponents

Opponent IDs such as `fighter` are names you choose; stages select a named mixture.

| `type` | `controller` | Behavior / required fields |
| --- | --- | --- |
| `scripted` | `stationary_target` | Idle target; no `options` |
| `scripted` | `basic_melee` | Approaches, aims and attacks; optional settings below |
| `scripted` | `random_walk` | Roams without attacking; `change_interval` (20 ticks), `wall_margin` (1.25 blocks); needs yaw and all eight world-ray observations |
| `scripted` | `stationary_melee` | Aims and retaliates without moving; accepts attack/distance/shield/weapon options, not locomotion options |
| `scripted` | `tactical_melee` | `basic_melee` options plus `style`: `evasive` retreats/circles, `aggressive` presses forward/jumps, `patient` retreats while recharging |
| `checkpoint` | Omit | Frozen saved policy; requires compatible checkpoint `path` |
| `self_play` | Omit | Current learner policy; weights stay fixed during each rollout |
| `frozen_self_play` | Omit | Rolling frozen learner copy; requires positive `every_updates` |

Policy opponents accept `deterministic` (default `false`; `true` uses argmax).
Frozen snapshots refresh every N global updates; ongoing episodes keep their assigned version.
Custom scripted `package.module:factory` controllers execute trusted Python code.

| `basic_melee.options` | Default | Meaning |
| --- | --- | --- |
| `attack_interval` | `13` | Ticks between attack attempts |
| `melee_distance` | `3` | Attack distance in blocks; strictly between 0 and 10 |
| `strafe` | `false` | Enable sideways movement |
| `strafe_interval` | `20` | Ticks between strafe direction changes |
| `sprint` | `false` | Enable sprinting |
| `shield` | `false` | Enable shield behavior |
| `axe_slot`, `sword_slot` | Omitted | Enable weapon switching; supply both, distinct slots 0-8 |

Shield/weapon options need matching policy observations/actions and actual loadouts.

## Example Plan

The example uses only the four example files: a walled arena, fixed iron swords,
a small movement/melee policy and one 100-update stage with periodic evaluation.
Its `combat` mixture samples `target` with weight 1 and `fighter` with weight 3:
25% stationary targets and 75% strafing/sprinting melee fighters, chosen per episode.
Weights are relative probabilities; two weight-1 entries would give a 50/50 mix.

The [advanced plan](curricula/example/advanced-melee.json) progresses through
aim, pursuit, stationary retaliation, moving melee, shield/axe combat and mixed
opponents with rolling frozen self-play. Pursuit enables walking, jumping,
sprinting, aiming and attacks; its evaluation scores only damage (10 points per
health bar) and win speed (up to 20), requiring mean return >=20 and win rate >=90%.
Critical hits count actual damage, not an extra jump/crit bonus. Later stages
advance on their own proficiency evaluations; thresholds are initial settings,
not evidence that a particular training budget will reach them.

Three evaluation-only tactical styles report separate win rates every 25 global
updates, 64 duels per style. They use independent fixed seeds and final-stage
conditions for comparable results, including a baseline during aim. Their IDs
never enter training mixtures or advancement criteria. These synchronous checks
add 192 evaluation duels per interval; monitoring does not prove human/general
opponent performance. See [controller/evaluation details](../training/AGENTS.md#controllers-and-constraints).

## Stages And Evaluation

| Stage field | Use |
| --- | --- |
| `name`, `opponents` | Unique stage name and a key from `mixtures` |
| `environment` | Optional replacement for the plan's default environment |
| `controls.fixed` | Fix learner/self-play heads, e.g. `{"jump":false}`; omitted heads remain learned |
| `task_override` | Change `max_ticks` or `rewards`; supplied reward maps replace, not merge; omitted rewards become zero |
| `spawns`, `equipment` | Patch sampling; `null` disables it; `sample` is `world` or `episode` |
| `advance` | Stage-local `updates`, `episodes`, `transitions`, or an `evaluation` threshold; combine with `any`/`all` lists |
| `evaluations` | Named suites with required `every_updates`, `episodes`, `seed` and optional parallel `worlds` |

Each stage starts from its declared environment, not the previous stage's patches.
Local evaluation suites inherit their stage and use stage-local update intervals.
Top-level suites use global intervals; `stage` pins conditions to a named stage.
Use suite `overrides` for changed conditions. Threshold metrics are `win_rate` or
`reward_mean` (mean episode return, not training's mean per-step reward).

Retained policy saves are opt-in: inside a stage, add
`"snapshots":[{"name":"combat-100","at_updates":100}]` to save after 100 stage updates.
At the plan root, `"snapshots":[{"name":"final","on_completion":true}]` saves when
the curriculum completes; stages also support `on_completion`. Saves go to the run's
`snapshots/<name>.pt` for checkpoint opponents or `--initialize`, not `--resume`.
See [snapshot rules](../training/AGENTS.md#retained-policy-snapshots) for details.

## Validate And Run

From the project root, with the Python environment activated:

```sh
python -m simulator.training.validate_plans simulator/setups/curricula/example/melee.json
bash rosie/submit.sh --dry-run
bash rosie/submit.sh
```

Validation checks JSON/references/semantics without training; omit paths to scan all curricula.
Use a fresh run name after edits; resume restores saved conditions, not edited JSON.
More: [plan reference](../training/AGENTS.md#research-training-plans),
[environment contract](../../docs/simulation_contract.md), [Rosie](../../rosie/README.md).