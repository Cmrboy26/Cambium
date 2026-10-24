# Create Training Setups

Use JSON to define a task, its policy and the sequence of training stages.
Start with the [advanced-melee example](curricula/example/advanced-melee.json).
Paths resolve relative to the JSON file containing them, not the working directory.

For a guided authoring and validation workflow, use the
[setup skill](../../../.github/skills/gpu-sim-setup/SKILL.md) (`/gpu-sim-setup`).

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

## Choose Observations

The policy's `observations` list selects inputs in order; unlisted features are
not supplied to the policy. Each entry is `{"feature": "self.velocity.x"}`.
The supported names come from the [shared catalogue](../../contracts/policy_catalogue.json).
In the table, `self`/`opponent` means both prefixes are supported; `x/y/z`
and slot `N` expand into separate feature names.

All model inputs are `float32` numbers. "Binary" below means `0.0` or `1.0`,
not a separate boolean tensor. By default, each feature produces one scalar input,
including integer-valued slot indices. Ranges below describe the original feature
before optional clipping and the `scale` multiplier (default `1`). Timer/progress
inputs are normalized floats, not raw tick counts.

Existing feature-only entries need no changes. Optional encodings can be mixed:

```json
"observations": [
	{"feature": "self.health_fraction"},
	{"feature": "self.weapon_slot", "encoding": "one_hot", "categories": [0, 1]},
	{"feature": "self.velocity.x", "clip": [-0.5, 0.5], "scale": 2}
]
```

This example produces four inputs: health, two selected-slot flags and clipped/
scaled velocity. `encoding` defaults to `scalar`; `one_hot` supports slot indices
and binary features, with distinct integer `categories` in the desired input order.
An unlisted category produces all zeros. Scalar `clip: [min, max]` runs before
`scale`; one-hot applies `scale` to its flags and does not allow clipping.
The total expanded layout is limited to 128 inputs. Encoding/clip changes need a
new compatible checkpoint and rebuilt native library/updated mod; plain scalar
policies retain their existing layout and fingerprint. See the
[encoding contract](../../docs/simulation_contract.md#policy-layout) for details.

| Observation | Feature names | Input type / range | Meaning |
| --- | --- | --- | --- |
| Health | `self.health_fraction`, `opponent.health_fraction` | Float, 0..1 | Current health / 20 |
| Alive | `opponent.alive` | Binary | 1 while opponent health is above zero |
| Facing | `self.yaw_fraction`, `opponent.yaw_fraction`, `self.pitch_fraction`, `opponent.pitch_fraction` | Signed float | Wrapped yaw / 180 ([-1,1)); pitch / 90 ([-1,1]) |
| Relative position | `relative.x/y/z` | Signed float | Opponent minus self feet position / 10 blocks; not clamped |
| Own velocity | `self.velocity.x/y/z` | Signed float | World-axis blocks per tick, not normalized |
| Relative velocity | `relative.velocity.x/y/z` | Signed float | Opponent minus self world-axis velocity; add own velocity to recover opponent velocity |
| Ground contact | `self.on_ground`, `opponent.on_ground` | Binary, 0 or 1 | Actual current ground-contact state; not inferred from vertical velocity or the jump action |
| Sprint state | `self.sprinting`, `opponent.sprinting` | Binary, 0 or 1 | Actual sprinting state after eligibility/cancellation checks, not the requested sprint action |
| Terrain | `arena.ray_world_0`, `arena.ray_world_45`, ..., `arena.ray_world_315` | Float, 0..1 | Eight world-fixed rays at 45-degree intervals; terrain hit distance / 50 blocks; 1 if no nearer hit |
| Target distance | `target.distance_fraction` | Float, 0..1 | Feet-to-feet distance / 10, capped at 1 |
| Target alignment | `target.aim_dot` | Float, -1..1 | Facing alignment: 1 toward opponent, 0 perpendicular, -1 away |
| Target angle errors | `target.yaw_error`, `target.pitch_error` | Signed float | Yaw error / 180 ([-1,1)); pitch error / 90 ([-2,2]) |
| Attack readiness | `self.attack_charge` | Recharge progress, 0..1 | Weapon-dependent recharge progress; 1 is fully charged, not elapsed time since a successful hit |
| Weapon type | `self.weapon_is_axe`, `opponent.weapon_is_axe` | Binary | 1 if the current weapon is an axe |
| Selected slot | `self.weapon_slot`, `opponent.weapon_slot` | Discrete index, 0..8 | Actual selected hotbar slot, encoded as one float |
| Weapon attributes | `self`/`opponent.weapon_damage_fraction`, `self`/`opponent.weapon_speed_fraction` | Float, 0..1 | Selected weapon damage / 10 and attack speed / 4, clamped; base attributes, not timers |
| Shield presence/use | `self`/`opponent.has_shield`, `self`/`opponent.shield_using` | Binary | Shield equipped and actively using it; using is not necessarily fully raised |
| Own shield disabled | `self.shield_disabled` | Binary | 1 while own disable cooldown is greater than zero |
| Opponent disable observed | `opponent.shield_disabled_observed` | Binary | Remembered observed axe-disable flag; not the opponent's exact remaining cooldown |
| Shield cooldown | `self.shield_cooldown_fraction` | Countdown, 0..1 | Remaining shield-disable ticks / 100, capped at 1; 0 means no disable cooldown |
| Shield raising | `self.shield_raise_fraction` | Elapsed progress, 0..1 | Completed shield-use ticks / 5, capped at 1; 0 when not using, 1 when fully raised |
| Armor | `self`/`opponent.armor_defense_fraction`, `self`/`opponent.armor_toughness_fraction` | Float, 0..1 | Defense / 20 and toughness / 12, clamped |
| Own slot flags | `self.weapon_slot_N.present`, `.is_axe` | Binary | Item present and axe flags for N=0..8; no hidden opponent inventory |
| Own slot attributes | `self.weapon_slot_N.damage_fraction`, `.speed_fraction` | Float, 0..1 | Slot damage / 10 and speed / 4, clamped; empty slots give zero |

At 20 simulation ticks per second, shield cooldown `0.5` means 50 ticks
(2.5 seconds) remaining; shield raising `0.4` means two of five use ticks
completed. No feature directly reports time since the last attack or hit.

`advanced-melee-v2` appends the four ground-contact/sprint flags to its previous
36 inputs, for 40 scalar inputs. Start a fresh run for this changed layout;
36-input checkpoints are not compatible with the updated policy. Rebuild the
native library and live mod for the expanded shared catalogue.

## Available Actions

The policy's `actions` list chooses which heads exist and the allowed `values`
for each head, in order. Each head selects one choice; all heads combine in the
same tick, so movement, aiming and attacking can happen together.

| Action head | Available choices | Effect |
| --- | --- | --- |
| `movement` | `neutral`, `forward`, `forward_right`, `right`, `backward_right`, `backward`, `backward_left`, `left`, `forward_left` | Movement relative to facing |
| `jump` | `false`, `true` | Request a jump |
| `sprint` | `false`, `true` | Request sprinting |
| `attack` | `false`, `true` | Attempt a melee attack |
| `yaw` | Configured degree deltas | Turn horizontally |
| `pitch` | Configured degree deltas | Look up/down |
| `weapon` | Configured slot indices 0..8 | Select a hotbar slot |
| `shield` | `false`, `true` | Release/use the shield |
| `sneak` | `false`, `true` | Release/hold sneak |

The model outputs one score (logit) per configured choice in each head, not a
direct angle, slot number or boolean. Training samples a choice index from that
head's softmax probabilities; deterministic inference selects the highest-scoring
index. The decoder looks up `values[index]`: with yaw `values: [-10, 0, 10]`,
index `2` means a `+10` degree turn. Boolean heads work the same way, not via a
threshold. Stage `controls.fixed` overrides selection for the named heads.

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
| `scripted` | `random_walk` | Roams without attacking; `change_interval` (20 ticks), `wall_margin` (1.25 blocks), optional `sneak` (false); needs yaw and all eight world-ray observations |
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