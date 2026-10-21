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

## Choose Opponents

Opponent IDs such as `fighter` are names you choose; stages select a named mixture.

| `type` | `controller` | Behavior / required fields |
| --- | --- | --- |
| `scripted` | `stationary_target` | Idle target; no `options` |
| `scripted` | `basic_melee` | Approaches, aims and attacks; optional settings below |
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