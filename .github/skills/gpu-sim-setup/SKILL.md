---
name: gpu-sim-setup
description: 'Create, modify, validate, and smoke-test GPU simulator setups in gpu-sim-minecraft/simulator/setups. Use for authoring arenas, environments, observation/action policies, curricula, opponent pools, rewards, and evaluation stages using existing supported features. Not for implementing new Minecraft mechanics, observation/action primitives, or controller types; use the contribution playbook for those.'
argument-hint: 'Setup owner, task, policy layout, and training goal'
---

# GPU Simulator Setup Authoring

## Scope And References

Identify the project root containing `contracts/policy_catalogue.json` and
`simulator/setups/`. In the Cambium workspace this is `gpu-sim-minecraft`.
Run all commands below there using the intended Python environment, not from
Cambium's parent package context. Read the applicable project instructions.

Use the [setup guide](../../../gpu-sim-minecraft/simulator/setups/README.md),
[training reference](../../../gpu-sim-minecraft/simulator/training/AGENTS.md#research-training-plans)
and [scenario contract](../../../gpu-sim-minecraft/docs/simulation_contract.md)
as the authoritative schemas. If the task needs an unsupported primitive or
mechanic, stop configuration work and use the
[contribution playbook](../../../gpu-sim-minecraft/CONTRIBUTING.md).

## 1. Establish The Experiment

- Determine the owner folder, task, fixed versus sampled conditions, loadouts,
  policy observations/actions, opponent types, rewards, stage goals and evaluation.
  Ask only for missing choices that materially affect the experiment; reuse
  agreed conditions rather than silently changing the research question.
- Inspect the nearest suitable example and the user's existing files. Preserve
  their edits. Do not edit shared examples or another person's setup as a shortcut.
- Check that the requested observations, actions, items, rules and controller
  prerequisites are supported. Native-only mechanics are not automatically
  available to setup authoring or live Minecraft.

## 2. Author The Four Layers

Create only the needed definitions; reuse intentionally shared definitions.
Use the same person-named folder under each category for personal work. The
`example/` directories are starter templates, not destinations for new experiments.
Do not depend on another person's files without explicitly agreeing to that reuse.

| Category | Owns | Format / Version |
| --- | --- | --- |
| `arenas/<owner>/` | Geometry, bounds and physical rules | `minecraft_sim_arena` / 1 |
| `environments/<owner>/` | Players, loadouts, spawn/gear sampling, limits and rewards | `minecraft_sim_environment` / 2 |
| `policies/<owner>/` | Ordered observations and semantic action choices | `minecraft_sim_policy` / 1 |
| `curricula/<owner>/` | Learner settings, opponents, mixtures, stages and evaluation | `minecraft_sim_training_plan` / 2 |

- Rename JSON `name` values and resolve each reference relative to the JSON file
  containing it, never relative to the shell working directory. Keep legacy
  concrete fixtures under `simulator/environments` separate from new authoring.
- Select existing features from the catalogue; preserve intended units/scales.
  Scalar is the default. Optional one-hot expands the input width and needs
  explicit supported categories; scalar clipping precedes scaling. Check the
  expanded dimension, not just the number of descriptors.
- Keep a single immutable policy layout throughout the plan. Stage fixed
  controls constrain semantic choices; they do not change head sizes. Check every
  opponent against the available observations, movement choices and weapon slots.
- Make task rewards explicit; omitted terms are zero. Verify selected weapon
  slots and sampled variants, spawn clearance, episode limits and supported rules.
- Give stages measurable advancement conditions and separate evaluation intent.
  Use documented snapshot keys exactly, including `at_updates` or `on_completion`;
  reject unknown keys rather than treating near-matches as valid.

## 3. Validate Without Training

Use the specific new plan, not the default whole-tree scan: unrelated personal
plans may be incomplete. Replace the example path with the authored plan.

```sh
python -B -m simulator.training.validate_plans simulator/setups/curricula/example/advanced-melee.json
```

Inspect the compiled, self-contained experiment without creating run artifacts:

```sh
python -B -c "import json; from simulator.training.plans import load_plan; print(json.dumps(load_plan('simulator/setups/curricula/example/advanced-melee.json'), indent=2))"
```

Check the resolved policy's expanded width, head choices, arena/loadouts,
per-stage fixed controls, rewards, opponent mixtures and evaluations. Compilation
is not proof of collision-free sampled worlds or runtime controller behavior.
If validation fails, repair the authored definition and rerun the same check;
do not weaken validators or modify unrelated plans to obtain a green result.

## 4. Optional Bounded Runtime Check

When runtime verification is needed, use an existing matching CPU native library
and a new, unused output directory. Check the output path first; never overwrite
a run or use `--resume` merely to test a new setup. Add `--library` with the
intended SO/DLL path if native discovery is ambiguous. Do not install toolchains,
rebuild CUDA jobs or launch a dashboard as part of configuration authoring.

```sh
python -B -m simulator.training.train --plan simulator/setups/curricula/example/advanced-melee.json --device cpu --envs 2 --updates 1 --rollout-steps 2 --max-ticks 4 --threads 1 --replay-every 0 --output simulator/training/runs/setup-smoke-UNIQUE
```

Replace both example paths before executing. The small rollout and explicit
episode-limit override are plumbing checks, not the authored experiment's
evaluation or evidence of policy quality. Restore authored limits for real runs.
Check saved resolved conditions and finite training metrics. If the library is
missing/stale, report the required build and validation result; do not claim a
runtime check succeeded. Changed policy layouts need new compatible weights.

## 5. Handoff

List created/changed definitions, reference reuse, observations/action dimensions,
important rewards/stage conditions, exact validation and optional smoke results,
and untested conditions. Provide the real training command for the user's plan
without smoke-only overrides. No long training run or live deployment is implied.
