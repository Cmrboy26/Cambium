# Phase 1 Bridge: Implementation Checklist

**Status:** In progress; Bridge mode is not implemented. This is the shared
implementation tracker for the seven Phase 1 milestones. Existing duel
features are foundations, not completed Bridge features.

## Next Action

**Owner:** Unassigned for the next slice. **Blocked by:** None recorded.
**First task:** P1-02-03.

1. Coordinate arena size, overflow checks and memory-cap ownership across the
  compiler, native and live paths before extending the scene limit. The first
  two-world overlay slice is low-level only and does not expose Bridge setups.
2. A static-arena match lifecycle may proceed independently against the
  [versioned rules](bridge_mode_v1.md); keep unverified external rules and
  the later setup/capability gate visible.

Starting points: [`BlockWorldView`](../simulator/include/minecraft_sim/geometry.hpp),
the native [scene loader](../simulator/apps/scene.hpp), [CUDA batch](../simulator/src/cuda_backend.cu),
and the existing [CPU](../simulator/tests/core_checks.cpp) and
[CUDA](../simulator/tests/observation_cuda_checks.cu) checks.

## Agent Handoff Rules

- Read this tracker, the [contribution playbook](../CONTRIBUTING.md), and the
  relevant component guides before editing. Work from the first unblocked task;
  coordinate ownership of the catalogue, ABI, setup schemas and shared files.
- Each task has a stable ID. `[ ]` means not complete; record **in progress**
  or **blocked**, an owner and the reason beside it when applicable. `[x]`
  means all stated acceptance criteria and applicable gates passed. Do not mark
  a milestone complete until its child tasks and gate are complete.
- For every completed task, record the date, commands, results including skips,
  relevant source/build or commit identity, hardware/runtime, durable evidence
  links when available, and remaining limitations. A skipped or unavailable
  check is not a pass. Existing local run artifacts may be ignored or removed;
  do not rely on them as the sole durable evidence.
- Separate source-derived expectations, CPU/CUDA parity, in-game captures,
  live integration and performance results. Agreement between simulator
  backends alone does not establish vanilla or Hypixel conformance.
- After each slice, update the task, evidence and **Next Action** in this file.
  Keep unresolved decisions and externally blocked checks visible. Do not
  claim another contributor has completed a dry run without their evidence.

## Scope And Baseline

Phase 1 delivers a playable Bridge-style subset on Minecraft Java 26.3 shared
CPU/CUDA mechanics, an opt-in representative GPU-resident PPO path, a model
extension interface, dynamic replay and live 26.3 inference. The initial
resident path targets one fixed arena per stage with stationary and current
self-play opponents; other configurations retain an explicit host fallback.

The proposal's default is first to five goals, an 18,000-tick match cap and
blocks persisting across goals. Cage duration, exact item counts, arrow
regeneration, and other Hypixel-specific details must be verified or explicitly
configured as provisional. Decide and test time-cap ties, simultaneous goals,
death/goal ordering and any other unspecified rules; do not present a proposed
choice as verified Hypixel behavior. Vanilla movement/combat remains 26.3, not
1.8.9 parity.

Out of scope: a production CNN or CNN comparison, Phase 2/3 research campaigns,
RUDDER/RRD, 1.8.9 physics, general Minecraft simulation and multi-GPU PPO. The
Phase 1 model gate includes the existing MLP and a small **test-only** alternate
model, not a trained CNN. Minimal score/win/loss signals for a smoke run do not
claim that the later progress-shaping reward has been implemented.

### Existing Duel Foundations

- Baseline inspected on 2026-11-12 at repository commit
  `725ea068c22b0152872e7f8758faf0e0baf15a47`. The validation below is a
  historical session record, not a claim that this tracker reran the tests.
- [x] P1-00-01: Shared 26.3 CPU/CUDA movement and combat, ordinary finite-ammo
  bow support, scalar/one-hot policy projections and legacy duel training exist.
  Evidence: [native guide](../simulator/AGENTS.md),
  [bow scope](bow_arrow_implementation.md), and
  [training guide](../simulator/training/AGENTS.md).
- [x] P1-00-02: An opt-in resident duel path has device observations, actions,
  rewards/endings, fixed selective resets and PPO collection with stationary or
  current self-play opponents. Unsupported plans use the host path or are
  rejected explicitly. Evidence: [training guide](../simulator/training/AGENTS.md)
  and [resident collector](../simulator/training/research.py).
- [x] P1-00-03: A bounded CUDA duel validation was recorded on 2026-11-12: the
  final T4 run with CUDA 12.9 and PyTorch 2.5.1+cu124 passed four native CUDA
  suites and 30 CUDA/PyTorch checks, including rollout, checkpoint and trace
  checks. The recorded commands were `ctest --test-dir simulator/build-phase1-t4`
  with the four CUDA suite names selected and `python -B -m unittest
  simulator.training.test_cuda_observation
  simulator.training.test_research.ResidentOpponentChecks` with the isolated
  CPU/CUDA libraries selected. Evidence: the existing
  [native CUDA fixtures](../simulator/tests/observation_cuda_checks.cu) and
  [Python CUDA tests](../simulator/training/test_cuda_observation.py); the run
  results were recorded in the implementation session, not a durable CI report.

The 2026-11-12 H100 duel tests and short aiming-only comparison preceded the
final checkpoint-counter and finite-data changes. They are useful baseline
evidence, not a final-source H100 gate, representative combat benchmark, Bridge
result or total-device-memory measurement. The final H100 gates remain open.

## Dependencies And Ownership

Rules and setup ownership (P1-01-01 through P1-01-04) precede terrain and match integration. Terrain
is needed for building and grid observations; the basic match lifecycle can
first be tested against a static arena. The model interface can proceed in
parallel once observation/action shapes are agreed. Resident Bridge collection
depends on the shared mechanics and observation contracts. Replay/live adapters
may start against agreed fixtures, but their final gate needs the completed
features. The positive setup validation gate P1-01-06 remains blocked until its
declared Bridge capabilities are implemented; do not make P1-02 through P1-05
wait for that final gate. Avoid simultaneous uncoordinated edits to shared
schemas or ABI.

| Layer | Bridge responsibility | Existing entry point |
| --- | --- | --- |
| Arena | Base geometry, protected regions, goal/cage/build-zone geometry | [Setup guide](../simulator/setups/README.md) |
| Environment | Teams, spawns, kits, match rules, timers and task rewards | [Scenario contract](simulation_contract.md) |
| Policy | Ordered scalar/grid observations and semantic action choices | [Policy catalogue](../contracts/policy_catalogue.json) |
| Curriculum | Learner, stages, opponents, controls and evaluations | [Training guide](../simulator/training/AGENTS.md) |

These are responsibilities for the extension, not fields accepted by today's
schema. The compiler must reject unsupported features until all relevant
adapters implement them. Existing authoring files compile into self-contained
resolved experiments; checkpoints and replays must retain the effective
conditions and distinguish compatible legacy duels from Bridge.

## 1. Rules And Setup Contract

- [ ] P1-01-00: Complete this milestone only after P1-01-01 through P1-01-06.
- [x] P1-01-01: Write and review a versioned Bridge mode specification covering
  teams/perspectives, coordinates, goals, cage regions, protected versus placed
  blocks, build limits, void depth, kits, timers and observability. Record each
  confirmed value, provisional value and unresolved external rule separately.
- [x] P1-01-02: Define tick phases and deterministic outcomes for death versus
  goal on one tick, both players scoring, simultaneous match-winning scores,
  goal reset, timeout and ties. Define whether timeout is termination or
  truncation and how PPO bootstraps it. Keep match end, goal/round reset and
  individual player death/respawn distinct.
- [x] P1-01-03: Define spawn, cage, refill and match-reset behavior, including
  which player, projectile, RNG, timer and terrain state is preserved or
  cleared. State explicitly whether and when blocks persist across goals.
- [x] P1-01-04: Specify the arena/environment/policy/curriculum fields and
  version/capability negotiation, including checkpoint and replay identity.
  Review [compiler.py](../simulator/training/compiler.py),
  [plans.py](../simulator/training/plans.py), the
  [scenario contract](simulation_contract.md) and native scene parsing.
  **Evidence, P1-01-01 through P1-01-04:** On 2026-11-13 GitHub Copilot
  reviewed the four tasks against [Bridge mode v1](bridge_mode_v1.md), the
  referenced compiler/plans/scenario/scene and current tick/GAE paths. The
  match-cap result and provisional external rules were agreed with the user.
  `python -B scripts/generate-contract.py --check` passed; proposed Bridge
  fields remain unsupported and no Hypixel capture was made. Source baseline:
  `951d573168851030f6d8ff8bcc13fab0b2dcdb16` plus this slice's changes;
  Linux x86_64, GNU 13.3.0. No vanilla-derived mechanics were added in this
  slice, so P1-01-05 and the positive setup gate remain open.
- [ ] P1-01-05: Trace each vanilla-derived addition to the actual controlling
  Minecraft 26.3 source class, method and verified line numbers. Put concise
  citations at the implementation; cite custom Bridge rules to the new mode
  specification, not to vanilla. Record the source/decompiler identity.
- [ ] P1-01-06: **Gate, after the declared capabilities exist:** validate and
  resolve a representative Bridge setup;
  reject invalid goals, overlapping/unsafe spawns, kits, zones, limits,
  unsupported capabilities and inconsistent versions. Confirm old duel
  setups/checkpoints still load or reject explicitly as designed.

## 2. Per-World Mutable Terrain

**Prerequisites:** P1-01 rule/geometry ownership and reset decisions. Coordinate
the composite terrain interface before parallel physics or observation work.

- [ ] P1-02-00: Complete this milestone only after P1-02-01 through P1-02-07.
- [x] P1-02-01: Define an immutable protected base plus an independently owned
  per-world editable overlay, or document and benchmark an equivalent design.
  Specify air, placed-block/team identity, protected blocks, collision,
  out-of-bounds behavior and deterministic simultaneous edits.
- [x] P1-02-02: Implement a two-world first slice sharing a protected base,
  with independently calculated expected collision and ray results. One world's
  mutation or selective match reset must never change another's state.
  **Evidence, P1-02-01 and P1-02-02:** On 2026-11-13 GitHub Copilot added the
  [overlay owner](../simulator/include/minecraft_sim/terrain.hpp), composite
  lookups and selective CPU/CUDA resets. Independent ray expectations 0.5,
  0.75 and 1.0, collision, ownership, protected cells and selected resets are
  exercised in [CPU](../simulator/tests/core_checks.cpp) and
  [CUDA](../simulator/tests/observation_cuda_checks.cu) fixtures. With CPU and
  CUDA libraries built from native source hash
  `5589e48fda1c2f5607f43b8bafd01a58b2b3c17c3cd42e98e099fcf2f8bd2261`
  at baseline `951d57316885` plus this slice, `ctest --test-dir
  simulator/build-bridge-reissmannc-cpu --output-on-failure` passed 6/6 on
  Linux x86_64, GNU 13.3.0. On a Tesla T4 (driver 610.43.02, CUDA toolkit
  12.9.86), `ctest --test-dir simulator/build-bridge-reissmannc-t4
  --output-on-failure -R '^(shared_core_cuda_checks|training_api_cuda_checks|projectile_cuda_checks|observation_cuda_checks)$'`
  passed 4/4. With those libraries selected via `MCSIM_TEST_CPU_LIBRARY` and
  `MCSIM_TEST_CUDA_LIBRARY`, `.venv-rosie/bin/python -B -m unittest -v
  simulator.training.test_cuda_observation
  simulator.training.test_research.ResidentOpponentChecks` passed 29 checks;
  one two-device check was skipped. PyTorch was 2.5.1+cu124. GPU builds warned
  about cluster clock skew even after a clean target rebuild; the relevant
  source and test targets were recompiled. H100 was not requested. This is
  low-level isolation, not Bridge setup, building, replay, full-map support,
  in-game conformance or a performance result.
- [ ] P1-02-03: Extend arena dimensions and cell budget for an approximately
  32 x 48 x 96 map. Validate sizes, multiplication overflow, fill counts and a
  documented configurable memory cap across compiler, native and live paths.
- [ ] P1-02-04: Make collision, raycasting, targeting and observations read the
  same per-world composite terrain on CPU and CUDA. Preserve old static-duel
  behavior and avoid full terrain transfers on each tick.
- [ ] P1-02-05: Restore only selected overlays on full match reset; preserve
  placed blocks across goals by default and test the explicit alternate rule.
- [ ] P1-02-06: Cover protected-block rejection, team identity, boundaries,
  simultaneous edits, nonaligned batch sizes, world isolation and selective
  reset with focused CPU fixtures and matching CUDA sequences.
- [ ] P1-02-07: **Gate:** CPU and CUDA observe identical intended terrain and
  collision/ray results after edits and resets; static duel regression passes;
  recorded memory use respects the configured cap.

## 3. Match Lifecycle

**Prerequisites:** P1-01 semantics. A static-arena vertical slice can proceed
before building; the complete milestone also requires P1-02 persistence.

- [ ] P1-03-00: Complete this milestone only after P1-03-01 through P1-03-07.
- [ ] P1-03-01: Add per-world team scores, match clock, goal/cage phase and
  per-player respawn/kit state in shared CPU/CUDA-compatible ownership.
- [ ] P1-03-02: Implement cages and their opening, goal detection and scoring,
  configurable first-to-N result, time cap and specified simultaneous/tie
  behavior. Emit explicit score and match outcome events.
- [ ] P1-03-03: Implement void/combat death and player-only respawn/refill
  without ending the match; clear the affected player's relevant state and
  preserve the other player's state according to the specification.
- [ ] P1-03-04: Implement goal/round reset of both players and their kits while
  preserving scores, match clock and terrain as configured; implement a
  separate full match reset that restores the selected world's full state.
- [ ] P1-03-05: Connect match outcomes and minimal score/win/loss task events
  to the training API and metrics. Preserve final observations and the specified
  termination/truncation semantics; a death or nonwinning goal is not a match
  terminal. Do not silently substitute duel rewards or endings.
- [ ] P1-03-06: Add independently specified cases for both sides, cage timing,
  player death, void, goal, simultaneous events, time-cap ties, selective reset
  and persistence, including shared CPU/CUDA state and result comparisons.
- [ ] P1-03-07: **Gate:** a complete basic Bridge match runs against a fixed
  opponent on CPU and CUDA with expected scores, resets and match outcome;
  existing duel termination and checkpoint behavior remain unchanged.

## 4. Building And Items

**Prerequisites:** P1-01 action/tick ordering and P1-02 composite terrain. Use
existing movement, combat and ordinary-bow behavior where applicable.

- [ ] P1-04-00: Complete this milestone only after P1-04-01 through P1-04-08.
- [ ] P1-04-01: Implement 26.3-derived eye-based full-cube face targeting,
  range, occlusion and boundary/tie behavior, with explicit entity/block action
  arbitration and independently calculated expected hits.
- [ ] P1-04-02: Implement held-use placement and repeat timing, selected-item
  and block counts, valid faces/support, occupied space, player collision,
  height/no-build restrictions and consumption only on success.
- [ ] P1-04-03: Implement held-attack pickaxe breaking with the supported
  hardness/tool/speed subset, progress and completion; cover release,
  cancellation, target/item switches and protected-block rejection.
- [ ] P1-04-04: Define hotbar selection and per-player counts/refills for sword,
  bow, pickaxe, blocks and apples. Resolve use/attack/slot precedence without
  regressing existing melee, shields and projectiles.
- [ ] P1-04-05: Implement supported golden-apple eating time, cancellation,
  absorption and regeneration; separately specify and implement the custom
  single-arrow refill timer. Do not claim general effects, inventory,
  enchantment, durability, fluid or special-block simulation.
- [ ] P1-04-06: Trace and cite actual 26.3 methods/lines for targeting,
  placement, breaking and consumption. Keep Bridge-specific refill and other
  mode rules attributed to the Bridge specification.
- [ ] P1-04-07: Test normal, boundary, cancellation and simultaneous-player
  cases on CPU and CUDA, including inventory updates, switched slots, damage
  interactions, goal/death refill and selective match reset.
- [ ] P1-04-08: **Gate:** source-derived fixtures and CPU/CUDA parity pass;
  recorded real 26.3 placement, breaking and eating cases cover the implemented
  subset, with any unverified behavior explicitly blocked rather than claimed.

## 5. Observations And Extensible Models

**Prerequisites:** Agree observation/action contracts with P1-01; grid extraction
depends on P1-02 terrain and relevant P1-03/P1-04 state. The model interface can
be developed in parallel once tensor shapes and compatibility are agreed.

- [ ] P1-05-00: Complete this milestone only after P1-05-01 through P1-05-08.
- [ ] P1-05-01: Specify optional local voxel and lane observations: axes, origin,
  rotation, categories, ownership, outside bounds, team perspective, sampling
  phase, visibility and numeric types. A 9 x 5 x 9 local window and a
  3 x 4 x 7 x 16 lane summary are candidates, not fixed requirements. Declare
  lane data global/omniscient if it includes information outside local view.
- [ ] P1-05-02: Add permitted score, inventory, arrow, cage/respawn and match
  observations; define held place/break actions and any finer look choices.
  Preserve semantic action order and avoid hidden opponent inventory leakage.
- [ ] P1-05-03: Extend the shared [catalogue](../contracts/policy_catalogue.json),
  generated C++/Java consumers, [Python contract](../simulator/training/contract.py),
  native extraction and setup compiler together. The current 128-input scalar
  limit cannot represent these grids by simply flattening the old contract.
- [ ] P1-05-04: Carry structured observations through Python/native buffers,
  rollout storage and PPO minibatches; keep categorical grids compact until
  needed and budget native plus PyTorch memory at the target world counts.
- [ ] P1-05-05: Add a trusted, versioned model registry/factory and separate
  architecture descriptor. Preserve the existing MLP's legacy state-dict keys;
  implement a test-only alternate model to prove the extension interface.
  External Python factories execute trusted code and must be explicitly trusted.
- [ ] P1-05-06: Route training, PPO, frozen/checkpoint opponents, evaluation,
  serving and inference packaging through the same model contract. Reject
  incompatible model/policy descriptors rather than silently reinitializing or
  relabeling checkpoints; preserve supported old scalar MLP checkpoints.
- [ ] P1-05-07: Add CPU/CUDA/Java fixtures for both teams, empty/boundary/placed
  cells, negative coordinates and yaw 0/45/90; test inputs, semantic actions,
  checkpoint identity, unsupported descriptors and memory bounds.
- [ ] P1-05-08: **Gate:** MLP and test model train, save, reload, act as an
  opponent, evaluate and serve through the actual adapters; grid fixtures and
  legacy compatibility pass. A production CNN remains future work.

## 6. GPU-Resident Bridge Training

**Prerequisites:** P1-02 through P1-05 contracts and relevant shared mechanics.
Start with the existing resident duel implementation, not a separate trainer.

- [ ] P1-06-00: Complete this milestone only after P1-06-01 through P1-06-08.
- [ ] P1-06-01: Extend [CUDA batch](../simulator/src/cuda_backend.cu),
  [environment.py](../simulator/training/environment.py) and
  [ResidentCollector](../simulator/training/research.py) for per-world terrain,
  Bridge state, structured observations, item events and scoring.
- [ ] P1-06-02: Implement device player, round and match resets with correct
  state/terrain preservation, per-world RNG and final observation handling.
  Preserve terminal-versus-truncation masks and timeout bootstrap in
  [compute_gae](../simulator/training/ppo.py).
- [ ] P1-06-03: Preserve pointer/shape/dtype/device validation, tensor lifetime,
  same-stream ordering, explicit cross-stream fences, delayed sticky errors and
  safe teardown. Keep graph-capture restrictions explicit; check device errors
  and nonfinite rollout values before an optimizer update.
- [ ] P1-06-04: Support one fixed arena per stage, one CUDA device, stationary
  and current-policy self-play, with device-side seat/opponent selection. Record
  backend/RNG provenance and fresh-episode checkpoint semantics. Explicitly
  reject unsupported strict configurations or report the host fallback.
- [ ] P1-06-05: Expose the backend selection through the Rosie submission
  workflow; [train.sbatch](../rosie/train.sbatch) currently invokes the trainer
  without the resident selector. Validate submission and monitoring paths.
- [ ] P1-06-06: Compare fixed-action CPU/CUDA multi-world trajectories through
  placement, break, goal, respawn, match ending and selective reset; compare
  observations, rewards, flags, values, GAE/returns, counters and PPO updates.
  Test checkpoint/resume, fallback and incompatible-contract rejection.
- [ ] P1-06-07: Capture a full decision-to-reset native CUDA runtime trace on
  the target H100: no per-decision host/device transfer, host scalar wait or
  device-wide synchronization. Device-to-device copies are allowed; setup,
  rollout-boundary metrics/evaluation/replay/checkpoint transfers and profiler
  shutdown must be labeled separately.
- [ ] P1-06-08: **Gate:** run actual resident Bridge PPO and the final current
  code on H100. Compare at least three warmed, unprofiled, matched host/resident
  duel runs, then report a separate Bridge workload. Record source/dirty state,
  build, GPU/Slurm/CPU resources, seed, model, rollout/optimizer settings,
  resets, learner transitions/s, collection/update time, variance and total
  device peak including native and PyTorch allocations. Report physics-only
  throughput separately; do not call the old aiming-only run Bridge performance.
  Record whether resident collection improves the matched duel workload; keep
  this performance gate open if it does not, unless the owner revises the goal.

## 7. Replay, Live Integration And Final Validation

**Prerequisites:** Agreed contracts and fixtures can be used early; final
validation requires the applicable mechanics, observations and resident path.

- [ ] P1-07-00: Complete this milestone only after P1-07-01 through P1-07-08.
- [ ] P1-07-01: Extend replay serialization with initial base terrain, ordered
  block deltas and score/cage/respawn/match events. Capture selected worlds
  without downloading all terrain on every training tick.
- [ ] P1-07-02: Update the dashboard and Fabric replay consumers to reconstruct
  terrain and match state deterministically, including pause, restart, cleanup
  and legacy replay compatibility. Test place/break, goals and deaths.
- [ ] P1-07-03: Extend the Python service and Java policy contract, encoder,
  control application and [live adapter](../cambiummod/AGENTS.md) for the same
  Bridge observations, actions and model descriptors. Reject unsupported modes
  or features explicitly instead of fabricating observations.
- [ ] P1-07-04: Implement or explicitly integrate a private-singleplayer Bridge
  rule controller for live scores, cages, respawns, protected blocks, kits and
  arrow refill. Vanilla Minecraft alone does not provide these custom rules.
- [ ] P1-07-05: Run shared Java fixtures, the actual Python/Java inference
  bridge and JDK 25 mod checks/build. Install only into an explicitly authorized
  disposable instance; record JAR/service identities and restart verification.
- [ ] P1-07-06: Capture reproducible real 26.3 placement, breaking, eating and
  respawn-related behavior, stating initial state, controls, tick order,
  tolerances and limitations. Validate goals/cages as custom Bridge rules.
- [ ] P1-07-07: Demonstrate a complete supported live 26.3 match and replay;
  record each teammate's setup, submission and monitoring dry run separately.
- [ ] P1-07-08: **Final gate:** the gates for P1-01 through P1-06 and tasks
  P1-07-01 through P1-07-07 pass; a four-layer
  Bridge setup validates; CPU/CUDA parity, duel compatibility, PPO
  update/save/resume/fallback, replay, live game, real captures, H100 trace and
  separate matched benchmarks have recorded evidence. Otherwise leave the
  affected tasks unchecked and list concrete blockers.

## Verification Reference

Run from the `gpu-sim-minecraft` root with an appropriate Python environment and
an isolated build directory for each contributor. The commands below are future
verification recipes, not claims that Bridge tests have already passed. Record
the commands actually used, selected native libraries and skipped checks.

```sh
CPU_BUILD="simulator/build-bridge-${USER}-cpu"
cmake -S simulator -B "$CPU_BUILD" -DBUILD_TESTING=ON
cmake --build "$CPU_BUILD" -j 4
ctest --test-dir "$CPU_BUILD" --output-on-failure
python -B scripts/generate-contract.py --check
python -B -m unittest simulator.training.test_training \
  simulator.training.test_research simulator.training.test_controllers \
  simulator.training.test_experiments simulator.training.test_cuda_observation
```

Select the newly built CPU library for the tests that use it; native library
discovery and explicit test-library environment variables are described in the
[training guide](../simulator/training/AGENTS.md). In particular, verify which
library each test actually loads rather than assuming an environment variable
overrides all discovery. Add targeted Bridge fixtures as they are implemented.

On a CUDA allocation, configure a separate build using
`-DMCSIM_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DBUILD_TESTING=ON` for
H100, then build and run the relevant CUDA CTest suites. Set
`MCSIM_TEST_CPU_LIBRARY` and `MCSIM_TEST_CUDA_LIBRARY` to the freshly built
libraries for [CUDA Python checks](../simulator/training/test_cuda_observation.py).
Use a compatible CUDA-enabled PyTorch interpreter; setup/status/teardown may
synchronize, so trace the decision region separately. See the
[Rosie guide](../rosie/AGENTS.md) for Slurm allocation and environment details.

For live integration, use JDK 25 and run from `cambiummod`:

```sh
bash gradlew check replayAnalysisChecks build --no-daemon --console=plain
```

This build is not an installed or validated game mod. Follow the
[live-mod completion gate](../CONTRIBUTING.md#live-mod-completion-gate) and
record separately what was built, installed, captured and tested in-game.
