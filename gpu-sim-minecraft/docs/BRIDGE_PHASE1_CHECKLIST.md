# Phase 1 Bridge: Implementation Checklist

**Status:** In progress; complete Bridge mode is not implemented. An experimental
fixed-arena host-staged Bridge match and PPO path run on CPU/CUDA. This is the shared implementation
tracker for the seven Phase 1 milestones; existing duel features are foundations,
not completed Bridge features.

## Next Action

**Owner:** GitHub Copilot for the current Bridge slice.
**Blocked by:** Full authored capabilities, resident collection and replay/live
adapters are not implemented; in-game conformance remains open.
**First task:** P1-01-05, alongside the remaining P1-04 item/building work.

1. Finish source citations and the supported building/item subsets; extend the
  fixed authored setup only when each capability and adapter is ready.
2. Complete structured observations, the model interface and resident Bridge
  integration; the current PPO path is host-staged and stationary-opponent only.
3. Complete the remaining replay/live, in-game and H100 validation gates before
  claiming full Bridge support.

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
  setups/checkpoints still load or reject explicitly as designed. **In progress:**
  a fixed static arena and stationary-opponent plan validate and train; kits,
  build zones, sampling and the full declared Bridge capability gate remain open.

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
- [x] P1-02-03: Extend arena dimensions and cell budget for an approximately
  32 x 48 x 96 map. Validate sizes, multiplication overflow, fill counts and a
  documented configurable memory cap across compiler, native and live paths.
  **Evidence:** On 2026-11-13 GitHub Copilot added a 262144-cell ceiling and
  `MCSIM_TERRAIN_BUDGET_BYTES` preflight in the compiler, native scene/terrain,
  CUDA batch and live/replay validator; see the
  [budget contract](../simulator/AGENTS.md#json-environments). The 147456-cell
  arena resolves and loads through CPU/CUDA; tests cover exact and insufficient
  caps, dimensions, overflow, fills and shared CUDA static bases. CPU CTest
  passed 6/6; `.venv-rosie/bin/python -B -m unittest -v
  simulator.training.test_experiments` passed 36/36. T4 CUDA CTest passed 4/4;
  the CUDA observation/resident-duel Python suites passed 29, with one
  two-device test skipped. `JAVA_HOME=/tmp/cambium-jdk25 bash gradlew check
  build --no-daemon --console=plain` passed in `cambiummod`; the mod was built
  but not installed or tested in-game. The generated catalogue check passed.
  CPU and CUDA libraries reported native source hash
  `242c6e2b432fa2c156a0d2eac00cddace191701f3734a0910f34064d4a6eac7e`,
  baseline `f45f9375ed49` plus this slice, on Linux x86_64, GNU 13.3.0,
  Tesla T4/CUDA 12.9.86, PyTorch 2.5.1+cu124 and Temurin JDK 25.0.5.
  CUDA builds reported cluster clock-skew warnings; affected targets were
  compiled and tested. The cap bounds specified terrain payloads, not total
  process/device memory; measured-memory and Bridge gameplay gates remain open.
- [x] P1-02-04: Make collision, raycasting, targeting and observations read the
  same per-world composite terrain on CPU and CUDA. Preserve old static-duel
  behavior and avoid full terrain transfers on each tick. Evidence below covers
  the implemented host-staged path; resident Bridge is separately tracked in P1-06.
- **P1-02-04 host-staged evidence, 2026-11-19:** The shared
  [match fixture](../simulator/tests/bridge_match_checks.hpp) now creates and
  breaks a block through Bridge ticks, then probes the same overlay for
  targeting, rays, collision, movement, melee and arrows before/after edits.
  These physics probes use separate test player/projectile state, not the
  training batch's actual subsequent actions. The
  [CUDA observation fixture](../simulator/training/test_cuda_observation.py)
  checks independently expected rays in host, raw CUDA and configured CUDA
  observations after real place/break actions and selective resets in two
  worlds. CPU native CTest passed 6/6; T4 native CTest passed 4/4. The Python
  CUDA/duel regression ran 31 tests: 30 passed, one two-device check skipped.
  The focused CPU Python Bridge building fixture also passed with the CPU
  library explicitly selected. Commands from the project root:

  ```sh
  cmake --build simulator/build-bridge-validation-20261119-cpu -j 4
  ctest --test-dir simulator/build-bridge-validation-20261119-cpu --output-on-failure
  cmake --build simulator/build-bridge-validation-20261119-t4 --target minecraft_sim_observation_cuda_checks -j 4
  cmake --build simulator/build-bridge-validation-20261119-t4 --target minecraft_sim_checks minecraft_sim_training_checks minecraft_sim_projectile_cuda_checks -j 4
  ctest --test-dir simulator/build-bridge-validation-20261119-t4 --output-on-failure -R '^(shared_core_cuda_checks|training_api_cuda_checks|projectile_cuda_checks|observation_cuda_checks)$'
  MCSIM_TEST_CPU_LIBRARY="$PWD/simulator/build-bridge-validation-20261119-cpu/minecraft_sim_training.so" MCSIM_TEST_CUDA_LIBRARY="$PWD/simulator/build-bridge-validation-20261119-t4/minecraft_sim_training.so" .venv-rosie/bin/python -B -m unittest -v simulator.training.test_cuda_observation simulator.training.test_research.ResidentOpponentChecks
  ```

  T4 job `302282` used `dh-node8`, Tesla T4 (15360 MiB), driver 610.43.02,
  CUDA toolkit 12.9.86, GNU 13.3.0 and PyTorch 2.5.1+cu124. The CPU and CUDA
  native libraries report source hash
  `daadbc54c6578b6580697a0c520f941d5d3a815dcf6c09352841790d37798a5b`,
  revision `5b4c02db424b` plus dirty worktree. The match and Python fixture
  SHA-256 values were respectively
  `1cd28bdeae7c9dc4b1d7f8e29974bdaa56d94b6e84d5f6373893ae4a4227a94c` and
  `c639d5280f195391a5b735d1df45e2f3e170d4a0cd964f65bfe83f1c8f4396ce`.
  NVCC reported `#20013-D` host/device constexpr warnings.

  Nsight Systems 2025.1.3.140 profiled the new Python test on that T4 with
  `--trace=cuda,nvtx --sample=none`; an NVTX `bridge_step` range wrapped each
  `Environment.step`. An exported SQLite trace showed five ranges, each with
  one 112-byte host-to-device copy and device-to-host copies of 96, 112, 256,
  256 and 1504 bytes. The two-world 7 x 4 x 7 fixture has a 196-byte shared
  base and two 196-byte overlays. No copy of those full-grid sizes appeared
  inside the marked steps; source inspection identifies control/state/event
  transfers and no per-step terrain download. Initialization, resets and
  observation calls were outside the ranges. The temporary trace is not a
  durable artifact, and this bounded observation does not prove behavior for
  all arena sizes, resident Bridge or real-game conformance.
- **P1-02-04 completion, 2026-11-19:** The actual CPU/CUDA
  [training API fixture](../simulator/tests/training_checks.cpp) now compares
  two worlds through same-tick placement and player movement or melee: only
  the edited world clips movement and blocks a fully charged sword hit. It also
  fires actual bows in both worlds, verifies the placed block stops an arrow,
  breaks that block through the match tick and confirms a later arrow hits.
  The earlier shared and device-observation fixtures above cover targeting,
  independently expected rays, selected resets, nonaligned CUDA batches and
  host/device observation agreement. The recorded T4 trace and source inspection
  cover the host-staged per-tick terrain transfer criterion. From the project root:

  ```sh
  cmake -S simulator -B simulator/build-bridge-validation-20261119-cpu -DBUILD_TESTING=ON
  cmake --build simulator/build-bridge-validation-20261119-cpu -j 4
  ctest --test-dir simulator/build-bridge-validation-20261119-cpu --output-on-failure
  cmake -S simulator -B simulator/build-bridge-validation-20261119-t4 -DMCSIM_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DBUILD_TESTING=ON
  cmake --build simulator/build-bridge-validation-20261119-t4 --target minecraft_sim_training_checks -j 4
  ctest --test-dir simulator/build-bridge-validation-20261119-t4 --output-on-failure -R '^(shared_core_cuda_checks|training_api_cuda_checks|projectile_cuda_checks|observation_cuda_checks)$'
  MCSIM_TEST_CPU_LIBRARY="$PWD/simulator/build-bridge-validation-20261119-cpu/minecraft_sim_training.so" MCSIM_TEST_CUDA_LIBRARY="$PWD/simulator/build-bridge-validation-20261119-t4/minecraft_sim_training.so" .venv-rosie/bin/python -B -m unittest -v simulator.training.test_cuda_observation simulator.training.test_research.ResidentOpponentChecks
  ```

  CPU CTest passed 6/6, T4 CTest passed 4/4, and the CUDA/Python duel and
  observation suite passed 30 of 31 tests; the two-device check was skipped.
  T4 job `302283` used `dh-node8`, Tesla T4 (15360 MiB), driver 610.43.02,
  CUDA 12.9.86, PyTorch 2.5.1+cu124 and GNU 13.3.0. Both native libraries
  report source hash `daadbc54c6578b6580697a0c520f941d5d3a815dcf6c09352841790d37798a5b`
  and revision `42968419ef63` with dirty test/documentation worktree. The
  final training fixture SHA-256 is
  `db2e39cb451bdb4531f8f67b26189fb7b3a523956e4f18d6a883019528dc7380`;
  the other fixture hashes and NVCC warnings are recorded above. This does not
  complete the separate measured-memory gate, resident Bridge, H100, authored
  setup, replay/live or real-game conformance work.
- [x] P1-02-05: Restore only selected overlays on full match reset; preserve
  placed blocks across goals by default and test the explicit alternate rule.
  Evidence: the CPU/T4 validation below covers both policies and selective reset.
- [x] P1-02-06: Cover protected-block rejection, team identity, boundaries,
  simultaneous edits, nonaligned batch sizes, world isolation and selective
  reset with focused CPU fixtures and matching CUDA sequences. Evidence: the
  CPU/T4 validation below runs shared sequences and a 137-world CUDA fixture.
- **P1-02-05/06 evidence, 2026-11-19:** [Shared terrain fixtures](../simulator/tests/terrain_checks.hpp),
  [match fixtures](../simulator/tests/bridge_match_checks.hpp),
  [CUDA sequences](../simulator/tests/observation_cuda_checks.cu) and
  [training API fixtures](../simulator/tests/training_checks.cpp) cover ordered
  edits and resets, per-world isolation, and an actual placement and nonwinning
  goal on the same tick with both persistence settings. The training API cases
  check observations after goal and selected full resets. From the project root:

  ```sh
  cmake -S simulator -B simulator/build-bridge-validation-20261119-cpu -DBUILD_TESTING=ON
  cmake --build simulator/build-bridge-validation-20261119-cpu -j 4
  ctest --test-dir simulator/build-bridge-validation-20261119-cpu --output-on-failure
  cmake -S simulator -B simulator/build-bridge-validation-20261119-t4 -DMCSIM_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DBUILD_TESTING=ON
  cmake --build simulator/build-bridge-validation-20261119-t4 -j 4
  cmake --build simulator/build-bridge-validation-20261119-t4 --target minecraft_sim_training_checks -j 4
  ctest --test-dir simulator/build-bridge-validation-20261119-t4 --output-on-failure -R '^(shared_core_cuda_checks|training_api_cuda_checks|projectile_cuda_checks|observation_cuda_checks)$'
  MCSIM_TEST_CPU_LIBRARY="$PWD/simulator/build-bridge-validation-20261119-cpu/minecraft_sim_training.so" MCSIM_TEST_CUDA_LIBRARY="$PWD/simulator/build-bridge-validation-20261119-t4/minecraft_sim_training.so" .venv-rosie/bin/python -B -m unittest -v simulator.training.test_cuda_observation simulator.training.test_research.ResidentOpponentChecks
  ```

  CPU CTest passed 6/6; T4 CTest passed 4/4. The Python CUDA/duel regression
  ran 30 tests: 29 passed and the two-device check was skipped. T4 Slurm job
  `302274` used `dh-node8`, Tesla T4 (15360 MiB), driver 610.43.02, CUDA
  toolkit 12.9.86, and PyTorch 2.5.1+cu124 (CUDA 12.4). Both native libraries
  report source hash `daadbc54c6578b6580697a0c520f941d5d3a815dcf6c09352841790d37798a5b`,
  revision `5b4c02db424b` plus dirty worktree, and GNU 13.3.0; CPU used Linux
  x86_64 on an Intel Xeon Gold 6240. Final fixture SHA-256 values, in link
  order above: `a580a3503cb6144f904155ef8270077bd4e364a91bf35c59c52c7335cac1ad75`,
  `0e46d055bca3d715f86fd3613ac178baf7a8a9df5c5664eea67259646db45597`,
  `4f5f04e81b3993406d58dcd0d2f9b2fcbd46acb6435ee63d5487e9b0ddb18489`,
  `de393de4209e523ebd5b9f2373e5dfb29125c034dfae08f575a2e728547194dc`.
  NVCC reported `#20013-D` host/device constexpr warnings. Results are fixture
  evidence for the experimental host-staged path, not resident Bridge, measured
  memory, H100, live integration or real-game conformance.
- [x] P1-02-07: **Gate:** CPU and CUDA observe identical intended terrain and
  collision/ray results after edits and resets; static duel regression passes;
  recorded terrain preflight estimates respect the configured per-preflight cap.
  **Decision:** preserve the existing per-preflight contract; it does not cap
  aggregate live allocations or physical host/GPU residency.
- **P1-02-07 measurement, 2026-11-19:** For a 32 x 48 x 96 base and 16
  independent overlays, allocation tracing on the integrated source measured
  147456 bytes for the host base and 2359296 for host overlays. The CPU
  preflight accepted a 2506752-byte cap and rejected 2506751. On T4, the
  7520256-byte CUDA preflight accepted and 7520255 was rejected. Its source,
  staging and new device payloads each accounted for 2506752 bytes, but the
  old 147456-byte device base remained allocated during replacement: peak
  attributed requested terrain payload was 7667712 bytes. Host usable sizes
  included allocator overhead; GPU allocation extents were queried via the
  CUDA driver, not measured physical residency. The cap documented in the
  [native guide](../simulator/AGENTS.md#json-environments) is per preflight,
  not an aggregate process/device limit. The observed aggregate peak is not
  within the cap. The temporary probe and traces are not durable repository artifacts.
- **P1-02-07 decision and validation, 2026-11-19:** Repeated the representative
  allocation trace on native source hash
  `a52a1324d3dee7e2a197c17ebd7a3541d4ae549a5aebd5242d1f3008bfec945d`
  at baseline `ecef3904e801` plus worktree changes. CPU accepted 2506752 bytes
  and rejected 2506751; T4 accepted 7520256 and rejected 7520255, with the same
  7667712-byte aggregate requested-payload overlap described above. Exact and
  insufficient preflights and rollback are durably exercised by
  [training checks](../simulator/tests/training_checks.cpp); CUDA also tests
  distinct `(base pointer, cell count)` sharing in
  [observation checks](../simulator/tests/observation_cuda_checks.cu).
  `ctest --test-dir simulator/build-bridge-host-20261119-cpu --output-on-failure`
  passed 6/6; the four CUDA suites passed 4/4 in
  `simulator/build-bridge-host-20261119-t4`. The focused
  `minecraft_sim_training_checks simulator/tests/reset_environment.json --terrain-budget`
  check also ran with `--cuda` on T4. Slurm job 302316 used `dh-node8`, Tesla
  T4, driver 610.43.02, CUDA 12.9.86 and GNU 13.3.0. These measurements do not
  include physical GPU backing, allocator overhead, other batches or models.

## 3. Match Lifecycle

**Prerequisites:** P1-01 semantics. A static-arena vertical slice can proceed
before building; the complete milestone also requires P1-02 persistence.

- [ ] P1-03-00: Complete this milestone only after P1-03-01 through P1-03-07.
- [ ] P1-03-01: Add per-world team scores, match clock, goal/cage phase and
  per-player respawn/kit state in shared CPU/CUDA-compatible ownership.
  **In progress:** GitHub Copilot; static CPU/CUDA host batches own match state;
  kit counts and refill state are written and host fixtures pass, but resident
  integration and its validation remain open.
- [x] P1-03-02: Implement cages and their opening, goal detection and scoring,
  configurable first-to-N result, time cap and specified simultaneous/tie
  behavior. Emit explicit score and match outcome events. The fixed authored
  host-staged subset is validated below; resident support remains tracked in P1-06.
- [ ] P1-03-03: Implement void/combat death and player-only respawn/refill
  without ending the match; clear the affected player's relevant state and
  preserve the other player's state according to the specification.
  **In progress:** GitHub Copilot; internal player/projectile reset helpers are
  wired to the host-batch tick with kit restoration and host fixtures pass;
  resident integration and its validation remain open.
- [ ] P1-03-04: Implement goal/round reset of both players and their kits while
  preserving scores, match clock and terrain as configured; implement a
  separate full match reset that restores the selected world's full state.
  **In progress:** GitHub Copilot; internal player, projectile, RNG and terrain
  reset helpers and kit restoration are wired to host batches; resident integration remains open.
- **Partially validated slice, 2026-11-19:** Source baseline `5b4c02d` plus worktree
  changes on Linux x86_64. The internal [match tick](../simulator/include/minecraft_sim/bridge_match.hpp)
  follows the [Bridge mode proposal](bridge_mode_v1.md). Shared
  [terrain](../simulator/tests/terrain_checks.hpp) and
  [match](../simulator/tests/bridge_match_checks.hpp) fixtures cover edits,
  reset policies, score/timeout ordering, lifecycle isolation and a short match.
  The P1-02-05/06 evidence above records focused CPU/T4 runs; these do not
  establish complete lifecycle coverage, in-game conformance or playable Bridge.
- **Partially validated adapter slice, 2026-11-19:** The experimental direct
  [training batch](../simulator/apps/training.cpp) and
  [Python environment](../simulator/training/environment.py) expose explicit
  static-arena Bridge configuration, score/outcome rewards, match events and
  CPU/CUDA host-staged ticking. Focused native and Python fixtures passed; see
  P1-02-05/06 above. Mutable host-staged terrain was added later; the
  fixed authored and research slice is recorded below; resident,
  replay and live controllers remain unsupported.
- [x] P1-03-05: Connect match outcomes and minimal score/win/loss task events
  to the training API and metrics. Preserve final observations and the specified
  termination/truncation semantics; a death or nonwinning goal is not a match
  terminal. Do not silently substitute duel rewards or endings.
- **P1-03-02/05 authored training evidence, 2026-11-19:** The [static-lane
  plan](../simulator/setups/curricula/bridge/static-lane.json) uses one fixed
  arena, fixed spawns/loadouts, scalar observations and a stationary opponent.
  Validation rejects unsupported sampling, task overrides, evaluation, replay,
  resident collection and live serving. Research collector fixtures exercise
  scoring and outcomes from the learner's perspective, task rewards, selective
  resets, final observations, terminal versus truncated bootstrapping, and
  checkpoint reset-counter continuity. The driven scoring fixture records a
  training return of 9; per-rollout goal and match counters are learner-relative.

  Fresh CPU and T4 native builds reported source hash
  `a52a1324d3dee7e2a197c17ebd7a3541d4ae549a5aebd5242d1f3008bfec945d`
  at baseline `ecef3904e801` plus worktree changes. CPU native CTest passed
  6/6; the four relevant T4 native tests passed 4/4. The affected CPU Python
  suites ran 210 tests with four environment-dependent skips before the final
  metric additions; 36 focused collector/dashboard tests and two metric-selector
  tests passed afterward. T4 Python observation/resident-opponent checks ran
  34 tests with one two-device skip; 50 mocked Rosie submission tests passed.
  The authored plan validator passed. Fixture SHA-256 values are
  `115b4936da83953e90155b7b7b564c53d15157de66467d060df60768609a3b94`
  for [research checks](../simulator/training/test_research.py) and
  `eb5c9c89a6cee1891357ba2348f15f9ae47bf13964692e55d5e8bff61595bb28`
  for [experiment checks](../simulator/training/test_experiments.py).

  CPU and host-staged T4 PPO each ran two 64-step rollouts in two worlds,
  saving a checkpoint after the first update and resuming for the second.
  Both reached 256 samples with finite losses and explicit Bridge event metrics;
  no matches finished or goals were scored, so these runs demonstrate plumbing,
  not learning. The T4 runs used `dh-node8` (Slurm jobs 302316 and 302317),
  driver 610.43.02, CUDA 12.9.86 and PyTorch 2.5.1+cu124. No H100 or in-game
  conformance is claimed. From the project root, a fresh CPU smoke and resume
  can be reproduced with:

  ```sh
  OUTPUT=$(mktemp -d /tmp/cambium-bridge-smoke-XXXXXXXX)
  .venv-rosie/bin/python -B -m simulator.training.train --plan simulator/setups/curricula/bridge/static-lane.json --device cpu --rollout-backend host --envs 2 --updates 1 --rollout-steps 64 --max-ticks 100 --threads 1 --replay-every 0 --library simulator/build-bridge-host-20261119-cpu/minecraft_sim_training.so --output "$OUTPUT"
  .venv-rosie/bin/python -B -m simulator.training.train --resume "$OUTPUT/latest.pt" --device cpu --rollout-backend host --updates 1 --threads 1 --replay-every 0 --library simulator/build-bridge-host-20261119-cpu/minecraft_sim_training.so
  ```
- [x] P1-03-06: Add independently specified cases for both sides, cage timing,
  player death, void, goal, simultaneous events, time-cap ties, selective reset
  and persistence, including shared CPU/CUDA state and result comparisons.
- [x] P1-03-07: **Gate:** a complete basic Bridge match runs against a fixed
  opponent on CPU and CUDA with expected scores, resets and match outcome;
  existing duel termination and checkpoint behavior remain unchanged.
- **P1-03-06/07 evidence, 2026-11-19:** The experimental host-staged batch
  completed a three-goal match against a stationary fixed opponent on CPU and
  T4, with independently specified score, cage, reward, reset and terminal
  observation expectations. A separate two-goal fixture drives both scoring
  perspectives across the arena and compares CPU/CUDA state, kits,
  observations, rewards and endings. Mirrored combat and void deaths,
  simultaneous scores, timeout ties, truncation and selective resets are also
  compared. Shared fixtures independently cover projectile, terrain, RNG and
  player preservation. Respawning either player now clears both observer
  shield-disable caches without resetting the surviving player's shield state;
  a CUDA host/device observation fixture covers both perspectives and world
  isolation. These are synthetic static-arena matches, not authored setups,
  resident Bridge, full item validation or in-game conformance.

  On Linux x86_64, GNU 13.3.0 and Tesla T4 (Slurm job 302294 on `dh-node8`,
  driver 610.43.02, CUDA 12.9.86, PyTorch 2.5.1+cu124), CPU and CUDA native
  libraries both reported source hash
  `ad96884af86b0dea6ca00c3f76fd2400676f466d878f33105070e863d5dee84c`
  at baseline `b85153061ef5` plus worktree changes. CPU `ctest --test-dir
  simulator/build-bridge-integrated-20261119-cpu --output-on-failure` passed
  6/6. T4 `ctest --test-dir simulator/build-bridge-integrated-20261119-t4
  --output-on-failure -R '^(shared_core_cuda_checks|training_api_cuda_checks|projectile_cuda_checks|observation_cuda_checks)$'`
  passed 4/4. The Python `simulator.training.test_experiments` and
  `simulator.training.test_training.TrainingChecks` suites ran 85 tests with
  one CUDA-only skip, selecting the CPU library for default environments;
  this includes legacy duel termination and checkpoint/resume coverage. With
  `MCSIM_TEST_CPU_LIBRARY` and `MCSIM_TEST_CUDA_LIBRARY` selecting those two
  builds, `.venv-rosie/bin/python -B -m unittest -v
  simulator.training.test_cuda_observation
  simulator.training.test_research.ResidentOpponentChecks` ran 34 tests with
  one two-device skip. NVCC emitted existing host/device constexpr warnings.
  The fixture SHA-256 values are `ab1752aef1293deca5d64c293ff23756daeccce5ab238329cb6ecad8d8f9a837`
  for [shared match checks](../simulator/tests/bridge_match_checks.hpp),
  `e6067bd8edc09663d52503a8b46493cc51270f41db676072949b654facbe298f` for
  [training checks](../simulator/tests/training_checks.cpp), and
  `f72c4edb8d8c341f282f8e2c094243cf6be0c4eb55dc803bc84ed926a1916b5f` for
  [CUDA Python checks](../simulator/training/test_cuda_observation.py).

  The CPU Python suite was run from the project root with:

  ```sh
  MCSIM_TEST_CPU_LIBRARY="$PWD/simulator/build-bridge-integrated-20261119-cpu/minecraft_sim_training.so" .venv-rosie/bin/python -B -c 'import os, unittest; import simulator.training.environment as env; original = env.load_library; env.load_library = lambda library=None, device="cpu": original(library or os.environ["MCSIM_TEST_CPU_LIBRARY"], device); unittest.main(module=None, argv=["unittest", "-v", "simulator.training.test_experiments", "simulator.training.test_training.TrainingChecks"])'
  ```

## 4. Building And Items

**Prerequisites:** P1-01 action/tick ordering and P1-02 composite terrain. Use
existing movement, combat and ordinary-bow behavior where applicable.

- [ ] P1-04-00: Complete this milestone only after P1-04-01 through P1-04-08.
- [ ] P1-04-01: Implement 26.3-derived eye-based full-cube face targeting,
  range, occlusion and boundary/tie behavior, with explicit entity/block action
  arbitration and independently calculated expected hits. **In progress:**
  GitHub Copilot; shared targeting and expected-hit fixtures are written;
  validation and unsupported item-specific reach variants remain open.
- [ ] P1-04-02: Implement held-use placement and repeat timing, selected-item
  and block counts, valid faces/support, occupied space, player collision,
  height/no-build restrictions and consumption only on success. **In progress:**
  GitHub Copilot; an opt-in full-cube subset is wired to host batches and
  fixtures; authored setups, live adapters and validation remain open.
- [ ] P1-04-03: Implement held-attack pickaxe breaking with the supported
  hardness/tool/speed subset, progress and completion; cover release,
  cancellation, target/item switches and protected-block rejection.
  **In progress:** GitHub Copilot; configurable full-cube mining is wired to
  host batches; exact material/tool coverage and validation remain open.
- [ ] P1-04-04: Define hotbar selection and per-player counts/refills for sword,
  bow, pickaxe, blocks and apples. Resolve use/attack/slot precedence without
  regressing existing melee, shields and projectiles. **In progress:**
  GitHub Copilot; experimental kits track counts and slots; building-enabled
  batches implement a limited block/pickaxe subset, and golden-apple use is
  experimental; pickaxe entity combat and broader validation remain open.
- [ ] P1-04-05: Implement supported golden-apple eating time, cancellation,
  absorption and regeneration; separately specify and implement the custom
  single-arrow refill timer. Do not claim general effects, inventory,
  enchantment, durability, fluid or special-block simulation. **In progress:**
  GitHub Copilot; the custom arrow refill timer and an experimental golden-apple
  eating, absorption and regeneration subset are implemented. Authored setup,
  real-game conformance and broader validation remain open.
- **P1-03/P1-04 integration, 2026-11-19:** Shared
  [lifecycle](../simulator/tests/bridge_match_checks.hpp) and
  [golden-apple](../simulator/tests/golden_apple_checks.hpp) fixtures cover
  mirrored respawns, 32-tick eating, cancellation, effects, damage and resets.
  The [training API fixture](../simulator/tests/training_checks.cpp) also checks
  consumption, cancellation, regeneration, world isolation and selected reset.
  CPU CTest passed 6/6; T4 CTest passed 4/4. The CPU training class ran 49
  checks, with its CUDA-only case skipped; 36 setup checks passed. CUDA/Python
  observation and resident-duel suites ran 31 checks, with one two-device
  check skipped. From the project root:

  ```sh
  cmake -S simulator -B simulator/build-bridge-integrated-20261119-cpu -DBUILD_TESTING=ON -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON="$PWD/simulator/build-bridge-validation-20261119-cpu/_deps/nlohmann_json-src"
  cmake --build simulator/build-bridge-integrated-20261119-cpu -j 4
  ctest --test-dir simulator/build-bridge-integrated-20261119-cpu --output-on-failure
  .venv-rosie/bin/python -B -m unittest -v simulator.training.test_experiments
  cmake -S simulator -B simulator/build-bridge-integrated-20261119-t4 -DMCSIM_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DBUILD_TESTING=ON -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON="$PWD/simulator/build-bridge-validation-20261119-t4/_deps/nlohmann_json-src"
  cmake --build simulator/build-bridge-integrated-20261119-t4 -j 4
  ctest --test-dir simulator/build-bridge-integrated-20261119-t4 --output-on-failure -R '^(shared_core_cuda_checks|training_api_cuda_checks|projectile_cuda_checks|observation_cuda_checks)$'
  MCSIM_TEST_CPU_LIBRARY="$PWD/simulator/build-bridge-integrated-20261119-cpu/minecraft_sim_training.so" MCSIM_TEST_CUDA_LIBRARY="$PWD/simulator/build-bridge-integrated-20261119-t4/minecraft_sim_training.so" .venv-rosie/bin/python -B -m unittest -v simulator.training.test_cuda_observation simulator.training.test_research.ResidentOpponentChecks
  ```

  T4 job `302292` ran on `dh-node8`, Tesla T4 (15360 MiB), driver 610.43.02,
  CUDA 12.9.86, GNU 13.3.0 and PyTorch 2.5.1+cu124. Both native libraries
  report source hash `c2490332db3b52e0584216d83790e1444c471b46b6ea9a47f593976dfb48e8a9`
  and revision `b85153061ef5` plus worktree changes. The new shared apple,
  lifecycle, training API and CUDA observation fixture SHA-256 values are
  `2755c19609e4683fb7726553b6d74f2038aa37e6bd6384c261f6b62a5bb59022`,
  `57a5d463b1c8cb23de5d7c4ac602d2d4821ae4e732a17921b85436d0e60deb54`,
  `b68659543eba2fc0f04784505bf62237b6bcfc318f38c16df95e1149faf72b4d`, and
  `78226e7dcc6650959e53c3150b7113a58280eef62b3685cad686ce493db3ec4e`.
  NVCC reported `#20013-D` host/device constexpr warnings. The CPU training
  checks explicitly selected the integrated native library; fresh raw capture
  creation also passed, but the changed player layout requires old captures to
  be regenerated. Source basis: supplied decompiled Minecraft 26.3 tree;
  decompiler identity remains unknown. These results are not live-game or
  Hypixel conformance, resident Bridge or a complete lifecycle gate.
- **Partially validated kit slice, 2026-11-19:** The experimental host batch tracks
  per-player kit counts, selected slots and the custom arrow refill timer on
  CPU/CUDA; focused shared and adapter fixtures passed. Item actions, real-game
  conformance, authored setups and resident collection remain open.
- **Partially validated building slice, 2026-11-19:** The shared
  [targeting](../simulator/include/minecraft_sim/geometry.hpp),
  [building](../simulator/include/minecraft_sim/bridge_building.hpp) and
  [host batch](../simulator/apps/training.cpp) contain an opt-in full-cube
  subset with ordered edits and compact CUDA-to-host deltas. Focused CPU/T4
  fixtures passed; see P1-02-05/06 above. Source basis: supplied decompiled Minecraft
  26.3 tree, protocol 777, world version 5023, Java 25; decompiler identity and
  source content hash are not recorded. No in-game capture, performance or
  vanilla/Hypixel conformance is claimed. Tool effects, fluids, special shapes,
  pickaxe entity combat, complete eating validation, replay and resident
  collection remain open.
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

**Developer experience requirement:** A teammate experimenting with an
architecture should be able to implement a Python model or factory, register it
through a documented trusted extension point, select it and its options in
setup JSON, validate the setup, and train and reload it without changing native
physics, CUDA kernels, PPO internals, the training loop or the mod merely to
switch models. This applies when the model uses an already-supported
observation/action contract; genuinely new observations or actions remain
separate simulator and contract work. Provide clear input/output shape,
device and checkpoint requirements so this workflow is usable without an agent
or repository-wide programming changes.

- [ ] P1-05-00: Complete this milestone only after P1-05-01 through P1-05-08.
- [ ] P1-05-01: Specify optional local voxel and lane observations: axes, origin,
  rotation, categories, ownership, outside bounds, team perspective, sampling
  phase, visibility and numeric types. A 9 x 5 x 9 local window and a
  3 x 4 x 7 x 16 lane summary are candidates, not fixed requirements. Declare
  lane data global/omniscient if it includes information outside local view.
  **In progress:** GitHub Copilot; a detailed
  [provisional specification](bridge_mode_v1.md#proposed-structured-terrain-observations)
  needs agreement on dimensions, yaw quantization and omniscient visibility.
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
  architecture descriptor selectable with model options in setup JSON. Define
  the Python model's observation inputs, named action logits and value output,
  shapes, device behavior and validation. Preserve the existing MLP's legacy
  state-dict keys; implement a test-only alternate model to prove the extension
  interface. External Python factories execute trusted code and must be
  explicitly trusted.
- [ ] P1-05-06: Route training, PPO, frozen/checkpoint opponents, evaluation,
  serving and inference packaging through the same model contract. Reject
  incompatible model/policy descriptors rather than silently reinitializing or
  relabeling checkpoints; preserve supported old scalar MLP checkpoints.
- [ ] P1-05-07: Add CPU/CUDA/Java fixtures for both teams, empty/boundary/placed
  cells, negative coordinates and yaw 0/45/90; test inputs, semantic actions,
  checkpoint identity, unsupported descriptors and memory bounds.
- [ ] P1-05-08: **Gate:** MLP and test model train, save, reload, act as an
  opponent, evaluate and serve through the actual adapters; grid fixtures and
  legacy compatibility pass. Document and exercise a teammate workflow that
  adds a Python model, selects it and its options in JSON, validates the setup,
  trains on CPU and a supported CUDA backend, and reloads the checkpoint without
  editing simulator or PPO internals. A production CNN remains future work.

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
