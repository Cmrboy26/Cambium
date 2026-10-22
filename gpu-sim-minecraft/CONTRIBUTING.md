# GPU Simulator Contribution Playbook

Use this guide to choose an implementation path and its verification gates.
Follow the [root rules](AGENTS.md) and the applicable component guide first.
Component guides and specifications own detailed contracts; this file connects
the workflows rather than duplicating their full references.

## Start With The Smallest Supported Change

Existing observations, action choices, loadouts, terrain, rewards and stages
usually need [setup JSON](simulator/setups/README.md), not new native mechanics
or another policy preset. Identify the missing behavior before changing code.
For guided configuration work, use the workspace's
[setup-authoring skill](../.github/skills/gpu-sim-setup/SKILL.md).

For an implementation change, write down the observable result, the owning
layer, a minimal reproducer and the cheapest check that could disprove the
proposed implementation. Keep existing user edits and experiment definitions.
Do not start long training runs to establish basic mechanics correctness.

## Ownership Lookup

Paths and command examples in this guide are relative to `gpu-sim-minecraft`.
Run Python there, not from the parent Cambium directory: both contain `backend`.

| Task | Owning Files / Symbols | First Focused Check |
| --- | --- | --- |
| Port player/tick mechanics | [core.hpp](simulator/include/minecraft_sim/core.hpp): `PlayerState`, `PlayerView`, `WorldView::processTick`; [native guide](simulator/AGENTS.md#numeric-and-tick-contracts) | Source-derived case in [core_checks.cpp](simulator/tests/core_checks.cpp); CPU/CUDA comparison when applicable |
| Add an observation | [catalogue](contracts/policy_catalogue.json), [generator](scripts/generate-contract.py), [observation.hpp](simulator/apps/observation.hpp), [training.cpp](simulator/apps/training.cpp): `TrainingBatch::observe` | [shared fixtures](contracts/policy_fixtures.json), native `shared_core_checks`, [Python encoding checks](simulator/training/test_experiments.py) |
| Sample actual live state | [LivePolicy.java](cambiummod/src/main/java/net/cambium/LivePolicy.java): `encode`; [ObservationEncoder.java](cambiummod/src/main/java/net/cambium/ObservationEncoder.java) | [ReplayAnalysisChecks.java](cambiummod/src/test/java/net/cambium/ReplayAnalysisChecks.java), then an in-game check |
| Add an action or encoding rule | [configuration.hpp](simulator/apps/configuration.hpp): `PolicyLayout`; [contract.py](simulator/training/contract.py): `validate_contract`; [SimulationContract.java](cambiummod/src/main/java/net/cambium/SimulationContract.java) | Shared projection/decoding/rejection fixtures and reordered layouts |
| Add a scripted controller | [controllers.py](simulator/training/controllers.py): `register_controller`, `create_controller`; [serve.py](simulator/training/serve.py): live duel admission | [test_controllers.py](simulator/training/test_controllers.py), then live admission/inference checks if exposed |
| Extend state, rules or serialization | [native guide](simulator/AGENTS.md), [scenario contract](docs/simulation_contract.md), [environment.py](simulator/training/environment.py), [mod guide](cambiummod/AGENTS.md) | Load/reset/step round trips; native and live replay consumers |
| Change rewards or episode endings | [training.cpp](simulator/apps/training.cpp), [environment.py](simulator/training/environment.py), [reward recipe](simulator/training/AGENTS.md#reward-changes) | One-event totals, terminal versus timeout, selective reset and resume tests |
| Author a setup | [setup guide](simulator/setups/README.md), [compiler.py](simulator/training/compiler.py): `resolve_experiment`; [plans.py](simulator/training/plans.py): `load_plan` | Validate only the new plan, inspect resolved conditions, then a bounded smoke run |

## Port A Minecraft Mechanic

1. Find the controlling methods and callers in the supplied Minecraft 26.3
   source. Record class/method names, relevant constants, eligibility checks,
   client/server distinction and the supported subset. Trace who updates the
   state; a similarly named getter or isolated formula is not enough.
2. Establish tick phase and ordering: inputs, look, attacks, movement, timers
   and observation sampling. Preserve Java float/double conversions, comparison
   boundaries, lookup-table behavior and ordered RNG draws where relevant.
3. Choose the owning native abstraction. CPU and CUDA should use the same
   device-compatible core logic, not separate approximations. Policies and
   scripted controllers choose controls; they do not implement physics.
4. Add independently calculated or captured regression expectations. Cover a
   normal case, a boundary and a cancellation/interaction case. Do not generate
   expected results with the implementation being tested.
5. Validate the narrow CPU case first, then relevant interactions, CUDA parity
   and game captures. Record the exact supported conditions and remaining gaps.

Reference: [physics specification](docs/minecraft_simulation_specs.md) and
[native numeric/tick contracts](simulator/AGENTS.md#numeric-and-tick-contracts).
Passing CPU/CUDA parity proves agreement between backends, not vanilla fidelity.

## Add Observations Or Actions

1. Decide whether the primitive already exists. Configuration-only selection,
   ordering, scalar clipping/scaling and allowed action values need JSON only.
2. Define units, bounds, perspective, sampling phase and availability in both
   simulation and Minecraft. Expose actual state rather than requested controls;
   do not leak hidden opponent information. Grounded and sprinting state are
   independent flags, not deductions from jump/sprint requests or velocity.
3. Append catalogue entries; never reorder existing indices or legacy prefixes.
   Run `python -B scripts/generate-contract.py`. Do not hand-edit generated
   C++/Java tables. Generation supplies names and identity, not implementation.
4. Implement native extraction/decoding and live extraction/decoding. New
   actions also need actual mechanics, neutral/held/one-shot semantics and the
   correct application phase. Replay controls store semantic values, not head
   indices. Expand capabilities and reject unsupported configurations explicitly.
5. For categorical/binary observations, update eligibility and validation in
   Python `categorical_values`, C++ `PolicyLayout` and Java `SimulationContract`.
   Keep scalar/one-hot projection, category order, clipping and expanded widths
   consistent. Consumers must count expanded columns, not descriptor entries.
6. Extend shared fixtures with both perspectives, boundary/unknown categories,
   reordered/scaled projection and invalid descriptors as applicable. Also test
   extraction through the native adapter; pure projection fixtures do not prove
   that the state is actually sampled.
7. Select the new primitives in the intended authored policy only. A changed
   input layout or action contract requires compatible new weights or an explicit
   migration, not an unchecked resume. Rebuild each used native backend and
   complete the [live-mod completion gate](#live-mod-completion-gate). Updated
   source or passing simulator tests alone do not establish mod compatibility.

See [extension recipes](simulator/training/AGENTS.md#extension-recipes) and the
[encoding contract](docs/simulation_contract.md#policy-layout).

## Live Mod Completion Gate

An observation/action contribution is incomplete until all relevant live paths
are implemented and checked. Do not leave Java support as an implicit follow-up.

1. Regenerate `PolicyCatalogue.java` from the shared catalogue. Implement real
   entity extraction/application in `LivePolicy` and the appropriate live control
   adapter, pure encoding in `ObservationEncoder`, and descriptor validation,
   projection/decoding and binary-category eligibility in `SimulationContract`.
   Adding generated names does not populate observations or apply actions.
2. Load a representative checkpoint using the updated Python inference service,
   including the new policy's full input/action layout and two-player metadata.
   Test the new features, both actor perspectives and legacy compatibility in
   Java/shared fixtures, then run the explicit Python/Java HTTP bridge. A bridge
   that exercises only an old policy does not cover the new contract.
3. Run mod checks and build the installable artifact. From project root on
   Windows, use `./scripts/build-mod.ps1 -Task check`,
   `./scripts/build-mod.ps1 -Task replayAnalysisChecks` and
   `./scripts/build-mod.ps1`. On Linux with JDK 25, run
   `bash gradlew check replayAnalysisChecks build --no-daemon --console=plain`
   from `cambiummod`. Check/compile tasks alone are not an installed mod update.
4. Deploy only with an explicitly selected Minecraft game directory and
   installation authorization. Close Minecraft; install the newly built
   `cambiummod/build/libs/cambiumfabric-1.0.0.jar`, not the sources JAR. Use the
   [installation helper](scripts/build-mod.ps1) with `-Install -MinecraftDirectory`
   on Windows. Check for differently named old copies; do not delete them or
   overwrite an arbitrary instance without approval.
5. Update the inference-side Python checkout and shared catalogue too; restart
   the inference process after code/catalogue changes. Restart Minecraft after
   replacing its JAR. A current JAR with an older local Python service can still
   fail, independently of the remote training checkout.
6. Compare the service `/info` catalogue hash with the hash logged by Minecraft;
   verify the loaded JAR path/build identity and `/simduel status`. Exercise an
   actual round using the new checkpoint and check feature/action behavior.
   Never bypass catalogue, descriptor or shape checks to make a model load.
7. Report source implementation, fixture/bridge results, built artifact path,
   installation/restart verification and in-game evidence separately. If the
   local game instance is unavailable, say "implemented/built, not installed or
   verified in-game" and provide deployment steps. Do not claim a user's mod was
   updated merely because the repository or a build output changed.

### Diagnose The Rejecting Layer

- `Checkpoint is not a compatible standalone two-player PPO policy` is raised
  by Python [Policy.__init__](simulator/training/serve.py), not the Java
  catalogue guard. Inspect the inference traceback and chained cause: an older
  Python catalogue can reject new feature names, or checkpoint format/schema,
  player-count or learner metadata may be incompatible. The message alone does
  not establish that Java observation extraction is missing.
- `Policy catalogue differs from mod` is raised by Java
  [LivePolicy.info](cambiummod/src/main/java/net/cambium/LivePolicy.java). Compare
  the running service and installed JAR, not just files in the source checkout.
- For descriptor, tensor-shape or model-weight errors, compare the checkpoint's
  saved contract and dimensions with the service/model. Do not relabel old
  weights, replace their metadata or fabricate missing observations as a fix.

## Add A Scripted Controller

1. Start from the nearest existing controller. Define required observations,
   available semantic action choices, options, state and unsupported layouts.
   Validate prerequisites before acting; do not hard-code preset column indices.
2. Implement the [controller contract](simulator/training/AGENTS.md#controllers-and-constraints):
   factory arguments `contract`, `count`, `seed`, `options`; `act`, `reset`,
   `state_dict`, `load_state_dict`, `describe` and `provenance`. Decode configured
   scales/one-hot columns correctly; clipped information cannot be recovered.
3. Keep state and RNG per persistent actor slot. Sub-batch order must not change
   random sequences. Selective reset must not clear unrelated actors or rewind
   their generators. Save only checkpoint-safe state and test restoration.
4. Register the factory or use an explicitly trusted module factory. A new
   training registry entry is not automatically admitted to live duels: inspect
   `serve.py` and update builtin admission/tests deliberately if live use is needed.
5. Test missing features/choices, reordered and expanded layouts, options,
   action bounds, seeded repeatability, sub-batches, reset and state round trips.
   Controllers see their actor's observation row, not privileged simulator state.

## Extend State, Worlds Or Rules

Treat a new state field as a lifecycle change, not just a struct member. Check
initialization, episode sampling, full/selective reset, CPU/CUDA upload/download,
scenario parsing, replay recording/readback and live initialization. Distinguish
requested initial conditions from current runtime state and transient events.

For terrain/collision changes, also check geometry representation, raycasts,
spawn clearance, boundaries and the supported live block shapes. Native support
does not imply setup/compiler, policy, replay or live support. Explicitly expose
or reject the new condition at each boundary; do not silently substitute defaults.

Follow the [compatibility checklist](simulator/training/AGENTS.md#compatibility-checklist)
for catalogue identity, C ABI/buffer changes, replay/HTTP envelopes and checkpoint
semantics. Change the relevant identity and rejection checks, not every version.

## Change Rewards Or Episode Rules

Define event versus tick versus episode-budget units and termination versus
truncation first. Keep native evaluation, Python names/order, buffer counts,
metrics and displayed totals aligned. Authored task rewards are explicit; omitted
terms are zero, not historical defaults. Test isolated events, boundaries,
partial health/absorption, timeouts and selective reset before PPO behavior.

A weight override is configuration; a changed reward meaning needs the
[reward semantics procedure](simulator/training/AGENTS.md#reward-changes).
Preserve inference compatibility separately from training-resume compatibility.

## Verification And Handoff

Use the [validation matrix](simulator/training/AGENTS.md#validation-matrix).
PowerShell checks rebuild an existing configured tree; set `-BuildDirectory` to
your own build. Native, Python and Java toolchains are separate prerequisites.
On Linux, use the component guides' CMake/unittest/Gradle equivalents; invoke
`bash gradlew` when the wrapper is not executable and select JDK 25 explicitly.

- Start with the reproducing test, then the touched slice. Do not begin with a
  full training run or require unrelated optional tools for a pure unit check.
- Run `python -B scripts/generate-contract.py --check` for catalogue consumers.
- Run the explicit HTTP/replay bridge for inference-boundary changes:
  `python -B -m unittest simulator.training.test_training.InferenceChecks.test_java_bridge -v`,
  with `MCSIM_LIVE_JAVA_CHECK=1` and JDK 25 `JAVA_HOME` in the environment.
- Run CUDA checks on a capable device for shared physics/state changes. CPU-only
  results are not CUDA certification. Do not rebuild another contributor's or
  running job's build tree.
- Test actual game sampling/application for live changes. Headless Java fixtures
  and the HTTP bridge do not validate full Minecraft behavior.
- For hot-path changes, compare the same workload, device, world count and seed
  before/after. Report physics throughput separately from observation/reward,
  host/device transfer and end-to-end training timings. Profiling must not
  change RNG draws, state transitions or reset order.

A contribution is ready to hand off when it includes the source/contract basis,
minimal reproducer and independent expectations, required implementation paths,
compatibility decision, exact checks/results, and explicit untested gates.
Distinguish pre-existing failures from regressions; do not hide or fix unrelated
failures as part of the contribution. Update the owning reference when the
supported behavior changes, without copying exhaustive contracts into this guide.
