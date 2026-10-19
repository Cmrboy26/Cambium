# Environment Interface and Next Build Steps

Status: design and implementation priorities, updated 2026-10-17. The full
interfaces below are not implemented or validated; the bounded shared movement
slice and current verification limits are recorded in the
[simulator README](../simulator/README.md). This document does not replace the
[source-derived physics specification](minecraft_simulation_specs.md).

A first [host-staged PPO trainer](../simulator/training/README.md) now provides
batched CPU/CUDA selection, basic duel rewards and selective resets. CPU training
and resume are checked locally; the new CUDA training/reset path still needs a
device run. The more general interfaces and GPU-resident path below remain goals.

## Goal and Boundary

Build a GPU-batched replacement for selected Minecraft training environments,
with a shared C++/CUDA physics core, minimal CPU runner and real-game adapter.
CPU and GPU execute the same physics functions; do not maintain independent
physics implementations. CPU execution is a debug/replay path, not additional
ground truth. Minecraft-to-GPU fixtures remain the game-fidelity check, while
CPU/GPU comparisons diagnose execution, numeric and batching differences.
Keep Python/PPO above a model-independent game interface:

```text
Game state -> observation encoder -> policy -> action decoder -> player controls
    ^                                                               |
    +-------------------- environment simulation step <----------------+
```

The game backend owns physics, world state, control execution and tick order.
The encoder chooses model observations; the decoder converts policy outputs
into controls. Reward and episode rules belong in the task wrapper rather than
changing physical rules. Start/stop and map reset are externally configurable.

Do not freeze the existing 22-value observation vector, six policy heads or
eight movement bins into the simulator. Model architecture, observation space
and output heads may change. A legacy input/output adapter is optional; it would
not make a 1.8.9 policy behaviorally equivalent on 26.3. Old-checkpoint compatibility
and legacy-bug reproduction are not requirements for the core engine.

## PPO and Self-Play Direction

Confirmed direction: PPO, with two agent-controlled players in each independent
environment and thousands of environments advancing in parallel. The initial
host-staged wrapper trains both players with shared policy weights; thousands of
worlds with end-to-end CUDA training have not yet been validated locally.
Opponent pooling, opponent snapshots and their sampling strategy are undecided;
do not make a pool a requirement or bake it into the physics interface.

Batch both players' observations and policy inference, then queue both actions
before advancing each world. Both decisions observe the same pre-step phase;
player physics still executes in the target build's sequential order. Parallelize
worlds, not interacting players whose updates depend on that order. Agent
instances need not own separate neural networks or optimizers: shared policy
parameters can control many player instances. Model/critic architecture remains open.

The initial PPO schedule should collect a rollout with fixed policy weights,
then run minibatch update epochs over that rollout. Episodes may end/reset during
collection; a rollout boundary or PPO epoch is not an episode boundary. Overlapping
collection and learning is deferred until policy versioning/staleness is defined.
Keep physics state, observations/actions and rollout buffers GPU-resident where
practical. Multi-GPU scaling should shard environments per device and synchronize
training as needed, rather than distribute one small world across devices.

Measure physics throughput as environment-ticks per wall-clock second: one
environment-tick advances both players once. Two players therefore give twice as
many player-ticks, not necessarily twice as many policy decisions when actions
are held. CUDA-event measurements exclude setup/transfers and must identify
launch count, ticks per launch, batch size, warmup and workload. A physics-only
number is not end-to-end PPO throughput; measure inference, observation/reward,
resets, rollout storage and updates separately when implemented.

## Confirmed Target and Legacy Boundary

User-confirmed target: Minecraft Java 26.3, using the supplied source/classes and
the physics specification. The CPU reference, CUDA implementation, real-game
validation fixtures and deployment adapter must identify that same target build.
This is not a claim that an existing Cambium deployment already runs 26.3.

The Cambium audit identifies Minecraft 1.8.9 as its legacy integration target,
with retained client/server evidence for Forge 11.15.1.2318. That report is useful
for understanding control/state integration, not as the modern physics oracle.
Do not import 1.8.9 combat, key-processing quirks, reset effects, enchantment IDs,
or timing assumptions into the 26.3 backend. Do not add a second physics version
to the initial implementation merely to reproduce the old training stack.

Port or replace the small state/control adapter for a 26.3 client and, where
needed, add server instrumentation for state capture and tick correlation.
Instrumentation must not alter the gameplay mechanics being validated. Old
1.8.9 traces can test a legacy encoder/decoder, not certify 26.3 physics or transfer
of an old policy. New training and deployment evaluations should use the declared
26.3 control, observation, reward and timing profiles.

## Current Integration Evidence

The following comes from the user-provided Cambium audit of a separate checkout;
it has not been independently checked in this repository or certified at runtime.
The expanded report names source revision
`a9f841fde8706ebf4a57c15520436782acc30fda` plus preserved local launcher changes.
Its source, artifact and synthetic-test evidence must not be relabeled as new
26.3 gameplay captures.

- Client source and launch configuration reportedly target Minecraft 1.8.9;
  retained historical logs identify client and server Forge 11.15.1.2318. Source
  build targets and staged binaries are not proven equivalent, and a currently
  running deployment was not certified by that audit.
- The mod reportedly reads observations and applies controls at client tick START,
  with a blocking Python HTTP request between those operations.
- PhysicsController reportedly writes rotation and Minecraft key bindings rather
  than providing a separate movement integrator. Vanilla client prediction is
  therefore part of the behavior to examine, not just the server's world tick.
- Movement bin 0 reportedly means forward, including objects named idle; the old
  policy has no neutral movement bin. New controls should allow neutral input.
- Each true legacy attack/use command reportedly both holds the binding and adds
  a key press on every execute call. Look adds float degree deltas once, clamps
  accumulated pitch to [-90,90], and does not scale deltas by elapsed time. Hotbar
  flags reportedly persist across commands. These are adapter behaviors to replace
  or explicitly profile, not universal vanilla control semantics.
- Observations can reportedly include opponent positions synthesized from backend
  caches. Availability, age and provenance need explicit treatment.
- The audit reports reward attribution without action IDs and fight resets that
  do not reliably mark the actual episode boundary. Fix these separately from
  physical simulation; a correct engine cannot repair mislabeled PPO experience.

The audit supplies enough legacy model/control detail for the initial design;
another broad model audit is not the next dependency. What remains is the 26.3
adapter, target build/world identity and real capture/apply timing evidence.

## Proposed Backend Interface

```text
read_state(environment_ids) -> StateBatch
apply_controls(environment_ids, ControlBatch)
advance_ticks(environment_ids, count) -> StepBatch
reset(environment_ids, ResetBatch) -> StateBatch
```

These are conceptual operations, not committed C++ or Python signatures.

- read_state exposes named state fields/buffers without advancing simulation.
- apply_controls sets held controls and queues discrete events for the next defined
  input phase. It does not itself move a player or apply arbitrary velocity.
- advance_ticks executes the requested positive number of physical ticks for the
  selected environments. Held controls persist; discrete events execute once at
  their assigned phase rather than once per advanced tick.
- reset initializes selected environments, including world, players, inventory,
  RNG and control state. Start a new episode identity and clear pending events.
- StepBatch identifies the environment, episode and resulting tick. The task
  wrapper can expose observations, reward, terminated, truncated and diagnostics.

Maintain independent environment tick counters and explicit player/entity IDs.
Validate indices, batch shapes, duplicate/conflicting submissions and unsupported
controls. Specify errors rather than silently accepting or reinterpreting them.
The exact real-game stepping/barrier mechanism remains to be designed; blocking
HTTP requests are not evidence that the server advances in synchronous lockstep.

### Player Controls

Separate persistent state from events; do not collapse both into one action index.

| Control group | Proposed semantics |
| --- | --- |
| Movement | Forward/back/left/right held buttons, including all released; vanilla input shaping resolves their combination. |
| Jump/sprint/crouch | Held controls, processed through the documented eligibility and movement rules. |
| Look | Explicit yaw/pitch delta events in degrees, applied once; document wrapping, pitch clamping and order relative to targeting. |
| Attack/use | Distinguish held bindings from press events/counts and release transitions, preserving vanilla click arbitration and mining/use behavior. |
| Hotbar | Optional selection command for slot 0..8; no command means retain the current selection. |
| Hand swap | Discrete swap event, not a continuously held command. |

Define omitted-field defaults and simultaneous-control ordering before freezing
the first schema. Derive the 26.3 execution path from its actual input processing;
do not treat repeated true commands as either repeated clicks or hold-only without
an explicit rule. If a policy acts every k ticks, define press-event cadence and
held-state persistence separately. Do not expose teleportation, direct velocity writes or attacks
by arbitrary entity ID as ordinary controls unless the real adapter can perform
the same legal action. A neutral control frame releases held buttons; leaving
controls unchanged is a different operation.

### State Access and Extensibility

Expose named fields for player pose/motion/orientation, collision flags, health,
combat counters, equipment/inventory and supported world/block/entity queries.
Keep simulator-only debug state separate from information permitted to the actor.
The critic may have a separately configured view; this is not currently implemented.

Specify field units, coordinate frames, dtype, shape, validity and sample phase.
Track entity identity and deterministic ordering when returning nearby entities.
Client-predicted, server-authoritative and backend-cached values are distinct;
do not merge them without a documented adapter. Preserve freshness/visibility
masks where information is missing or stale.

Observation tensors may use float32, but that does NOT select the engine's numeric
precision. Retain Java double position/velocity and the documented float/double
conversions and operation order internally. Packed policy tensors are a separate
representation, with declared origin/scale where world-position precision matters.

Separate semantic fields from packed model tensors. Use a small versioned schema
manifest with explicit optional-field defaults and supported capabilities. New
encoders/decoders may add observations or heads without rewriting the engine.
New gameplay actions still require implementation and validation of their actual
mechanics. Store schema and preprocessing identifiers with training checkpoints;
do not silently reinterpret an old model's inputs or outputs.

## Timing and Real-Game Evidence Needed

Minecraft physical integration uses 20 ticks per simulated second. GPU execution
need not sleep in real time. The policy decision interval can be one or several
ticks, but held actions, repeated clicks and observation sampling must be explicit.

Proposed training profile: accept joint controls, apply look/events once at the
declared input phase, advance a fixed number of physics ticks, and return aligned
state/reward/outcome. This is a new interface, not reproduction of legacy blocking
HTTP cadence. Real deployment may remain asynchronous; record and evaluate its
observation/action delay rather than assuming a shared server barrier exists.

Next evidence and implementation tasks for the confirmed target:

1. Pin the 26.3 client/server artifacts, adapter build, game rules and small world
  fixture. The old Forge binaries and old arena logs are not target captures.
2. Implement a small 26.3 collector for the selected actor-visible fields and a
  separate diagnostic capture of velocity, grounded/collision and combat state.
  Reuse semantic field ideas from Cambium, not its stale caches or proxy damage.
3. Implement 26.3 controls with neutral movement, explicit held/press/release
  behavior, look deltas and valid hotbar events. Clear controls on reset and
  define timeout/disconnect handling without replaying a forward-moving idle.
4. Locate capture/control hooks relative to 26.3 targeting, input shaping,
  movement and network updates; correlate client/server ticks and decision IDs.
  Add exact tick control only if it is actually needed and supported.
5. Record authorized target-build movement/collision fixtures and their initial
  states/actions. Synthetic schemas and legacy packing tests remain useful but
  do not replace these. Model weights and layer definitions are not needed.

Instrument the real adapter to record initial state/configuration, episode and
decision IDs, requested controls, application tick/phase, and resulting state.
Associate reward events with explicit intervals or decisions. Capture client and
server state separately where possible; distinguish known ticks from estimated
clock alignment. Include game version, RNG/seed information where available and
any uncontrolled timing/randomness. Do not label synthetic traces game captures.

## Implementation Order and Acceptance Gates

| Stage | Build | Acceptance evidence |
| --- | --- | --- |
| Interface and traces | Minimal state/control schema, 26.3 real adapter and recorder | Target artifacts pinned; units, action persistence, timing and provenance documented; 26.3 replay fixtures available or their absence explicitly recorded. |
| Shared movement slice | Host/device-compatible state views and ordered physics; minimal CPU runner for standing, falling, walking, jumping and full-cube floor/wall collision | Deterministic source-derived checks; target-game per-tick fixture comparison still required. |
| CUDA execution | GPU-resident state/controls running that same core across independent environments | Device compilation/execution; CPU/CUDA comparisons across batch sizes, boundary/index checks and environment-isolation/reset tests. |
| Two-player core | Contact, aim/hit selection, attack charge, damage gates and knockback | Real/CPU/CUDA fixtures for simultaneous interactions and ordering, without legacy/plugin assumptions. |
| Mode expansion | Bridge-like placement/mining/bows, then ordinary armor/shield/bucket kit PvP | Regression coverage for each enabled mechanic and mixed-system cases; unsupported profiles rejected explicitly. |
| PPO and scale | Python batched wrapper, rollout buffers, task rewards and episode flags; then multi-A100 training | Measured single-GPU correctness/throughput first, then measured distributed behavior and transfer evaluation. |

Parallelize environments first; preserve required sequential order within an
environment until measurements justify more complex kernels. Keep state, actions
and observations resident on GPU where practical. Use batched device buffers,
not per-player Python calls or mandatory full-state CPU copies every tick.
Logging/capture is a diagnostic path, not a requirement to synchronize the hot path.

Keep rollout boundaries separate from actual termination and time-limit
truncation. A PPO collection/update timer is not automatically an episode end.
Retain explicit transition/episode identity when replacing the old HTTP pipeline.

## Validation Claims and Immediate Next Step

Distinguish source-derived expectations, CPU/CUDA agreement and independently
observed Minecraft agreement. Two implementations sharing a bug can agree.
Check discrete decisions exactly and set justified numeric tolerances in advance;
preserve Java float/double operations and avoid fast-math initially. Compare the
first divergent tick, phase and field, not only final trajectories.

The shared movement foundation now has host-side checks, a synthetic trace
runner, a Fabric 26.3 client recorder and a continuous CSV replay comparator.
One ordinary client trajectory matches exactly for 250 transitions. Sprinting,
sprint-jumping, crouch/ledge protection, pose clearance and ordinary player contact
have source-derived checks. The fixed-diamond-sword profile now adds legal aim
targeting, ordered melee execution, health/damage cooldowns, crit/sprint/sweep
branches and a synthetic duel trace. These additions need real-game interaction
captures; inventory, equipment mitigation and episode/reset rules remain separate.
The synchronized action barrier is explicit, not arbitrary network packet timing.
The user reports prior CUDA checks and movement-only benchmarks passing on a
3060 Ti. The new combat paths and extended CUDA checks still need a GPU run;
CUDA remains unavailable on the development machine. Next work is these game/GPU
fixtures alongside the missing reset/training facilities.
Use the expanded legacy audit as integration context, not a requirement to retain
its model layout or bugs. Missing target game captures need not block foundation
work, but they do block a claim of Minecraft-validated parity. No complete stage,
Minecraft runtime-parity claim or performance target is marked finished.

## Real-Game Melee Validation

The existing Fabric mod and C++ CSV comparator validate **local movement only**.
F9 does not record clicks, target selection, health, the opponent's authoritative
state or server damage decisions. Recording a fight with it today cannot validate
melee. The CPU/CUDA duel tests are synthetic, not Minecraft reference captures.
The extension below is planned, not implemented.

The separate [simulation playback viewer](../cambiummod/README.md#simulation-playback)
can now consume CPU/CUDA duel exports using real singleplayer replay players.
Absolute mode visualizes simulated state; input mode runs a bounded vanilla
server-side control adapter and exposes trajectory drift. Neither adds the missing
authoritative Minecraft combat capture stream described below.

### What Works Today

Install the current recorder build into the matching 26.3 client. Use the exact
floor/wall terrain and supported state described in the simulator README, start
at rest with jump released, then toggle F9 to record a short isolated movement
fixture. Preserve both the raw CSV and its JSON metadata. Replay with the existing
[movement comparator](../simulator/AGENTS.md#recorded-physics-comparison), using
the declared current-input phase and a new report prefix. It seeds once and then
free-runs; inspect numeric errors and discrete mismatches, not just exit zero.

One real 250-transition movement capture matched all nine numeric fields and
ground state exactly. That is evidence for that trajectory, not all mechanics.
The expanded schema-2 sprint/crouch/pose hooks still need fresh game captures.
Keep initial movement checks separate from two-player contact and attacks.

### Recorder Extension Needed

Use two real players/clients with an instrumented 26.3 integrated or dedicated
server in a controlled world. Add common/server recording hooks as well as client
hooks; the current client-only recorder is insufficient even in singleplayer.
Remote-player rendered positions are interpolated, not authoritative snapshots.

Record the following without changing vanilla targeting or damage calculations:

| Stream | Required evidence |
| --- | --- |
| Client intent and selection | Player ID, refreshed movement controls, actual attack press/consumption, yaw/pitch and eye ray, chosen entity/block/miss, charge and local tick/phase. Include attempted clicks rejected before a packet is sent. |
| Server order and decisions | Packet receipt/handling, per-tick event sequence, attacker/victim IDs, reach/eligibility result, attack charge reset, damage attempt/accept/reject, crit/sprint/sweep and secondary hits. Log rejected attacks as well as accepted ones. |
| Before/after snapshots | Both players' position/velocity, pose/ground/sprint/fall state, movement timers, health/absorption, last hurt, damage/hurt/death cooldowns, attack charge counters, attributes and fixed equipment. Snapshot immediately around each relevant event and at declared tick boundaries. |
| Network impulse | Server knockback and velocity before/after, emitted velocity packet and the affected client's receipt/application phase. Distinguish server velocity restoration after sending a packet from actual client movement. |
| Manifest and identity | Exact game/mod build, terrain, rules, initial state, stable player IDs, session ID, client/server ticks and event sequence. Correlate attack packets and decisions across streams; if correlation is ambiguous, mark the fixture incomplete. Record RNG state/draws when required. |

Do not equate a client tick with a server tick or infer an attack from health
loss alone. Multiple attack packets may be handled in one server tick. Trace
hooks must not call random generators or mutate timers to obtain diagnostics.
Preserve raw float/double precision and both accepted and rejected events.

### Small Fixtures Before Duels

Use the same fixed-sword profile as the simulator: survival PvP enabled, no
teams/protection, unarmored players with unenchanted unbreakable diamond swords,
no shields/effects and natural regeneration off. Pin attributes/food; keep clips
short enough to avoid unsupported hunger changes and reject out-of-profile state.

1. Stationary fully charged hit: target, charge, damage, health and default knockback.
2. Cooldown gates: equal/weaker repeat rejection, stronger differential damage,
  and the boundary at 10 remaining ticks. Include absorption and lethal damage.
3. Critical and sprint hits: eligibility, damage, knockback ordering and attacker
  slowdown; then a separate sweep fixture with controlled additional targets.
4. Legal targeting: off-axis miss, nearest target, reach boundary, movement-based
  reach extension, wall occlusion and block versus air miss behavior.
5. Two players moving and attacking: contact, simultaneous inputs, lethal ordering,
  then a longer duel. Include degenerate knockback only with matching RNG evidence.

### Replay and Acceptance

First add a versioned combat trace reader and event-level comparator beside the
existing replay path, reusing the shared core. Isolated event checks may seed
from the captured before-state to diagnose targeting, damage or knockback. Label
them event checks, not continuous trajectory validation. This requires no PPO.

For a continuous two-player replay, seed both players and terrain once, apply only
captured legal controls/events at the documented phases, then compare every tick
without resetting position, velocity or combat counters to the recording. Compare
target/rejection decisions, branch flags, timers and health before inspecting the
first divergent motion field. Set numeric tolerances before checking the capture;
never fit action phases or tolerances merely to pass a trace.

The current core uses a synchronized look/target barrier, player-index attack
order, then sequential movement/contact. Vanilla network packet scheduling is
not automatically equivalent. The recorder must establish that a fixture matches
this order; otherwise report the unsupported schedule or implement a separately
specified adapter before claiming end-to-end parity. Record and apply a network
impulse exactly once; do not add packet velocity on top of the same simulated hit.

Keep three results separate: real-game event/trajectory agreement, CPU/CUDA
agreement on those fixtures, and synthetic checks. Passing shared CPU/CUDA tests
cannot substitute for the first. A run with missing events, unsupported state or
unknown timing is incomplete evidence, not a passing Minecraft conformance run.