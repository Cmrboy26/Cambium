# Native Simulator: Development And Reference

Follow the repository [development guide](../AGENTS.md). Training has additional
[local guidance](training/AGENTS.md). Public build entry points are in
[README.md](README.md).

## Ownership And Fidelity

C++20/CUDA-compatible functions target supplied Minecraft Java 26.3 behavior.
CPU/GPU call the same ordered tick code. Synthetic scenarios need no Minecraft
runtime/assets. `include/` owns shared physics/views, `src/` backend/support,
`apps/` executables/C ABI entry points, `setups/` authored experiment definitions,
`environments/` legacy concrete scenario fixtures, `tests/` native checks, and
`tools/` recording conversion.

## Experiment Setups

Keep reusable authoring JSON together under [setups/](setups/):

- [arenas/](setups/arenas/): geometry and physical rules.
- [environments/](setups/environments/): players, gear/spawn sampling and task settings.
- [policies/](setups/policies/): observation and action layouts.
- [curricula/](setups/curricula/): training plans, opponents, stages and evaluation.

References resolve relative to their containing JSON file. Legacy concrete
[environments/](environments/) are separate native/regression fixtures, not the
new authoring environment definitions.

Recorded status from 2026-10-18: CPU/source-derived checks pass locally; a prior
ordinary client capture has zero numeric/ground error across 250 movement steps.
Prior RTX 3060 Ti CUDA checks/movement benchmarks were user-reported, not evidence
for newer melee/equipment/food changes. Those need current device runs and combat
conformance captures. Do not present historical observations as fresh validation.

Implemented slice:

- Queued keyboard controls: opposing keys cancel, held controls persist, relative
  look events apply once. Pitch clamps to [-90,90]; yaw is not wrapped.
- Ordered bookkeeping, float keyboard normalization/square adjustment, ground
  jumping/repeat delay, relative acceleration, swept movement, ordinary
  zero-restitution collision, gravity/drag and position clamping.
- Sprint key/double-tap eligibility/persistence, acceleration/jump impulse;
  crouch slowdown/delayed flag, pose clearance and ledge backoff.
- Float width 0.6; standing/crouching/crawling heights 1.8/1.5/0.6; default
  attributes; air/ordinary solid full cubes with friction 0.6. Dense finite grid
  is X contiguous, then Y, then Z. Outside is loaded air.
- Y-first clipping, horizontal magnitude ordering with X-first ties; ordered
  player velocity pushes and independently callable knockback.
- Aim-ray melee, one-shot attack presses, sword/axe attributes and charge,
  critical/sprint/sword-only sweeps, armor/toughness, damage cooldowns,
  health/absorption/invulnerability, offhand shields and axe disabling.
- Opt-in food/saturation/exhaustion, regeneration, Peaceful restoration and
  difficulty-dependent starvation.

Not implemented: step-up, penetration resolution, fluids, special blocks,
climbing, effects, enchantments, inventory/durability, food consumption and timed
effects. Flight eligibility/movement attributes are supplied state, not the
systems underlying them. Spawn states must be finite and collision-free; views
need valid buffers/dimensions. Task rewards/resets belong in the training wrapper.

## Native Equipment Mechanics

Policy metadata comes from [the shared catalogue](../contracts/policy_catalogue.json)
through generated `apps/policy_catalogue.generated.hpp`. Regenerate from root
with `python scripts/generate-contract.py`; do not hand-edit generated tables.
Pure base observation math is in `apps/observation.hpp`; raycasts and equipment
sampling stay in the training adapter. Native `shared_core_checks` consumes the
same [fixture file](../contracts/policy_fixtures.json) as Java/Python.

The training library exports ABI 2 plus JSON `mcsim_build_info()` and
`mcsim_batch_info(handle)`. Build info contains catalogue/capabilities, scalar
widths, reward input/total counts, compiler/revision/compile time, CUDA build flag
and SHA-256 of native headers/support sources/training entry point/CMake input.
CMake reconfigures when these inputs change. Batch info declares actual world/
player/observation/action sizes; Python validates before passing output buffers.
These checks protect the supported adapter, not arbitrary raw C callers.
Selected library paths/device are recorded by Python; source/build identity is
diagnostic, not a physics-parity or checkpoint-compatibility claim.

The additive ABI 2 profiling calls are `mcsim_set_profiling(handle, enabled)`
and `mcsim_profile(handle)`. The setter accepts only 0/1, returns 0/-1, defaults
off and clears all counters whenever enabled. Disabling preserves the last
counters. The getter returns a thread-local JSON string with `version: 1` and
named wall-clock `seconds`, or null on error. It does not clear counters; the
string lasts until the next getter call on that thread. Only profiled CUDA
physics/reset calls wait for kernel completion. Profiling never draws random
numbers or changes reset/physics ordering. See the
[collection timing schema](training/AGENTS.md#collection-profiling).

The additive ABI 2 `mcsim_losses(handle, uint8_t* losses)` fills one loss flag per
player in world-major order (`[world_count, 2]`), returning 0 or -1. It reads current
host player state with the same alive/outside predicate as terminal rewards,
independently of reward weights; it neither steps nor resets worlds. Call after
step and before reset to classify fight outcomes. Python uses this bulk path for
completed-episode metrics and a terminal-snapshot fallback for older libraries.

The additive ABI 2 `mcsim_set_max_ticks(handle, limit)` returns 0 on success,
-1 on error, and accepts limits in 1..1,000,000. Apply it at a stage reset
boundary; it changes the existing batch's timeout without resetting players,
counters or immutable terrain. Python reapplies timeout-normalized shaping
weights. The setter does not rewrite the original scenario metadata.

The additive `mcsim_step_reward_totals(handle, player_mask, totals)` returns the
last successful tick's 12 reward contributions for selected actors. The mask
has `worlds*2` boolean entries; null selects both players. Resets clear selected
worlds' last-step contributions, not cumulative batch totals. Opponent-pool
training selects its learner actor before episode reset or side reassignment.
Existing reward calculations and ABI 2 buffer layouts are unchanged.

The default remains an unarmored diamond sword without shield; v1 policies stay
sword-only. [V2](training/AGENTS.md#swordaxeshield-v2) adds weapon/shield controls.
[Descriptive contracts](../docs/simulation_contract.md) configure slots/armor and
controls independently of presets. Envelope 3 carries equipment/use state;
movement CSV remains movement-only.

The split experiment compiler supplies concrete `minecraft_sim_environment`
version 1 plus a policy descriptor. Native code does not resolve authoring
arena/environment/policy references or equipment/spawn distributions. Compiled
players have nine physical hotbar slots, independently of equipped items;
repeated `minecraft:air` entries are valid, duplicate nonair entries are not.
Historical concrete fixtures with shorter loadouts remain supported.

`TrainingBatch` tracks each player's selected slot separately from the physical
`PlayerState::weapon`. Initialize it from concrete `loadout.selected`, update
from the weapon head's semantic slot value, and use it for self/opponent slot
observations. Never recover a selected empty slot by matching item IDs. Slot
changes retain the existing item-based charge behavior: air-to-air selection
does not reset attack charge. Episode scenario serialization retains the
requested initial slot, not a reconstructed or current selection.

Training replay frames add optional actor `weapon_slot` and controls
`select_slot` metadata. The latter is the requested slot when `select_weapon`
is true, otherwise -1; existing `weapon` item IDs remain for mechanics and old
readers. This is additive within envelope 3 and does not change native physics
state, `Controls`, or ABI size. Legacy standalone replay producers without
tracked slot identity may omit these fields.

Obtain `PlayerView` from `WorldView::player()` to preserve the world's sine table
for directional checks. Configure through setters:

```cpp
auto attacker = world.player(0);
auto defender = world.player(1);
attacker.equipWeapon(minecraft_sim::WeaponItem::diamond_axe);
defender.equipShield(true);
for (int slot = 0; slot < 4; ++slot) {
    defender.equipArmor(static_cast<minecraft_sim::ArmorSlot>(slot),
        minecraft_sim::ArmorMaterial::diamond);
}
minecraft_sim::Controls defense;
defense.use_item = true;
defender.setInputs(defense);
world.processTick();
```

- `WeaponItem`: unarmed and wooden, stone, copper, golden, iron, diamond and
  netherite swords/axes. Setters preserve widened Java float damage/speed
  modifiers. Changing weapon resets charge; setting the same weapon does not.
  Axes/unarmed never sweep; ordinary crit/sprint/quadratic charge rules apply.
- Armor slots: boots, leggings, chestplate, helmet, including mixed materials.
  Materials: none, leather, copper, chainmail, iron, golden, diamond, netherite
  and turtle helmet. Setters recompute defense/toughness/armor knockback
  resistance; `knockback_resistance` remains an additional base modifier.
- Immunity compares incoming pre-armor damage first. Armor/toughness reduce the
  accepted full hit or cooldown differential before absorption. Base reduction
  uses `CombatRules.getDamageAfterAbsorb`; no enchantment/potion protection,
  armor wear or damage-source armor bypass.
- `use_item` is held; `attack` is one-shot. Use starts/retries after combat,
  before movement. Releasing stops use before combat but consumes that tick's
  attack. Simultaneous attack/new-use attacks first, then raises the shield.
- `PlayerState::blocking()` needs five completed use ticks. Ordinary blocking
  ignores pitch/height and uses a horizontal 90-degree half-angle, including
  boundary and coincident source.
- Full blocks preserve health/absorption and victim default knockback, return
  `accepted=false`, and may seed hurt timers with `last_hurt=0`.
  `AttackOutcome::blocked_damage` distinguishes blocks from immunity rejection.
- Unenchanted axe blocks disable shields for 100 ticks even at low charge. The
  disabling hit is still blocked, without successful-hit sprint slowdown.
  Disable is evaluated before repeated-hit rejection; `shield_disabled` reports it.
- Cooldowns decrement at movement end, including disable tick (100 becomes 99).
  Held use retries every four ticks and raises again after expiry. The 72,000-tick
  use duration ends active use, but held input can restart it.
- Use scales movement by `0.2F` before crouch/square adjustment and prevents
  starting sprint; it does not itself cancel existing sprint.

Sources: 26.3 `Items`, `ToolMaterial`, `ArmorMaterials`, `CombatRules`,
`BlocksAttacks`, `LivingEntity`, `Player`, `UseEffects`, `LocalPlayer` and
`Minecraft.handleKeybinds`. Native `hurt` is ordinary melee with optional
axe-disable seconds and explicit shield bypass. No projectile/custom partial
blocks, inventory/hotbar protocol or other item-use systems are implied.

## Native Food And Regeneration

Defaults freeze food and disable regeneration. Set backing `WorldState` before
stepping or constructing/uploading a CUDA batch:

```cpp
world_states[0].natural_regeneration = true;
world_states[0].difficulty = minecraft_sim::Difficulty::normal;
```

Natural regeneration enables food simulation. `simulate_food=true` with
regeneration false enables hunger/starvation only. These are native settings;
environment JSON, training contracts and replay/CSV do not expose them.

- Initial food=20, saturation=5, exhaustion=0, food timer=0. `FoodData.tick`
  consumes one saturation (or food outside Peaceful) when exhaustion is strictly
  greater than 4, then runs healing/starvation on a shared timer.
- At food 20 and positive saturation, every 10 food ticks heal
  `min(saturation,6)/6` and add that much exhaustion. Otherwise food at least 18
  heals one health every 80 food ticks for 6 exhaustion. Exhaustion caps at 40.
- Supported attacks/health damage add 0.1 exhaustion; ordinary/sprinting jumps
  add 0.05/0.2. Grounded sprint distance uses rounded centimeters/source float
  arithmetic. Other movement branches are not implemented.
- With regeneration enabled, Peaceful additionally restores one health and one
  saturation every 20 player ticks, and one food every 10 ticks below 20.
- Starvation attempts one damage every 80 food ticks, stops at health 10 on
  Easy/Peaceful and 1 on Normal, and can kill on Hard. It bypasses armor/shields,
  not absorption/repeated-hit immunity, and adds no knockback/exhaustion.
- `heal()` clamps health without resurrection/absorption. Absorption is explicit
  state consumed after armor before health. No golden-apple/effect lifecycle.
- Replay actors set maximum absorption before seeding it. Replay/live adapters
  freeze food and both regeneration paths without changing world rules; reject
  unsupported rule changes. This prevents healing masking differences, not
  missed attacks or death outcomes.

## Numeric And Tick Contracts

Yaw uses Minecraft table lookup: 65,536 floats shared across batch worlds.
Java conversion/masking includes cosine offset before truncation. Host generation
uses double `std::sin` then float; bits are not certified against target Java.
CPU/CUDA share the generated table, uploaded once per CUDA batch.

Use [shared views](include/minecraft_sim/core.hpp) to separate intent/advancement:

```cpp
auto world = batch.world(environment_id);
world.player(player_id).setInputs(controls);
world.processTick();
```

Queue all players before advancing once. `processTick()` latches controls, calls
ordered ticks, increments world time. Controllers choose inputs, not physics.
Backing storage must outlive views and movement requires a valid sine table.
After each travel, overlapping pushable players are queried in array order;
pairs may push twice and are not deduplicated. Push changes velocity, not position.
Deterministic array order is not arbitrary Minecraft spatial-query order. No
vehicle pushes, special entity pushes or cramming.

`knockback(power, direction_x, direction_z)` applies resistance-scaled horizontal
momentum halving/subtraction and grounded vertical cap, without advancing time
or resolving an attack. Callers own hit eligibility/interaction phase. Negative
resistance is not clamped. Degenerate directions consume ordered draws from the
player's Java-compatible 48-bit legacy RNG. `state().random.setSeed(seed)` uses
Java scrambling; default zero is not a captured entity seed. Matching fallback
randomness needs matching state/draw history.

## Fixed Sword Melee Profile

The base conformance fixture is survival/PvP, no teams/spawn protection, default
reach, unenchanted fixed diamond sword, no armor/shield/effects. Use unbreakable
gear and disable regeneration for long real-game comparisons. Food eligibility
is supplied state unless native food simulation is explicitly enabled.

- Submit `Controls::attack=true` each requested press; inspect
  `state().last_attack` after `processTick()`. Held movement/sprint/sneak differs
  from consumed one-shot attack. Air misses start a ten-tick miss delay; a frame
  without attack releases it. Blocked outline rays do not damage through terrain
  or reset charge like an air miss. No mining occurs.
- Eyes: standing 1.62F, crouching 1.27F, crawling 0.4F. Float lookup direction
  selects nearest live-player AABB with zero margin. Reach is 3 plus positive
  look-projected current movement. Full-cube OUTLINE blocks clip first. Server
  validation buffer is not extra usable reach; no target-ID policy action.
- Latch controls, apply all look events, decrement combat timers, then select
  requested targets before any movement/impulse. Resolve attacks in player-index
  order; an earlier lethal hit cancels later dead attacks. This synchronized
  headless barrier does not reproduce arbitrary packet scheduling.
- Movement/contact remain sequential after attacks. Sprint eligibility updates
  during movement; attacks use carried sprint/ground/fall state at the barrier.
  Extra knockback clears sprint; later held-sprint processing may restart it.
- Diamond damage=7; speed=`4.0 + double(-2.4F)`, not idealized 1.6. Charge uses
  float delay/half-tick offset, quadratic damage, `charge > 0.9F` full gate.
  Eligible falling/airborne/nonsprinting full hits get 1.5F crit damage. Charged
  successful sprint hits add knockback, scale attacker X/Z by 0.6, clear sprint.
  Grounded slow full sword hits can sweep other nearby players for 1 damage;
  two-player duels have no secondary sweep candidate.
- Health starts 20. Accepted full hits set cooldown=20/hurt time=10. At cooldown
  greater than 10, equal/weaker hits fail; stronger hits apply only excess damage
  without default knockback or timer refresh. Accepted damage differs from
  health lost. Full impact marks synchronization even if resistance removes impulse.
- Charge increments once after movement. `last_attack` includes target,
  attempted/accepted, charge, incoming damage, health lost, crit/sprint/sweep and
  secondary sweep count. `hurt`/`resolveMelee`/`knockback` are debug/implementation
  primitives, not legal policy controls.
- Zero health disables targeting, damage, voluntary travel/jump and contact.
  Dead slots remain allocated and may retain corpse velocity; death time caps
  at 20. Native physics does not remove corpses, drop items, respawn, terminate
  episodes, reward or reset. The task wrapper stops at death.

`minecraft_sim_runner --duel --ticks 80` uses two players walking together and
attacking every 13 ticks, initially charged, stopping at first death. It prints
motion, health, target, charge and damage, not a captured fight or learned policy.
CUDA per-decision upload/download is diagnostic. The benchmark remains movement-
only. CSV recording lacks attack/health/both-player/network timing and cannot
certify combat. See the [melee validation plan](../docs/environment_interface_next_steps.md#real-game-melee-validation).

## Windows Build

Requires CMake 3.24+, C++20 compiler and build driver. From a compiler-enabled
terminal at repository root:

```powershell
cmake -S simulator -B simulator/build -G "Visual Studio 17 2022" -A x64
cmake --build simulator/build --config Release --parallel
ctest --test-dir simulator/build -C Release --output-on-failure
.\simulator\build\Release\minecraft_sim_runner.exe
```

Separate MSYS2 MinGW build, with GCC and `make` on PATH:

```powershell
cmake -S simulator -B simulator/build-mingw -G "MinGW Makefiles" `
  -DCMAKE_CXX_COMPILER=g++ `
  -DCMAKE_MAKE_PROGRAM="$((Get-Command make).Source)" `
  -DCMAKE_BUILD_TYPE=Debug
cmake --build simulator/build-mingw --parallel
ctest --test-dir simulator/build-mingw --output-on-failure
.\simulator\build-mingw\minecraft_sim_runner.exe
```

First configure needs Git/network for MIT-licensed
[fast-cpp-csv-parser](https://github.com/ben-strasser/fast-cpp-csv-parser), pinned
to `574a9fe4d323ba63416877a4a5fe59088d37aa34`. It is a private host recording-I/O
dependency fetched into the build, never device physics. Offline override:
`-DFETCHCONTENT_SOURCE_DIR_FAST_CSV=/path/to/checkout`.

MinGW stages available C++/GCC/pthread DLLs beside executables to prevent loading
incompatible PATH runtimes (for example Anaconda). No global PATH/admin changes.
Use `scripts/check.ps1 -BuildDirectory simulator/build-mingw` for that tree.
Training discovers `simulator/build*`; use `-Library`/`--library` when ambiguous.
Visual Studio examples use `build/Release`; MinGW executables are in the tree root.

## Linux Build

```sh
cmake -S simulator -B simulator/build -DCMAKE_BUILD_TYPE=Debug
cmake --build simulator/build --parallel
ctest --test-dir simulator/build --output-on-failure
./simulator/build/minecraft_sim_runner
```

Requires CMake, C++20 and Make/Ninja. Use fresh trees for changed toolchains;
configure `Release` for optimization. GCC/Clang disable fast-math/contraction;
MSVC uses strict floating point. These flags do not prove Java numeric parity.

## CUDA Build

Windows: `./scripts/run-gpu.ps1` builds, requires a passing non-skipped device
test, and runs a duel; `-Mode All` also benchmarks; `-DryRun` needs no CUDA.
Use a C++20 CUDA toolkit and supported host compiler, not the existing MinGW CPU
build. [GPU runbook](../scripts/AGENTS.md#run-gpu).

Linux/A100:

```sh
cmake -S simulator -B simulator/build-cuda -DMCSIM_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release
cmake --build simulator/build-cuda --parallel
ctest --test-dir simulator/build-cuda --output-on-failure
./simulator/build-cuda/minecraft_sim_runner --cuda --ticks 40
```

CUDA defaults off. One kernel thread owns one environment, preserving sequential
player order. `CudaBatch` owns state, controls, immutable terrain and sine table.
`step(k)` advances without host download; queued look applies only on tick one.
`deviceView()` pointers are device-only, never host-dereferenceable. Default
stream ordering must be preserved when integrating other streams.

`uploadControls()`/`download()` are synchronous diagnostic conveniences, used by
the trace runner each decision, not a GPU-resident training hot path. New batches
reset buffers. `resetEnvironments()` restores selected player states, clears world
ticks/controls using preallocated device scratch and preserves terrain. Initial
PPO remains host-staged.

## CUDA Throughput Benchmark

`minecraft_sim_benchmark` times actual CUDA physics with two players/world using
CUDA events plus a separate host completion timer. No CPU fallback: unavailable
CUDA/device fails. CPU-only builds still support help/argument validation.

RTX 3060 Ti uses architecture 86, A100 uses 80. Windows supported MSVC example:

```powershell
cmake -S simulator -B simulator/build-cuda -G "Visual Studio 17 2022" -A x64 `
  -DMCSIM_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build simulator/build-cuda --config Release --parallel
ctest --test-dir simulator/build-cuda -C Release --output-on-failure
4096, 16384, 65536 | ForEach-Object {
  & .\simulator\build-cuda\Release\minecraft_sim_benchmark.exe `
    --environments $_ --launches 1000 --ticks-per-launch 1 --warmup-launches 10
}
```

Linux uses the same flags on `./simulator/build-cuda/minecraft_sim_benchmark`.
`--help` lists defaults; `CUDA_VISIBLE_DEVICES` controls visibility/current device.
Workload: enclosed floor/wall fixture, initially overlapping players, fixed held
sprint/sneak/repeated-jump controls and varied orientations. It avoids falling out
but is not learned combat. Compare matching settings; warmup/evolving state affect
timings. Terrain is currently copied per world, so allocation may limit batches.

- One environment-tick advances both players. Numerator:
  `environments * launches * ticks_per_launch`, excluding warmup. Player-ticks/sec
  is twice environment-ticks/sec, not fresh decisions/sec for held controls.
- Allocation, initialization/upload, warmup, final download and validation are
  outside both timers. CUDA events bracket launches on the default stream and
  include gaps between submissions; host time includes submission/completion wait.
- Check final world/player tick counters and finite motion after timing. Run
  parity first; benchmark sanity checks prove neither CPU/GPU nor game parity.
- Start at one tick/launch; compare 8 or 32 for amortization with comparable total
  ticks. Held-control steps do not run fresh inference.
- Rates exclude observations, rewards, resets, inference, rollout storage and
  PPO. No benchmark substitutes for current target-GPU tests or full PPO metrics.

Shared-policy PPO exists; opponent pooling remains undecided/unimplemented. See
[training direction](../docs/environment_interface_next_steps.md#ppo-and-self-play-direction).

## Tick API And Replay

Runner flags: `--ticks 1..100000` (default 40), `--cuda`, `--duel`, `--help`.
Default output is per-tick position/velocity/yaw/ground/collision for a synthetic
standing/walking/jump/wall fixture, not an external capture loader. Tick rate is
20 simulated Hz, without sleeping. Ordinary input ownership is described above.

## JSON Environments

Default app terrain/players: [duel.json](environments/duel.json). Benchmark:
[benchmark.json](environments/benchmark.json), with varied orientations/controls.
Comparator uses default duel terrain. Runner accepts `--environment FILE.json`.

Host setup parses JSON once, caches the default per process, and keeps definition,
dense grid and initial states in the scene. Batches copy host buffers; CUDA uploads
terrain/state on construction and keeps them resident. Never read JSON per tick.

Schema 1: `format: minecraft_sim_environment`, Minecraft `26.3`, legacy profile
or `configured`, `bounds.min/size`, ordered inclusive `blocks` fills, two ordered
players. Coordinates are origin-relative. Later fills override; unspecified
cells are air. Only `minecraft:air`/`minecraft:stone` full cubes. Dimensions
1..128, total volume <=32768, coordinates -4096..4096, <=256 fills.

Players declare position/velocity/yaw/pitch/on_ground; health/absorption/charge
default 20/0/0. Optional `loadout` supplies Minecraft-ID weapon slots, selection,
offhand and armor; `policy` supplies ordered observations/controls. Positions must
be inside bounds; invalid sizes/fills/blocks/states fail, never silently coerce.
See [research-duel.json](environments/research-duel.json) and
[full contract](../docs/simulation_contract.md).

## Minecraft Visual Replay

Add `--replay` to a duel for unique timestamped JSONL in the current directory;
an optional path selects a file and refuses overwrite. Print the path. Omit the
flag for no file. CMake fetches nlohmann/json v3.12.0 for host loading/export,
not the CUDA core.

Envelope 3 identifies game/backend/20 Hz/two players and embeds resolved blocks,
spawns/loadouts. Frame zero is initial state; subsequent frames contain both
resulting states plus causal look/attack/semantic weapon/use controls, equipment
state and diagnostic outcomes. Playback is independent of policy head ordering.
Viewer also supports sword-only envelope 2, not 1.

Loading reconstructs arena, pauses at frame zero/speed 1. Restart resets/resumes;
reset pauses. Neither rereads files. Input mode is bounded vanilla diagnostics;
absolute mode forces state and cannot validate physics. Use real player entities.
[Playback/restoration details](../cambiummod/AGENTS.md#simulation-playback).

## Command-Block Recordings

```powershell
.\simulator\tools\parse_recording.ps1 -Path .\recorded_sessions\wall.txt
```

Parser writes sibling CSV with source line, server tick, mask/seven buttons,
position, absolute yaw/pitch and ground state. Preserve numeric text without NBT
suffixes, not default PowerShell double rounding. Ignore unrelated lines; malformed
samples fail before export. Report duplicate/discontinuous ticks without sorting
or dropping rows. Existing output needs `-Force` or `-OutputPath`. Never write raw
recordings. These are last-received server samples, not client-phase captures:
displacement is not stored velocity, absolute rotation is not raw mouse input,
and action/state boundary is not guaranteed.

## Recorded Physics Comparison

```powershell
.\simulator\build\Release\minecraft_sim_compare.exe .\recorded_sessions\wall.csv
```

CPU comparator uses [scene.hpp](apps/scene.hpp): floor X/Z=-8..7,Y=0; wall
X=-2..2,Y=1..3,Z=2; outside loaded air. CSV contains no terrain. Matching fixture,
default attributes and supported mechanics are caller responsibilities. It uses
the shared core, not a second simulator.

Default outputs: sibling `.comparison.csv`/`.summary.csv`; summary also prints.
`--output-prefix` changes base; `--force` permits report replacement. Never
overwrite input. Validate input/replay before opening reports; parents must
already exist. Detail text is buffered in memory.

- Row zero seeds state and is unscored. Later motion evolves freely, never
  corrected. Schema 2 additionally seeds pose/sprint/crouch, prior controls,
  sprint timer/window, food/flight, fall distance, jump delay, horizontal/minor
  collision once. Legacy defaults: standing, nonsprinting, released controls,
  food=20, no flight, window=7, zero timers/fall/collision. Start legacy neutral
  with jump released/cooldown expired.
- Velocity priority: `--initial-velocity VX VY VZ`, recorded `vx/vy/vz`, then
  assumed settled standing `(0, -0.08 * double(0.98F), 0)`. Last mode requires
  neutral grounded row zero, not proof of rest.
- `--input-phase current` (default) applies destination buttons/absolute angles
  before movement; `previous` uses source row. Never auto-fit phase. Imposed
  angles are inputs, not independent mouse-processing validation.
- Ticks must be consecutive/ordered. Parser columns required except optional
  `source_line`; extra columns, quoted fields, CRLF/BOM supported. Nonfinite
  data, inconsistent masks, partial velocity or schema-2 state fail. State block:
  `sprinting,crouching,pose,previous_forward,previous_sneak,sprint_trigger_time,`
  `sprint_window,food_level,may_fly,fall_distance,jump_delay,horizontal_collision,`
  `minor_horizontal_collision` plus velocity. Pose 0/1/2=standing/crouching/crawling.
  Legacy diagnostic columns stay compatible.
- Detail has recorded/simulated, signed `simulated-recorded`, absolute and
  percentage errors. Missing reference velocity leaves reference/errors blank,
  not simulated velocity. `dx/dy/dz` is displacement, not stored velocity.
  Ground/recorded movement flags compare without forcing state.
- Summary: mean/min/max absolute errors and percentages, eligible/excluded
  counts, worst tick, first above `--tolerance` (default 1e-9). Position is
  blocks; displacement/velocity blocks/tick. Ties pick first. Missing velocity
  is `NA`, not zero error. Flags have mismatch count/rate/first tick.
- Percentage=`100*abs(simulated-recorded)/abs(recorded)`. References with
  `abs(recorded) <= --percent-floor` (default 1e-9) are excluded only from
  percentages. Never add denominator epsilon. Position percentages are
  coordinate-origin dependent.

Use `--input-phase previous --output-prefix ./recorded_sessions/wall-previous`
for a separate phase probe. `--help` lists flags. Exit 0 means replay/report
generation succeeded even with differences; exit 1 means invalid input/arguments
or I/O. Differences alone prove neither a physics defect nor parity. Reference
velocity only means anything at matching phases.

## Checks And Next Work

- `shared_core_checks`: trig, queued input, world isolation/ownership, movement,
  sprint/jump/crouch/ledge/contact, knockback/RNG, thresholds/determinism, item
  tables/mixed armor, cooldown ordering, shield boundaries/delay/release/retry,
  weak axe disable and attack/use order; CSV precision/validation/statistics,
  missing references/phase/state seeding/free replay with perturbed references.
- `runner_build_info`: launch/target/default trace. `runner_duel`: lethal fixed
  sword trace. `benchmark_help`: CLI/metric identification without device/fallback.
- CUDA builds add `shared_core_cuda_checks`: 137 two-player worlds, multi-tick
  decisions, exact discrete state, initial 1e-12 numeric tolerance. Scenarios
  include equipment/shields and food/regen/starvation. No-device is skipped, not
  passed. Require a current device run before claiming those changes verified.
- Disable checks with `-DBUILD_TESTING=OFF` if needed. Keep target-device runs,
  exact sine-table certification and first-divergent-tick game captures separate.
  Shared code can share bugs. Follow [specs](../docs/minecraft_simulation_specs.md)
  and [implementation gates](../docs/environment_interface_next_steps.md).