# CambiumFabric: Development And Reference

Follow the repository [development guide](../AGENTS.md). Keep installation and
common commands in [README.md](README.md).

## Runtime And Build

Client-installed mod for Minecraft Java 26.3, Fabric Loader >=0.19.5 and matching
Fabric API (template 0.163.0+26.3). Never install on a dedicated server. Singleplayer
worlds must be private; use disposable worlds for replay/live experiments.

Requires JDK 25. From repository root: `./scripts/build-mod.ps1`, or `-Task check`
for checks, `-Task replayAnalysisChecks` for headless bridge/replay checks.
Installable output: `build/libs/cambiumfabric-1.0.0.jar` within this component,
not `-sources.jar`. Only `src/main` is built; root template copies are inactive.

`-JavaHome` overrides cached `.gradle/java25`, then `JAVA_HOME`, then PATH. Fresh
clones have no portable cache. Helper does not download/change system Java;
direct Gradle works with JDK 25 `JAVA_HOME`. Install requires explicit `-Install`,
Minecraft closed and a selected game folder. It does not provision Fabric/API.
Replace older JAR; remove differently named copies to avoid duplicates, then
reopen Minecraft. [Selection rules](../scripts/AGENTS.md#minecraft-folder).

Template license remains [CC0](LICENSE).

Gradle packages `cambium-build.properties` with mod version, source SHA-256 and
UTC build time. Java/resources/Gradle input changes regenerate it; unchanged
builds retain identity. Initialization logs identity, catalogue hash and loaded
artifact location; `/simduel status` also prints identity. Compare this log with
the intended build before diagnosing stale installed JARs. Rebuild/reinstall
after source/catalogue changes; restart Minecraft. No script deletes old JARs.

`PolicyCatalogue.java` is generated from the root catalogue, not hand-maintained.
`ObservationEncoder` owns pure base normalization; `LivePolicy` owns Minecraft
state extraction, equipment features and HTTP negotiation. The service catalogue
hash is checked when supplied; older descriptor-compatible services may omit it.
Gradle check loads shared fixtures via an explicit system property, independent
of working directory. Fixtures cover base math, projection, decoding and rejection,
not real world raycasts or full live behavior.

## Live Policy Duels

Two real actors share a checkpoint policy and use vanilla movement/targeting/
combat, never simulator physics or recorded future states. Python inference runs
on the same machine, loopback-only `127.0.0.1:8766`. Trusted standalone PPO formats
only; v1/v2/custom descriptive contracts/loadouts share the adapter. Unsupported
primitives/items fail before spawning. [Contract](../docs/simulation_contract.md).

Launch `./scripts/serve-policy.ps1 -List`, then `./scripts/serve-policy.ps1`.
CPU default supports GPU-trained weights without native DLL. Python requirements
are needed, not a native build when using somebody else's checkpoint.
`-Device cuda`, `-Port`, `-Seed`, `-Threads`, `-Deterministic`, `-Checkpoint` and
`-RunsDirectory` customize service. Default sampling is stochastic; print selected
checkpoint. Weights load once; restart for newer ones.

### Checkpoint Arena

`/simduel arena` builds the embedded checkpoint arena near the current origin,
uses its spawns and starts play. Default is normally enclosed 5x5. Snapshot/restore
blocks like replay; round resets rebuild arena. Choose a loaded area without
containers. Starting live replaces loaded replay and vice versa.

### Existing World

`/simduel pos1`/`pos2` mark current feet position and look direction. Absolute
`x y z` may be supplied, still capturing current yaw/pitch. Move clear/spectator
before `/simduel world`. Both marks belong to the current singleplayer session/
dimension. Validate clear, loaded feet positions within height/border; probe
ground contact at spawn. World mode never builds/restores terrain: resets reuse
marked spawns/current terrain, not arbitrary world/entity/hazard side effects.
Small-flat-arena policies may fail on stairs/water/cliffs/special blocks/distances.

| Command | Behavior |
| --- | --- |
| `/simduel arena [port]` | Embedded arena/spawns. |
| `/simduel pos1 [x y z]`, `/simduel pos2 [x y z]` | Mark feet/look. |
| `/simduel world [port]` | Both marks, no arena construction. |
| `/simduel join [1\|2]` | Replace default slot 1 or slot 2; reset round. |
| `/simduel leave` | Restore participant and return to bot versus bot. |
| `/simduel pause`, `/simduel resume` | Pause/resume controlled steps, discard pending actions/request fresh observations. |
| `/simduel reset` | Reset both at original spawns, resume. |
| `/simduel status` | Round/tick, health, wins/draws, request-to-application latency. |
| `/simduel stop` | Cancel pending work, remove actors, restore arena-mode blocks. |

Death, arena exit (arena mode), falling below world height or checkpoint tick
limit ends round; reset both after 20 server ticks. World mode has no artificial
X/Z boundary. Bots do not drop issued gear. Observers may punch/damage/knock back
bots without joining; bots still target each other, not observer.

### Fight The Bot

Join either arena/world duel with `join`/`join 2`. Human normal movement/attacks
replace that actor; opponent observes/targets the actual player. Reset both,
temporarily switch participant to survival, teleport to spawn, equip checkpoint
loadout. Weapons use hotbar, shields offhand.

Save once per participation: inventory/armor/offhand/selected slot, game mode,
position, health, absorption, food. Restore on leave/stop/replacement/normal
disconnect. Round resets keep slot and fresh kit. Count lethal damage as loss
without killing/dropping real inventory. During pause/reset delay, participant
cannot take damage but movement is not frozen; hunger freezes during participation.
World interactions, dropped items, effects and other side effects are not restored.
An in-memory snapshot cannot survive a crash; this is not a complete backup.

### Observation And Request Invariants

- Actual entity state and eight horizontal world-relative OUTLINE rays to 50
  blocks match training feature order. Both observation radii must be loaded;
  pause rather than force-load chunks. Custom-world vanilla block shapes differ
  from full-cube training. Other entities/rules still apply; PvP must be enabled.
- Freeze bots' food/both natural-regeneration paths; never change world rules.
  Human game mode changes temporarily, but Peaceful may still regenerate humans.
- HTTP is asynchronous with two-second timeout and session/tick tags. Apply at
  most one action pair/server tick. Waiting neither repeats prior inputs nor
  blocks server thread. Timeout/error pauses; fix issue then resume.
- Latency can reduce controlled actor tick rate; other entities/world clocks
  do not pause. Server-side adapter is not a networked `LocalPlayer` client.
- Bridge/schema/action/report/JAR checks are not live validation. In-game checks:
  both join slots, weapon/shield, loss/reset, pause/resume, leave/stop, disconnect
  restoration and observer punches during bot-versus-bot.

## Rosie Artifact Commands

Configure locally using `scripts/rosie-play.ps1 -Setup` and install the updated
mod once. See [setup/authentication](../rosie/README.md#local-replays-and-duels).
`RosieLauncher` reads only local game `config/cambium-rosie.json` and invokes
the configured Python module using literal `ProcessBuilder` arguments, never
a shell. Configuration is trusted executable configuration, not server input.
All commands are restricted to private singleplayer. Network/process work runs
on a virtual thread; JSON result handoffs return to the client/server controllers
and are guarded against superseded operations and changed worlds/owners.

| Command | Behavior |
| --- | --- |
| `/simrosie runs` | List saved remote runs, newest first; click a row to suggest replay command. |
| `/simrosie replays [run]` | List that run's samples; click to select one. |
| `/simrosie replay [run] [file]` | Download/cache one replay, load and start absolute playback. |
| `/simrosie duel [run]` | Download/cache weights, start local CPU policy on a free loopback port, start arena. |
| `/simrosie sync [run]` | Cache checkpoint plus all samples without changing the world. |
| `/simrosie status` | Current helper operation. |
| `/simrosie stop` | Cancel helper; stop its live duel and restore participant state; retain cache. |

Omitted run/file means `latest` by modification time, never best. Only one helper
operation runs at a time; stop it before switching. `/simduel join` remains the
participation command. `/simduel stop` alone stops actors, not the helper-owned
policy process; use `/simrosie stop` for both. Disconnect, client shutdown, world/
dimension change or opening to LAN cancels the owned helper and SSH descendants.
Python stderr goes to game `logs/rosie-launcher.log`; readable errors appear in
chat. Terminal-launched inference processes are not owned or terminated by the mod.
Headless fixtures check process arguments, not real SSH or Minecraft lifecycle;
manually test playback, join/stop restoration, cancellation and disconnect cleanup.

## Simulation Playback

Export with `./scripts/export-replay.ps1 -Ticks 80 -ToMinecraft`; this is scripted,
not learned policy (training's `-ReplayEvery` samples learned policy). `-Output`
selects another file; `-Device cuda` needs CUDA runner, not GPU for viewing.
[Export options](../scripts/AGENTS.md#replay-and-analysis).

Put JSONL in game `sim-replays` alongside `mods`. In private test world stand
above desired origin and load. `latest`/bare load choose newest last-modified
JSONL with any filename prefix. Quoted names/absolute paths select a file.
Embedded arena is reconstructed without original environment JSON; spawn two
actors, pause at frame zero, absolute mode, speed 1. Envelope 3/equipment and
legacy sword-only 2 supported; version 1 must be re-exported.

| Command | Behavior |
| --- | --- |
| `/simreplay load latest`, `/simreplay load [file]` | Load/reconstruct/spawn paused at frame zero. |
| `/simreplay absolute` | Reset paused, force recorded motion/orientation/pose/health; animate attacks without recomputing damage. |
| `/simreplay input` | Reset paused, seed once, apply controls to vanilla; never force later position/health. |
| `/simreplay analyze` | Reset input mode, start/export comparison at chosen speed; finish at final frame. |
| `/simreplay analyze stop` | Finalize partial report without stopping playback. |
| `/simreplay pause`, `/simreplay resume` | Freeze/resume frames, not rest of world. |
| `/simreplay step` | Pause/advance one recorded tick. |
| `/simreplay restart` | Rebuild actors/arena at frame zero, clear fractional progress, resume. |
| `/simreplay reset` | Same, paused. |
| `/simreplay speed 0.1` | Range 0.05..8; 1=20 replay ticks/sec. |
| `/simreplay stop` | Remove actors/restore original block states. |

Reset/restart/mode switches reuse loaded memory without rereading, preserve speed.
Reset/reload/mode changes/stop also finalize partial analysis.

Action bar: tick/mode/rate/buttons/actual health; absolute mode adds recorded hit
outcomes; input adds positional error in blocks. Swings visualize attempts;
default player skins.

Input reports first target-selection and first state difference once/reset in
chat/log. `logs/latest.log` includes filename/tick/actor/actual/expected. Target
details include ray origin/direction, motion-adjusted reach, opponent bounds,
eye-inside. Position/velocity vector threshold 1e-6; health/absorption 1e-5.
Earlier movement drift may cause later ray differences. Do not correct/pause on
these diagnostics. Sprint input runs in movement phase after attacks use carried
sprint state, matching simulator order.

### Accuracy Analysis

`load latest`, then `analyze` compares recorded CPU/GPU results against vanilla,
never reruns CUDA or substitutes later reference states. Pause/resume/step/speed
work during analysis. Reports are unique under game `sim-replays/analysis`;
chat paths copy on click. Share both:

- `.summary.json`: complete/partial/failed, source metadata, compared ticks,
  actor-tick mismatch counts, first movement/target/combat/state difference,
  per-actor/field mean/max absolute error, worst/first ticks. Counters/flags exact;
  motion tolerance 1e-6, health/damage/angles 1e-5.
- `.jsonl`: source header/environment/initials, every tick's controls/reference/
  actual/differences, before-prepare/before-attack/after-combat snapshots, ray
  origin/look/reach/hit positions/bounds/eye-inside/near-coincident-knockback,
  acceptance and per-attack motion.

Coordinates are replay-local; frame zero unscored. Keep reference crit/sprint/
sweep flags, but never guess/score vanilla counterparts. Reference only contains
post-tick states, not matching intermediate phases. No single accuracy percentage.
Earlier drift can explain later target differences.

Freeze actor food/both regen paths even in Peaceful without changing world rules.
Report world regeneration and effective `actor_natural_regeneration=false`/
`actor_food_simulated=false`. Set maximum absorption before seeding; include
`max_absorption` to distinguish seed clamping from damage. No absorption effect
lifecycle implied. Other entities/changes and adapter sprint/network/RNG limits
still apply. Stream long reports to disk; pause does not finalize, `analyze stop`
does. Normal cleanup flushes partial reports; crash may leave only flushed JSONL.

### Arena Restoration And Limits

Initial translation: `(floor(viewer.x), floor(viewer.y)-1, floor(viewer.z))`.
Restart/same-dimension switches retain origin. Stop/move/reload to choose another.
Replace all cells in bounds, including air (default 7x4x7 including floor/walls).
Snapshot original block states before edits; restore on stop/reload/normal cleanup.
Reject containers/block entities, unloaded chunks, invalid height/border before
initial edits. Do not build in loaded arena. Memory snapshots cannot restore
after crash; time/entities/side effects are not rolled back. Viewer game mode/
rules remain unchanged. Observe outside arena or in spectator to avoid contact.

Two players, supported configured swords/axes/empty hand, armor/offhand shields;
unbreakable resolved gear, JSONL <=64 MiB, same
[bounded environment format](../simulator/AGENTS.md#json-environments). Absolute
applies gear/use state; input feeds semantic controls without forcing damage.
PvP required. Attacks target replay players; other entities are allies for sweep
filtering. Cleanup on stop/reload/dimension change/open-to-LAN/server shutdown.

Input is diagnostic, not full network-client parity: no full local sprint
eligibility, arbitrary packet timing, RNG reconstruction or unsupported kits.
Block clicks do not mine. Playback clocks differ from world/client animation
clocks. Absolute forced-state mode cannot validate physics. Build/export/reader
checks do not replace in-game spawning/rendering/playback verification.

## Recording

F9 starts/stops only with no menu open. Rebind under Options > Controls > Key
Binds > Cambium > Toggle Movement Recording. Use matching floor/wall world,
start jump released/cooldown expired, preferably at rest. Walk/sprint/jump/crouch/
turn normally; never replace controls. Chat reports step count and absolute CSV
path, clickable to copy already quoted for comparator input.

Files live in game `movement-recordings`, not repository. Unique UTC+UUID CSV
and `.csv.json` sidecar; never overwrite. Buffered writer closes on toggle,
disconnect/player/world change/normal shutdown; crashes can lose buffered data.

```powershell
.\simulator\build\Release\minecraft_sim_compare.exe "C:\path\to\instance\movement-recordings\movement-....csv"
```

MinGW executable lives directly in its build tree, not Release. Use actual game
path. [Comparison contract](../simulator/AGENTS.md#recorded-physics-comparison).

## Capture Format

[LocalPlayerRecordingMixin](src/main/java/net/cambium/mixin/LocalPlayerRecordingMixin.java)
captures before-state at `LocalPlayer.aiStep` entry, refreshed inputs at inherited
movement call, result at `LocalPlayer.tick` return after pose updates. Pair used
angles/buttons with resulting state, not server or render interpolation.

- `tick` is local sequential step, not server time. Row zero seeds before-state,
  unscored; later rows include post-movement motion/ground/collision.
- Existing comparator mask/buttons/yaw/pitch format; use `--input-phase current`.
  Java float/double round-trip text, independent of decimal locale.
- Schema 2 adds pose (0=stand,1=crouch,2=dry crawl), prior forward/sneak, sprint
  timer/window, food/flight, fall distance/minor collision. Seed bookkeeping
  only from row zero; freely compare later sprint/crouch/pose. Jump delay and
  before-state diagnostics remain; start with zero jump cooldown.
- Sidecar identifies schema/game/dimension/phase, not terrain/attributes.
  Reconstruct matching world/default attributes separately.
- Stop before a step if position/velocity/ground/pose/jump delay changed since
  last post-state. Flags possible corrections/teleports/external forces without
  silently incorporating them, not proof of a server correction's cause.
- Supports ordinary land/sprint/jump/crouch/ledge/dry crawling. Flight/other
  poses/liquids/riding/item use/climbing/effects stop with chat reason. Special
  terrain/step-up unsupported. Single-player CSV lacks other entities/external
  impulses and cannot validate contact/combat. Hunger evolution/nondefault
  attributes are not reconstructed.

Historical 250-step ordinary client capture matched exactly. Synthetic schema-2
writer/parser replay passed, but expanded movement/hook boundaries require fresh
captures. A build proves neither timing/keybinding nor general parity.