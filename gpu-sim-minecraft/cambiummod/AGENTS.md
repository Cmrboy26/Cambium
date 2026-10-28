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

## Bundled Inference

Build the helper on each target OS/architecture using a CPU-only PyTorch Python
environment and the pinned packaging dependency. From the repository root:

```powershell
python -m pip install -r simulator/training/requirements-inference-build.txt
python -m simulator.training.build_inference
./scripts/build-mod.ps1
```

The build stages a PyInstaller executable and SHA-256 under
`cambiummod/build/runtime-bundle/`; Gradle embeds staged platform bundles in the
JAR. This generated directory is not shared across fresh checkouts. Build each
target on that target OS/architecture and stage its bundle before assembling a
multi-platform JAR. A normal mod build without a bundle retains the configured
Python helper but is not a standalone distribution. The Windows x86-64 CPU
bundle is approximately 180 MB. `--dry-run` prints the packaging command.

For local import and duels, the mod prefers the matching embedded executable,
extracts it into a checksum-named directory under the game folder's
`sim-runtime`, verifies it and runs it over loopback. Missing bundles fall back
to an existing configured Python helper. Rosie commands still require the local
Rosie configuration and SSH setup. The helper loads trusted standalone PPO
checkpoints only; `weights_only=True` is not a sandbox for untrusted files.
Only the selected concrete checkpoint environment is imported, not the training
curriculum. Validate each distributed platform and an in-game duel separately.

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

Two real actors use vanilla movement/targeting/combat, never simulator physics or
recorded future states. Legacy service mode shares one checkpoint; panel duels
select independent policy/scripted controllers with one common semantic layout.
Python inference runs
on the same machine, loopback-only `127.0.0.1:8766`. Trusted standalone PPO formats
only; v1/v2/custom descriptive contracts/loadouts share the adapter. Unsupported
primitives/items fail before spawning. [Contract](../docs/simulation_contract.md).

The mod consumes only concrete `minecraft_sim_environment` version 1 and the
resolved policy descriptor. It does not parse split experiment authoring
references, distributions, or training plans. Compiled loadouts have nine
physical hotbar slots; repeated air is allowed, duplicate nonair is rejected.
Historical concrete loadouts with fewer slots remain supported.

Optional `/info.experiment` deployment metadata supplies experiment name,
stage name/index, environment and arena names, `controllers=shared_learner`,
and `sampling=fixed_checkpoint_episode`. Startup and `/simduel status` show
the supplied experiment/stage/environment (displaying the zero-based stage
index as one-based). Missing metadata remains compatible with historical
services. Live rounds run the fixed embedded checkpoint scenario with a shared
learner, not a curriculum, opponent pool, or reset-time distribution sampler;
world mode still substitutes its marked spawns/current terrain.

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
| `/simmenu`, `/simduel menu` | Open the F8 native control panel. |
| `/simduel hud` | Toggle the dedicated live/replay sidebar; persist preference. |
| `/simduel arena [port]` | Embedded arena/spawns. |
| `/simduel pos1 [x y z]`, `/simduel pos2 [x y z]` | Mark feet/look. |
| `/simduel world [port]` | Both marks, no arena construction. |
| `/simduel join [1\|2]` | Replace default slot 1 or slot 2; reset round. |
| `/simduel leave` | Restore participant and return to bot versus bot. |
| `/simduel pause`, `/simduel resume` | Pause/resume controlled steps, discard pending actions/request fresh observations. |
| `/simduel reset` | Reset both at original spawns, resume. |
| `/simduel status` | Round/tick, health, wins/draws, latency, optional deployment stage and fixed-scenario scope. |
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
- Weapon-slot observations read the actual inventory selected slot for bots
  and humans, including any empty slot; do not infer it by matching item IDs.
  Descriptor decoding carries the semantic slot in controls `select_slot`.
  Bots keep the resolved hotbar and select that physical slot. Charge resets
  only when the selected item changes, not when switching between empty slots.
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
Remote commands request automatic session reuse. On Windows the helper opens a
visible login terminal if needed; passwords/MFA stay there, never in Minecraft.
An established login terminal remains open for reuse after an operation finishes;
close it separately with Ctrl+C. Offline local commands never start SSH.
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
| `/simrosie sync [run]` | Cache policies/config only; panel offers latest/all replay scopes. |
| `/simrosie status` | Current helper operation. |
| `/simrosie stop` | Stop the match and restore participant state; retain ready inference and cache. Cancel unfinished startup/downloads. |
| `/simrosie shutdown` | Stop the match and terminate the owned helper; retain cache. |

Omitted run/file means `latest` by modification time, never best. Only one helper
operation runs at a time; shut it down before switching to another remote helper.
Local matches reuse retained inference. `/simduel join` remains the
participation command. `/simduel stop` alone stops actors, not the helper-owned
policy process; use `/simrosie shutdown` for both. Disconnect, client shutdown, world/
dimension change or opening to LAN cancels the owned helper and SSH descendants.
Python stderr goes to game `logs/rosie-launcher.log`; readable errors appear in
chat. Terminal-launched inference processes are not owned or terminated by the mod.
Headless fixtures check process arguments, not real SSH or Minecraft lifecycle;
manually test playback, join/stop restoration, cancellation and disconnect cleanup.

### Native Panel And Local Library

F8 is rebindable under Cambium Menus. `ControlPanel` uses Minecraft screens and
widgets, paginated searchable lists and tooltips for clipped names. It does not
pause the world. Command opening is deferred to the client tick because chat
closes its screen after submitting a command. Private-world checks apply again
before opening. Closing the panel leaves the current duel running.

The Rosie tab explicitly lists remote runs and syncs a selected one. Its scope
selector defaults to policy only (latest checkpoint, retained snapshots and
config); latest replay adds one sample, all adds every replay and metrics.
Three bounded concurrent downloads report per-file preparation, transfer and
verification, aggregate bytes/files/cache hits, then separate library indexing.
The bar measures bytes processed, including unchanged files reused from cache;
indexing can continue after the transfer bar reaches 100%.
Opening the
panel, browsing cached entries and starting a local duel never contact Rosie.
Library Import selects a local `.pt`, validates it in the offline helper and
copies it into its own cache entry with its policy contract and environment.
It requires no Rosie configuration when a matching bundled runtime is present.
Login still happens in the local terminal connection flow, never a password UI.
Helpers retain generation/world/owner guards, literal process arguments and
disconnect/LAN/shutdown cleanup. Live Stop Match stops actors and keeps a ready
owned inference process warm. Shutdown Inference on Live/Rosie also kills the
owned helper. Externally served actors stop without killing their external server.

Game `sim-rosie-cache/library.json` version 1 stores readable policy/environment
entries. Policy IDs use `run:latest.pt` or `run:snapshots/<name>.pt`; each has a
local checkpoint path, update/stage, semantic contract hash and policy descriptor.
Environments include checkpoint scenes and resolved stage/arena templates from
synced `config.json`. These are fixed templates, not distribution sampling or a
live curriculum. The index stores structurally equal environments once per
contract hash and retains source IDs/names as aliases. The duel selector resolves
saved IDs through those aliases, including IDs from older version-1 libraries.
Genuinely different geometry, loadouts, duration or policy layouts remain
separate and display their useful variant details; source labels disambiguate
distinct setups with the same name. Re-indexing needs no library version bump.
Sync preserves other runs and existing policy aliases.
Library rows select runs; the bottom selector chooses Latest or a retained policy
version. Local run and per-version display names are independent of remote IDs,
paths and files; sync preserves both. Renaming is disabled while the helper runs.
Cached checkpoints must exist inside
the cache's real path; mismatched contracts are rejected before starting.
Arena environment candidates must match the semantic contract hashes of both
selected policy fighters; scripted fighters add no layout constraint. World
mode does not select a cached environment preset, but two policy fighters must
still share one contract.

`selections.json` stores both fighter modes/policy IDs, scripted options, chosen
environment, arena/world mode, sync scope, optional round duration, and World
mode's per-fighter nine-slot hotbar, selected slot, offhand and armor. Duration
uses seconds in the panel, rounded to twenty controlled ticks per second;
positive limits up to 1,000,000 ticks are supported. Arena may use Environment
Default; World requires a chosen duration and both current-dimension spawn marks.
Changing selections affects the next launch, not the running round. Inference
delays can extend wall-clock duration.
`duel.json` is a `minecraft_sim_duel` version 1 configuration with two fighter
entries. Arena keeps its concrete cached policy/environment; World carries the
shared policy descriptor, explicit duration and loadouts in a runtime-only
empty scenario. Marked positions replace its placeholder spawns before actors
are created. World terrain is not edited.
Launching invokes the configured local interpreter/repository with Rosie action
`local --duel <file> --offline`; CPU inference uses a free loopback port and no
native simulator library. `/simrosie local` starts the last saved arena setup.
Scripted controllers include Basic melee, Stationary melee, Tactical melee,
Random walk and Stationary target. Basic melee options cover attack/strafe
intervals, melee distance, strafe, sprint, shield and optional zero-based
axe/sword slots. Stationary melee keeps attack timing/aim/equipment options but
does not move. Tactical melee adds evasive/aggressive/patient style selection;
Random walk exposes direction-change interval and wall margin. Unsupported
features/actions are rejected by the service for the selected policy contract.
Stationary target requires no options. No arbitrary Python factories or training
opponent pools are executed.

Cold startup reports PyTorch import and fighter loading with elapsed time.
Online `/simrosie duel` still reads remote metadata; the library path never
connects to SSH. Replacing a library duel keeps the owned Python process warm
and POSTs `{}` to its authenticated `/reload` route. Only its configured local
duel file is read. The new controllers are validated before swapping; failure
keeps the prior service configuration. Stop Match retains the process, reload
port/token and lifecycle monitor, but invalidates in-flight reload handoffs so
they cannot restart actors after Stop. The next local launch reloads fighters
into that same warm process. Repeated Stop is safe; unfinished cold startup or
downloads are canceled rather than retained. Shutdown Inference and disconnect,
world/dimension changes, LAN sharing and client exit still kill the helper.
Warm-reload handoffs check generation, reload generation, world, server and owner.
The service occupies the helper slot until explicit shutdown, so remote sync/list
operations require shutting it down first. Fresh launches after shutdown pay
cold import cost; no startup speed is guaranteed across machines. Status reports
cold-start or warm-reload timing.

`DuelHud` publishes copied server-tick telemetry to a Fabric HUD layer: P1 at
left, P2 at right, shared round/time/score/latency centrally. Health, absorption,
attack charge and remaining-time bars are bounded. Native item rendering shows
main/offhand, helmet, chest, legs and boots. Columns adapt to GUI width; long
labels are clipped with ellipsis. It hides with vanilla HUD or an open screen.
The Live tab retains textual details (including replay errors/hit outcomes) from
the same snapshot. Preference lives in `config/cambium-hud.json`; disconnect or
session close clears telemetry. No live/replay telemetry uses action-bar text.

Validation: `replayAnalysisChecks` covers local duel files, aliases, cache bounds
and offline helper arguments; `InferenceChecks`/`RosieTests` cover per-side
controllers, offline service startup and snapshot/library sync. Actual F8/command
opening, GUI scales, HUD rendering and join/reset/stop/disconnect restoration
still require Minecraft testing; headless compilation does not prove them.

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

Sidebar: tick/mode/rate/buttons/actual health; absolute mode adds recorded hit
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

Envelope 3 accepts optional actor `weapon_slot` and controls `select_slot`.
Explicit indices must match the concrete loadout/item; frame-zero selection
must match `loadout.selected`. `select_slot` is -1 when no slot is requested.
New native training frames preserve exact empty-slot identity in both fields;
absolute playback and input playback honor it, and analysis compares supplied
slot state against actual inventory selection. Historical frames without these
fields use -1 internally and retain item-based fallback: keep a matching
current slot, otherwise choose the first matching item. Unknown historical
slot identity is not scored as an actual slot mismatch.

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