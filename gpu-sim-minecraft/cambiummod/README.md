# CambiumFabric

Movement recording, replay comparison and live policy duels in Minecraft Java
**26.3**. Client-installed Fabric mod for private singleplayer, not dedicated
servers. Use a disposable test world.

## Installation

Install JDK 25, Fabric Loader 0.19.5+ and matching Fabric API for Minecraft 26.3.
Close Minecraft, then run from the repository root:

```powershell
$env:MCSIM_MINECRAFT_DIRECTORY = Read-Host 'Minecraft game folder (containing mods)'
.\scripts\build-mod.ps1 -Install -DryRun
.\scripts\build-mod.ps1 -Install
```

The helper does not install Fabric/API. Remove differently named older Cambium
JARs to avoid duplicates, then reopen Minecraft. No particular launcher or instance
name is required. [Game-folder selection](../scripts/README.md#minecraft-folder).

A distribution JAR with a bundled runtime needs no Python installation for local
policy duels. Developers must [build the runtime](AGENTS.md#bundled-inference)
before building that JAR. Runtime bundles are specific to their OS and architecture.

## In-Game Panel

Press **F8** or use `/simmenu` in a private singleplayer world. After the
[one-time local Rosie setup](../rosie/README.md#local-replays-and-duels), the
**Rosie** tab lists runs and syncs policies only, policies plus the latest replay,
or all outputs, with transfer progress. **Library** groups policies by run, with
a version selector and local-only run/policy display names. **Import** adds a
trusted compatible `.pt` checkpoint and its embedded environment without Rosie.
In **Duel**, choose an environment and independently select a policy or scripted
controller for each fighter, set the round duration, then start. Synced setups
run locally while offline. **Stop match** keeps Python/PyTorch warm for the next
launch; **Shutdown inference** releases it. **Live** also has pause/resume/reset
and join/leave controls. The debug
HUD places P1 left, P2 right and match status centrally, with health/charge/time
bars and gear icons. Toggle it there or with `/simduel hud`, never the action bar.
Selections and aliases persist in the game folder. Existing commands still work.

## Live Policy Commands

Start inference from a trusted checkpoint and leave it running:

```powershell
.\scripts\serve-policy.ps1 -List
.\scripts\serve-policy.ps1
```

In-game, `/simduel arena` starts two policy-driven fighters. `/simduel join`
replaces player 1 with you; `join 2` takes the other slot. `leave` restores your
saved player state. `pause`, `resume`, `reset`, `status` and `stop` control the duel.

To use existing terrain, mark `/simduel pos1` and `/simduel pos2` at the desired
feet positions/look directions, then `/simduel world`. Use a command port argument
to match nondefault inference ports, e.g. `/simduel arena 8768`.

Default inference is CPU at **127.0.0.1:8766**, including GPU-trained checkpoints.
Restart the service to load newer weights. [Full commands and limits](AGENTS.md#live-policy-duels).

## Rosie Outputs

After [one-time local setup](../rosie/README.md#local-replays-and-duels), use
`/simrosie runs`, `/simrosie replay <run>` or `/simrosie duel <run>` to fetch
artifacts over SSH and play locally. `/simduel join` lets you fight the model;
`/simrosie stop` stops the match but keeps ready inference warm;
`/simrosie shutdown` also terminates the helper. No remote inference server is needed.

## Simulation Playback

```powershell
.\scripts\export-replay.ps1 -ToMinecraft
```

This exports a scripted duel. For learned-policy samples use training's
`-ReplayEvery N -ReplayToMinecraft` instead. In-game:

```text
/simreplay load latest
/simreplay restart
```

`pause`, `resume`, `step`, `reset`, `speed` and `stop` control playback.
`input` runs controls through vanilla mechanics; `absolute` displays recorded
states. `analyze` writes comparison reports under the game's `sim-replays/analysis`.
Share the summary JSON and detailed JSONL together. [Playback reference](AGENTS.md#simulation-playback).

## Recording

Press **F9** to start/stop movement capture in the matching floor/wall test world.
Start at rest with jump released. Click the saved CSV path in chat to copy it;
files live in the game's `movement-recordings` folder, not this repository.
[Capture and comparator details](AGENTS.md#recording).

## Safety And Development

Arena modes replace blocks and restore them on normal cleanup, not after crashes.
World mode and human participation do not roll back all world/player side effects.
Do not use valuable worlds. Absolute playback cannot validate physics, and input
playback is diagnostic rather than proof of full client/server parity.

[AGENTS.md](AGENTS.md) holds implementation details, command tables, capture
formats, restoration rules and verification limits. Template license: [CC0](LICENSE).