# Bow and Arrow Integration

Ordinary bows are available to descriptive PPO policies, CPU/CUDA training,
replay, and the live Fabric adapter. The public C ABI remains version 3: its
buffers are described by the negotiated policy. Rebuild native libraries and
the mod after the catalogue change. Legacy 22/31-input layouts retain their
feature order and dimensions, but cannot supply new bow inputs.

## Controls and State

`use_item` is a boolean right-click head: hold to draw, release to shoot.
The historical `shield` head is an alias; a policy must not contain both.
Selected equipment determines the behavior. Selecting another physical slot
cancels a draw before processing button-up, without firing or consuming ammo.
The internal `BowCommand` enum is not a PPO head.

Loadouts accept `minecraft:bow` and optional `arrows: 0..64` (default 0),
representing one finite ordinary-arrow stack. Live kits place ammo outside the
hotbar; human inventory restoration retains the participation snapshot.

Each world owns `ProjectileWorldState` and a 128-slot arrow buffer. Full and
selective resets clear the selected worlds' draws, ammo, arrows and RNG only.
`processWorldTick` cancels switched slots, resolves release against pre-look,
pre-movement shooter state (matching `ReplayPlayer.prepare`), advances players,
then advances previously spawned arrows. New arrows first move next tick.
Release suppresses that sample's melee attack. Actual drawing applies item-use
movement slowdown and prevents sprint start.

## Observations

Catalogue indices 89..101 are appended, never inserted into legacy layouts.
Bow-specific observations require the `bow` capability:

| Features | Units and availability |
| --- | --- |
| `self.arrow_count_fraction` | Own ordinary arrows / 64; scalar, not one-hot |
| `self.bow_using`, `self.bow_equipped` | Actual use state and selected bow |
| `self.bow_power_fraction` | Actual draw power, 0 when not drawing |
| `opponent.bow_equipped`, `opponent.bow_using` | Visible equipment/use, not opponent ammo or charge |
| `arrow.present` | Nearest moving, nonembedded opponent-owned ordinary arrow within 50 blocks |
| `arrow.relative.x/y/z` | Arrow minus own feet position, divided by 10 blocks |
| `arrow.relative.velocity.x/y/z` | Arrow minus own world-axis velocity, blocks per tick |

All seven arrow values are zero when absent. Equal distances use stable native
slot order; the live adapter uses entity IDs. These are state observations like
existing opponent positions, not a pixel/line-of-sight visibility model.

## Rewards and Replay

Delayed arrow hits add owner-attributed health removed to `doing_damage` and
`taking_damage` once per tick; blocked, immune and absorption-only hits do not
earn health-damage rewards. Self-hits retain damage-taken penalties but earn no
damage-dealt credit and do not satisfy the episode's damage requirement.
Melee outcomes remain separate. Existing terminal
win/loss rules apply; no general vanilla kill-credit system is claimed.

Envelope 3 optionally carries actor `arrow_count`, `bow_using`, `bow_draw_ticks`,
`bow_power`, and frame `projectiles` with ID, owner, position, velocity, age,
owner-clearance, embedded lifetime and critical state. Older replays default to
no arrows. Dashboard playback interpolates only continuous arrow identities;
Fabric absolute playback uses non-damaging display entities. Live play releases
real vanilla arrows; round cleanup is restricted to captured owned spawns.

## Mechanics Scope

The supported slice is ordinary unenchanted bow shots with finite ordinary
arrow counts, two standing PvP-enabled players, static ordinary solid full cubes,
air flight, and existing player damage/armor/absorption/invulnerability/shield
state. Charge uses Java's float formula; each `hold` call contributes one held
tick. Ammunition is consumed only when a slot is allocated. A full buffer
returns `capacity_full` without consuming ammunition. Bow use is rejected while
the player is using a shield. Arrow spread and critical draws use explicit
`LegacyRandom` state for reproducibility; Minecraft's per-projectile seed
assignment is external and is not claimed to match.

The tick phase clips the swept centerline against full-cube terrain before
player boxes, applies the arrow age-ramped entity margin and owner clearance,
then applies air drag/gravity or embedded lifetime. Successful hits reuse
`PlayerView::hurt` as arrow damage, including armor, damage cooldown, absorption,
shield arc/use delay and projectile-direction default knockback. Rejected hits
reverse velocity at one-tenth speed; the random display rotation is consumed but
not represented as render state.

Not implemented: general inventory lookup/stack semantics, bow durability, client
prediction/packets, teams or PvP rejection, mounted/shared-vehicle owners,
non-standing players, entity iteration tie ordering beyond the two-player
contract, exact arrow entity-size/broadphase edge behavior, pickup, water/fluid,
fire, tipped/spectral arrows, enchantments, portals, special blocks, and a
Minecraft-seed-compatible projectile RNG.
Geometry endpoint tie behavior inherits the existing native `rayBoxHit`
contract. Comments above mechanics methods cite the controlling supplied
Java 26.3 code; [the source reference](bow_arrow_source_reference.md) gives detail.

## Sniper Setup

[The sniper curriculum](../simulator/setups/curricula/colten/sniper-duel.json)
uses one 41-input, six-head layout across stationary target, moving target and
scripted/frozen-self-play duel stages. Players spawn 32 blocks apart with
64 arrows each and side cover. `good_aim: 2` enables at most two points per full
episode for bow aim, normalized to each stage's tick limit. The full-draw
intercept heuristic accounts for air drag, gravity, target velocity and inherited
shooter velocity; it requires ammo and a clear eye-to-eye line of sight. It is a
directional hint, not a hit guarantee or a prediction of future target actions.
Melee proximity and level-pitch shaping remain disabled. `archer` is a scripted 20-tick draw/release controller
with heuristic lead/drop, not a learned policy or independent physics reference.
Start a fresh run for the new layout.

## Verification

From `gpu-sim-minecraft`, configure/build/test with:

```sh
cmake -S simulator -B simulator/build -DBUILD_TESTING=ON
cmake --build simulator/build --target minecraft_sim_projectile_checks
ctest --test-dir simulator/build -R '^projectile_checks$' --output-on-failure
```

With CUDA enabled and a usable device, also build and run
`minecraft_sim_projectile_cuda_checks`; its CTest entry skips with code 77 when
no GPU is available. It compares the same seeded launch and air-flight inputs
on CPU and CUDA; it does not cover every collision or damage interaction.
Training adapter checks also cover use, cancellation, delayed damage, replay
and selective resets on CPU and CUDA. Java headless checks and the explicit
Java/Python HTTP bridge exercise the bow descriptor and finite kit. CPU/CUDA
agreement alone is not a Minecraft parity claim. The mod is built, not installed
or verified in-game; deployment needs a selected private client instance.
