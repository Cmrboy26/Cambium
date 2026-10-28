# Mace and Wind Charge Implementation Plan

**Status:** In progress, not feature-complete. Native unenchanted mace damage,
accepted-hit bounce, fall-distance reset and player-only splash are implemented;
mace loadouts are recognized by native, training and Fabric validation. Ordinary
landing damage and mace impulse protection are implemented behind the optional
`simulate_fall_damage` rule, with default player attributes and full-cube blocks;
their game parity and live-world rule compatibility remain unverified. Wind launch,
full-cube hit, blast and isolated per-world projectile ticks have focused checks,
including CUDA buffer relocation and selective reset. Wind storage is not allocated
by the application or training adapter, and item use is not connected. Full-cube
hit face, inside-block, short-ray and endpoint
fixtures were also checked against Minecraft's `VoxelShape.clip`; multi-block
traversal and live impacts remain unverified. Focused source-derived mace and wind-helper checks and
fall-damage training resets passed on CPU and an RTX 3060 Ti CUDA device; no
live round or Minecraft capture has verified these mechanics. See the
[scope](mace_smash_wind_charge_spec.md), [reference](MACE_FORMULA_REFERENCE.md)
and [research checklist](MACE_WIND_RESEARCH_PLAN.md).

## 1. Resolve the Remaining Vanilla Gates

Trace `Player.attack`, `ItemStack.hurtEnemy`/`postHurtEnemy`, target damage
return values, damage-source tags and shield/cooldown behavior. Determine which
effects run for blocked, immune, rejected, absorption-only and lethal hits.
Verify the implemented creation, replacement and clearing of fall-damage impulse
immunity, including repeated smashes and landing. Record exact tick phase, numeric types
and independently derived fixture expectations. Capture matching Minecraft
26.3 cases; do not infer game parity from simulator-only tests.

## 2. Implement Mace Mechanics

1. In `simulator/include/minecraft_sim/core.hpp`, integrate the item bonus into
   the existing charged melee and critical pipeline, without replacing ordinary
   melee processing. Preserve attack-target selection, damage-source distinctions,
   hit outcome, and success gates.
2. Implement sourced smash hit effects and impulse fall protection with their
   full reset lifecycle. Preserve fall distance until bystander knockback and
   all other consumers have used it. Compare attack-before-movement behavior
   with the real client/server ordering instead of assuming they match.
3. Use a world-aware operation for bystander splash. The direct target still
   receives ordinary melee effects; a third player is necessary to test splash.
   Keep fixed storage and host/device compatibility. Do not silently use the
   ordinary two-dimensional living-entity knockback routine for the splash push.
4. Add independent expectations to `simulator/tests/core_checks.cpp`, including
   boundary values below, at and above 1.5, 3, 5 and 8; non-smash, partial and
   full charge; critical and sprinting; shield, invulnerability, damage
   cooldown, absorption and death; fall protection and landing; and multiple
   actors with distance, vertical separation and resistance variations. Cover
   simultaneous attacks and ordering. Check CPU and CUDA separately.

## 3. Expose Mace Once Implemented

Update item-name parsing/serialization in `simulator/apps/configuration.hpp`
and the applicable scenario compiler/validation and loadout paths together.
Audit `simulator/apps/scene.hpp`, `simulator/apps/training.cpp`, reset, rewards,
replay, C ABI buffers and any state-copy paths for new state or outcomes. Add
focused native and training tests for full and selective resets, serialization,
older configurations and supported/unsupported items.

Use the existing controls where they suffice. If policy capabilities or
observations actually need extending, update `contracts/policy_catalogue.json`,
regenerate its consumers, and test descriptor/checkpoint compatibility. Update
the Fabric live adapter and replay handling only when the item is exposed there;
test loadout validation, inventory restoration and private-world round cleanup.
Do not hand-edit generated catalogue consumers.

## 4. Wind Charges as a Separate Milestone

First close the flight, hit and explosion research items. Then extend
`simulator/include/minecraft_sim/projectiles.hpp` and relevant orchestration
with finite storage, actor ownership, launch RNG, the item cooldown, collision,
direct damage and explosion push. Respect the player-use launch path and
source-derived tick ordering. Define capacity exhaustion, terrain restrictions,
item consumption, self-effects, reset and projectile retirement explicitly.
Reuse existing arrow infrastructure where appropriate without assuming arrows
and wind charges share their flight or hit semantics.

Add native and training checks for item selection/use, cooldown boundaries,
seeded launch and inherited motion, collision/occlusion, direct-hit versus
blast damage, each affected actor including the owner, exposure, immunity,
deflection timing, capacity, lifecycle, replay and selective reset. Add CUDA
fixtures and private-game captures before a parity claim. Triggered special
blocks remain excluded unless separately implemented and verified.

## 5. Additional Projectiles

Snowballs, eggs and pearls have finite loadout stacks, shared dry-air flight,
represented-player and full-cube impacts, training storage/reset, and typed
replay snapshots. Pearls additionally teleport and damage their owner and have
a use cooldown. CPU and CUDA checks cover these paths. The mod creates vanilla
projectiles, tracks round-owned ones for cleanup, and suppresses chicken
hatching for tracked eggs. The mixins and live play still require an observed
round; no Minecraft capture establishes simulator parity. Fluids, special
blocks, mobs, projectile deflection, cross-dimension travel and chunk tickets
are outside the simulator slice. Vanilla endermite spawning is not suppressed
in the mod. Training batches must start with at least one throwable loadout to
allow a later reset to introduce throwable items.

## Acceptance Gates

- Source-derived and independently calculated fixtures pass on CPU; CUDA has
  separately passing checks on a usable device, not merely a skipped test.
- Matched, reproducible Minecraft 26.3 captures validate the supported slice,
  including tick, position, velocity, damage and reset effects. Record version,
  initial state, controls, random conditions, tolerances and limitations.
- Scenario, training, reset, replay, ABI and live-mod tests pass for each
  interface actually exposed; generated contracts are current and old layouts
  remain compatible or fail explicitly.
- Document exclusions, unsupported states and any remaining approximation.
  CPU/CUDA agreement alone does not establish vanilla equivalence.