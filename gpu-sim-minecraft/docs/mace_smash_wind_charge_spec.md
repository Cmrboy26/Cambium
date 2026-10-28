# Mace Smash and Wind Charge Scope

**Status:** Partial implementation. Mace loadouts are accepted and the native
simulator implements damage, accepted-hit bounce, fall-distance reset and
player-only splash. Ordinary fall damage and impulse protection are available
behind an opt-in rule for default player attributes and full-cube terrain, but
are not verified against a live world. This is not full mace parity. Wind charges
are available as finite loadout items and projectiles in the represented player
and full-cube terrain slice. Focused source-derived calculations have passed on
CPU and CUDA; no actual game round has validated either mechanic.

## First Milestone: Unenchanted Mace

Support an ordinary unenchanted mace in the existing player combat model, first
on CPU and then CUDA, with static ordinary arena blocks and the existing modeled
player states. Ordinary non-smash attacks must continue to use normal melee
charge, critical, damage, armor, shield and knockback handling. Eligible smash
attacks add the fall-based bonus and reproduce the sourced hit effects and
bystander knockback with the correct gates and ordering. The underlying world
must be testable with at least three players even if public training remains
two-player.

The controlling formulas, strict thresholds and callback distinctions are in
the [source reference](MACE_FORMULA_REFERENCE.md). The simulator must reproduce
the lifecycle of fall-distance reset and impulse-related fall-damage protection,
not merely cancel fall damage on every attack. Explicitly document any
unsupported player state or source behavior rather than presenting it as vanilla.

The current scenario, training and live interfaces accept `minecraft:mace`;
fall mechanics' limited scope and unverified live behavior remain explicit
limitations until the validation gates pass. Existing
attack and equipment-selection controls should be assessed first; no separate
smash action or fabricated smash cooldown is required. Add observations or
contract changes only for demonstrated needs, preserving compatibility and
generating shared catalogue artifacts through the existing workflow.

## Later Milestone: Wind Charge Item

Model finite ordinary wind-charge items, use/cooldown, projectile state,
collision and explosion-driven motion as a separate feature. The owner is the
launching actor, not a hotbar slot. The player-use launch path differs from a
dispenser, and the 1.2F explosion power is not its entity influence radius.
Define the supported inventory, terrain, collision, self-knockback and reset
behavior before exposing it to policies or live play.

## Later Milestone: Other Thrown Items

Support ender pearls, snowballs and eggs after mace and wind charges. Chicken
spawning from eggs is explicitly excluded, including in live mod play.

## Deferred and Excluded

The first milestone excludes enchantments, Wind Burst, durability, arbitrary
inventory behavior, fluids, special blocks, non-player entity types, teams,
mounts, creative flight and elytra. Wind Burst is an enchantment effect, not a
separate projectile. Wind charges and their triggered block interactions are
not part of the initial mace milestone. Client/server latency, networking and
unmodeled world behavior are not implied by simulator parity.

The [implementation plan](MACE_WIND_IMPLEMENTATION_PLAN.md) defines integration
and verification gates; the [research plan](MACE_WIND_RESEARCH_PLAN.md) lists
the evidence still needed before exact mechanics can be claimed.