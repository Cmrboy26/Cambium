# Java 26.3 Bow and Arrow Source Reference

This is a source-derived mechanics note for a later GPU simulator contribution.
It describes the local decompiled Minecraft source; it is not an implementation,
test result, CPU/GPU parity claim, or claim of complete vanilla PvP equivalence.

## Scope and Source Identity

Target the ordinary unenchanted `BowItem` firing the ordinary `Items.ARROW`
stack, with finite ammunition, two standing players, PvP damage enabled, and
static ordinary solid-cube arena blocks. The first simulator milestone excludes
arrow pickup, durability, water and other fluids, flames/fire aspect, tipped or
spectral arrows, enchantments, portals, special blocks, and multiplayer packet
prediction. Keep the vanilla source behavior separate from those deliberate
simulator restrictions.

The supplied tree identifies itself as Minecraft `26.3`, protocol `777`, world
version `5023`, build time `2026-09-15T11:20:48+00:00`, Java 25
([version.json, id](../minecraft-26.3-src/version.json#L2),
[protocol/world version](../minecraft-26.3-src/version.json#L4),
[build and Java version](../minecraft-26.3-src/version.json#L11)). No content
hash is recorded here; do not infer one from the version descriptor.

## Call Path and Item-Use Timing

For an ordinary player shot, the controlling path is:

1. `BowItem.use` checks for a supported projectile and begins item use; it does
   not consume an arrow.
2. `LivingEntity.startUsingItem` saves the hand/stack and sets the remaining
   use duration. Each use update calls `onUseTick` and then decrements that
   remaining count.
3. A release action reaches `LivingEntity.releaseUsingItem`, which verifies the
   same item type is still in the used hand, rebinds the current stack, and calls
   `BowItem.releaseUsing` with the remaining time.
4. `BowItem.releaseUsing` recomputes available ammunition and charge. It calls
   `ProjectileWeaponItem.draw`; for the ordinary unenchanted case this consumes
   one arrow at release and creates one projectile. The server creates/shoots
   the arrow; the client-side sound/stat path is not projectile authority.

Relevant source: [BowItem.releaseUsing](../minecraft-26.3-src/net/minecraft/world/item/BowItem.java#L26),
[BowItem.use](../minecraft-26.3-src/net/minecraft/world/item/BowItem.java#L95),
[LivingEntity use update](../minecraft-26.3-src/net/minecraft/world/entity/LivingEntity.java#L3473),
[LivingEntity use start](../minecraft-26.3-src/net/minecraft/world/entity/LivingEntity.java#L3500),
[LivingEntity release](../minecraft-26.3-src/net/minecraft/world/entity/LivingEntity.java#L3605),
[release packet dispatch](../minecraft-26.3-src/net/minecraft/server/network/ServerGamePacketListenerImpl.java#L1383),
[hotbar switch cancellation](../minecraft-26.3-src/net/minecraft/server/network/ServerGamePacketListenerImpl.java#L1551),
[offhand-swap cancellation](../minecraft-26.3-src/net/minecraft/server/network/ServerGamePacketListenerImpl.java#L1366),
[projectile draw](../minecraft-26.3-src/net/minecraft/world/item/ProjectileWeaponItem.java#L105),
[projectile ammo use](../minecraft-26.3-src/net/minecraft/world/item/ProjectileWeaponItem.java#L124).

`BowItem.getUseDuration` is 72000 ticks, while the charge formula has a 20-tick
full-draw duration. Held time is the integer difference between use duration and
remaining time. The Java `float` expression is:

```text
float p = timeHeld / 20.0F;
p = (p * p + 2.0F * p) / 3.0F;
if (p > 1.0F) p = 1.0F;
```

The release predicate is strictly `p < 0.1`; equality would shoot. With integer
held ticks, `t=2` gives approximately `0.07` and does not shoot, while `t=3`
gives approximately `0.1075` and does. `t=20` gives `1.0`; later values remain
capped at `1.0`. The launch speed argument is `p * 3.0F` blocks per tick,
uncertainty is `1.0F`, and critical status is exactly `p == 1.0F`
([charge and release](../minecraft-26.3-src/net/minecraft/world/item/BowItem.java#L33),
[power function](../minecraft-26.3-src/net/minecraft/world/item/BowItem.java#L74),
[server shot arguments](../minecraft-26.3-src/net/minecraft/world/item/BowItem.java#L41)).

For a bow, supported held projectiles are arrow-tag items. Ammo lookup checks
offhand before main hand, then scans the inventory in slot order. This note
narrows that path to ordinary arrows. Ammo is rechecked at release; ordinary
finite ammo is consumed then, not on initial use. Do not model client prediction
and server consumption as two mutations of one inventory
([held projectile preference](../minecraft-26.3-src/net/minecraft/world/item/ProjectileWeaponItem.java#L33),
[player inventory scan](../minecraft-26.3-src/net/minecraft/world/entity/player/Player.java#L1860),
[ammo use](../minecraft-26.3-src/net/minecraft/world/item/ProjectileWeaponItem.java#L124)).

Cancellation is not release: switching away from a main-hand bow and swapping
hands call `stopUsingItem`, which clears use state without calling the bow's
release callback. A headless simulator must specify its own action boundary
(start, held-tick updates, release, or cancel) rather than assume every control
sample maps to one vanilla client/server packet sequence
([release callback gate](../minecraft-26.3-src/net/minecraft/world/entity/LivingEntity.java#L3605),
[stop implementation](../minecraft-26.3-src/net/minecraft/world/entity/LivingEntity.java#L3618)).

## Aim, Launch, Units, and Randomness

The ordinary arrow constructor places the arrow at `(shooter X, shooter eye Y -
0.1F, shooter Z)` and assigns the shooter as owner. `ArrowItem.createArrow`
constructs an `Arrow`, the ordinary subclass of `AbstractArrow`
([arrow constructor](../minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java#L116),
[ArrowItem factory](../minecraft-26.3-src/net/minecraft/world/item/ArrowItem.java#L17)).

For bow offset zero, `shootFromRotation` builds the float direction components
in this order:

```text
xd = -sin(yawRadians) * cos(pitchRadians)
yd = -sin(pitchRadians)
zd =  cos(yawRadians) * cos(pitchRadians)
```

The vector is normalized, then independent X/Y/Z noise is added, and only then
is the result scaled by launch speed. Do not renormalize after adding noise.
Each component uses `RandomSource.triangle(0.0, 0.0172275 * uncertainty)`;
the double overload is `mean + spread * (nextDouble() - nextDouble())`, so one
shot consumes six `nextDouble` calls from the projectile's `random` stream for
spread. Bow aim uses `Mth.sin/cos`; those helpers index a 65536-entry sine
lookup table, so replacing them with host-library trigonometry can change
results. The exact projectile RNG seed/state assignment is an external initial
condition and is not specified by this document
([direction/noise](../minecraft-26.3-src/net/minecraft/world/entity/projectile/Projectile.java#L161),
[rotation and inherited motion](../minecraft-26.3-src/net/minecraft/world/entity/projectile/Projectile.java#L183),
[triangle draw](../minecraft-26.3-src/net/minecraft/util/RandomSource.java#L59),
[trig lookup](../minecraft-26.3-src/net/minecraft/util/Mth.java#L50)).

After spread and scaling, the projectile adds the shooter's `getKnownMovement`
X/Z components, and adds Y only when the shooter is not on ground. For an
ordinary unmounted player, known movement resolves to that player's known
movement; it must not be silently replaced with zero. Thus `p*3` is the
projectile's aim-relative launch speed, not necessarily its final world-space
speed. Positions are blocks and velocities are blocks per tick
([inherited velocity](../minecraft-26.3-src/net/minecraft/world/entity/projectile/Projectile.java#L188),
[known movement default](../minecraft-26.3-src/net/minecraft/world/entity/Entity.java#L4163)).

The random path that affects this subset is projectile-local triangular spread
and, on a critical hit, projectile-local `nextInt`. Other projectile behaviors
(for example a rejected-hit deflection or a block-change fall-out) consume
additional random values. The source excerpt establishes which stream/method is
used, but not a portable seed mapping between a future GPU RNG and Minecraft.

## Arrow Tick and Trajectory

`AbstractArrow.tick` is not living-entity travel. Its important order is:

1. Capture `movement = getDeltaMovement()` and current block state. If the arrow
   point is already inside a nonempty block collision AABB, zero velocity and
   set in-ground.
2. Decrement shake; clear fire in water/rain. For an embedded arrow, check for a
   changed block state and clearance; otherwise advance the in-ground lifetime.
3. If airborne, save its position. In water, multiply STORED velocity by
   `0.6F`; the swept segment still ends at `position + movement` using the
   captured pre-inertia vector.
4. Clip the captured segment against collider blocks and the world border, with
   fluids ignored. Process entities only along the segment clipped to that
   block result. Move to the selected hit/end point and run callbacks.
5. If the arrow is still airborne, apply `0.99F` air inertia only when its
   resulting position is not in water, then subtract gravity `0.05` from Y.
   Finally call the base projectile/entity tick.

For ordinary unobstructed air flight, position advances by the current velocity
before drag/gravity update. A useful recurrence is `xNext = x + v`, then
`vNext = 0.99F * v + (0, -0.05, 0)` (with the source's float-to-double
conversion for `0.99F` and double gravity). Water is intentionally outside the
first milestone; source behavior differs as noted above and must not be
implicitly treated as air
([tick order](../minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java#L173),
[water inertia and captured segment](../minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java#L222),
[block clip and post-hit update](../minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java#L245),
[air drag/gravity](../minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java#L262)).

## Swept Collision, Eligibility, and Ordering

The arrow first clips its current movement segment to collider block shapes
(fluid mode `NONE`) and the world border. The entity broadphase is the arrow
AABB expanded toward current velocity and inflated by `1.0`; this is a search
area, not an arrow-radius hit test. Exact entity hits clip the centerline segment
against the target AABB. `ProjectileUtil.computeMargin` ramps the entity margin
as `max(0, min(0.3F, (tickCount - 2) / 20.0F))`. A margin-only hit is accepted
only after a block-clipped trace toward the target center reaches the actual
target AABB. For an arrow, the recorded location is the inflated-box entry
location, not the surface-hit point
([arrow sweep/search area](../minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java#L278),
[margin formula](../minecraft-26.3-src/net/minecraft/world/entity/projectile/ProjectileUtil.java#L153),
[many-hit exact/margin path](../minecraft-26.3-src/net/minecraft/world/entity/projectile/ProjectileUtil.java#L201)).

Eligibility includes `canBeHitByProjectile`, owner/shared-vehicle exclusion
until `leftOwner`, and (for arrows) the piercing-ignore set. Owner clearance is
tested against the owner's root vehicle and passengers that pass
`EntitySelector.CAN_BE_PICKED`, using the arrow AABB expanded toward current
velocity and inflated by `1.0`. Player-owned arrows also apply
`Player.canHarmPlayer`, so
team/PvP setup is part of the fixture, not an incidental hit detail. In the
ordinary two-player case, neither player should be configured as unable to harm
the other
([left-owner update](../minecraft-26.3-src/net/minecraft/world/entity/projectile/Projectile.java#L141),
[owner clearance](../minecraft-26.3-src/net/minecraft/world/entity/projectile/Projectile.java#L148),
[base entity eligibility](../minecraft-26.3-src/net/minecraft/world/entity/projectile/Projectile.java#L379),
[arrow/player eligibility](../minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java#L591),
[team damage gate](../minecraft-26.3-src/net/minecraft/world/entity/player/Player.java#L722)).

For standing players, `Avatar.STANDING_DIMENSIONS` is width `0.6F`, height
`1.8F`, eye height `1.62F`; with the normal feet-position AABB this is
horizontal half-width `0.3` on either side. Crouching is `0.6 x 1.5` and
swimming/fall-flying is `0.6 x 0.6`; freeze the standing pose for initial
fixtures or include the actual pose/AABB in state
([avatar dimensions and poses](../minecraft-26.3-src/net/minecraft/world/entity/Avatar.java#L21)).

Entities are collected along the block-clipped segment and sorted by squared
distance from the arrow's current point to each candidate's ENTITY POSITION,
not by the segment intersection. There is no explicit secondary comparator for
equal distances; ties inherit candidate iteration order. Because the entity
segment ends at the block hit, an entity candidate at that endpoint can be
processed before the block fallback if the geometry clip admits it. Exact
coincident block/AABB endpoint inclusion depends on the clip helpers and is a
boundary fixture to preserve, not an assumed universal tie rule
([candidate sort and block fallback](../minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java#L278),
[entity clipping/margin acceptance](../minecraft-26.3-src/net/minecraft/world/entity/projectile/ProjectileUtil.java#L220)).

## Hit Damage, Armor, Shield, and Attribution

For an unenchanted arrow, `AbstractArrow.onHitEntity` takes the length of
CURRENT STORED velocity, casts it to `float`, multiplies by the `double`
`baseDamage` (default `2.0`), clamps to `[0, 2147483647]`, and applies `ceil`
to get integer damage. If critical, it adds
`random.nextInt(damage / 2 + 2)`; `damage / 2` is integer division. Full bow
charge sets the critical flag. Consequently, launched speed alone is not enough
to predict damage when shooter inheritance or spread changes current velocity;
the arrow's tick drag/gravity and hit timing matter too
([base damage and defaults](../minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java#L74),
[arrow hit and critical calculation](../minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java#L423)).

The damage source is `ARROW` with the arrow as direct entity and current owner
as causing entity (falling back to the arrow if owner is absent). The target's
normal living-entity damage path then checks invulnerability/death, item
blocking, cooldown, armor, effects, absorption, health, and knockback. In
`LivingEntity.hurtServer`, when `damageCooldownTime > 10` and the source does
not bypass cooldown, an incoming post-blocking hit no greater than `lastHurt`
is rejected; a stronger hit applies only the difference. Otherwise the source
sets cooldown to 20 and applies the damage. `ServerPlayer.tick` decrements that
timer. The supplied `bypasses_cooldown` damage-type tag is empty, and the
ordinary arrow is not added to it
([ARROW source construction](../minecraft-26.3-src/net/minecraft/world/damagesource/DamageSources.java#L207),
[direct projectile field](../minecraft-26.3-src/net/minecraft/world/damagesource/DamageSource.java#L59),
[causing entity field](../minecraft-26.3-src/net/minecraft/world/damagesource/DamageSource.java#L63),
[hurt/cooldown path](../minecraft-26.3-src/net/minecraft/world/entity/LivingEntity.java#L1178),
[player timer decrement](../minecraft-26.3-src/net/minecraft/server/level/ServerPlayer.java#L590),
[cooldown bypass tag](../minecraft-26.3-src/data/minecraft/tags/damage_type/bypasses_cooldown.json#L2)).

For ordinary armor, the source computes `toughness = 2.0F + armorToughness / 4.0F`,
`realArmor = clamp(totalArmor - damage / toughness, totalArmor * 0.2F, 20.0F)`,
then multiplies damage by `1 - realArmor / 25`. Armor durability, resistance,
absorption, and other effects are additional state and are outside an
unarmored/no-effects fixture. Arrow damage does not bypass armor
([armor gate](../minecraft-26.3-src/net/minecraft/world/entity/LivingEntity.java#L1908),
[armor formula](../minecraft-26.3-src/net/minecraft/world/damagesource/CombatRules.java#L16),
[effects/absorption/health](../minecraft-26.3-src/net/minecraft/world/entity/LivingEntity.java#L1960)).

An ordinary shield's `BLOCKS_ATTACKS` component has a `0.25F` second delay
(five ticks), a 90-degree horizontal blocking angle, reduction `base=0,
factor=1`, and the `BYPASSES_SHIELD` tag as its bypass set. Once the item has
been used for at least five ticks, the defender compares the horizontal source
direction with head yaw; angles over 90 degrees block nothing. Within the arc,
ordinary arrow damage is fully reduced. `LivingEntity.applyItemBlocking`
explicitly lets arrows with pierce level greater than zero bypass this path;
ordinary unenchanted arrows have pierce level zero. Shield durability is
intentionally not part of this milestone
([vanilla shield component](../minecraft-26.3-src/net/minecraft/world/item/Items.java#L2420),
[block delay and arc reduction](../minecraft-26.3-src/net/minecraft/world/item/component/BlocksAttacks.java#L127),
[active-use delay](../minecraft-26.3-src/net/minecraft/world/entity/LivingEntity.java#L3636),
[arrow/piercing shield gate](../minecraft-26.3-src/net/minecraft/world/entity/LivingEntity.java#L1311),
[shield bypass tag](../minecraft-26.3-src/data/minecraft/tags/damage_type/bypasses_shield.json#L2)).

On an unblocked successful living hit, normal damage calls default knockback
with `0.4F`; projectile direction comes from current arrow X/Z velocity and is
negated before the knockback method subtracts it, producing an impulse along
travel. The arrow's additional `doKnockback` is an
enchantment callback with base zero, so ordinary unenchanted arrows add none.
If full shield blocking makes `hurtOrSimulate` return false, a normal entity
inherits `projectileReceivesSideEffectsOnHit == true`; the arrow takes its
rejected-hit `REVERSE` deflection path. Scalar power `0.2` multiplies each
velocity component by `-0.1`, then adds a random 170-to-190 degree display yaw
rotation. The generic entity deflection hook runs before arrow damage; the
supplied `DEFLECTS_PROJECTILES` entity tag lists Breeze, not Player
([default knockback](../minecraft-26.3-src/net/minecraft/world/entity/LivingEntity.java#L1293),
[arrow extra knockback](../minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java#L512),
[entity deflection hook](../minecraft-26.3-src/net/minecraft/world/entity/Entity.java#L3637),
[rejected arrow hit](../minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java#L499),
[reverse deflection](../minecraft-26.3-src/net/minecraft/world/entity/projectile/ProjectileDeflection.java#L11),
[deflecting entity tag](../minecraft-26.3-src/data/minecraft/tags/entity_type/deflects_projectiles.json#L2)).

Preserve the owner separately from the arrow entity: damage sources carry both
direct projectile and causing entity, and living damage attribution resolves
the responsible player from the causing entity. A simulator reward/event layer
that needs shooter attribution must use that owner identity, not infer the
shooter from the arrow's position or the target's health delta
([direct projectile field](../minecraft-26.3-src/net/minecraft/world/damagesource/DamageSource.java#L59),
[causing entity field](../minecraft-26.3-src/net/minecraft/world/damagesource/DamageSource.java#L63),
[player damage attribution](../minecraft-26.3-src/net/minecraft/world/entity/LivingEntity.java#L1365)).

## Block Impact, Stuck State, and Reset State

On a block hit, the arrow stores the hit block state, runs the base block
callback, backs its position off by `sign(currentVelocity) * 0.05F` on each
axis, zeros velocity, sets in-ground, sets shake to seven, clears critical and
pierce flags, and clears piercing hit sets. An embedded arrow increments its
`life` counter unless a changed block state and clear `shouldFall()` check
trigger fall-out; otherwise it reaches the 1200-tick discard threshold. If the
block changes and a point AABB inflated by `0.06` is clear,
it falls out, scales each velocity component by an independent
`nextFloat() * 0.2F`, and resets `life`. This is not a generic 1200-tick
airborne timeout
([block impact](../minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java#L526),
[stuck lifetime](../minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java#L379),
[embedded clearance/fall-out](../minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java#L348)).

For a clean episode reset or pooled projectile reuse, clear active arrows and
their owner references, position/velocity, tick count (it affects collision
margin), `leftOwner`, critical/pierce flags, in-ground and last-block state,
shake/lifetime/in-ground counters, weapon/item metadata, and projectile RNG
state. Also reset each player's bow-use hand/countdown, ammunition, damage
cooldown/last-hit state, health/absorption, armor, shield-use delay, pose, team
damage eligibility, and movement. Vanilla player-owned arrows become
pickup-allowed by default, but pickup is explicitly excluded here; do not
silently add inventory returns to the finite-ammo model
[projectile leftOwner state](../minecraft-26.3-src/net/minecraft/world/entity/projectile/Projectile.java#L54),
[projectile owner field](../minecraft-26.3-src/net/minecraft/world/entity/projectile/Projectile.java#L53),
[arrow persisted state](../minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java#L600),
[player owner pickup default](../minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java#L630),
[pickup operation](../minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java#L641)).

## Suggested Source-Derived Fixtures

These are proposed independent checks for a future implementation, not results
already obtained:

- Charge boundaries: integer held ticks 2 (no projectile), 3 (shot at about
  `0.3225` blocks/tick before spread/inherited velocity), 20 (speed `3.0`,
  critical), and 21 (still capped at `3.0`, critical).
- Launch vector: stationary grounded shooter at yaw/pitch zero, with a scripted
  projectile RNG supplying equal pairs for all six spread draws; expected
  unperturbed full-draw velocity `(0, 0, 3)`. Repeat with nonzero shooter X/Z
  movement, then airborne Y movement, to check inherited-velocity conditions.
- Free flight: for the zero-spread full-draw case with no collision, positions
  after ticks 1-3 are approximately `(0,0,3)`, `(0,-0.05,5.97)`,
  `(0,-0.1495,8.9103)`; compare Java float/double operation order, not only
  rounded display decimals.
- Collision boundaries: target before a cube wall, target fully behind it,
  exact target/block endpoint contact, two target centers at equal sort distance,
  and margin-only grazing hits on early versus mature arrows.
- Owner/PvP: owner overlapping the initial arrow, owner after `leftOwner`,
  shared-root-vehicle exclusion, two teammates versus eligible opponents.
- Shield/damage: front versus rear impact, shield use at four versus five ticks,
  ordinary arrow versus piercing-level positive control, full-block rejected-hit
  reverse velocity, armor reduction, and equal/lower versus stronger damage
  while `damageCooldownTime > 10`.
- Stuck/reset: block impact and zero velocity, 1199 versus 1200 embedded ticks,
  block-state change fall-out, then episode reset followed by a shot to detect
  stale owner/RNG/cooldown/use state. Separately assert no pickup and no
  durability mutation in the deliberately restricted simulator milestone.

For fixture expectations, preserve the exact source call order and Java numeric
types. The endpoint and equal-distance cases have no explicit secondary tie
policy in the source; record their concrete clip-helper outcome before encoding
an expectation.

Copy-ready source comment references for later code (references only, not GPU
implementation):

```cpp
// Java 26.3 BowItem.getPowerForTime: minecraft-26.3-src/net/minecraft/world/item/BowItem.java:74
// Java 26.3 Projectile.shootFromRotation: minecraft-26.3-src/net/minecraft/world/entity/projectile/Projectile.java:183
// Java 26.3 AbstractArrow.tick: minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java:173
// Java 26.3 ProjectileUtil.computeMargin: minecraft-26.3-src/net/minecraft/world/entity/projectile/ProjectileUtil.java:153
```
