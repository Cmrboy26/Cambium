# Minecraft Java 26.3 CPU Physics Specification

## PvP Coverage and Missing Features

**Bottom line:** this document can guide a CPU implementation of a restricted
vanilla movement/combat arena, including ordinary damage, survival mining,
item-use timing, fluid hazards and a selected unenchanted kit/block data profile
(section 10). It is NOT a complete specification for arbitrary vanilla kits,
mace combat, or end-crystal PvP, and no working simulator is claimed here.
Sumo/bridge-like scenarios use Java 26.3 mechanics, not older combat or plugins.
Their normal vanilla damage remains; a special nonlethal sumo rule is not implied.

The coverage below describes DOCUMENTED behavior, not implemented or runtime-
verified functionality. "Specified core" means the relevant algorithm is described;
"partial" means only part of the feature is described; "missing" means additional
source research and specification are needed. A mentioned callback or attribute
does not mean its whole feature has been specified.

### What Could Be Simulated

Assume explicitly supplied block shapes/tags, fixed player/item attributes,
loaded terrain, a defined action schedule, and no unlisted special mechanics.
Under those constraints, an implementer could reproduce these physical cores:

| Capability | Documented behavior | Important limit |
| --- | --- | --- |
| Player locomotion | Walking, strafing, sprinting, jumping/sprint jumps, crouching, edge protection, gravity/drag, poses, ladder movement | Raw controls and transformed inputs are distinguished; auto-jump geometry is not specified. Sections 2 and 8. |
| Terrain collision | Swept axis collision, corner ordering, candidate-height stepping, ceiling impacts, support sampling, restitution | Requires actual state/context-dependent shapes; not a complete shape database for every block. Section 3. |
| Player/entity contact | Ordinary living overlap pushes and their sequential timing | Not all entity-specific hard-collision, vehicle, or AI behavior. Section 3.5. |
| Melee positioning, damage and knockback | Hit direction, grounded/airborne response, sprint bonus, charge/critical eligibility, sweep selection, repeated-hit gates, player armor/absorption/health pipeline | Ordinary weapon/armor data is in section 10.5; arbitrary enchantment callbacks and complete death/recovery lifecycle are not supplied. Sections 5 and 10.1. |
| Shield physical gates | Facing arc, raise delay, full/partial blocking, piercing-arrow bypass, ordinary axe disable=100 ticks, durability and cooldown behavior | Default shield/ordinary axes are specified, not every custom component/enchantment profile. Sections 5.4, 8.4 and 10.3-10.5. |
| Basic bow/arrow physics | Draw-dependent launch, spread RNG, inherited motion, trajectories, hit damage/critical RNG, ammo/durability, embedding and deflection | Ordinary bow/arrow core, not complete crossbow, firework, tipped-arrow, or enchantment systems. Sections 5.5-5.6 and 10.3/10.5. |
| Ordinary block, mining and bucket actions | Placement/replacement/body obstruction; mining formulas, cancellation/completion and removal ordering; buckets, fluid spread and currents | Selected block/tool data only; loot pickup, special callbacks and all waterlogging/container variants are not exhaustively specified. Sections 4, 6 and 10.2-10.5. |
| Item use and survival feedback | Hotbar/offhand transitions, attack/use/mining arbitration, release versus cancellation, cooldown groups, lava/fire/fall/drowning, hunger/regeneration/sprint feedback | Not a complete container-inventory, consumable, totem, timed-effect or world-fire-spread implementation. Sections 10.3-10.5. |
| Fluid and air movement variants | Swimming, lava movement, fluid exit, gliding integration, basic creative flight, selected surface effects | Not complete elytra equipment/rocket gameplay or every special-block callback. Sections 2, 6 and 8.5. |
| Deterministic processing | Relevant entity/world phases, scheduled ticks, numeric helpers, random draws and required state | A headless action schedule must be chosen explicitly; actual client/server packet timing is not one universal tick sequence. Sections 7-8. |

For example, a fixed-map sword/bow/bucket arena using the section 10.5 block/kit
subset and ordinary damage is a reasonable target. The same physical foundation
supports vanilla bridge-like or unarmed ring-out scenarios. Goals, start/stop and
map reset are external and are not counted as physical gaps here. This remains a
restricted environment, not proof of complete vanilla PvP equivalence.
"Completely from this document" does NOT include all registry data: additional
states, tags, components, loot and kit values must be extracted when enabled.

### What Could Not Yet Be Reproduced Faithfully

These are missing SPECIFICATION MODULES, not features absent from Minecraft.
The entries identify research needed; they do not establish untraced formulas,
constants, or execution order for those mechanics.

| Missing or incomplete module | What still needs to be specified |
| --- | --- |
| Complete damage/effect and death lifecycle | Ordinary mitigation, absorption, health loss, environmental damage and natural regeneration are specified in section 10. Arbitrary damage-source/enchantment/effect callbacks, full lethal-hit recovery, death and respawn lifecycle still need their own specification. |
| Full item/kit and enchantment coverage | Section 10 supplies ordinary weapons/armor/tools/shield, ammo/durability, hotbar/offhand and use/release/cooldown behavior. Other components, container inventory operations, loot pickup/replenishment and arbitrary enchantment/effect lifecycles remain dependencies. |
| Mace smash attacks | Smash eligibility and fall-state sampling, fall-dependent damage, smash-specific direct/nearby knockback, fall-distance/damage resets, and mace enchantments such as Density, Breach and Wind Burst. Ordinary melee and gravity do not supply these rules. |
| Wind charges and Wind Burst | Launch/use/cooldown, projectile collision/detonation, radial force and occlusion rules, self-launch versus target launch, fall-state interactions and the enchantment-triggered launch path. |
| Explosions | Source-specific blast parameters, entity exposure/line-of-sight sampling, damage and impulse, explosion resistance/protection, block destruction, source attribution and chain-reaction ordering. Covers end crystals, TNT, explosive beds/anchors and wind-related variants only after their distinct rules are researched. |
| End crystals | Placement on eligible bases, clearance/entity obstruction, spawned crystal hitbox/state, hit/detonation triggers and crystal-to-crystal interactions, plus the explosion module. A crystal is not an ordinary BlockItem placement. |
| Totems and combat recovery | Lethal-hit interception, held-item selection/consumption, resulting health/effects, chained lethal events and replenishment timing; also golden apples, food and potions where the kit uses them. |
| Ender pearls and other utility projectiles | Pearl launch/trajectory/collision, teleport destination and state changes, damage/cooldown and interactions; fishing-rod casts/hooks/pulls, snowballs and similar items each need their own rules. Do not substitute arrow physics. |
| Special blocks, world fire and loot | Ordinary survival mining/removal is specified in section 10.2, with a five-block data subset in 10.5. Full loot/pickup, flammable terrain/fire propagation, falling blocks, cobweb/powder-snow/bubble-column callbacks, redstone/pistons and interactable blocks are not implied by that subset. |
| Other weapons and mobility | Crossbow loading/multishot/fireworks, trident throws/Loyalty/Riptide/Channeling, spear attacks, elytra rocket boosting and vehicle-specific physics. |
| Complete multiplayer behavior | Packet quantization, movement correction, inventory prediction/reconciliation and latency-dependent action ordering. Physical core reproduction is not networking or anti-cheat parity. |
| Match framework (external, not a physics blocker) | Objectives, scoring, start/stop, map reset, rounds and RL observation/action/reward definitions are separate. Select actual vanilla game rules and legal kit/state data; do not substitute plugin behavior for a missing physical rule. |

### Readiness by PvP Game Type

"Core usable" below means usable as a foundation, NOT an already complete mode.
Readiness here concerns physics/behavior only; goals, episode control and map
resets are assumed external. The section 10 profile is vanilla Java 26.3.

| Game type | Readiness from this document | Additions needed beyond the documented core |
| --- | --- | --- |
| Knockback / sumo / void duels | Ordinary vanilla unarmed physical core specified | Retain vanilla damage/charge/cooldowns. Nonlethal rules and custom knockback are not vanilla mechanics. Sections 5, 8.4 and 10. |
| Sword-only duels | Ordinary unenchanted damage/armor/knockback core specified | Extend only for enabled enchantments, recovery items and unlisted equipment/effects. Sections 5 and 10. |
| Axe-and-shield duels | Ordinary axe/default-shield core specified | Default disable/use/swap/durability now documented; custom components/enchantments and expanded kits need their actual behavior. Section 10.5. |
| Bow duels | Ordinary flight/hit/damage/ammo core specified | Arrow pickup, special arrows, enchantments, crossbows and rockets require their own modules. Sections 5.5-5.6 and 10. |
| Sword/bow/bucket kit PvP | Ordinary physical/behavior core specified for the selected profile | Mining/use/hazards/armor are now included; loot pickup, flammable terrain, consumables and arbitrary enchanted kits remain bounded out. Section 10.6. |
| Bridge / block-placement / SkyWars-style games | Ordinary movement, placement, mining and combat subset specified | Unlisted blocks, loot pickup/replenishment, containers and optional consumables/projectiles still need coverage; objectives/reset are not counted as physics gaps. Section 10.6. |
| Mace combat | Not yet supported as a faithful mode | Mace smash module; wind charges and Wind Burst if allowed; fall-damage/state interactions and enchantment/armor data. Pearl or elytra variants additionally need their mobility modules. |
| End-crystal PvP | Not yet supported as a faithful mode | Crystal placement/entity/detonation, explosions with exposure and block destruction, chained-event ordering, armor/enchanted damage, totems, inventory swaps; pearls/apples/anchors where allowed. |
| Respawn-anchor / bed / TNT combat | Not yet supported as a faithful mode | Explosions plus each source's placement, arming/charging, activation, dimension conditions, fuse/trigger and chain behavior; combat recovery as required. |
| UHC / potion / full survival-kit PvP | Not yet supported as a faithful mode | Healing/absorption, potions/status effects, food, armor/enchantments, pearls, mining and other kit-specific tools; existing movement/melee/buckets are only part of the mode. |
| Trident / crossbow / rocket combat | Not yet supported as a faithful mode | Weapon-specific launch/use/reload, projectiles, enchantments, propulsion/explosion rules and kit lifecycle. |

### Practical Expansion Roadmap

1. Establish the restricted CPU reference: fixed map/kit, documented movement,
   ordinary collision, basic melee/shields/bow and simple placement, with explicit
   action boundaries and source-versus-simulator trajectory/state checks.
2. Implement the ordinary PvP foundation documented in section 10: selected data,
   mining, hotbar/use timing, player damage/armor and environmental/food feedback.
   Extend loot, consumables, arbitrary effects and death/recovery only when enabled.
3. For mace games, add smash/fall-state behavior, then the allowed wind-charge,
   Wind Burst, pearl and elytra interactions. Gliding alone is not mace mobility.
4. For crystal games, add source-aware explosions and exposure/block destruction,
   crystal placement/detonation and chained events, then totems and the permitted
   utility/consumable kit. Generic melee knockback is not explosion knockback.
5. Add remaining mode-specific systems and compare representative multi-action
   sequences against Minecraft before calling that mode faithful. Merely listing
   a missing feature here does not promote it to specified or validated status.

## 1. Authority, Scope, and Reproduction Contract

Ground truth: the decompiled source in `minecraft-26.3-src`, version `26.3`, protocol
`777`, world version `5023`, build time `2026-09-15T11:20:48+00:00`, Java 25.
See [version.json](../minecraft-26.3-src/version.json). Do not substitute formulas
from earlier Minecraft releases.

This document specifies player locomotion, block/entity collision, ordinary block
and bucket placement, survival mining, physical combat/projectiles, player damage,
item-use transitions, environmental/food behavior and simulation ordering. It is
an implementation specification for a bounded CPU simulator, not a claim that
such a simulator has been implemented or validated against the game.

For reproducibility, fix world/block states, loaded chunks, entity identities and
iteration order, attributes, effects, equipment/components, game rules, random
state, and action arrival order. Position alone is not sufficient state. Unless a
branch explicitly says otherwise, assume an alive, unmounted survival player in a
loaded world. Specialized blocks, items, entities, enchantments, and networking
must be enabled explicitly rather than silently approximated as ordinary cubes.

### 1.1 Numeric and Coordinate Contract

- Positions are feet positions; positive Y is up. Positive Z is forward at yaw 0;
  positive yaw turns toward negative X. Pitch is positive looking down.
- Movement vectors are blocks per simulation tick, not blocks per second. Never
  multiply the per-tick constants below by a variable frame duration.
- Preserve Java `double` versus `float` operations. An `F` suffix below means
  binary32 rounding before promotion. Entity positions/velocities and AABBs use
  binary64; many inputs, angles, attributes, and coefficients use binary32.
- Preserve source evaluation order, comparisons, floor behavior at negative
  coordinates, and collision tolerances. Do not replace float expressions with
  algebraically equivalent all-double expressions when targeting exact parity.
- Trigonometry and random draws must match the source helpers, not merely have
  similar distributions. Details and remaining parity boundaries are listed below.

## 2. Core Living-Entity Movement

Source: [LivingEntity.java](../minecraft-26.3-src/net/minecraft/world/entity/LivingEntity.java),
methods `aiStep`, `travel`, `travelInAir`, `travelInWater`, `travelInLava`,
`handleRelativeFrictionAndCalculateMovement`, `getFrictionInfluencedSpeed`,
`jumpFromGround`, `handleOnClimbable`, `getFluidFallingAdjustedMovement`.

### 2.1 Pre-Travel and Jump Order

In `LivingEntity.aiStep`, in this order:

1. Decrement positive `noJumpDelay`. Non-authoritative, non-interpolating entities
   multiply velocity by `0.98` (not the normal locally simulated player branch).
2. Tick equipment. For players, set BOTH horizontal velocity components to zero
   when `vx*vx + vz*vz < 9.0E-6`; nonplayers zero X and Z independently when their
   absolute values are below `0.003`. Independently zero Y if `abs(vy) < 0.003`.
3. Apply input. The base implementation multiplies strafe/forward fields by
   `0.98F`; player/client overrides must be considered before feeding these fields.
   Immobility clears jump, strafe, and forward; otherwise run applicable server AI.
4. Resolve jumping BEFORE travel. When jump is not held or fluids do not affect
   the entity, reset `noJumpDelay=0`. Deep water/lava jumping adds `0.04F` to Y
   every eligible tick. Shallow fluid/ground branches use a ground jump with
   `noJumpDelay==0`, then set `noJumpDelay=10`. Fluid height and ground state choose
   the branch; do not substitute an unconditional `onGround && jump` test.
5. Update gliding state, construct `(xxa,yya,zza)`, reset fall distance for slow
   falling/levitation, then call ridden travel or ordinary authoritative travel.
6. Apply block effects AFTER travel. Later perform freezing, spin attack, entity
   pushing, and water-sensitive damage. These can modify next tick's state.

Ground jump: `J = float(JUMP_STRENGTH) * multiplier * blockJumpFactor + jumpBoost`;
`jumpBoost = 0.1F*(amplifier+1)` if present. If `J <= 1.0E-5F`, do nothing.
Otherwise set `vy=max(J,vy)`; if sprinting, add
`(-sin(yawRadians)*0.2, 0, cos(yawRadians)*0.2)`. Mark velocity for synchronization.

Exact jump branch decision, after decrementing the jump delay: when jumping and
affected by fluids, let `H` be lava height if in lava, otherwise water height;
`W=inWater && H>0`, `T=fluidJumpThreshold`. If `W && !(onGround && H<=T)`, do
water jump. Otherwise if `inLava && !(onGround && H<=T)`, do lava jump. Otherwise,
if `(onGround || (W && H<=T)) && noJumpDelay==0`, do ground jump and set delay 10.
If not jumping or not affected by fluids, reset delay to zero. A failed height
or delay condition does not itself reset delay.

### 2.2 Travel Dispatch and Air/Ground Integration

Dispatch first to fluid movement when in liquid, affected by fluids, and unable to
stand on the fluid at the entity's block position; otherwise glide when fall
flying; otherwise air/ground. Creative flying is a player-specific override.

Air/ground procedure, with the start-of-travel ground flag:

1. Sample the movement-support block. Set `f` to modified block friction on ground,
   otherwise `1.0F`. Friction modification itself is specified later.
2. Ground acceleration coefficient is `speed * (0.21600002F/(f*f*f))` ONLY if
   `f > 0.6` (comparison with the double literal); otherwise it is `speed`.
   Airborne acceleration uses `getFlyingSpeed()` instead.
3. Add yaw-rotated normalized input times that acceleration to velocity using
   `moveRelative`. Apply climbable constraints, then call the swept collision
   movement routine with this velocity. This changes position and may zero or
   otherwise change velocity through collision/block callbacks.
4. Read back velocity. If `(horizontalCollision || jumping)` and on a climbable
   (or eligible powder-snow condition), replace Y by `0.2`.
5. With levitation, `vy += (0.05*(amplifier+1)-vy)*0.2`. Otherwise subtract effective
   gravity. Only the client's unloaded-chunk branch substitutes `-0.1` above the
   world's minimum Y, or zero below it.
6. Unless friction is discarded, multiply X/Z by
   `f * modifiedFriction(0.91F,AIR_DRAG_MODIFIER)` and Y by
   `modifiedFriction(0.98F,AIR_DRAG_MODIFIER)`. Omnidirectional air movers use
   modified `0.91F` vertically too. When friction is discarded, keep X/Z and the
   gravity-adjusted Y without these multipliers.

Important: displacement uses pre-gravity velocity. Gravity and drag establish the
next velocity; they are NOT applied before the collision displacement.
Effective gravity is `min(GRAVITY,0.01)` while falling (`vy<=0`) with slow falling,
otherwise `GRAVITY` (subject to the entity's no-gravity flag).

Climbable constraint: reset fall distance; clamp X/Z to `[-0.15F,0.15F]`, clamp Y
to at least `-0.15F`. A player suppressing ladder sliding sets negative Y to zero
unless inside scaffolding.

### 2.3 Water and Lava Integration

Capture `falling=(vy<=0)`, old position Y, and effective gravity `g` before moving.

Water:

1. `d = sprinting ? 0.9F : getWaterSlowDown()` (base `0.8F`), acceleration `a=0.02F`.
   Let `w=float(WATER_MOVEMENT_EFFICIENCY)`, halved when not on ground. If `w>0`,
   `d += (0.54600006F-d)*w`, `a += (speed-a)*w`. Dolphin's Grace replaces `d` with
   `0.96F` after this interpolation.
2. Add relative input using `a`; collide/move. Horizontal collision on a climbable
   sets Y to `0.2`. Multiply velocity by `(d,0.8F,d)`.
3. Apply fluid falling adjustment below, then the fluid-exit test.

Lava:

1. Add relative input using `0.02F`; collide/move.
2. If lava height is no greater than the fluid-jump threshold, multiply velocity
   by `(0.5,0.8F,0.5)` and apply fluid falling adjustment. Otherwise multiply all
   components by `0.5` without that adjustment.
3. If `g!=0`, add `-g/4` to Y. Then perform the fluid-exit test.

Fluid falling adjustment runs only with `g!=0` and NOT sprinting: if captured
`falling`, `abs(vy-0.005)>=0.003`, and `abs(vy-g/16)<0.003`, set `vy=-0.003`;
otherwise subtract `g/16`.

Fluid exit: with current post-drag velocity `v`, if horizontally colliding and
`isFree(vx, vy+0.6F-currentY+oldY, vz)`, set Y velocity to `0.3F`. `isFree` requires
both collision clearance and no liquid in the translated bounding box.

### 2.4 Player Input, Attributes, and Poses

Sources: [LocalPlayer.java](../minecraft-26.3-src/net/minecraft/client/player/LocalPlayer.java)
(`applyInput`, `modifyInput`, `modifyInputSpeedForSquareMovement`, `aiStep`),
[Player.java](../minecraft-26.3-src/net/minecraft/world/entity/player/Player.java)
(`createAttributes`, `getFlyingSpeed`, `travel`, `updatePlayerPose`),
[Attributes.java](../minecraft-26.3-src/net/minecraft/world/entity/ai/attributes/Attributes.java).

Distinguish two APIs: raw player controls need the LOCAL input transformation;
already transformed `(xxa,yya,zza)` must NOT receive it again. A headless server
player is not an ordinary AI-controlled living entity running keyboard travel.

Raw keyboard axes: forward minus backward and left minus right, each in {-1,0,1};
opposing keys cancel. `KeyboardInput.tick` NORMALIZES this Vec2 before the local
transformation below. Left is positive strafe. Analog actions must declare whether
they enter before or after this raw-input stage.

For a controlled local player, transform the input Vec2 in float arithmetic:

1. If its squared length is zero, return it. Scale by `0.98F`.
2. If using an item and not riding, scale by `itemUseSpeedMultiplier()` from that
   item's use-effects component. If crouching or visually crawling, also scale by
   float `SNEAKING_SPEED` (default `0.3`).
3. Compute length `L`; if positive let `d=input/L`,
   `t=min(abs(dx),abs(dz))/max(abs(dx),abs(dz))`,
   `squareDistance=sqrt(1.0F+t*t)`. Return
   `d * min(L*squareDistance,1.0F)`. This square adjustment is part of 26.3 behavior.
4. Assign this result to strafe and forward and assign held jump. The local
   override does NOT additionally call the base `applyInput` damping.

`Entity.moveRelative(a,input)` computes squared length in double; below `1.0E-7`
it adds nothing. Otherwise normalize only if squared length exceeds 1, scale by
float `a`, and rotate using float yaw sine/cosine:
`worldX=ix*cos-iz*sin`, `worldY=iy`, `worldZ=iz*cos+ix*sin`.

Defaults: player base movement speed `0.1F`, jump strength `0.42F`, gravity `0.08`,
step height `0.6` (read as float by `maxUpStep`), friction modifier and air-drag
modifier `1`, knockback resistance `0`. Player airborne acceleration is `0.02F`,
or `0.025999999F` while sprinting. Do not use the attribute registry's generic
movement-speed default `0.7` as the player default.

`setSprinting` removes the previous sprint modifier and, when enabled, adds a
`0.3F` ADD_MULTIPLIED_TOTAL movement-speed modifier. This is approximately a 1.3x
ground-speed multiplier, not an independent horizontal impulse; sprint jump is
the separate impulse in section 2.1. Attribute modifiers apply in this order:
add all ADD_VALUE to base, add that resulting base times each ADD_MULTIPLIED_BASE,
then multiply the running result by `(1+amount)` for each ADD_MULTIPLIED_TOTAL;
finally sanitize/clamp to the attribute's range. Preserve modifier iteration order.

`modifiedFriction(f,m)=clamp(1.0F-(1.0F-f)*m,0.0F,1.0F)`.
Living block-speed factor interpolates the entity's ordinary block-speed factor
toward 1 using float `MOVEMENT_EFFICIENCY`.

Player swimming first adjusts Y velocity toward look-vector Y with coefficient
`0.085` if look Y `<-0.2`, otherwise `0.06`. Do this only if look Y `<=0`, jump is
held, or fluid is nonempty at `(x,y+1.0-0.1,z)`.

Creative flight disables fluid influence. Local pre-travel ascent/descent adds
`(jump-shift)*abilities.flyingSpeed*3.0F` to Y. Player travel saves this Y, calls
ordinary living travel, then replaces Y with saved Y times `0.6`. Flying air
acceleration is ability flying speed, doubled for sprint. Non-spectator flight
turns off after landing. Spectator uses no-physics positioning.

Sprint state is an input state machine, not merely a speed toggle. Local start
requires forward impulse, sufficient food/flight ability, no blindness mobility
restriction, no disallowing item-use slowdown, no incompatible crouch/glide, and
eligible water depth. Sprint key or the configured double-forward window starts
it. Running stops on loss of eligibility/forward input or a non-minor horizontal
collision; swimming has its own stop predicate. For deterministic RL actions,
declare whether sprint actions bypass keyboard double-tap/auto-jump settings;
otherwise preserve these local state machines and their timers.

`isUnderWater` means PREVIOUS eye-water flag AND current in-water; shallow water
means current in-water AND not underwater. Food eligibility is `foodLevel>6.0F`
or `abilities.mayfly`. Sprint's water-depth predicate is this shallow-water state,
not just the numeric fluid-jump height.

For local minor-horizontal-collision classification, rotate current strafe/forward
fields into world X/Z WITHOUT speed scaling. If desired horizontal squared length
or accepted horizontal displacement squared length `<1.0E-5F`, return false.
Otherwise compute `acos(dot(desired,accepted)/sqrt(desiredLenSq*acceptedLenSq))`;
minor means angle `<0.13962634F` (about 8 degrees). Use accepted displacement,
not post-collision velocity. This result feeds the following sprint stop decision.

Pose update: if even swimming dimensions do not fit, leave pose unchanged.
Otherwise desired pose priority is sleeping, swimming, fall flying, spin attack,
crouching (shift and not flying), standing. Use desired pose if spectator/riding
or it fits; else crouching if it fits; else swimming. Fit test is collision of the
new-pose AABB at current feet, deflated by `1.0E-7`. Pose dimensions and eye heights
come from `Avatar.POSES` (section 8.2), not camera animation or only the base
player entity-type registration.

### 2.5 Gliding

`LivingEntity.updateFallFlyingMovement` is sequential. Let `l` be look vector,
`p` float pitch in radians, `H=sqrt(lx*lx+lz*lz)`, `S=horizontalSpeed(v)` captured
at entry, `q=Math.cos(p)^2` (standard double cosine, NOT lookup-table cosine), and
`g=effectiveGravity`:

1. `v.y += g*(-1+0.75*q)`.
2. If new `vy<0` and `H>0`, let `c=vy*(-0.1)*q`; add `(lx*c/H,c,lz*c/H)`.
3. If `p<0` and `H>0`, let `c=S*(-sin(p))*0.04`; add
   `(-lx*c/H,c*3.2,-lz*c/H)`.
4. If `H>0`, add `((lx/H*S-vx)*0.1,0,(lz/H*S-vz)*0.1)`.
5. Multiply by `(0.99F,0.98F,0.99F)`, assign velocity, then collision-move.

On climbables, use air travel and stop gliding instead. Server wall damage is
based on lost horizontal speed (`loss*10-3`), not a separate impact impulse.

## 3. Collision, Stepping, and Contact

Sources: [Entity.java](../minecraft-26.3-src/net/minecraft/world/entity/Entity.java)
(`move`, `collide`, `collideWithShapes`, `restituteMovementAfterCollisions`,
`checkSupportingBlock`, `push`),
[Direction.java](../minecraft-26.3-src/net/minecraft/core/Direction.java)
(`axisStepOrder`),
[VoxelShape.java](../minecraft-26.3-src/net/minecraft/world/phys/shapes/VoxelShape.java)
(`collideX`),
[EntityGetter.java](../minecraft-26.3-src/net/minecraft/world/level/EntityGetter.java)
(`getEntityCollisions`, `isUnobstructed`), and `Player.maybeBackOffFromEdge`.

### 3.1 Geometry and Swept Axis Resolution

Entity AABBs are axis-aligned regardless of yaw. Blocks contribute their STATE-
and collision-context-dependent VoxelShape, not necessarily a unit cube. Empty
collision shapes (including ordinary liquid) do not block walking. Outline/pick
shapes are a DIFFERENT query. Entity hard-collision shapes are filtered by
`source.canCollideWith(other)` and spectator/vehicle rules; do not treat every
living entity as an immovable obstacle.

For requested displacement `d`, query entity colliders in
`AABB.expandTowards(d).expandTowards(0,maxUpStep,0)`. Combine entity shapes, the
world-border shape when close enough, and block collision shapes over the swept
area. Preserve contextual block shape selection and block iteration semantics.

Axis order: Y first, then Z,X if `abs(dx)<abs(dz)`; otherwise X,Z (ties choose X).
For each axis, clip that axis's ORIGINAL requested amount against every shape
using the AABB translated by already accepted axis displacements. This is swept
clipping, not final-position overlap correction and not simultaneous XYZ motion.

Voxel clipping uses `1.0E-7`: displacement absolute value below this becomes zero
for a nonempty tested shape. The transverse occupancy indices use min+epsilon
and max-epsilon. For positive displacement, scan occupied voxels beyond the
leading max face and clamp to `obstacleMin-movingMax` if that gap `>=-epsilon`;
for negative displacement scan backward and clamp to `obstacleMax-movingMin`
if gap `<=epsilon`. Occupied transverse cells and internal shape coordinate
planes must be preserved. For full-cube-only worlds these reduce to ordinary
axis-separated AABB gap clipping with the same tolerances. Starting inside a
block does not invoke a general penetration solver.

### 3.2 Step-Up Selection

First compute ordinary clipped displacement `c`. Attempt steps only if
`maxUpStep>0`, (`onGround` OR downward requested Y was clipped), and X or Z was
clipped (exact comparisons here).

1. If downward Y clipped, start from original box translated by `c.y`; otherwise
   original box. Call this grounded box `G`.
2. Collect colliders in `G.expandTowards(dx,maxUpStep,dz)`; if no downward landing,
   also extend it downward by `1.0E-5F`.
3. For every collider Y coordinate plane, convert `planeY-G.minY` to float. Keep
   unique candidates in `[0,maxUpStep]`, excluding `float(c.y)`. Sort ascending.
4. For each candidate `h`, clip `(dx,h,dz)` from `G` with the same axis resolver.
   Return the FIRST result whose horizontal squared displacement is strictly
   greater than the ordinary result. Subtract
   `(0,originalBox.minY-G.minY,0)` before returning.
5. Otherwise keep ordinary result. Do not replace this with a fixed 0.6-block
   upward teleport or an arbitrary best-distance candidate search.

### 3.3 Move Side Effects and Restitution

`Entity.move` in order:

1. No-physics: directly add displacement, record movement, clear collision flags.
   Otherwise apply piston limiting when applicable. A nonzero stuck-speed
   multiplier (`lengthSquared>1.0E-7`) scales displacement except for pistons;
   then clear multiplier AND velocity.
2. Apply player edge backoff; compute collision/step result. Update feet position
   only when result length squared `>1.0E-7` OR requested length squared minus
   result length squared `<1.0E-7`. Record swept movement for block effects.
   Long falling movements also trace fall-distance-resetting blocks/water.
3. X/Z collision flags use `!Mth.equal(requested,accepted)`; Y uses exact inequality.
   When requested Y is nonzero OR entity is local-instance-authoritative, set
   vertical collision, downward collision, and ground flag. Ground means Y was
   clipped while trying to move downward, not simply touching a floor.
4. Update supporting block; run fall-damage handling; if still present, resolve
   velocity restitution when movement simulation is permitted and a relevant
   collision occurred. Then movement events and block-speed multiplication.
5. Multiply X/Z velocity (NOT displacement already applied) by block speed factor.
   Water/bubble-column use their current-cell factor; otherwise use current-cell
   factor if not 1, else the support factor. Jump factor follows a similar
   current-cell-then-support lookup.

Ordinary player entity restitution is zero. Blocked horizontal components become
zero. For Y, start with entity restitution `r`; on a downward hit set it to zero
if downward speed is no greater than effective gravity, bounce is suppressed,
or block suppresses bounce; otherwise use max(entity r, block bounce restitution).
With `r>0`, set `t=acceptedDy/oldVy`, `gravityComp=t*effectiveGravity`,
`effectiveDrag=lerp(t,1,getAirDrag())`; new Y is
`(gravityComp-oldVy)*effectiveDrag*r`. With `r=0`, new Y is zero. Horizontal bounce
is `-oldComponent*entityRestitution`. Invoke block `bounceOn` and sync when bounced.
Nonliving entities scale block restitution by `0.8F`. Subsequent travel gravity
and drag STILL run after this callback.

Supporting block search uses a slab from `AABB.minY-1.0E-6` to `AABB.minY`.
On failure after movement it can retry the slab shifted by `(-acceptedDx,0,
-acceptedDz)` and retain `onGroundNoBlocks`. Movement support sampling uses
offset `0.500001F`; legacy impact effects use `0.2F`; ordinary on-position uses
`1.0E-5F`. A stored support position and fence/wall/gate cases change the lookup.
Do not collapse these to `floor(feetY-1)`.

### 3.4 Sneaking at Edges

For self/player moves, not flying, requested `dy<=0`, staying-on-ground-surface,
and above-ground eligibility: reduce X toward zero in increments of `0.05` while
the X-only offset could fall at least step height; then reduce Z independently;
then reduce both together while the diagonal offset could fall. Snap a component
to zero when its absolute value is `<=0.05`. Preserve this X, Z, diagonal order.

`canFallAtLeast(dx,dz,h)` tests `noCollision` for the thin prism from
`box.minY-h-1.0E-7` to `box.minY`, with horizontal min inset `1.0E-7`, max inset
`1.0E-7`, shifted by `(dx,dz)`. Above-ground means on ground OR fall distance
`<stepHeight` and unable to fall `stepHeight-fallDistance` at zero horizontal
offset. This changes attempted displacement without directly damping velocity.

### 3.5 Entity Contact Pushes Versus Solid Collision

After travel and block effects, `LivingEntity.pushEntities` queries pushable
entities overlapping its current AABB and invokes `doPush` in returned order.
The ordinary `Entity.push(other)` skips shared-vehicle pairs and no-physics:

1. `dx=other.x-self.x`, `dz=other.z-self.z`, `m=max(abs(dx),abs(dz))`.
2. If `m<0.01F`, no push. Otherwise `s=sqrt(m)` (NOT Euclidean distance),
   `k=min(1,1/s)*0.05F`, `impulse=(dx/s*k,0,dz/s*k)`.
3. Add negative impulse to self if pushable and not a vehicle; add positive
   impulse to other under the same independent conditions. Mark changed motion.

This is neither mass-based rigid-body resolution nor combat knockback. It can
run for both members during their ticks; do not globally deduplicate unordered
pairs unless reproducing a source rule that does so. Specialized entities can
override pushing. Building obstruction is also distinct: placement checks
`blocksBuilding` and actual shape intersection, not just pushability.

## 4. Block and Bucket Placement

Sources: [BlockItem.java](../minecraft-26.3-src/net/minecraft/world/item/BlockItem.java),
[BlockPlaceContext.java](../minecraft-26.3-src/net/minecraft/world/item/context/BlockPlaceContext.java),
[BucketItem.java](../minecraft-26.3-src/net/minecraft/world/item/BucketItem.java),
[Item.java](../minecraft-26.3-src/net/minecraft/world/item/Item.java)
(`getPlayerPOVHitResult`),
[BlockGetter.java](../minecraft-26.3-src/net/minecraft/world/level/BlockGetter.java)
(`clip`, `clipWithInteractionOverride`, `traverseBlocks`),
[BlockBehaviour.java](../minecraft-26.3-src/net/minecraft/world/level/block/state/BlockBehaviour.java)
(`canBeReplaced`), and
[Level.java](../minecraft-26.3-src/net/minecraft/world/level/Level.java) (`setBlock`).

### 4.1 Targeting and Ordinary Blocks

Item POV ray starts at the current eye position and ends at
`eye+calculateViewVector(pitch,yaw)*blockInteractionRange`. Default survival block
range is `4.5`; entity interaction range is a different attribute (default `3`).
Use OUTLINE block shapes for item targeting, not movement COLLIDER shapes.
`BlockGetter.clip` traverses voxels in ray order, tests block and permitted-fluid
shapes, and chooses block on equal block/fluid hit distance. Interaction shapes
may replace the hit FACE without replacing the collision location. Face direction,
inside-hit flag, exact hit point, selected hand, and player orientation are inputs
to placement; selecting the nearest block center is not equivalent.

Block item placement, after interaction permissions and block-use handling:

1. Construct `BlockPlaceContext`. `replaceClicked` is the clicked state's
   `canBeReplaced(context)`. If true, destination is clicked cell; otherwise it is
   clicked cell plus hit-face normal. Placement requires replacement of clicked
   cell OR `destination.canBeReplaced(context)`.
2. Require block feature enabled; run item's `updatePlacementContext` (ordinary
   BlockItem returns same context); obtain `block.getStateForPlacement(context)`.
   For a plain non-directional cube this is the default state. Specialized blocks
   select orientation, waterlogging, slab half, etc.; do not invent generic rules.
3. Require nonnull state, `canSurvive` if item requires it (ordinary BlockItem does),
   and `level.isUnobstructed(state,destination,placementContext(player))`.
   This checks placement collision SHAPE against nonremoved entities with
   `blocksBuilding`, with the obstruction query's vehicle rules. It is NOT just a
   player-center check, and ordinary blocks must not be placed through a player's
   body. `CollisionGetter` passes a null excluded entity to this obstruction check.
4. Execute `setBlock(destination,state,11)`; fail if it returns false. Read resulting
   world state again. If still the same block type, apply item block-state
   component, block-entity data/components, then `setPlacedBy` and placement trigger.
5. Sound/event and consume one item (creative/infinite-material handling remains
   in the item-consumption logic). Placement is immediate, not deferred to the
   next entity movement tick.

Default block replacement is `state.canBeReplaced()` AND (held item empty OR not
the same block's item). Fluid replacement instead defaults to replaceable OR
not solid. Individual block overrides control exceptions.

`setBlock` rejects invalid world bounds and server debug-world mutation. After
chunk mutation, only when the read-back state is the requested state: mark dirty,
send updates if bit 2 permits, notify neighbors if bit 1 set, then (unless bit 16
set and while recursion budget remains) run old indirect shape updates, new
neighbor shape updates, new indirect shape updates, and POI change handling.
Default shape recursion budget is 512. Flag 11 includes neighbor/client updates;
do not implement placement as a raw voxel-array write. Chunk mutation/on-place
callbacks can schedule fluid/block ticks before `setPlacedBy` runs.

### 4.2 Water/Lava Buckets and Pickup

Filled bucket targeting uses `Fluid.NONE`; empty bucket uses `Fluid.SOURCE_ONLY`.
Both still trace OUTLINE block shapes. Require a block hit, `level.mayInteract`
at clicked cell, and `player.mayUseItemAt` at the face-adjacent cell.

For water on a `LiquidBlockContainer`, initially try clicked cell; otherwise try
face-adjacent cell. `emptyContents` accepts only `FlowingFluid` contents:

1. Let `replace=state.canBeReplaced(content)`. Let `accept=replace OR container
   canPlaceLiquid(...)`. A destination is eligible if AIR OR
   (`accept` AND (shift NOT held OR fallback hit-result is null)).
2. On ineligible first target, retry exactly the original face-adjacent cell with
   a null hit result, then stop on failure. Thus shift can bypass filling an
   initial occupied/container cell, but is not a global prohibition on placement.
3. If water evaporates according to the environment attribute at the target,
   report success with effects but do NOT place water.
4. If water and target is a liquid container, call `placeLiquid` with nonfalling
   source state and return success. Otherwise, server destroys replaceable
   nonliquid blocks with drops, then sets the fluid's default legacy block using
   flags 11. A failed write can still count as success if old fluid was a source.
5. Apply bucket result/inventory handling. There is no BlockItem-style entity
   obstruction test for a normal liquid source placement.

If emptying failed and content is EMPTY, invoke clicked block's `BucketPickup`.
Only a nonempty pickup result succeeds; source-liquid and waterlogged-block
pickup are governed by their block implementations, not arbitrary removal of a
flowing cell. Filling/emptying modifies the world synchronously; propagation is
scheduled (section 6).

## 5. Combat Physics

Sources: `LivingEntity.hurtServer`, `dealDefaultKnockback`, `knockback`,
`applyItemBlocking`, `getKnockback`, `getItemBlockingWith`; `Player.attack`,
`causeExtraKnockback`, `doSweepAttack`, `blockUsingItem`; and
[BlocksAttacks.java](../minecraft-26.3-src/net/minecraft/world/item/component/BlocksAttacks.java),
[UseEffects.java](../minecraft-26.3-src/net/minecraft/world/item/component/UseEffects.java),
[Items.java](../minecraft-26.3-src/net/minecraft/world/item/Items.java) (shield registration).
Damage magnitude is relevant only insofar as it selects success, cooldown,
blocking, death, and effect branches; these branches cannot be dropped even if
health reward design is outside the simulator's focus.

### 5.1 Reusable Living Knockback Primitive

`knockback(power,xd,zd,source,damage,comesFromEffect)`:

1. `power *= 1-KNOCKBACK_RESISTANCE`; return if `power<=0`. Resistance can be
   negative in this version; do not clamp it to [0,1] for this primitive.
2. While `xd*xd+zd*zd < 1.0E-5F`, redraw each component as
   `(nextDouble()-nextDouble())*0.01` from the victim's random source.
3. `k=normalize((xd,0,zd))*power`; read existing velocity `v`.
4. Assign `(vx/2-kx, onGround ? min(0.4,vy/2+power) : vy, vz/2-kz)`.
   Mark motion synchronization. This SUBTRACTS the supplied horizontal direction.

There is no position teleport, immediate collision move, mass division, or
airborne Y impulse here. Later movement ticks integrate the changed velocity.

### 5.2 Damage Gating and Default Knockback

After basic invulnerability/alive checks, apply item blocking and subtract blocked
damage. Ordinary damage cooldown branch: if `damageCooldownTime>10` and source
does not bypass cooldown, reject when incoming remaining damage `<=lastHurt`;
otherwise apply only excess and mark `tookFullDamage=false`. New full-damage hits
set cooldown 20, last damage, hurt duration/time 10.

Only `tookFullDamage` hits perform default knockback, and only if damage type does
not have NO_KNOCKBACK and the hit was NOT fully blocked. NO_IMPACT separately
controls the hurt/synchronization marker, not that knockback predicate.

Default power is `0.4F`. For a direct projectile, horizontal input direction is
NEGATIVE its `calculateHorizontalHurtKnockbackDirection` (base projectile returns
its current X/Z velocity). For other positioned sources, input is
`sourcePosition-victimPosition` in X/Z. Pass through the primitive above, so the
result pushes away from source / along projectile travel. A missing/degenerate
direction uses the primitive's random fallback.

### 5.3 Player Sword/Melee Sequence

Attack charge delay is `float(1/ATTACK_SPEED*20)`. Charge
`c=clamp((attackStrengthTicker+0.5F)/delay,0,1)`; fully charged means `c>0.9F`.
Base damage scales by `0.2F+c*c*0.8F`; enchantment bonus scales by charge. Reset
attack ticker on attack. Item minimum-charge restrictions also gate the outer
interaction handler. Critical conditions include positive fall distance,
airborne, not climbable/in water/blindness-restricted/riding/sprinting, and a
living target. Critical damage is multiplied by 1.5; it is NOT a separate impulse.

1. Check attackability/skip interaction, compute charge/damage, and handle eligible
   redirectable projectiles by aim deflection rather than ordinary entity damage.
2. Sprint knockback bonus is selected only if sprinting AND fully charged.
3. Record victim's old velocity; invoke victim damage pipeline, which can apply
   default knockback FIRST. Continue extra knockback/sweep only on successful hurt.
4. Extra power is `getKnockback(target,source)+(sprintBonus ? 0.5F : 0)`.
   `getKnockback` is enchantment-modified ATTACK_KNOCKBACK divided by `2.0F`.
   If positive and victim living, call primitive with
   `(sin(attackerYaw),-cos(attackerYaw))`. Nonliving victim receives additive
   `(-sin(yaw)*power,0.1,cos(yaw)*power)` instead.
5. A positive extra power also multiplies attacker's X/Z velocity by `0.6` and
   clears sprint. No extra power means this slowdown/sprint clearing does not run.
6. Evaluate sweep/post-hit effects in source order. For a server-player victim
   requiring velocity sync, send the motion packet and restore saved victim
   velocity on the server; the receiving local player's simulation applies it.
   A unified headless sim must apply the impulse ONCE, not mimic the restore
   while omitting the client application.

Sweep requires fully charged, not critical, not sprint-knockback, grounded, main
hand in SWORDS tag, and attacker known horizontal speed squared
`<(getSpeed()*2.5)^2`. Search victim AABB inflated `(1,0.25,1)`; exclude attacker,
primary target, allies, marker armor stands; require attacker-to-candidate feet
distance squared `<9`. Each successfully hurt candidate receives an additional
`0.4F` yaw-directed primitive knockback, after its own damage response.

### 5.4 Shields, Direction, Delay, and Disabling

Default shield component: block delay `0.25F` seconds -> `round(seconds*20)=5`
ticks; horizontal half-angle `90.0F`; reduction base 0, factor 1; bypass tag
BYPASSES_SHIELD. Blocking is active only while using the item and elapsed use
ticks `>=blockDelayTicks`. Ordinary item use defaults to movement multiplier
`0.2F` and disallows sprint through `UseEffects.DEFAULT`.

`applyItemBlocking` returns zero for nonpositive damage, no active blocking item,
bypassing damage type, or direct AbstractArrow with pierce level >0. Otherwise:

1. Compute horizontal head view with pitch ZERO and defender HEAD yaw. Normalize
   horizontal `sourcePosition-defenderFeet`; `angle=acos(dot(toSource,headView))`.
   Missing source position uses float PI. Do not use pitch, attacker look
   direction, or body yaw in place of head yaw.
2. For each reduction matching damage type and with
   `angle <= float(PI/180)*horizontalBlockingAngle`, add
   `clamp(base+factor*damage,0,damage)`, then clamp total to incoming damage.
3. Damage blocking item according to its component. Default shield durability
   function: below 3 damage no cost, otherwise `floor(1+blockedDamage)`.
4. Positive blocked damage from direct living, nonprojectile attacker invokes
   `blockUsingItem`. The base attacker's `blockedByItem` only knocks the DEFENDER
   with power `0.5` when the hit is PARTIALLY blocked, using
   `defenderPosition-attackerPosition` as primitive input. Preserve this literal
   direction; do not replace it with a remembered older-version shield recoil.
5. Player defender additionally checks attacker's `getSecondsToDisableBlocking()`;
   positive duration disables using its component scale, rounded seconds*20,
   sets item cooldown, and stops using. The active weapon's WEAPON component
   controls duration; ordinary axe/default-shield values are in section 10.5.

A fully blocked hit suppresses the normal damage knockback and reports failed
hurt, so ordinary successful-hit sword bonuses and arrow post-hit knockback do
not run. Partial blocks can still receive the later default knockback.

### 5.5 Bow Launch and Arrow State

Sources: [BowItem.java](../minecraft-26.3-src/net/minecraft/world/item/BowItem.java),
[Projectile.java](../minecraft-26.3-src/net/minecraft/world/entity/projectile/Projectile.java),
[AbstractArrow.java](../minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java),
[ProjectileUtil.java](../minecraft-26.3-src/net/minecraft/world/entity/projectile/ProjectileUtil.java).

For held bow time `t`, float `p=t/20.0F`, then `p=(p*p+2*p)/3`, capped at 1.
Release below `0.1` does not shoot. Arrow speed is `p*3.0F`, uncertainty `1.0F`;
full power marks critical. Spawn is `(shooterX,eyeY-0.1F,shooterZ)`.

`shootFromRotation` forms float direction
`(-sin(yaw)*cos(pitch), -sin(pitch+yOffset), cos(yaw)*cos(pitch))` (bow offset 0).
Normalize it, independently add triangle noise to XYZ with deviation
`0.0172275*uncertainty`, then scale by speed. Do NOT renormalize after noise.
Triangle draw is two uniform draws' difference times deviation. Finally add
shooter KNOWN X/Z movement and known Y only when shooter is not on ground.
Known movement is not always the raw velocity field, especially for server players.

Persist position, velocity, owner reference, left-owner flag, tick count,
in-ground flag, last block state, shake/life timers, critical/pierce flags,
pierced-entity IDs, pickup mode, weapon/components, and RNG.

### 5.6 Arrow Tick, Sweep, Hit, and Deflection

Arrow tick is NOT the living-entity movement routine:

1. Capture velocity `m` at tick entry. If point position is inside a current block
   collision shape and physics enabled, zero velocity and mark in ground.
   Decrement shake; extinguish in water/rain. Embedded arrows check changed block
   state and a `0.06`-inflated point-box clearance before falling, else despawn
   counting. Start-falling scales each current velocity component by an independent
   `nextFloat()*0.2F`; it does not recreate impact velocity.
2. If airborne and currently in water, apply water inertia `0.6F` to STORED velocity
   BEFORE hit processing. The swept segment still uses captured `m`: position
   to `position+m`. This distinction affects impact speed and impulses.
3. Trace COLLIDER blocks including border, fluid NONE, to that segment end. Check
   owner exclusion. Entity broadphase is arrow AABB expanded by CURRENT velocity,
   inflated 1, with the segment clipped to block-hit position.
4. Entity margin is `max(0F,min(0.3F,(tickCount-2)/20.0F))`, not constant 0.3.
   Prefer exact target-AABB segment clip. If only inflated-margin box clips, trace
   from margin hit toward target box center, clipped by blocks, and require it
   reaches the actual target box; for arrows record the margin-hit location.
   Respect projectile hittability, owner/shared-vehicle exclusion until outside
   owner collision range, team PvP rules, and already-pierced exclusions.
5. Sort candidate hits by squared distance from arrow's current position to each
   ENTITY POSITION (not necessarily to segment intersection). Move arrow to the
   first hit location, apply traversed block effects and portal handling. Process
   candidate hit/deflection callbacks in this sorted order while alive. Piercing
   can repeat the loop when not deflected; preserve the ignore-ID set.
6. With no entity candidates, move to block hit/end; apply effects; if an actual
   block hit remains valid, invoke hit/deflect callback. Arrow block hit remembers
   block state, invokes base block callback and enchantment effects, backs position
   off by `sign(currentVelocityComponent)*0.05F` on EACH axis, zeros velocity,
   sets in-ground, shake 7, clears critical/pierce and pierced-entity tracking.
7. After callbacks, if NOT currently in water multiply velocity by `0.99F`.
   If physics enabled and not embedded, subtract arrow gravity `0.05` from Y.
   Then call base projectile/entity tick. No-physics arrows use direct displacement
   and block effects instead of hit sweeps; no gravity.

Entity hit damage uses current velocity length. Successful hurt first triggers
the victim's default `0.4F` knockback; then arrow's `doKnockback` adds an optional
weapon/enchantment knockback: normalize current horizontal arrow velocity and
scale by `modifiedKnockback*0.6*max(0,1-victimResistance)`, then ADD
`(kx,0.1,kz)` only if horizontal scaled vector has positive squared length.
This is not the living primitive and does not halve existing velocity. Ordinary
nonpiercing arrows discard after success. Rejected hurt can restore target fire
state and REVERSE-deflect the arrow at power `0.2`; very small resultant motion
(`lengthSquared<1.0E-7`) can produce pickup and discard. REVERSE multiplies each
velocity component by `-powerComponent*0.5` (so scalar power 0.2 yields -0.1),
rotates display yaw by `170F+nextFloat()*20F`, and marks sync. AIM_DEFLECT replaces
velocity with deflector's look vector times power; it does not preserve old speed.
Entity deflection behavior may intercept before the ordinary hit callback.
Preserve projectile-class rules;
snowballs, tridents, rockets, and other projectiles are not all arrows.

## 6. Liquid States, Propagation, and Currents

Sources: [FlowingFluid.java](../minecraft-26.3-src/net/minecraft/world/level/material/FlowingFluid.java),
[WaterFluid.java](../minecraft-26.3-src/net/minecraft/world/level/material/WaterFluid.java),
[LavaFluid.java](../minecraft-26.3-src/net/minecraft/world/level/material/LavaFluid.java),
[LiquidBlock.java](../minecraft-26.3-src/net/minecraft/world/level/block/LiquidBlock.java),
[EntityFluidInteraction.java](../minecraft-26.3-src/net/minecraft/world/entity/EntityFluidInteraction.java),
`Entity.updateFluidInteraction`, and
[GameRules.java](../minecraft-26.3-src/net/minecraft/world/level/gamerules/GameRules.java).

### 6.1 Representation and Delays

Fluid state and liquid block LEVEL are not interchangeable. Source amount is 8,
flowing amount is its fluid LEVEL, and FALLING is separate. Fluid own height is
`amount/9.0F`; height is `1.0F` when same fluid is above, otherwise own height.
Legacy block LEVEL is 0 for source; otherwise
`8-min(amount,8)+(falling ? 8 : 0)`. Liquid block states 1..7 map to nonfalling
amount 7..1; block LEVEL >=8 maps to falling amount 8. Keep source identity even
when a falling column has the same amount.

| Parameter | Water | Lava, normal | Lava, FAST_LAVA |
| --- | --- | --- | --- |
| Scheduled delay, ticks | 5 | 30 | 10 |
| Side amount drop-off | 1 | 2 | 1 |
| Slope search bound | 4 | 2 | 4 |
| New-source conversion game rule default | true | false | false |
| Entity current scale | 0.014 | 0.0023333333333333335 | 0.007 |

FAST_LAVA is an environment/dimension attribute, not a hardcoded dimension-name
test. Lava spread delay multiplies by 4 when old/new states are nonempty,
nonfalling, new surface height is greater, and `level.random.nextInt(4)!=0`.

Liquid `onPlace` and neighbor changes run reaction checks then schedule a fluid
tick at its delay; shape updates also schedule when self OR changed neighbor is
a source. Same-type/position pending ticks are deduplicated by the scheduler.
Source pickup requires liquid block LEVEL==0 and writes AIR with flags 11.

### 6.2 Recompute Fluid at a Cell

`getNewLiquid(level,pos,blockState)`:

1. Visit horizontal neighbors NORTH,EAST,SOUTH,WEST. Across passable shared faces,
   count same-fluid sources and find maximum same-fluid amount.
2. If at least two sources and source-conversion rule enabled, return nonfalling
   source if below is solid OR a source of this same fluid.
3. Otherwise, if same fluid above with a passable shared face, return flowing
   amount 8, FALLING=true.
4. Otherwise `amount=maxNeighbor-dropOff`; return EMPTY if `amount<=0`, else
   flowing with that amount and FALLING=false.

Shared-face passage is based on the TWO block collision shapes and their merged
face occlusion, not just destination air. Full-cube collision on either side
blocks passage; two empty shapes allow it. Preserve dynamic-shape/context rules.
`canHoldAnyFluid` is true for a LiquidBlockContainer or a state in the
WASHED_AWAY_BY_FLUIDS tag, NOT every nonsolid state. Specific-fluid container
acceptance adds another gate. Read tags from this version's data pack.

### 6.3 Scheduled Spread Procedure

On a fluid tick, first validate that current cell's fluid still has the scheduled
type. Non-sources recompute their state: if EMPTY, replace with AIR; if changed,
write new legacy block and schedule its type at spread delay. Sources retain
state. Then call spread with the current/new state.

1. If nonempty, try DOWN first. Require non-source destination of same type,
   generic holding eligibility and shared-face passage; compute new below state,
   then require destination-fluid replacement permission and specific acceptance.
2. On successful downward spread, also spread sideways only if current cell has
   at least THREE horizontal source neighbors, then return.
3. If downward spread did not succeed, spread sideways if source OR below is NOT
   a water-hole. A water-hole requires downward face passage and either same
   fluid below or below able to hold this flowing fluid.
4. Side spreading runs only if `amount-dropOff>0` (falling overrides this local
   test amount to 7). For each eligible horizontal neighbor compute new fluid and
   distance to a drop: 0 if immediate water-hole; otherwise depth-first bounded
   slope search excluding the immediate reverse direction, starting pass=1.
   An encountered drop returns pass; no route returns sentinel 1000. Recurse only
   while pass<slope bound.
5. Keep all directions tied for minimum distance, clearing earlier choices on a
   strictly smaller distance. Check target fluid replacement permission before
   inserting a direction; the minimum distance STILL updates when that permission
   rejects it. Result is an enum map, so application order follows Direction enum
   order (NORTH,SOUTH,WEST,EAST for these horizontal keys), not search order.
6. Apply each spread: containers use `placeLiquid`; other targets invoke
   before-destroy callback if nonair then `setBlockAndUpdate` with legacy fluid.
   These writes run neighbor callbacks and enqueue future ticks synchronously.

Liquid reactions also run synchronously on relevant placement/neighbor callbacks.
Lava adjacent to water in `LiquidBlock.shouldSpreadLiquid` becomes OBSIDIAN if
source, otherwise COBBLESTONE; its examined neighbors are described by that
method's POSSIBLE_FLOW_DIRECTIONS: iterate DOWN,SOUTH,NORTH,EAST,WEST and inspect
the OPPOSITE neighbor, i.e. above,north,south,west,east in that order. Lava over soul soil with
an examined blue-ice neighbor becomes BASALT. Separately, lava spreading DOWN
into water makes STONE when water block is a LiquidBlock, fizzes, and returns
without ordinary spread. Do not implement all water/lava contact as one rule.

### 6.4 Flow Vector at a Fluid Cell

Visit horizontal neighbors N,E,S,W; only same fluid or empty fluid affects flow.
For positive neighbor own-height, slope=`selfOwnHeight-neighborOwnHeight`.
For zero height and neighbor not BLOCKS_FLUID_FLOW, inspect below neighbor:
if same/empty eligible fluid with positive height, slope=
`selfOwnHeight-(belowNeighborOwnHeight-0.8888889F)`; otherwise zero.
Accumulate `faceNormalXZ*slope` in double. If FALLING and an eligible solid
neighbor face at the same height OR one above exists, normalize horizontal sum
and add `(0,-6,0)` once. Return normalized resulting vector. Ice is specially
excluded from those solid-face tests. A flat uniform pool has zero current.

### 6.5 Fluid Contact and Applying Currents to Entities

During `Entity.baseTick`, before living travel, call `updateFluidInteraction`:

1. Reset trackers/currents. Interaction box is entity AABB deflated `0.001`, with
   vehicle modifications if any. Require all relevant chunks FULL/loaded; missing
   required chunks make update return without ordinary fluid detection.
2. Scan X outermost, then Y, then Z, from floor(min) to ceil(max)-1 inclusive.
   Nonempty fluid contributes only if its top is NOT below interaction-box minY.
   Track maximum fluid top minus ORIGINAL entity AABB minY, separately by type.
   In-fluid means tagged maximum height>0. Eyes use the eye column and
   `getHeightForCamera`, not this feet-height test.
3. Accumulate each flow vector. Update the current's running MAX immersion height;
   if that running height<0.4, scale THIS cell's vector by it. This accumulation
   depends on the scan order; do not average per-cell depths first.
4. If current count nonzero and squared SUM magnitude is not `<1.0E-5F`, use
   SUM/count for players; normalized SUM for nonplayers. Multiply by current scale.
5. If old `abs(vx)<0.003` AND `abs(vz)<0.003` and resulting impulse length is
   `<0.0045000000000000005`, normalize and scale impulse to that minimum.
   Add impulse to velocity; preserve separate water then lava application order.

Water contact resets fall distance. Lava halves fall distance later in base tick.
`isInLava` additionally excludes first tick. Swimming state is updated after
fluid interaction: existing swim requires sprint+water+not passenger; starting
requires sprint+underwater+water at feet+not passenger. Flying player overrides
swimming off. Other movement callbacks (notably living fall checks) can refresh
fluid interaction too; do not assume there is exactly one sample per world tick.

## 7. Processing and Tick Order

Sources: [MinecraftServer.java](../minecraft-26.3-src/net/minecraft/server/MinecraftServer.java)
(`tickServer`, `tickChildren`),
[ServerLevel.java](../minecraft-26.3-src/net/minecraft/server/level/ServerLevel.java)
(`tick`, `tickTime`, `tickNonPassenger`, `tickPassenger`),
[ServerPlayer.java](../minecraft-26.3-src/net/minecraft/server/level/ServerPlayer.java)
(`tick`, `doTick`),
[ServerGamePacketListenerImpl.java](../minecraft-26.3-src/net/minecraft/server/network/ServerGamePacketListenerImpl.java),
[Minecraft.java](../minecraft-26.3-src/net/minecraft/client/Minecraft.java)
(`tick`),
[LevelTicks.java](../minecraft-26.3-src/net/minecraft/world/ticks/LevelTicks.java),
[ScheduledTick.java](../minecraft-26.3-src/net/minecraft/world/ticks/ScheduledTick.java).

### 7.1 Server World Timeline

Normal, unpaused, unfrozen operation, at the default 20 simulation ticks/second:

1. Server tick-rate bookkeeping, command functions; clocks when running normally.
2. Iterate server levels in their stored level order. For EACH `ServerLevel.tick`:
   invalidate environment cache; world border and weather; sleep/wake handling;
   sky brightness; increment game time and run due scheduled server events.
3. Unless debug/frozen: process scheduled BLOCK ticks (cap 65,536), then scheduled
   FLUID ticks (separate cap 65,536), using the new game time.
4. Raid processing; chunk-source tick (includes relevant chunk/random ticking,
   loading/tickets and tracking work); block-event queue.
5. Active-world entity loop: skip removed/frozen entities; check despawn; require
   player OR entity-ticking range. Mounted entities defer to their vehicle's
   recursive passenger tick. Nonpassenger: `commonTick()` then `tick()`, then
   recursively passengers via `commonTick()` and `rideTick()`. This is sequential
   entity mutation, NOT a simultaneous snapshot update of every entity.
6. Tick block entities AFTER entity loop, then entity management/debug maintenance.
   The entity/block-entity branch is gated by empty-world inactivity (`emptyTime<300`).
7. After ALL worlds, tick network connections, player-list maintenance, debug/test
   systems, chunk sending and flush. Connection tick invokes server-player
   `doTick` as described next.

Tick-rate freeze, sprinting the server, inactive/unloaded chunks, and pause change
eligibility/timing. Never model block random ticks as scheduled fluid ticks.

### 7.2 Living/Player Internal Timeline

`commonTick` decrements general invulnerability time, snapshots old position and
rotation, performs client interpolation if appropriate, increments tickCount.
`tick()` is separate and must not increment tickCount a second time.

For ordinary living `tick`: `super.tick` dispatches living `baseTick`, which
updates swing/enchantment ticks, then entity base tick (known speed, cached state,
portal/cooldowns, fluid contact/current, swim state, fire/lava/below-world), then
living environment damage, hurt/cooldown timers and status-effect ticks.
After base tick: update use-item countdown and swim amount, equipment detection,
sleep validity, then `aiStep` in section 2.1, then orientation/glide counters and
dirty attribute refresh. Effect expiry and item use thus precede travel.

`Player.tick` wraps that path: set spectator/no-physics and mounted ground state,
sleep/underwater bookkeeping; living tick; clamp X/Z to +/-29,999,999; increment
attack/item-swap tickers; reset charge when main-hand item type changed; tick item
cooldowns; update pose LAST. `LocalPlayer.aiStep` adds keyboard input, crouch,
sprint, fly/glide transitions, water descent, and creative vertical input before
calling `super.aiStep`.

SERVER player is different: `ServerPlayer.tick` in world loop does game mode,
damage cooldown decrement, attributes, containers/advancements, but does NOT call
ordinary `Player.tick`. During later connection tick, `tickPlayer` resets saved
positions, calls `ServerPlayer.doTick` -> `Player.tick`, then snaps back to the
connection's accepted position. Normal ServerPlayer does not own local keyboard
movement simulation; movement packets are processed by `handleMovePlayer` and
`handlePlayerPositionChange`, which collision-move their displacement and either
accept the reported position/flags or correct it.

### 7.3 Actions, Client Order, and Networking Boundary

Client tick: game-mode maintenance; `pick(1.0F)`; keybind actions (attack/use);
client entity ticks (including local input+travel); client block entities; send
local player changes; then client-level tick. Attack/use can therefore occur
before that tick's new local movement. Do not automatically move the agent first
and aim/place from its post-move eye position.

Server packet handlers enqueue to/execute on the server thread. Melee, block use,
bucket use and move acceptance mutate state when their handlers execute, not in
a universal final 'combat phase'. For genuine multiplayer parity, the action and
movement packet sequence and its position among server tasks are part of input.
A fixed universal ordering such as 'all movement, all attacks, all placement' is
NOT a fact established by Minecraft's source.

The use-on-block handler validates eye/block interaction range with margin 1,
hit position components relative to block center `<1.0000001`, bounds, spawn
protection, pending teleports, and game-mode eligibility. It consumes the reported
hit result rather than always replacing it with a fresh server eye raycast.
Movement acceptance and client prediction/reconciliation are network behavior;
they are not additional gravitational integrations in a unified CPU model.

For a headless single-authority RL environment, choose and record an explicit
schedule, for example: apply ordered pre-move actions using old pose/eyes, advance
world scheduled/block/fluid work, then tick each entity using local locomotion
and server combat callbacks exactly once. This is a DEFINED ADAPTER, not a claim
of an existing universal Minecraft client/server tick. Compare against a game
reference with the same observation/action boundaries and event ordering.

### 7.4 Scheduled-Tick Ordering

Persist `(type,position,triggerTick,priority,subTickOrder)` plus per-chunk queues.
Chunk-local priority queue comparator: trigger time, then priority, then sub-tick
order. At collection, only due queues whose chunks are eligible enter the global
container queue. That queue compares each chunk's NEXT tick by priority THEN
sub-tick order, WITHOUT comparing trigger time (all selected entries are due).
Drain a chunk while its next entry is due, capacity remains, and it does not rank
after another eligible queue's next entry by this intra-tick comparator.

Collect the tick batch before running callbacks; newly scheduled same-kind ticks
do not recursively join that already collected batch. Block callbacks can affect
the subsequent fluid collection, since block and fluid queues are collected/run
separately. Execute a scheduled callback only if current cell still matches its
scheduled block/fluid type. Same-type/position uniqueness is independent of due
time. This matters for liquid frontier shape and reactions at boundaries.

## 8. CPU Implementation Details

### 8.1 Math Helpers and Exactness

Sources: [Mth.java](../minecraft-26.3-src/net/minecraft/util/Mth.java),
[Vec3.java](../minecraft-26.3-src/net/minecraft/world/phys/Vec3.java),
[Vec2.java](../minecraft-26.3-src/net/minecraft/world/phys/Vec2.java),
[AttributeInstance.java](../minecraft-26.3-src/net/minecraft/world/entity/ai/attributes/AttributeInstance.java),
[RandomSource.java](../minecraft-26.3-src/net/minecraft/util/RandomSource.java).

- `Vec3.normalize`: double length; if length `<1.0E-5F`, return zero, else divide
   components by length. Do not confuse this with squared-length cutoffs elsewhere.
- `Mth.equal(a,b)`: `abs(b-a)<1.0E-5F`, even for double arguments. This is why
   move collision flags can disagree with exact-comparison step-selection tests.
- `Mth.sqrt(float)` is double square root CAST BACK to float. Floor/ceil use
   mathematical floor/ceil, then integer conversion. Java casts to integer/long
   truncate toward zero (relevant for negative lookup-table inputs).
- Sine table: `table[index]=float(Math.sin(index/10430.378350470453))`,
   indices 0..65535. `sin(angle)` reads
   `table[long(angle*10430.378350470453)&65535]`; `cos(angle)` uses
   `long(angle*10430.378350470453+16384.0)&65535`. Do not substitute native sin/cos
   for these helpers. Some formulas explicitly use `Math.cos` instead; keep that.
- Ground `f>0.6` compares float f promoted to double against DOUBLE 0.6.
   Ordinary `0.6F` is slightly greater than double 0.6 and enters the formula.
   Preserve this instead of changing the literal to `0.6F` during a port.
- Disable reassociation/fast-math/FMA contraction where it changes Java operation
   results. For bitwise claims also pin platform math/transcendental behavior and
   reference table bits. 'Same formulas within tolerance' is a weaker target.
- Random state must match the owning random SOURCE and draw ordering. Do not use
   one global seeded RNG for all entities/worlds. Particle/sound-only branches can
   consume the same RNG later used for physics. Triangle noise is not Gaussian.

`Entity.calculateViewVector` computes float `pitchRad=pitch*float(PI/180)` and
`negativeYawRad=-yaw*float(PI/180)`, then returns FLOAT products promoted to double:
`(Mth.sin(negativeYawRad)*Mth.cos(pitchRad), -Mth.sin(pitchRad),
Mth.cos(negativeYawRad)*Mth.cos(pitchRad))`. Evaluate the negative-yaw lookup itself;
do not assume a native trig identity reproduces quantized lookup bits. Projectile
launch direction has its separately specified source expression.

`Vec2.normalized` uses float squared length and float square root; below length
`1.0E-4F` returns zero, otherwise float component division. This differs from the
Vec3 normalization threshold and precision.

Entity RNG default (`RandomSource.create`) is
[LegacyRandomSource.java](../minecraft-26.3-src/net/minecraft/world/level/levelgen/LegacyRandomSource.java)
with the primitives in
[BitRandomSource.java](../minecraft-26.3-src/net/minecraft/world/level/levelgen/BitRandomSource.java):

1. Seeding stores `state=(externalSeed XOR 25214903917)&((1<<48)-1)`.
2. Each `next(bits)` updates `state=(state*25214903917+11)&((1<<48)-1)` and
   returns the upper `bits` of the 48-bit state. Use Java overflow semantics.
3. `nextFloat=next(24)*5.9604645E-8F`. `nextDouble` combines
   `(long(next(26))<<27)+next(27)`, multiplied by `1.110223E-16F` promoted to double.
4. `nextInt(bound)` for positive power-of-two bound is
   `(long(bound)*next(31))>>31`. Otherwise repeatedly sample `s=next(31)`,
   `r=s%bound` until signed-32-bit `s-r+(bound-1)>=0`, then return r.
5. `nextLong` is `(long(signedNext32)<<32)+signedNext32` with long overflow,
   not unsigned concatenation. Fork uses the nextLong as a new external seed.
6. Double triangle is `mean+spread*(nextDouble()-nextDouble())`; float overload
   uses float draws instead. Arrow uncertainty uses the double overload in X,Y,Z
   order (six double draws total). Co-located knockback consumes victim draws.

Unseeded creation uses a unique-seed generator, so world seed alone does not imply
reproducible entity random states. Capture entity RNG states or explicitly seed
the fixture; Gaussian cache and other RNG implementations need their own state
when enabling mechanics that use them.

### 8.2 Player Geometry and Primitive Queries

Sources: [Avatar.java](../minecraft-26.3-src/net/minecraft/world/entity/Avatar.java),
[EntityTypes.java](../minecraft-26.3-src/net/minecraft/world/entity/EntityTypes.java),
[EntityDimensions.java](../minecraft-26.3-src/net/minecraft/world/entity/EntityDimensions.java),
[AABB.java](../minecraft-26.3-src/net/minecraft/world/phys/AABB.java).

Default scale=1 pose values (all F):

| Pose | Width | Height | Eye height above feet |
| --- | --- | --- | --- |
| Standing | 0.6 | 1.8 | 1.62 |
| Crouching | 0.6 | 1.5 | 1.27 |
| Swimming / fall flying / spin attack | 0.6 | 0.6 | 0.4 |
| Sleeping | 0.2 | 0.2 | 0.2 |
| Dying | 0.2 | 0.2 | 1.62 |

Construct half-width as FLOAT `width/2.0F`, then double bounds:
`(x-halfWidth,y,z-halfWidth,x+halfWidth,y+height,z+halfWidth)`. Apply living SCALE
to scalable dimensions (sleeping/fixed dimensions follow their fixed rules).
Fluid-jump threshold is `eyeHeight<0.4 ? 0 : 0.4` (double comparison). Ordinary
player poses including float 0.4 eye height therefore have threshold 0.4.

AABB intersection is STRICT overlap on all three axes; mere face touching is not
intersection. Point containment includes min and excludes max on each axis.
Eye-to-box squared distance sums squared coordinate gaps to each closed interval;
it is not eye-to-center distance. Player entity range checks use `<(range+buffer)^2`;
block range uses the unit cell box similarly. AttackRange has its own inclusive
min/max distance tests described below.

Default move-collision predicate is other.canBeCollidedWith(source), excluding
same-vehicle pairs; ordinary base entities return false for hard collision.
Ordinary living overlap primarily uses pushes. Specialized solid entities must
carry their own collision/push flags rather than being approximated by mass.

### 8.3 Exact Ray Traversal and Face Selection

`BlockGetter.traverseBlocks` expands the segment slightly for VOXEL VISITATION:
use `from'=lerp(-1E-7,from,to)`, `to'=lerp(-1E-7,to,from)` componentwise, but clip
shapes against ORIGINAL from/to. Test floor(from') cell first. For each axis,
`step=sign(to'-from')`, `tDelta=step/delta` or double max if step zero;
first boundary time is `tDelta*(step>0 ? 1-frac(from') : frac(from'))`.
Advance the least time axis while any time<=1. Tie rule follows strict nested
comparisons: X wins only if less than BOTH Y and Z; Y wins if not X<Y and Y<Z;
otherwise Z. Thus full ties advance Z, then Y, then X over successive iterations.
Return first nonnull per-cell hit. Equal original endpoints return miss.

VoxelShape raycast returns no hit when empty or segment squared length<1E-7.
Test `from+(to-from)*0.001` for occupied shape interior; if inside, return that
point, opposite approximate-nearest movement direction, with inside=true.
Otherwise clip its component AABBs. For AABB clipping, test entry faces in X,Y,Z
order; ignore axis magnitude<=1E-7; candidate parameter must be STRICTLY >0 and
<currentBest (initial 1). Both transverse coordinates must lie strictly between
face limits extended by 1E-7. Retain first face on equal parameter. Consequently
an exact endpoint-only hit or origin-on-face is not automatically an AABB hit;
the shape interior rule and voxel traversal are relevant.

### 8.4 Action Routing and Attack Range

Sources: [ServerPlayerGameMode.java](../minecraft-26.3-src/net/minecraft/server/level/ServerPlayerGameMode.java),
[AttackRange.java](../minecraft-26.3-src/net/minecraft/world/item/component/AttackRange.java),
[KeyboardInput.java](../minecraft-26.3-src/net/minecraft/client/player/KeyboardInput.java),
[ProjectileDeflection.java](../minecraft-26.3-src/net/minecraft/world/entity/projectile/ProjectileDeflection.java).

Use-on-block first allows the block's item-aware interaction unless shift is
held AND at least one hand nonempty. If it consumes action, stop; if it requests
empty-hand interaction on MAIN hand, try that and stop if consumed. Only then,
with nonempty noncooled item, invoke item `useOn`. Thus clicking a chest is not
unconditionally a block-placement attempt. Spectator is handled separately.

An item's ATTACK_RANGE overrides living default. Default is min=0, max=entity
interaction attribute, same creative range, margin=0 (different from the CODEC's
optional-field defaults). Effective component ranges select survival/creative
values for players and multiply by mobFactor for nonplayers.

Client closest attack hit uses head-look ray from eye+look*effectiveMin to
eye+look*(effectiveMax+max(0,knownMovement.dot(look))). Clip OUTLINE blocks first
and reject targets beyond an obstruction or before minimum reach. Candidate
entity/margin checks follow `ProjectileUtil.getManyEntityHitResult`, with
include-from=true and project-surface-hit=true. Select smallest squared distance
from eye to returned hit location, not the arrow entity-position sort.

Server attack packet does NOT re-run that aim ray. It checks loaded client,
nonspectator, valid target/border/item/charge, excludes piercing-weapon routing,
and checks eye-to-target-AABB distance within
`[effectiveMin-margin-buffer,effectiveMax+margin+buffer]`, INCLUSIVELY; buffer=3.
Invalid targets such as item entities/self/experience orbs and nonattackable arrows
are rejected/disconnected. Preserve this distinction if accepting raw packet-like
actions. A normal RL 'attack where looking' action should perform client targeting
first, not exploit the server's generous acceptance buffer as normal reach.

Shield disable duration is the active weapon's WEAPON component value, only when
`weaponItem==getActiveItem()` by identity; active item is use item while using,
otherwise main hand (spectator empty). Do not assume a universal random axe-disable
chance from older versions.

### 8.5 Ordinary Surfaces and Special-Block Hooks

Sources: [Blocks.java](../minecraft-26.3-src/net/minecraft/world/level/block/Blocks.java),
[SlimeBlock.java](../minecraft-26.3-src/net/minecraft/world/level/block/SlimeBlock.java),
[HoneyBlock.java](../minecraft-26.3-src/net/minecraft/world/level/block/HoneyBlock.java),
[WebBlock.java](../minecraft-26.3-src/net/minecraft/world/level/block/WebBlock.java).

| Surface | Friction F | Speed factor F | Jump factor F | Bounce restitution F |
| --- | --- | --- | --- | --- |
| Default block | 0.6 | 1 | 1 | 0 |
| Ice | 0.98 | 1 | 1 | 0 |
| Blue ice | 0.989 | 1 | 1 | 0 |
| Slime | 0.8 | 1 | 1 | 1 |
| Soul sand | 0.6 | 0.4 | 1 | 0 |
| Honey | 0.6 | 0.4 | 0.5 | 0 |

These are properties, NOT complete behavior replacements or claims of unit-cube
geometry. For example honey and soul sand have special collision shapes.
Standing slime `stepOn` additionally scales horizontal velocity by
`0.4+abs(vy)*0.2` if `abs(vy)<0.1` and not stepping carefully, AFTER travel.
Base `isSteppingCarefully` and `isSuppressingBounce` both follow shift-key state.
Honey sliding checks airborne, feet below blockY+0.9375-1E-7, pre-gravity-equivalent
`oldY=vy/0.98F+0.08<-0.08`, and outside the inner horizontal overlap threshold
`0.4375+width/2F` (with epsilon). It sets Y to `(-0.05-0.08)*0.98F`; when
`oldY<-0.13` it also scales X/Z by `-0.05/oldY`; reset fall distance.

Web inside callback sets stuck multiplier `(0.25,0.05F,0.25)`, or
`(0.5,0.25,0.5)` with Weaving. The multiplier is consumed by a later `move`,
which clears stored velocity. This is not ordinary friction. Ladder/climbable
recognition checks feet block's CLIMBABLE tag or an open trapdoor over a ladder
with matching facing; spectators and flying players do not climb, and fall
flying through CAN_GLIDE_THROUGH-tagged states is exempt.

Entity movement records are also used to apply effects along swept paths, not
just at the final box. `applyEffectsFromBlocks` calls grounded `stepOn`, then
collects inside effects along recorded axis-ordered movement and applies the
collector. Paths are capped/merged when recordings exceed 100. Preserve this
phase when adding webs, bubble columns, powder snow, berry bushes, portals, etc.
This document does not enumerate every specialized block's callback.

Fall distance also selects critical-hit and glide branches. During ordinary
`Entity.checkFallDamage`, when not in water and accepted `dy<0`, subtract
`float(dy)` from fall distance. On ground, run block `fallOn` if distance positive,
emit ground event, then reset distance. Keep this before collision restitution
in `move`. The living override can refresh fluid interaction first, so newly
entered water can cancel distance accumulation/reset a fall on the same move.

### 8.6 Iteration, State, and CPU Structure

Source: [EntityTickList.java](../minecraft-26.3-src/net/minecraft/world/level/entity/EntityTickList.java).
Tick entities in active linked-map insertion order, NOT sorted ID/spatial order.
Mutation during iteration switches the active map via copy-on-write, so newly
added entities do not enter the already iterated snapshot; removed entity objects
still receive the outer removed-state checks. Entity queries have their OWN
returned order; preserve it for pushes, equal hit candidates, and effects.

At minimum the simulator state must persist:

| Owner | Required physical/branch-selecting state |
| --- | --- |
| World | Block STATE IDs, fluid source/amount/falling, shapes/tags, bounds/border, loaded and ticking chunks, game time, dimension/environment attributes, game rules, random sources |
| Scheduler | Per-chunk block/fluid queues, type/position uniqueness, priorities/sub-tick counters, block events and neighbor-update recursion/order |
| Entity | Type/ID/order, feet/current and old positions, known movement, velocity, AABB/dimensions/scale, yaw/pitch/head yaw, onGround/collision/support flags, noPhysics/noGravity/removed, relationships, RNG |
| Living | Attributes/modifiers/effects, movement inputs/jump/delay, fall distance, climb/glide/swim flags, fluid/eye history, stuck multiplier, movement records, equipment, use item/countdown, damage cooldown/lastHurt |
| Player | Pose/shift/sprint, food/abilities, selected item/hand/inventory, attack charge/item-swap timer, cooldowns, optional auto-jump/double-tap/fly timers, client authority/accepted movement when networked |
| Arrow | Owner/leftOwner, flags, timers, ignored pierced IDs, weapon state, in-ground/last block state, current/old motion, RNG |

Build the CPU reference as deterministic sequential stages first. Separate query
primitives (block shape lookup, swept axis clip, ray hit, overlap, fluid contact)
from entity integration and action handlers. Cache shapes only by all dependencies
that actually affect them. Avoid replacing source mutation order with a parallel
snapshot approximation before there is a passing CPU oracle.

## 9. Conformance Cases and Coverage Limits

### 9.1 Source-Derived Checkpoints

These are expected checkpoints derived from the source, NOT results of running
a completed replica or launching Minecraft here. Use loaded flat-world fixtures,
default attributes/no effects unless stated, fixed RNG, and compare full tick
state rather than only final positions.

| Case | Required observation |
| --- | --- |
| Stationary air fall from rest | First displacement Y=0; next velocity Y=`(0-0.08)*0.98F`; next tick moves using that velocity |
| Ground jump from rest | Movement uses `double(0.42F)` immediately; post-travel Y=`(double(0.42F)-0.08)*0.98F` when unobstructed |
| Sprint jump, yaw 0 | Add Z impulse `0.2`, not X; input acceleration is a separate addition |
| Cardinal versus diagonal controls | Before acceleration, unmodified forward is `0.98F`; normalized keyboard diagonal is square-adjusted toward unit length; neither gets base input damping again |
| Small player horizontal velocity | `(0.0029,0,0.0029)` survives horizontal snap because combined squared length exceeds 9E-6; `(0.002,0,0.002)` snaps both X/Z |
| Ordinary floor | Ground acceleration samples start ground state; horizontal next-velocity multiplier is `0.6F*0.91F`; Y collision occurs before new gravity |
| Equal X/Z motion into a corner | Resolve Y,X,Z; swapping to Y,Z,X can change result |
| Steps of 0.5 and >0.6 | Eligible 0.5 plane can step if horizontally better and overhead clear; over-limit plane cannot be a candidate |
| Sneak diagonal edge | X-only reduction, Z-only reduction, then diagonal reduction; visual crouch alone is not edge protection |
| Living contact | Overlap creates two-sided horizontal push where eligible, no vertical impulse; may occur again during other entity's tick |
| Grounded versus airborne knockback | Power 0.4F, zero velocity: grounded Y=`min(0.4,double(0.4F))`; airborne Y stays 0 |
| Sprint sword hit | Default damage knockback first, extra yaw knockback second; two horizontal halvings can occur |
| Shield just raised | Before 5 elapsed ticks not blocking; at 5 eligible frontal damage can be fully blocked; head yaw controls arc |
| Repeated hit during cooldown | Equal/lower damage in >10 branch rejected; stronger hit applies excess without default knockback, though successful caller bonuses may still apply |
| Piercing arrow versus shield | Pierce>0 bypasses item blocking; pierce also controls ignored entity IDs and discard limit |
| Arrow first air tick | Move by captured launch velocity; multiply stored velocity by 0.99F, then subtract 0.05 Y |
| Arrow in water | Sweep captured pre-inertia segment, process hit with stored velocity scaled by 0.6F; no ordinary air drag while water state remains true |
| Arrow rejected hit | REVERSE power 0.2 makes velocity -0.1 times current vector before later tick drag/gravity |
| Block placement into player | Collision shape obstruction fails; normal bucket source placement does not use this obstruction test |
| Replaceable clicked block | Replace clicked cell rather than always offset by clicked face |
| Water source on level floor | First scheduled spread delay 5 ticks; source remains; sideways amount normally 7; no instantaneous flood fill |
| Water source conversion | Two passable horizontal sources plus solid/same-source below and enabled rule create source; one source does not |
| Lava downward into water | Down-spread STONE path differs from source-adjacency OBSIDIAN and flowing-adjacency COBBLESTONE |
| Same-kind zero-delay scheduling inside callback | Cannot join that kind's already collected batch; block-to-fluid scheduling can precede fluid collection |
| Spawn during entity iteration | Newly inserted entity does not acquire a tick in that iteration's snapshot |

Also exercise negative coordinates, grazing AABB faces, exact range/angle limits,
ceiling jumps, stairs/slabs/fences, mixed support blocks, current across partial
fluid heights, fluid entry/exit, shield rear hits, co-located knockback RNG, and
multiple arrows/targets competing within one tick. Log first divergent STATE FIELD
and source phase; visual trajectory matching is insufficient.

### 9.2 Boundaries of This Specification

- Covered algorithms: ordinary player input/travel, jumping/sprinting/crouching,
   swimming/lava/climbing/gliding/basic creative flight, axis collision/steps/bounce,
   ordinary living contact, plain block and bucket placement, water/lava propagation
   and currents, sword/melee/shield/arrow physics, and relevant world/client ordering.
   Section 10 adds ordinary player damage/mitigation, survival mining, hotbar/offhand
   and use lifecycle, fire/lava/fall/drowning/food feedback and selected kit/block
   data. Its exact delayed-mining quirk has a targeted original-bytecode check.
- Data still required for a real implementation: exact block states and contextual
   VoxelShapes, tags, item components/enchantments, environment values, entity
   collision flags/overrides, game rules, and source-compatible random state. This
   is not a self-contained replacement for all registries/data packs. Relevant
   files include [washed_away_by_fluids.json](../minecraft-26.3-src/data/minecraft/tags/block/washed_away_by_fluids.json),
   [blocks_fluid_flow.json](../minecraft-26.3-src/data/minecraft/tags/block/blocks_fluid_flow.json),
   [no_knockback.json](../minecraft-26.3-src/data/minecraft/tags/damage_type/no_knockback.json),
   [no_impact.json](../minecraft-26.3-src/data/minecraft/tags/damage_type/no_impact.json),
   [bypasses_shield.json](../minecraft-26.3-src/data/minecraft/tags/damage_type/bypasses_shield.json).
- Not exhaustively specified: every special block callback, ridden vehicles,
   pistons/redstone machinery, mob-specific AI/collision overrides, trident/riptide,
   mace/spear attacks, crossbows/firework rockets, explosions/wind charges,
   fishing/leashes, portals/dimension transitions, all enchantment/effect hooks,
   world fire spread, loot/pickup and full container inventory, consumables/totems,
   auto-jump geometry, full death/respawn, and network quantization/reconciliation. Do not
   advertise whole-game exact parity with these features enabled on this document
   alone. Reject or explicitly scope them out of an initial RL environment.
- Verification performed for this document is static source inspection and local
   reference checks. No game launch, bytecode equivalence proof, CPU simulator,
   golden replay, or runtime/differential parity test has been performed. Decompiled
   source is the available evidence; suspicious/decompiler-failed branches should
   be checked against the original class files before freezing a bit-exact oracle.

The practical acceptance target is: for an explicitly supported world/item/action
profile and a fixed event schedule, the CPU reference reproduces source phase
transitions and per-tick state. Only after differential checks establish that
contract should a GPU version be judged against it.

## 10. Vanilla Arena Combat, Mining, and Item Behavior

This section extends the physical reference for sword/bow/bucket kit PvP,
axe/shield variants, bridge-like placement/mining combat, and unarmed ring-out
arenas. The target is the supplied Java 26.3 code, NOT legacy combat or a server
plugin. Starting/stopping episodes, goals, map resets and round management are
outside this section. There is no special vanilla "sumo" or "bridge" physics mode.

Use an explicit survival-player profile with named block/item states, equipment,
game rules, difficulty and loaded chunks. A useful initial supported subset is
ordinary solid building blocks, ordinary water/lava, unenchanted melee weapons,
ordinary bow/arrows and the default shield. This is a proposed restriction, not a
claim that other item states can be silently mapped onto it. Special blocks,
enchantments and effects require their actual data and callbacks when enabled.

### 10.1 Player Damage, Mitigation, and Hit Acceptance

Sources: [Player.java](../minecraft-26.3-src/net/minecraft/world/entity/player/Player.java)
(`hurtServer`, `actuallyHurt`, `isInvulnerableTo`),
[LivingEntity.java](../minecraft-26.3-src/net/minecraft/world/entity/LivingEntity.java)
(`hurtServer`, `getDamageAfterArmorAbsorb`, `getDamageAfterMagicAbsorb`,
`doHurtEquipment`, `setHealth`), and
[CombatRules.java](../minecraft-26.3-src/net/minecraft/world/damagesource/CombatRules.java).
Apply the outer ServerPlayer protections as well when reproducing a server player;
these formulas do not authorize bypassing spawn/ability/team protection gates.

Preserve this distinction: incoming hit damage, cooldown comparison damage,
post-mitigation damage, health lost, successful-hurt return value, and knockback
eligibility are NOT interchangeable quantities.

1. Player rejects applicable invulnerability, dead/dying, and ability-invulnerable
   cases. Drowning/fall/fire/freeze game rules can make their tagged sources
   invulnerable. Difficulty scaling runs only for sources that request it:
   PEACEFUL -> 0, EASY -> `min(damage/2.0F+1.0F,damage)`, HARD -> `damage*3.0F/2.0F`.
   Player returns false for zero damage before entering the living damage handler.
2. Living checks include fire resistance against IS_FIRE sources. Clamp negative
   damage to zero; apply item blocking BEFORE the cooldown comparison. Freezing
   extra-type scaling and helmet-damaging sources are separate branches: helmet
   equipment is hurt, then damage is multiplied by `0.75F`. Nonfinite damage is
   replaced by `Float.MAX_VALUE` before cooldown handling.
3. As in section 5.2, compare post-blocking/pre-armor damage with `lastHurt`.
   During `damageCooldownTime>10`, reject equal/lower damage unless the source
   bypasses cooldown. A stronger hit mitigates only `damage-lastHurt`, updates
   `lastHurt`, and does NOT take the full-hit/default-knockback branch. A new full
   hit stores incoming damage, sets cooldown 20 and hurt duration/time 10.
4. Player `actuallyHurt` executes armor, resistance/enchantment mitigation,
   absorption, and health loss in that order. Use the PLAYER override, not the
   superficially similar base LivingEntity implementation.
5. The living handler then applies full-hit blocking/impact/default-knockback
   branches and death/protection handling. Its successful-hurt return is
   `!blocked || damage>0`, using the handler's remaining damage, NOT health lost.
   Thus absorption can consume all damage while the hit still succeeds. Caller
   sprint/sweep/post-hit logic remains conditional on this return value.

Armor mitigation, with every arithmetic operation below in Java float except
the source attribute conversions:

```text
A = floor(ARMOR attribute), converted to float
T = float(ARMOR_TOUGHNESS attribute)
toughness = 2.0F + T/4.0F
realArmor = clamp(A - damage/toughness, A*0.2F, 20.0F)
armorFraction = realArmor/25.0F
modifiedArmorFraction = armorFraction
if a weapon item exists and this is the server:
    modifiedArmorFraction = clamp(enchantmentModifyArmorEffectiveness(...),0.0F,1.0F)
damage = damage * (1.0F - modifiedArmorFraction)
```

BYPASSES_ARMOR skips both armor damage and this mitigation. Otherwise armor
durability damage is computed BEFORE reduction: `int(max(1.0F,damage/4.0F))`
for positive damage, on each eligible feet/legs/chest/head stack. Eligibility
requires an EQUIPPABLE component with damageOnHurt, a damageable stack, and
`canBeHurtBy(source)`. Preserve equipment break/modifier-update ordering.

BYPASSES_EFFECTS returns immediately from magic mitigation. Otherwise Resistance
amplifier `r` applies `max((damage*(25-(r+1)*5))/25.0F,0.0F)` unless the damage
bypasses resistance. If damage is now nonpositive return zero. Unless the source
bypasses enchantments, obtain protection `P` from the actual equipment callbacks;
when `P>0`, apply `damage*(1.0F-clamp(P,0.0F,20.0F)/25.0F)`. This describes the
reduction primitive, not every enchantment's conditional protection contribution.

For the player override, with `D` the resulting damage and `B` current absorption:
set `healthDamage=max(D-B,0.0F)` and absorption to `B-(D-healthDamage)` through its
setter. If healthDamage is nonzero, add the damage source's food exhaustion,
record damage, and set health to `health-healthDamage`, clamped to `[0,maxHealth]`.
Do not subtract healthDamage from absorption a second time. Death and totem
interception are later branches; full totem/effect lifecycle remains out of scope
for the initial ordinary-kit profile.

An unarmed ring-out scenario still has vanilla health loss and attack cooldowns.
Do NOT enable creative/invulnerability or replace incoming damage with zero to
make a supposedly equivalent nonlethal sumo mode: Player can reject those hits
before knockback. Stop the episode at the desired boundary, or explicitly label
any external health-restoration rule as an environment modification, not vanilla
combat behavior.

Source-derived checkpoints (not executed Minecraft tests): unarmored normal full
hit loses incoming health; armor 20/toughness 0 with incoming 10 gives effective
armor 15 and approximately 4 health damage; absorption 5 with post-mitigation
damage 3 leaves absorption 2 and health unchanged but need not reject the hit;
an equal repeat during cooldown is rejected before armor/absorption processing;
a stronger repeat mitigates the DIFFERENCE rather than subtracting two separately
mitigated totals. Combine these with the existing shield/knockback checkpoints.

### 10.2 Survival Mining and Block Removal

Sources: [Player.java](../minecraft-26.3-src/net/minecraft/world/entity/player/Player.java)
(`getDestroySpeed`, `hasCorrectToolForDrops`),
[BlockBehaviour.java](../minecraft-26.3-src/net/minecraft/world/level/block/state/BlockBehaviour.java)
(`getDestroyProgress`),
[Tool.java](../minecraft-26.3-src/net/minecraft/world/item/component/Tool.java),
[MultiPlayerGameMode.java](../minecraft-26.3-src/net/minecraft/client/multiplayer/MultiPlayerGameMode.java)
(`startDestroyBlock`, `continueDestroyBlock`, `stopDestroyBlock`, `sameDestroyTarget`),
[ServerPlayerGameMode.java](../minecraft-26.3-src/net/minecraft/server/level/ServerPlayerGameMode.java)
(`tick`, `handleBlockBreakAction`, `incrementDestroyProgress`, `destroyBlock`).

Per-call progress `q` is computed in this order; do not round it to an integer
mining duration and discard float accumulation:

1. Get held stack's mining speed for the actual state. TOOL rules are ordered:
   first matching rule WITH a speed wins, otherwise use defaultMiningSpeed.
   Drop correctness separately takes the first matching rule WITH a correctness
   value, otherwise false. A rule lacking that field does not terminate its search.
2. If stack speed `>1.0F`, add float MINING_EFFICIENCY. Haste/conduit dig-speed
   selection then multiplies by `1.0F+(amplifier+1)*0.2F` when present.
3. Mining Fatigue multiplies by `float(Math.pow(0.3,amplifier+1))`. Do not use an
   older version's hardcoded fatigue table. Multiply by float BLOCK_BREAK_SPEED.
4. If EYES are in water, multiply by SUBMERGED_MINING_SPEED. If not on ground,
   divide by `5.0F`. These penalties compound; mere feet-in-water is not this test.
5. If block destroy speed/hardness is `-1.0F`, q=0. Otherwise q is
   `playerSpeed/hardness/(correctToolForDrops ? 30 : 100)` in that operation order.
   Player correctness is `!state.requiresCorrectToolForDrops || stackCorrectness`.
   Preserve float division, including zero-hardness behavior.

Client survival state machine:

- START rejects restricted action/out-of-border targets. A different target
   aborts an existing mining attempt. Invoke the block attack hook when applicable.
   If nonair and q>=1, predict removal immediately and send START. Otherwise set
   destroying=true, save position/direction/item, and set progress and destroyTicks
   to zero. START does NOT add q to the ordinary accumulated client progress.
- Held attack calls CONTINUE. First synchronize the selected item. If destroyDelay
   is positive, decrement it and return WITHOUT advancing mining. For the same
   target, add current q once and increment destroyTicks. At progress>=1, predict
   removal, send STOP, clear progress/ticks and set destroyDelay=5. A direction
   change alone updates direction; it does not reset progress.
- Same target requires equal position AND `ItemStack.isSameItemSameComponents`
   between selected stack and saved destroyingItem. A target/item mismatch routes
   through START again. Preserve stack alias/copy semantics; do not assume a full
   inventory snapshot comparison including count.
- Releasing/interruption invokes `stopDestroyBlock`: when destroying, send ABORT,
   clear destroying/progress and reset the player's attack-strength ticker. This
   local cancellation is NOT the successful-mining STOP packet.

Server state machine (separate from client prediction):

- Increment gameTicks in game-mode tick. Gate packet actions on block-interaction
   range with server buffer 1, height, and applicable world/protection/restriction
   checks. A normal RL aim action should use ordinary targeting, not treat the
   server tolerance as extra player reach.
- START saves destroyProgressStart=gameTicks, calls enchantment/block attack hooks
   and computes q. A nonair q>=1 target is removed immediately. Otherwise save
   active target/direction. Ordinary active tick updates cracking feedback; it
   does NOT automatically remove a fully progressed block without the relevant
   completion path.
- Matching STOP recomputes current q and `p=q*(gameTicks-destroyProgressStart+1)`.
   If p>=0.7F, clear active mining and destroy now, even though the client's own
   completion threshold is 1.0F. Otherwise, if no delayed attempt exists, clear
   active mining and save delayed target and delayedTickStart=destroyProgressStart.
- ABORT clears active mining and crack feedback; do not invent additional delayed
   cancellation that the source does not perform. An air target clears its active
   or delayed branch in game-mode tick.
- VERSION-SPECIFIC QUIRK: delayed tick calls
   `incrementDestroyProgress(state,pos,delayedTickStart)`, and that helper returns
   `q*(argument+1)`. It does NOT subtract delayedTickStart from current gameTicks.
   The ordinary active branch instead passes gameTicks-destroyProgressStart.
   Preserve this literal distinction when reproducing these exact classes.

That last branch was checked against the original
[ServerPlayerGameMode.class](../minecraft-26.3/net/minecraft/server/level/ServerPlayerGameMode.class):
tick loads delayedTickStart directly before calling incrementDestroyProgress;
the helper performs integer argument+1, converts to float, then multiplies q.
The installed javap cannot read class version 69, so a read-only constant-pool and
Code-attribute inspection was used. This is targeted branch verification, not a
whole-program bytecode-equivalence proof or evidence that later releases share it.

Successful server removal first validates held-item permission and restrictions,
calls playerWillDestroy, then `level.removeBlock(pos,false)`, and, if changed,
the block destroy callback. Do not replace that operation with an isolated write
of AIR: fluid legacy state and neighbor/support updates matter. If player does
not prevent drops, copy the current tool, compute drop correctness, run
`itemStack.mineBlock`, then invoke block.playerDestroy only if changed AND correct
for drops. A wrong tool can still remove the block; it changes progress/drops.
Loot tables, special break callbacks, spawned-item pickup and unlisted blocks
remain explicit data/behavior dependencies rather than implied universal support.

Source-derived checkpoints: same block/tool on ground versus airborne has a
fivefold speed difference before other modifiers; underwater-eye and airborne
penalties compound; component-changing tool swaps can restart mining; direction
changes do not; client START is distinct from the first accumulated CONTINUE;
progress just below/at 1.0F and server STOP just below/at 0.7F take different
branches; breaking a supporting block or water-containing cell must update the
next collision/fluid queries. Exercise the literal delayed branch separately.

### 10.3 Input Arbitration, Switching, and Item Lifecycle

Sources: [Minecraft.java](../minecraft-26.3-src/net/minecraft/client/Minecraft.java)
(`tick`, `handleKeybinds`, `startAttack`, `continueAttack`, `startUseItem`),
[MultiPlayerGameMode.java](../minecraft-26.3-src/net/minecraft/client/multiplayer/MultiPlayerGameMode.java),
[ServerGamePacketListenerImpl.java](../minecraft-26.3-src/net/minecraft/server/network/ServerGamePacketListenerImpl.java)
(`handleSetCarriedItem`, player-action handling),
[Player.java](../minecraft-26.3-src/net/minecraft/world/entity/player/Player.java),
[LivingEntity.java](../minecraft-26.3-src/net/minecraft/world/entity/LivingEntity.java)
(`updatingUsingItem`, `startUsingItem`, `releaseUsingItem`, `stopUsingItem`),
[ItemCooldowns.java](../minecraft-26.3-src/net/minecraft/world/item/ItemCooldowns.java),
[BowItem.java](../minecraft-26.3-src/net/minecraft/world/item/BowItem.java),
[ProjectileWeaponItem.java](../minecraft-26.3-src/net/minecraft/world/item/ProjectileWeaponItem.java),
[BucketItem.java](../minecraft-26.3-src/net/minecraft/world/item/BucketItem.java),
[BlockItem.java](../minecraft-26.3-src/net/minecraft/world/item/BlockItem.java), and
[ItemUtils.java](../minecraft-26.3-src/net/minecraft/world/item/ItemUtils.java).

For ordinary first-person controls without a GUI, the relevant client order is:

1. At client tick start decrement rightClickDelay if positive. Game-mode tick
   synchronizes selected slot; then pick/aim is computed before handleKeybinds.
2. Process hotbar-key selection before combat/use input. Offhand-swap key sends
   its action packet; it is not a local unconditional immediate two-stack swap.
3. If already using an item at the branch test, release if use is no longer held,
   then consume queued attack/use/pick clicks WITHOUT performing those actions.
   Releasing use this tick does not re-enter the idle queued-attack branch.
4. Otherwise process all queued attack clicks, then queued use clicks, then pick.
   Each action can change the state tested by subsequent actions. This is not
   simultaneous independent attack/place/use execution.
5. If use remains held, delay==0 and player is not using, call startUseItem again.
   Finally call continueAttack for held attack unless an instant-break result
   suppresses it. Ordinary held attack sustains block mining, not automatic
   repeated entity attacks without new attack clicks.
6. Client entity/player tick runs later. Keep this order relative to movement,
   attack-ticker increments, equipment changes and use countdown (section 7).

`startUseItem` does nothing while gameMode is destroying a block. Otherwise set
rightClickDelay=4 before the remaining checks. For each hand in MAIN then OFF
order, try targeted entity/block interaction followed by that hand's ordinary
item use. Block Success consumes the action, Fail stops without trying offhand;
Pass permits fallback. Enabled-feature, border, range, cooldown and hands-busy
checks still apply. Block interaction versus shift placement follows section 8.4.
Thus attack-started mining can prevent a same-tick queued placement, and a
main-hand consuming action can prevent use of an offhand shield/bucket.

`startAttack` is gated by missTime, hands-busy, enabled item and minimum charge.
An eligible entity hit invokes attack; a block hit starts mining; a miss can set
missTime=10 and resets attack strength. missTime decrements after handleKeybinds
in the client tick. continueAttack cannot mine while using an item. Preserve these
gates rather than interpreting every requested attack as an accepted melee hit.

Distinct switching/identity rules:

| State machine | Actual comparison or transition |
| --- | --- |
| Player attack tracking | Each Player.tick increments attackStrengthTicker and itemSwapTicker. If main-hand stack no longer matches the saved copy, reset BOTH only when the ITEM TYPE differs; then save a fresh copy. A count/component-only change is not that type-change reset. |
| Explicit reset | resetAttackStrengthTicker resets both timers; resetOnlyAttackStrengthTicker resets only attack strength, used by Player.onAttack. Keep both entry points. |
| Continuing item use | Compare same item TYPE in used hand, then rebind useItem to the current stack and update its countdown. A different type stops use. This is not the mining component-equality test. |
| Server selected-slot change | Validate slot 0..8. If selected index changes and used hand is MAIN, stopUsingItem BEFORE selecting the new slot. An offhand-use slot change does not take that stop branch. |
| Server offhand swap | For nonspectator, exchange main/offhand stacks, then stopUsingItem. This is cancellation, not a release callback. |
| Mining target continuity | Same position and same item/components as destroyingItem, as in section 10.2. |

Use start captures stack, hand and full duration if nonempty and not already
using. Each update invokes onUseTick with the current remaining count, then
predecrements it. At zero, server completes only items not marked useOnRelease.
Normal release verifies same item type in the used hand, rebinds the stack,
invokes its release callback with remaining time, handles useOnRelease update,
then stops. stopUsingItem merely clears use state/countdown and flags; it does
NOT invoke the release callback. Therefore switching away from a drawn bow via
the server hotbar path cancels it rather than firing automatically.

Cooldowns are separate from attack charge and rightClickDelay. ItemCooldowns
stores start/end times by USE_COOLDOWN group, otherwise item registry ID. Each
player cooldown tick increments its counter and removes entries with end<=now;
queried fraction is clamp((end-(now+partial))/(end-start),0,1). Swapping to another
stack in the same group does not bypass a cooldown. Shield disable uses this
mechanism, not the client four-tick repeat delay.

Ordinary survival consumption:

- Block placement's successful path consumes one stack item after its placement
  callbacks. Do not consume on a failed attempt.
- Bucket success uses ItemUtils.createFilledResult: consume one input; if now
  empty return the new bucket stack, otherwise insert the result into inventory,
  dropping it if insertion fails. Filled water/lava buckets produce an empty
  bucket; pickup produces the filled bucket returned by its pickup callback.
  Creative/infinite-material branches are different and excluded from this profile.
- Bow use requires available ammo for ordinary survival, then starts use for up
  to 72000 ticks. Release rechecks ammo, computes held time and charge (section
  5.5), and shoots only above its minimum. ProjectileWeaponItem.draw/useAmmo runs
  actual count/ammo enchantment hooks: the ordinary unenchanted single arrow
  consumes one ammo on release, not on initial draw; an emptied ammo stack is
  removed from inventory. Zero-use ammo paths mark an intangible projectile.
- Authoritative shooting spawns the projectile and damages the bow by one per
  ordinary projectile. Stop shooting if the weapon becomes empty. Weapon/ammo
  enchantments and alternative projectiles require their actual callbacks.

A unified CPU model must choose one authoritative inventory and apply successful
consumption/spawning once. Client prediction may be represented separately, but
do not run both client and server mutations on that same inventory. Likewise,
choosing a headless action API is not permission to silently erase use delays,
mining cancellation or item-specific release rules.

Source-derived checkpoints: held use repeats ordinary placement at delay expiry,
not every tick; switching bow to sword cancels without an arrow; ordinary bow
release consumes ammo only when a shot is drawn; offhand swap stops shield use;
changing hotbar while using an offhand shield differs from changing its hand;
attack+use against a mining target follows the idle attack-first arbitration;
same-type/component-changing stacks exercise different use/mining reset rules.

### 10.4 Liquid Hazards, Falling, Breathing, and Food

Sources: [Entity.java](../minecraft-26.3-src/net/minecraft/world/entity/Entity.java)
(`baseTick`, `applyEffectsFromBlocks`, `updateFluidInteraction`, `checkFallDamage`),
[LivingEntity.java](../minecraft-26.3-src/net/minecraft/world/entity/LivingEntity.java)
(`baseTick`, `calculateFallDamage`, `causeFallDamage`, `igniteForTicks`),
[Player.java](../minecraft-26.3-src/net/minecraft/world/entity/player/Player.java),
[ServerPlayer.java](../minecraft-26.3-src/net/minecraft/server/level/ServerPlayer.java)
(`tick`, `doTick`, `jumpFromGround`, `checkMovementStatistics`),
[LocalPlayer.java](../minecraft-26.3-src/net/minecraft/client/player/LocalPlayer.java),
[WaterFluid.java](../minecraft-26.3-src/net/minecraft/world/level/material/WaterFluid.java),
[LavaFluid.java](../minecraft-26.3-src/net/minecraft/world/level/material/LavaFluid.java),
[InsideBlockEffectApplier.java](../minecraft-26.3-src/net/minecraft/world/entity/InsideBlockEffectApplier.java),
[InsideBlockEffectType.java](../minecraft-26.3-src/net/minecraft/world/entity/InsideBlockEffectType.java),
[BaseFireBlock.java](../minecraft-26.3-src/net/minecraft/world/level/block/BaseFireBlock.java),
[Block.java](../minecraft-26.3-src/net/minecraft/world/level/block/Block.java), and
[FoodData.java](../minecraft-26.3-src/net/minecraft/world/food/FoodData.java).

Contact and burning:

- During baseTick, update fluid state before processing burning. On server, if
  remainingFireTicks>0, clear if fire-immune; otherwise when counter%20==0 AND
  not in lava, attempt ON_FIRE damage 1.0F, then decrement fire counter. This is
  a counter-phase check, not an independent once-per-second wall-clock timer.
- Lava contact registers CLEAR_FREEZE, LAVA_IGNITE and an after-LAVA_IGNITE
  callback lavaHurt. Ignite requests 15 seconds (300 ticks); living ignition
  scales ticks by BURNING_TIME and ceils before extending the existing counter.
  lavaHurt attempts LAVA damage 4.0F when not fire-immune. Contact callbacks and
  their damage acceptance remain distinct from baseTick's periodic fire damage.
- Water contact registers EXTINGUISH (clearFire); rain clears fire after collected
  contact effects. clearFire sets counter=min(0,counter), not always exactly zero.
- StepBasedCollector flushes each movement step in enum order: FREEZE,
  CLEAR_FREEZE, FIRE_IGNITE, LAVA_IGNITE, EXTINGUISH. For each type: before-list,
  one deduplicated type effect if requested, then after-list. Callback lists are
  NOT deduplicated just because their type effect is; preserve contact ordering.
  applyAndClear stops executing further effects if the entity is no longer alive.
- After contact/rain processing, if not burning and not newly ignited this tick,
  server sets counter to -getFireImmuneTicks(): -20 for Player. Do not clamp away
  negative counters. Fire-block contact increments a negative counter toward zero;
  for nonnegative ServerPlayer counters it draws level RNG nextInt(1,3), adds it,
  then requests 8 seconds of ignition when the result is nonnegative. Direct
  fire-block damage uses that block's fireDamage and the IN_FIRE damage source.

All these damage attempts go through source-specific tags, game rules, fire
resistance and section 10.1's shared cooldown/mitigation/knockback gates. Do not
grant a separate unconditional lava damage timer. Lava ignition of nearby blocks,
flammability and random fire spread are additional world behavior; a restricted
arena can use nonflammable blocks, but cannot advertise exact flammable terrain
without also porting those callbacks and random ticks.

Fall state and landing:

1. updateFluidInteraction resets fallDistance when body contact reports water.
   Entity.baseTick halves fallDistance when in lava. These are not equivalent.
2. checkFallDamage accumulates descending movement only when not in water:
   `fallDistance -= (float)resolvedY`, where resolvedY<0. Keep the float cast
   before subtracting from the double accumulator. On ground, call support
   block fallOn if distance>0, then reset distance.
3. Ordinary Block.fallOn uses
   `reducedDistance=fallDistance*(1.0F-fallDistanceReduction)` and calls entity
   causeFallDamage with modifier 1.0F. Special support blocks can override this.
4. Ordinary living calculation is
   `floor((distance+1.0E-6-SAFE_FALL_DISTANCE)*damageModifier*FALL_DAMAGE_MULTIPLIER)`.
   Only positive results trigger hurt; do NOT replace this with an older-version
   ceil formula. With defaults SAFE_FALL_DISTANCE=3, multiplier=1 and ordinary
   support, distance 3.5 gives no damage, distance 4 gives one damage.
5. Player abilities/game rules and damage-type mitigation still apply. Existing
   impulse-ignore contexts can substitute a bounded effective fall distance;
   wind/mace producers of those contexts remain outside the ordinary-kit profile.

For bucket clutches, use actual placement success, fluid update/contact phase,
resolved movement and support callbacks. Merely having a water block somewhere
below the player is not a valid fall-damage cancellation condition.

Breathing (ordinary player): max air=300. When eyes are in water outside a bubble
column, and no applicable breathing effect/ability/type exempts drowning, update
air by decreaseAirSupply. With OXYGEN_BONUS<=0, decrement by one. With bonus>0,
draw entity RNG nextDouble and retain air if draw>=1/(bonus+1), otherwise decrement.
At air<=-20 set air=0 and attempt DROWN damage 2.0F. Out of the drowning condition,
recover air by four per eligible tick, capped at max; effects can govern recovery
while underwater. Body-in-water alone is insufficient for the drowning test.

FoodData defaults: food=20, saturation=5.0F, exhaustion=0, tickTimer=0.
The server calls FoodData.tick after super.tick in ServerPlayer.doTick (section
7), NOT a duplicate tick in the client. ServerPlayer.damageCooldownTime decrements
in its world tick; LivingEntity skips that decrement for ServerPlayer.

Food processing order per tick:

1. If exhaustion>4.0F (STRICT), subtract 4 once. If saturation>0 subtract one
   saturation, clamped to zero; otherwise reduce food by one unless PEACEFUL.
   Do not drain the whole exhaustion backlog in one tick. addExhaustion caps
   the accumulator at 40.0F.
2. If natural regeneration is enabled, saturation>0, health is positive/below
   max, and food>=20: increment shared tickTimer. At >=10, let S=min(saturation,6),
   heal S/6.0F, add S exhaustion, and reset timer. Saturation is not directly
   subtracted here; later exhaustion processing spends it.
3. Otherwise if natural regeneration, food>=18 and hurt: increment timer; at >=80,
   heal one, add 6 exhaustion, reset timer.
4. Otherwise if food<=0: increment timer; at >=80 attempt STARVE damage one when
   health>10 OR difficulty HARD OR (health>1 AND difficulty NORMAL), then reset.
   These are pre-damage eligibility tests, not an invented post-hit health clamp.
   Otherwise reset timer. Keep the shared timer across branches exactly as source.

Exhaustion producers relevant here, on authoritative non-invulnerable players:

| Action | Exhaustion |
| --- | --- |
| Successful ordinary melee hurt | 0.1F on the attacker |
| Server jumpFromGround | 0.2F sprinting, otherwise 0.05F |
| Ordinary successful block playerDestroy | 0.005F |
| Taking nonzero health damage | Damage source food-exhaustion value, not a universal constant |
| Eligible swimming or eyes-underwater movement | Let N=round(float(sqrt(dx*dx+dy*dy+dz*dz))*100.0F); if N>0 add 0.01F*N*0.01F |
| Eligible surface-water movement | Same calculation using sqrt(dx*dx+dz*dz) |
| Eligible ground sprint | Same horizontal N; if N>0 add 0.1F*N*0.01F |
| Ordinary ground walk/crouch | Zero in the movement-statistics branch |

Server movement statistics use their existing passenger/no-movement guards and
branch order: swim, eyes underwater, body water, climb, ground, gliding, air.
Do not add sprint exhaustion again for movement already handled as swimming.
Use the source's rounded distance, not unrounded Euclidean distance.

Food>6 permits exhaustive maneuvers, or mayfly bypasses that food gate. Local
sprint start also requires forward impulse, allowed mobility/fluid/pose state and
not being slowed by item use (section 2.4). A frozen food value is an explicit
environment restriction; disabling regeneration does not disable exhaustion.
Food consumption, golden apples, potions and arbitrary timed effects still need
their own item/effect lifecycle when included in a kit; they are not supplied by
the FoodData regeneration state machine alone.

Source-derived checkpoints: periodic fire at positive counter 20 versus 19;
negative fire counter persists; same-step lava ignition/damage before water
extinguishing; lava and fire attempts share hit cooldown state; water resets fall
distance while lava halves it; distance just below/at the fall-damage boundary;
air 0 versus -20; exhaustion exactly 4 versus just above; food 6 versus 7 for
sprint eligibility; regeneration timers 9/10 and 79/80 with fixed saturation.

### 10.5 Ordinary Kit Data, Axe/Shield Details, and Block Subset

Sources: [Items.java](../minecraft-26.3-src/net/minecraft/world/item/Items.java),
[Item.java](../minecraft-26.3-src/net/minecraft/world/item/Item.java),
[ToolMaterial.java](../minecraft-26.3-src/net/minecraft/world/item/ToolMaterial.java),
[Tool.java](../minecraft-26.3-src/net/minecraft/world/item/component/Tool.java),
[Weapon.java](../minecraft-26.3-src/net/minecraft/world/item/component/Weapon.java),
[BlocksAttacks.java](../minecraft-26.3-src/net/minecraft/world/item/component/BlocksAttacks.java),
[ItemStack.java](../minecraft-26.3-src/net/minecraft/world/item/ItemStack.java),
[Player.java](../minecraft-26.3-src/net/minecraft/world/entity/player/Player.java),
[LivingEntity.java](../minecraft-26.3-src/net/minecraft/world/entity/LivingEntity.java),
[Attributes.java](../minecraft-26.3-src/net/minecraft/world/entity/ai/attributes/Attributes.java),
[ArmorMaterials.java](../minecraft-26.3-src/net/minecraft/world/item/equipment/ArmorMaterials.java),
[ArmorMaterial.java](../minecraft-26.3-src/net/minecraft/world/item/equipment/ArmorMaterial.java),
[ArmorType.java](../minecraft-26.3-src/net/minecraft/world/item/equipment/ArmorType.java),
[Blocks.java](../minecraft-26.3-src/net/minecraft/world/level/block/Blocks.java),
[BlockBehaviour.java](../minecraft-26.3-src/net/minecraft/world/level/block/state/BlockBehaviour.java),
[TransparentBlock.java](../minecraft-26.3-src/net/minecraft/world/level/block/TransparentBlock.java), and
[AbstractArrow.java](../minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java).

Player base ATTACK_DAMAGE is 1.0, not Attributes' generic living default 2.0.
Base ATTACK_SPEED=4.0, ATTACK_KNOCKBACK=0, armor/toughness/knockback resistance=0,
MAX_HEALTH=20, MAX_ABSORPTION=0. Default interaction ranges are block=4.5 and
entity=3.0; use actual targeting in section 8.4, not packet tolerance as reach.
BLOCK_BREAK_SPEED=1, SUBMERGED_MINING_SPEED=0.2, MINING_EFFICIENCY=0,
OXYGEN_BONUS=0, SAFE_FALL_DISTANCE=3, FALL_DAMAGE_MULTIPLIER=1, BURNING_TIME=1.
Positive absorption fixtures must also supply a compatible max-absorption
attribute/effect, since its setter clamps to [0,MAX_ABSORPTION].

Tool material data, directly from constructors:

| Material | Max durability | Efficient mining speed | Attack-damage bonus |
| --- | --- | --- | --- |
| WOOD | 59 | 2.0F | 0.0F |
| STONE | 131 | 4.0F | 1.0F |
| COPPER | 190 | 5.0F | 1.0F |
| IRON | 250 | 6.0F | 2.0F |
| DIAMOND | 1561 | 8.0F | 3.0F |
| GOLD | 32 | 12.0F | 0.0F |
| NETHERITE | 2031 | 9.0F | 4.0F |

The following are ordinary, unenchanted MAINHAND values before charge/critical
scaling. Displayed attack speeds are NOMINAL: compute the attribute as base
double 4.0 plus the promoted FLOAT modifier, not a freshly rounded decimal.

| Held item | Full-strength base attack damage | Speed modifier literal | Nominal attacks/second |
| --- | --- | --- | --- |
| Empty hand | 1 | none | 4.0 |
| Wooden / golden sword | 4 | -2.4F | 1.6 |
| Stone / copper sword | 5 | -2.4F | 1.6 |
| Iron sword | 6 | -2.4F | 1.6 |
| Diamond sword | 7 | -2.4F | 1.6 |
| Netherite sword | 8 | -2.4F | 1.6 |
| Wooden axe | 7 | -3.2F | 0.8 |
| Stone / copper axe | 9 | -3.2F | 0.8 |
| Iron axe | 9 | -3.1F | 0.9 |
| Golden axe | 7 | -3.0F | 1.0 |
| Diamond axe | 9 | -3.0F | 1.0 |
| Netherite axe | 10 | -3.0F | 1.0 |

Sword attack modifier is float(3.0F+materialBonus); axe registrations supply
baselines wood/gold=6, stone/copper=7, iron=6, diamond/netherite=5, plus bonus.
Pickaxes use baseline damage modifier 1.0F+bonus and speed modifier -2.8F, giving
nominal 1.2 attacks/second. These attributes apply in MAINHAND only.

Ordinary tools have TOOL rules in order: deny drops on that tier's incorrect
block tag, then mine/drop efficiently on the tool's mineable tag. Default speed
1.0F, damagePerBlock=1. Swords instead have cobweb speed 15/correct drops, then
SWORD_INSTANTLY_MINES speed Float.MAX_VALUE, then SWORD_EFFICIENT speed 1.5F;
default speed 1 and damagePerBlock=2. None of the five blocks below is in the
two sword speed tags. Correctness and speed scan rules independently (10.2).

Durability changes use ItemStack.processDurabilityChange, including damageable/
infinite-material checks and enchantment processing. Unenchanted ordinary weapon
attack cost is sword=1, axe/pickaxe=2; successful server mining with nonzero block
hardness costs its TOOL damagePerBlock. Bow max durability=384 and shot cost=1;
shield max durability=336. applyDamage updates DAMAGE, then when broken shrinks
stack by one and invokes the equipped-break callback. Do not leave a zero-count
weapon's modifiers active or charge durability again on the predicted client.

Equipment changes are phase-sensitive: LivingEntity.tick calls updatingUsingItem,
then on server detectEquipmentUpdates before aiStep. collectEquipmentChanges
compares actual equipped stacks with saved copies, stops previous location-based
effects/modifiers for changed slots, then installs current nonempty/nonbroken
stack modifiers by ID and invokes applicable location-change enchantments. Hand
swap detection also updates the saved copies. Preserve this schedule rather than
declaring every hotbar inventory assignment an instantaneous attribute rebuild.

Armor table gives HEAD/CHEST/LEGS/FEET defense, toughness and knockback resistance
PER PIECE, and durability multiplier. Only actual worn slots contribute.

| Material | Defense H/C/L/F | Toughness per piece | Knockback resistance per piece | Durability multiplier |
| --- | --- | --- | --- | --- |
| LEATHER | 1/3/2/1 | 0 | 0 | 5 |
| COPPER | 2/4/3/1 | 0 | 0 | 11 |
| CHAINMAIL | 2/5/4/1 | 0 | 0 | 15 |
| IRON | 2/6/5/2 | 0 | 0 | 15 |
| GOLD | 2/5/3/1 | 0 | 0 | 7 |
| DIAMOND | 3/8/6/3 | 2.0F | 0 | 33 |
| NETHERITE | 3/8/6/3 | 3.0F | 0.1F | 37 |

Maximum piece durability is multiplier times HEAD=11, CHEST=16, LEGS=15,
FEET=13. Full diamond gives armor 20/toughness 8; full netherite gives armor
20/toughness 12 and nominal resistance 0.4 (sum the promoted 0.1F modifiers,
not a substituted decimal constant). Recompute equipment effects at their source
phase when pieces break. Turtle helmet and nonhuman armor are not this table.

Default shield and ordinary axe:

- Item.use enters blocking via BLOCKS_ATTACKS after the consumable and swappable-
   equipment branches. Default shield is equippableUnswappable OFFHAND, so its
   generic use proceeds to blocking rather than equipment auto-swap. It can still
   be held/used in a hand and explicitly swapped as in section 10.3.
- Shield use duration=72000, block delay=round(0.25F*20)=5, horizontal half-angle
   90.0F, reduction base=0/factor=1 for matching nonbypassing damage. Default use
   slowdown is 0.2F and does not allow sprint; see sections 2.4 and 5.4.
- Blocked-damage durability cost: below 3.0F -> 0; otherwise floor(1.0F+blocked).
   Use actual blocked damage, not damage after the defender's armor.
- Every ordinary axe's WEAPON carries disableBlockingForSeconds=5.0F. Sword and
   pickaxe carry zero. The attacker returns this value only if its weapon stack
   is IDENTICAL to getActiveItem(), as specified in section 8.4.
- On a direct living, nonprojectile positive block, player blockUsingItem checks
   that value. Shield scale=1.0F -> round(5*1*20)=100 cooldown ticks, then stop use.
   This side effect happens in blocking before the later damage-cooldown decision.
   A fully blocked attack can therefore disable the shield while returning failed
   hurt and producing none of the ordinary successful-hit bonuses. Do not add a
   random chance, sprint requirement or full-charge test to the disable callback.
- Cooldown belongs to the shield item/group, so selecting a second ordinary
   shield does not remove the disable. Front/rear, delay and partial/full blocking
   behavior remain as section 5.4; bypass tags/piercing arrows skip the block path.

Ordinary arrow damage fills the damage-value input used by section 10.1:

```text
power = float(length(current stored arrow velocity))
baseDamage = 2.0 (double), modified by weapon damage enchantments when enabled
damage = ceil(clamp(power*baseDamage,0.0,2147483647.0))
if critical:
   bonus = arrowRng.nextInt(damage/2 + 2)
      damage = int(min(long(damage)+bonus,2147483647L))
```

The division in the RNG bound is integer division and nextInt's bound is
exclusive. The source damage type is ARROW; owner and direct projectile are separate source
fields. Resolve stored velocity after water inertia as in 5.6. A fixture with
speed exactly 3 and no enchantments has base damage 6 and critical bonus 0..4;
real bow spread/inherited movement can change that speed. Successful hurt causes
default damage knockback plus any positive enchantment arrow-knockback callback;
ordinary unenchanted arrows have zero EXTRA arrow knockback. Rejected-hit fire
restoration/deflection stays as section 5.6. Bow ammo preference is supported
offhand ammo first, then main hand, then inventory index order; do not arbitrarily
select a different tipped/spectral stack if those types are enabled.

Small nonflammable, non-falling building-block subset:

| Block | Destroy hardness literal | Requires correct tool for drops | Efficient tag |
| --- | --- | --- | --- |
| STONE | 1.5F | yes | mineable/pickaxe |
| COBBLESTONE | 2.0F | yes | mineable/pickaxe |
| DIRT | 0.5F | no | mineable/shovel |
| GLASS | 0.3F | no | none of pickaxe/axe/shovel |
| TERRACOTTA (uncolored) | 1.25F | yes | mineable/pickaxe |

These registrations use ordinary full-block outline/collision, friction=0.6F,
speed/jump factor=1.0F and bounce restitution=0. Glass noOcclusion/empty VISUAL
shape does NOT make its COLLISION empty. None of these states is in the checked
incorrect-for-wood/stone/copper/iron/diamond/gold/netherite-tool tags; a matching
wooden pickaxe can drop the three pickaxe blocks. Nonmatching tools on a
requires-correct block still use divisor 100 and do not earn its ordinary drops.
This table does not replace loot tables: stone/cobblestone/glass drops must not
be inferred from their placed state or the correctness flag. Drop entities,
pickup/replenishment and arbitrary loot/enchantment outcomes remain to be ported
if they are part of the action profile.

Tag evidence:
[pickaxe.json](../minecraft-26.3-src/data/minecraft/tags/block/mineable/pickaxe.json),
[shovel.json](../minecraft-26.3-src/data/minecraft/tags/block/mineable/shovel.json),
[axe.json](../minecraft-26.3-src/data/minecraft/tags/block/mineable/axe.json),
[incorrect_for_wooden_tool.json](../minecraft-26.3-src/data/minecraft/tags/block/incorrect_for_wooden_tool.json),
[incorrect_for_copper_tool.json](../minecraft-26.3-src/data/minecraft/tags/block/incorrect_for_copper_tool.json),
[incorrect_for_stone_tool.json](../minecraft-26.3-src/data/minecraft/tags/block/incorrect_for_stone_tool.json),
[incorrect_for_iron_tool.json](../minecraft-26.3-src/data/minecraft/tags/block/incorrect_for_iron_tool.json),
[incorrect_for_diamond_tool.json](../minecraft-26.3-src/data/minecraft/tags/block/incorrect_for_diamond_tool.json),
[incorrect_for_gold_tool.json](../minecraft-26.3-src/data/minecraft/tags/block/incorrect_for_gold_tool.json),
[incorrect_for_netherite_tool.json](../minecraft-26.3-src/data/minecraft/tags/block/incorrect_for_netherite_tool.json),
[sword_efficient.json](../minecraft-26.3-src/data/minecraft/tags/block/sword_efficient.json),
[sword_instantly_mines.json](../minecraft-26.3-src/data/minecraft/tags/block/sword_instantly_mines.json).
Nested #tag memberships were resolved from the supplied JSON, not guessed by name.

Damage data for the ordinary profile (Y means tagged; N means not tagged):

| Damage type JSON | Food exhaustion | NO_KNOCKBACK | BYPASSES_SHIELD | BYPASSES_ARMOR |
| --- | --- | --- | --- | --- |
| [player_attack](../minecraft-26.3-src/data/minecraft/damage_type/player_attack.json) | 0.1F | N | N | N |
| [arrow](../minecraft-26.3-src/data/minecraft/damage_type/arrow.json) | 0.1F | N | N | N |
| [lava](../minecraft-26.3-src/data/minecraft/damage_type/lava.json) | 0.1F | Y | Y | N |
| [on_fire](../minecraft-26.3-src/data/minecraft/damage_type/on_fire.json) | 0 | Y | Y | Y |
| [in_fire](../minecraft-26.3-src/data/minecraft/damage_type/in_fire.json) | 0.1F | Y | Y | N |
| [drown](../minecraft-26.3-src/data/minecraft/damage_type/drown.json) | 0 | Y | Y | Y |
| [fall](../minecraft-26.3-src/data/minecraft/damage_type/fall.json) | 0 | Y | Y | Y |
| [starve](../minecraft-26.3-src/data/minecraft/damage_type/starve.json) | 0 | Y | Y | Y |

All eight have scaling=when_caused_by_living_non_player; do not apply difficulty
scaling to player-caused ordinary PvP just because the world is HARD. None is in
BYPASSES_COOLDOWN, BYPASSES_RESISTANCE or BYPASSES_ENCHANTMENTS. Of these eight,
only DROWN is NO_IMPACT and only STARVE is BYPASSES_EFFECTS. Thus lava can be
armor-mitigated, ongoing ON_FIRE damage bypasses armor, and none of the listed
hazards grants a free physical damage knockback or independent hurt cooldown.

Damage tag evidence:
[no_knockback.json](../minecraft-26.3-src/data/minecraft/tags/damage_type/no_knockback.json),
[no_impact.json](../minecraft-26.3-src/data/minecraft/tags/damage_type/no_impact.json),
[bypasses_shield.json](../minecraft-26.3-src/data/minecraft/tags/damage_type/bypasses_shield.json),
[bypasses_armor.json](../minecraft-26.3-src/data/minecraft/tags/damage_type/bypasses_armor.json),
[bypasses_cooldown.json](../minecraft-26.3-src/data/minecraft/tags/damage_type/bypasses_cooldown.json),
[bypasses_effects.json](../minecraft-26.3-src/data/minecraft/tags/damage_type/bypasses_effects.json),
[bypasses_resistance.json](../minecraft-26.3-src/data/minecraft/tags/damage_type/bypasses_resistance.json),
[bypasses_enchantments.json](../minecraft-26.3-src/data/minecraft/tags/damage_type/bypasses_enchantments.json).

Source-derived checkpoints: empty-hand damage=1/speed=4, diamond sword=7 and
modifier -2.4F; full netherite changes knockback as well as mitigation; shield
use ticks 4/5, blocked damage below/at 3, axe disable 100 ticks, sword disable 0;
blocked axe still disables although successful-hurt is false; cooldown survives a
shield-stack swap; current damage equal to maximum breaks the equipped stack;
glass blocks movement despite its visual shape; lava versus ON_FIRE armor
eligibility differs. These fixtures are ordinary vanilla, not legacy PvP rules.

### 10.6 Coverage After These Investigations

The five investigated areas now have source-derived algorithms, state transitions
and a limited data profile: shared player damage; survival mining/removal;
hotbar/offhand/use timing; environmental damage/food feedback; and ordinary
weapon/tool/armor/shield/block data. This is additional specification, NOT a
completed implementation, game replay comparison or exhaustive registry export.

| Scenario | Documented physical/behavior foundation | Remaining constraints |
| --- | --- | --- |
| Vanilla unarmed ring-out arena | Movement, aim/hit selection, charge, hit acceptance, contact pushes, knockback, hunger and health | Normal vanilla damage still exists. Nonlethal health rules would be explicit modifications; episode management is external. |
| Vanilla bridge-like arena | Above plus ordinary placement/mining, bow flight/damage/ammo and hotbar action arbitration | Use the supported block/item set; loot pickup, special blocks, consumables and full packet prediction are not covered by inference. |
| Ordinary sword/bow/bucket kit PvP | Above plus fluid propagation/current, bucket stack transitions, armor, fire/lava/drowning/falling and regeneration | Nonflammable terrain and explicitly loaded data; arbitrary enchanted kits, healing items and world fire spread need their own modules. |
| Ordinary axe/shield variant | Above plus fixed axe component values, shield raise/facing/durability/disable and hand-use state | Same bounded kit/data limits; no legacy disable chance or custom knockback. |

Before freezing a CPU oracle, add differential fixtures that combine systems,
not only isolated formulas: sprint-hit while a fire damage attempt is due;
shield raise/axe hit at a delay boundary; bow release versus slot switch; mining
while becoming airborne or entering water; same-tick break/place and support
loss; water bucket placement immediately before landing; and food exhaustion
crossing a sprint threshold. Fix RNG and event phase/order and compare first
divergent state. These are proposed validation cases, not tests performed here.