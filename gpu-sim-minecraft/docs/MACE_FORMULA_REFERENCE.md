# Minecraft Java 26.3 Mace and Wind Charge Reference

This is a bytecode-derived research reference, not a Minecraft parity claim.
Unenchanted mace combat and wind-charge items are implemented in the represented
simulator slice; live-game behavior still needs verification. See the [scope](mace_smash_wind_charge_spec.md),
[implementation plan](MACE_WIND_IMPLEMENTATION_PLAN.md), and
[remaining research](MACE_WIND_RESEARCH_PLAN.md).

## Evidence

The inspected artifact is the locally installed Minecraft Java 26.3 client
JAR, SHA-256 `4508d006323f24fa02876310c192d739af56516eb259000ac50f0909a68c9a2d`.
Its Mojang-named classes were inspected with JDK 25 `javap -p -c -l`; the
enchantment definitions were read from the JAR's `data/minecraft/enchantment/`
JSON. The `minecraft-26.3-src/` tree cited by some older notes is not present
in this checkout. Method names below identify the inspected bytecode, not
links to a checked-in source tree. No in-game captures were made for this work.

## Mace Attributes and Damage

`MaceItem.createAttributes` adds 5.0 attack damage and `-3.4F` attack speed
to the player bases of 1.0 and 4.0. Thus the effective unenchanted values are
6.0 damage and `4.0 + double(-3.4F)` attacks per second, approximately 0.6;
the negative value is an additive modifier, not the effective speed. Preserve
Java's float-to-double widening. The mace does not sweep.

`MaceItem.canSmashAttack` requires `fallDistance > 1.5` and not fall-flying.
The comparison is strict: exactly 1.5 is not a smash. It does not impose a
separate smash cooldown or forbid sprinting. Normal attack charge and critical
eligibility still have their own rules.

For eligible attacks, `MaceItem.getAttackDamageBonus` computes this additional
damage from the attacker's double fall distance `d`:

| Distance | Additional damage before enchantments |
| --- | --- |
| `d <= 3` | `4 * d` |
| `3 < d <= 8` | `12 + 2 * (d - 3)` |
| `d > 8` | `22 + d - 8` |

The bonus is zero when smash eligibility fails and is returned as a float.
`Player.attack` scales ordinary base damage by `0.2F + charge * charge * 0.8F`,
then adds the item bonus; an eligible critical multiplies that sum by 1.5F.
Enchanted damage has its own path. For example, an unenchanted, fully charged,
noncritical hit with `d = 2` has incoming damage 6 + 8 = 14 before the target's
damage processing; an otherwise eligible critical produces 21. These are
formula fixtures, not observed in-game results.

In the server path, the fall-based enchantment modifier comes from the
attacker's weapon and contributes `modifier * d`, not a multiplier on the
piecewise bonus. Enchantments are deferred: the packaged Density definition
adds `0.5 * level` per fallen block, while Breach changes armor effectiveness
by `-0.15 * level`. Their context and integration must be traced before support.

## Smash Effects and Bystanders

For an eligible `MaceItem.hurtEnemy` callback, the attacker's Y velocity is
replaced with `double(0.01F)`, the attacker is marked to ignore fall damage
from the current impulse, and nearby knockback is applied. The impact position
can retain an existing lower impulse-impact position; it is not simply a
permanent no-fall-damage flag. `MaceItem.postHurtEnemy` resets fall distance
when eligible. The simulator models these callbacks for accepted hits and has
limited opt-in landing damage; repeated impulses and game parity still need
additional verification.

`MaceItem.knockback` queries living entities around the primary target's
bounding box inflated by 3.5. Its predicate excludes the attacker, primary
target, spectators, attacker's allies, animals tamed by the primary target,
marker armor stands and creative-flying players, and requires squared
target-to-bystander position distance `<= 3.5 * 3.5`. For each remaining
bystander, let `direction = bystander.position - target.position` and:

```text
power = (3.5 - length(direction)) * double(0.7F)
        * (attacker.fallDistance > 5.0 ? 2 : 1)
        * (1 - bystander.knockbackResistance)
```

If power is positive, vanilla adds `(normalize(direction).x * power,
double(0.7F), normalize(direction).z * power)` to the bystander's velocity.
Distance and normalization are three-dimensional. This splash is separate
from the primary target's ordinary melee knockback. Compute it before any
fall-distance reset that would change the heavy multiplier. A two-player
fixture cannot exercise this effect.

## Wind Charge Item and Projectile

`WindChargeItem.use` throws a separate item projectile; it is not a mace charge
or a Wind Burst projectile. The item registration sets a `0.5F` second use
cooldown, corresponding to 10 ticks at 20 ticks per second. Player use invokes
`Projectile.spawnProjectileFromRotation` with speed `1.5F` and inaccuracy
`1.0F`. `Projectile.getMovementToShoot` normalizes the look vector, adds
independent triangular noise with spread `0.0172275 * inaccuracy` to each
component, then scales by speed. `shootFromRotation` adds the shooter's known
X/Z movement and adds Y when the shooter is airborne. RNG state and exact spawn
position require their own fixtures. The player-use factory spawns at the
player's eye position. `WindChargeItem.asProjectile` is a
different dispenser path; do not use its direction/noise rules for player use.

`WindCharge` initializes `noDeflectTicks` to 5; its tick decrements the counter
after the superclass tick, and deflection is refused while it remains positive.
`AbstractWindCharge.onHitEntity` attempts 1.0F direct wind-charge damage and
then explodes. `onHitBlock` explodes at the hit location plus 0.25 times the
hit-face normal. A hit removes the projectile. In air,
`AbstractHurtingProjectile.tick` applies inertia before tracing the movement,
places the projectile at the hit location or movement endpoint, calls
`Projectile.tick`, then handles the hit (`AbstractHurtingProjectile.java:68-103`).
`Projectile.tick` checks owner clearance before `Entity.tick`
(`Projectile.java:135-143`). Complete owner clearance, collision/deflection,
lifetime and RNG seeding still require verification.

For a full-cube hit, `VoxelShape.clip` rejects movement with squared length
below `1.0E-7`, then checks a point 0.001 along the movement. If that point is
inside the shape, it returns that point and the opposite of the approximate
movement direction as the hit face (`VoxelShape.java:153-171`,
`Direction.java:331-345`). The existing generic simulator block ray does not
implement this inside-block behavior or return a face; it must not be used as
an exact wind-charge terrain hit without addressing those differences.

`WindCharge.explode` passes power `1.2F`, `false` for fire and `TRIGGER` block
interaction. Its `SimpleExplosionDamageCalculator` enables block explosion
handling, disables explosion entity damage, supplies knockback multiplier
`1.22F`, and uses the `blocks_wind_charge_explosions` tag as immune blocks.
The direct-hit damage above is separate. `ServerExplosion.hurtEntities` uses
twice the power for the nominal entity influence range, approximately 2.4
blocks, not 1.2. The blast samples exposure across the entity bounding box,
measures range from the entity position and directs its impulse from the blast
center toward the eye position. It can affect the owner. Block interactions
and matching game captures remain necessary before claiming exact parity.

The packaged Wind Burst enchantment is a post-attack explosion affecting the
attacker, not a separately launched projectile. Its level-dependent behavior
is outside the initial unenchanted scope.

## Additional Throwables

Player use of an ender pearl, snowball or egg launches at speed `1.5F` and
inaccuracy `1.0F`, then consumes one item (`EnderpearlItem.java:24-32`,
`SnowballItem.java:26-33`, `EggItem.java:26-33`). Unlike wind charges,
`ThrowableProjectile.tick` applies gravity, then inertia, before tracing and
moving (`ThrowableProjectile.java:43-67`). Its default gravity is `0.03`;
dry-air drag is the widened float `0.99F` (`ThrowableProjectile.java:70-88,
104`). Fluids have different behavior and remain outside the simulator scope.

An egg attempts zero thrown damage on entity hit (`ThrownEgg.java:54-56`);
a snowball attempts zero against ordinary players and three against blazes
(`Snowball.java:55-60`). Ordinary player damage rejects zero before living
damage handling (`Player.java:753-754`). Vanilla eggs can create chickens;
the simulator does not create mobs, and the mod suppresses hatching only for
tracked round-owned eggs. That mixin has not yet been verified in a live game.

Pearls have a 16-item stack limit and a one-second use cooldown
(`Items.java:1034`; `ItemStack.java:400-410`; `UseCooldown.java:37-40`).
On impact, a valid owner teleports to the projectile's pre-tick position
(`Entity.java:587-591`; `ThrownEnderpearl.java:106-114`), with zero velocity,
preserved rotation, cleared fall and impulse state, then takes five damage
(`ThrownEnderpearl.java:132-136`). The represented player-only simulator slice
implements these effects and retires pearls when their owner dies. It does not
model sleeping, dimensions, chunk tickets, mobs or pearl-spawned endermites.
Vanilla endermite spawning remains enabled in the live game.

All three items are exposed in scenario loadouts, training, replay and the mod.
CPU and CUDA checks are source-derived; collision ordering, RNG equivalence,
mixins and live gameplay have not been verified against game captures.