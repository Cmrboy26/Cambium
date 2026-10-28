# Mace and Wind Charge Research Checklist

## Evidence So Far

Minecraft Java 26.3's locally installed client JAR was inspected with `javap`
and its packaged enchantment JSON was read directly. The artifact identity,
method names, verified formulas and limitations are recorded in the
[reference](MACE_FORMULA_REFERENCE.md). The cited decompiled source tree is not
present in this checkout; bytecode is available, but no real-game captures were
performed. Focused mace and wind-helper checks have run on CPU and CUDA, not a
complete feature parity suite. Native mace combat and loadout
acceptance are partially implemented; ordinary fall damage and impulse protection
are implemented behind an opt-in rule with limited supported player and terrain
states. Wind launch and blast helpers have focused CPU and CUDA checks but are not yet
connected to projectile gameplay.

## Before Mace Implementation

- Extend the traced `Player.attack` and accepted-hit callback checks with
  matched game captures, including damage cooldown and simultaneous attacks.
- Verify `setIgnoreFallDamageFromCurrentImpulse`, `currentImpulseImpactPos`,
  landing and subsequent impulses against the implementation, including repeated
  impulses and grace-time boundaries. Confirm ordering relative to fall reset.
- Map vanilla action arrival and player updates onto the simulator's attack,
  movement and projectile phases; examine simultaneous attacks and target
  movement. Capture initial fall distance rather than inferring it from height.
- Build independent numeric expectations and matching private-game fixtures
  at and around each threshold, with explicit charge, critical and sprint
  conditions. Include three-player bystander positions, resistance, vertical
  separation and exact-zero/edge cases.

## Before Wind Charge Implementation

- Trace `WindChargeItem.use`, its factory and constructors for spawn position,
  item consumption, cooldown timing, owner identity and projectile RNG state.
  Separate player use from dispenser behavior.
- Trace `AbstractHurtingProjectile.tick`, inertia, gravity, collision/deflection,
  owner clearance and lifetime in air, including exact operation order and
  initial no-deflection counter semantics.
- Trace `AbstractWindCharge` hit callbacks and `ServerExplosion.hurtEntities`:
  explosion center, distance and exposure sampling, self/other motion, immunity,
  block resistance and trigger behavior. Identify exactly what static full-cube
  arenas can represent and what must remain unsupported.
- Capture seeded or otherwise controlled game trajectories and impacts, with
  open/occluded targets and owner self-launch. Distinguish direct damage from
  blast knockback and inspect cooldown/deflection boundary ticks.

## Later Enchantments

Density, Breach and Wind Burst are deferred. Their packaged definitions are
useful evidence, but modifier context, applicability, effect ordering and full
explosion behavior must be traced and independently tested before enabling them.
Do not treat Wind Burst as a thrown wind-charge entity.

Complete the integration and separate CPU, CUDA and Minecraft capture gates in
the [implementation plan](MACE_WIND_IMPLEMENTATION_PLAN.md) before advertising
any supported feature.