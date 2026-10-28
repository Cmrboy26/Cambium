#pragma once

namespace minecraft_sim {

// Minecraft 26.3: MaceItem.java:44-47, createAttributes.
inline constexpr double MACE_DAMAGE_MODIFIER = 5.0;
inline constexpr double MACE_SPEED_MODIFIER = double(-3.4F);

// Minecraft 26.3: MaceItem.java:128-143, 160, 164.
inline constexpr float MACE_SMASH_FALL_THRESHOLD = 1.5f;
inline constexpr float MACE_HEAVY_THRESHOLD = 5.0f;
inline constexpr float MACE_KNOCKBACK_RADIUS = 3.5f;
inline constexpr double MACE_KNOCKBACK_POWER = double(0.7F);

// Minecraft 26.3: MaceItem.java:101-124, getAttackDamageBonus.
inline constexpr double MACE_DAMAGE_THRESHOLD_1 = 3.0;
inline constexpr double MACE_DAMAGE_THRESHOLD_2 = 8.0;

// Minecraft 26.3: WindChargeItem.java:30-36, use.
inline constexpr float WIND_CHARGE_SHOOT_POWER = 1.5f;

// Minecraft 26.3: WindCharge.java:36, 52-63, 68-69.
inline constexpr float WIND_CHARGE_EXPLOSION_POWER = 1.2F;
inline constexpr int WIND_CHARGE_NO_DEFLECT_TICKS = 5;

// Minecraft 26.3: AbstractWindCharge.java:88-107, onHitEntity.
inline constexpr float WIND_CHARGE_DAMAGE = 1.0f;

// Minecraft 26.3: Items.java:1168, 0.5 seconds at 20 ticks per second.
inline constexpr int WIND_CHARGE_COOLDOWN_TICKS = 10;

}  // namespace minecraft_sim
