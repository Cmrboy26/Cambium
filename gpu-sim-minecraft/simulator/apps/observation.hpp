#pragma once

#include "minecraft_sim/projectiles.hpp"

#include <array>
#include <cmath>
#include <vector>

namespace minecraft_sim::app {

struct ObservationProjection {
    int source;
    float scale;
    float category;
    float lower;
    float upper;
    bool one_hot;
};

MCSIM_HD inline float projectObservation(const float* catalogue, const ObservationProjection& projection) {
    const float value = catalogue[projection.source];
    const float encoded = projection.one_hot ? (value == projection.category ? 1.0F : 0.0F)
        : (value < projection.lower ? projection.lower : (projection.upper < value ? projection.upper : value));
    return encoded * projection.scale;
}

MCSIM_HD inline void legacyEquipmentObservation(float* catalogue, const PlayerState& source,
                                                  const PlayerState& target, int observed_opponent_disable) {
    catalogue[22] = weaponAttributes(source.weapon).disable_shield_seconds > 0 ? 1.0F : 0.0F;
    catalogue[23] = weaponAttributes(target.weapon).disable_shield_seconds > 0 ? 1.0F : 0.0F;
    catalogue[24] = source.using_shield ? 1.0F : 0.0F;
    catalogue[25] = target.using_shield ? 1.0F : 0.0F;
    catalogue[26] = source.shield_cooldown > 0 ? 1.0F : 0.0F;
    catalogue[27] = observed_opponent_disable > 0 ? 1.0F : 0.0F;
    catalogue[28] = static_cast<float>(minimum(source.shield_cooldown / 100.0F, 1.0F));
    catalogue[29] = source.attackCharge();
    catalogue[30] = source.using_shield
        ? static_cast<float>(minimum(source.shield_use_ticks / 5.0F, 1.0F)) : 0.0F;
}

MCSIM_HD inline float equipmentFraction(double value, double maximum) {
    const double fraction = value / maximum;
    return static_cast<float>(fraction < 0.0 ? 0.0 : (fraction > 1.0 ? 1.0 : fraction));
}

MCSIM_HD inline void equipmentObservation(float* catalogue, const PlayerState& source, const PlayerState& target,
                                          const WeaponItem* own_slots, int own_slot_count) {
    for (int actor = 0; actor < 2; ++actor) {
        const auto& state = actor == 0 ? source : target;
        catalogue[33 + actor] = state.shield_equipped ? 1.0F : 0.0F;
        catalogue[35 + actor] = equipmentFraction(state.attack_damage, 10.0);
        catalogue[37 + actor] = equipmentFraction(state.attack_speed, 4.0);
        catalogue[39 + actor] = equipmentFraction(state.armor, 20.0);
        catalogue[41 + actor] = equipmentFraction(state.armor_toughness, 12.0);
    }
    for (int slot = 0; slot < 9; ++slot) {
        float* values = catalogue + 43 + slot * 4;
        for (int field = 0; field < 4; ++field) values[field] = 0.0F;
        if (slot >= own_slot_count || own_slots[slot] == WeaponItem::unarmed
            || (own_slots[slot] == WeaponItem::snowball && source.snowball_count == 0)
            || (own_slots[slot] == WeaponItem::egg && source.egg_count == 0)
            || (own_slots[slot] == WeaponItem::ender_pearl && source.ender_pearl_count == 0)
            || (own_slots[slot] == WeaponItem::wind_charge && source.wind_charge_count == 0)) continue;
        const auto attributes = weaponAttributes(own_slots[slot]);
        values[0] = 1.0F;
        values[1] = attributes.disable_shield_seconds > 0.0F ? 1.0F : 0.0F;
        values[2] = equipmentFraction(attributes.damage, 10.0);
        values[3] = equipmentFraction(attributes.speed, 4.0);
    }
}

inline void equipmentObservation(float* catalogue, const PlayerState& source, const PlayerState& target,
                                 const std::vector<WeaponItem>& own_slots) {
    equipmentObservation(catalogue, source, target, own_slots.data(), static_cast<int>(own_slots.size()));
}

MCSIM_HD inline void velocityObservation(float* catalogue, const PlayerState& source, const PlayerState& target) {
    catalogue[79] = static_cast<float>(source.velocity.x);
    catalogue[80] = static_cast<float>(source.velocity.y);
    catalogue[81] = static_cast<float>(source.velocity.z);
    catalogue[82] = static_cast<float>(target.velocity.x - source.velocity.x);
    catalogue[83] = static_cast<float>(target.velocity.y - source.velocity.y);
    catalogue[84] = static_cast<float>(target.velocity.z - source.velocity.z);
}

MCSIM_HD inline void movementObservation(float* catalogue, const PlayerState& source, const PlayerState& target) {
    catalogue[85] = source.on_ground ? 1.0F : 0.0F;
    catalogue[86] = target.on_ground ? 1.0F : 0.0F;
    catalogue[87] = source.sprinting ? 1.0F : 0.0F;
    catalogue[88] = target.sprinting ? 1.0F : 0.0F;
}

// Samples actual bow use and the nearest moving opponent-owned arrow from each actor's feet position.
// Source: minecraft-26.3-src/net/minecraft/world/entity/projectile/Projectile.java, getOwner (379);
// minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java, tick (173-266).
MCSIM_HD inline void arrowObservation(float* catalogue, const PlayerState& source, const PlayerState& target,
                                      const ArrowObservationSummary& summary) {
    catalogue[89] = static_cast<float>(summary.ordinary_arrows) / 64.0F;
    catalogue[90] = source.using_bow ? 1.0F : 0.0F;
    catalogue[91] = source.using_bow ? bowPower(summary.draw_ticks) : 0.0F;
    catalogue[92] = source.weapon == WeaponItem::bow ? 1.0F : 0.0F;
    catalogue[93] = target.weapon == WeaponItem::bow ? 1.0F : 0.0F;
    catalogue[94] = target.using_bow ? 1.0F : 0.0F;
    for (int index = 95; index < 102; ++index) catalogue[index] = 0.0F;
    if (!summary.found) {
        return;
    }
    catalogue[95] = 1.0F;
    catalogue[96] = static_cast<float>(summary.relative_position.x / 10.0);
    catalogue[97] = static_cast<float>(summary.relative_position.y / 10.0);
    catalogue[98] = static_cast<float>(summary.relative_position.z / 10.0);
    catalogue[99] = static_cast<float>(summary.relative_velocity.x);
    catalogue[100] = static_cast<float>(summary.relative_velocity.y);
    catalogue[101] = static_cast<float>(summary.relative_velocity.z);
}

MCSIM_HD inline void arrowObservation(float* catalogue, const PlayerState& source, const PlayerState& target,
                                      const ProjectileWorldState& projectiles, int source_player) {
    arrowObservation(catalogue, source, target, summarizeArrowObservation(source, projectiles, source_player));
}

MCSIM_HD inline double wrapObservationAngle(double angle) {
    return angle - 360.0 * ::floor((angle + 180.0) / 360.0);
}

MCSIM_HD inline void baseObservation(const PlayerState& source, const PlayerState& target,
                                     const float* rays, float* result) {
    constexpr double radians = 0.017453292519943295;
    const Vec3 relative{target.position.x - source.position.x, target.position.y - source.position.y,
                        target.position.z - source.position.z};
    const double distance = ::sqrt(relative.lengthSquared());
    for (int index = 0; index < 22; ++index) result[index] = 0.0F;
    result[0] = source.health / 20.0F;
    result[1] = static_cast<float>(wrapObservationAngle(source.yaw) / 180.0);
    result[2] = source.pitch / 90.0F;
    result[3] = target.alive() ? 1.0F : 0.0F;
    result[4] = target.health / 20.0F;
    result[5] = static_cast<float>(relative.x / 10.0);
    result[6] = static_cast<float>(relative.y / 10.0);
    result[7] = static_cast<float>(relative.z / 10.0);
    result[8] = static_cast<float>(wrapObservationAngle(target.yaw) / 180.0);
    result[9] = target.pitch / 90.0F;
    for (int index = 0; index < 8; ++index) result[index + 10] = rays[index];
    result[18] = static_cast<float>(minimum(distance / 10.0, 1.0));
    if (distance > 0.01) {
        const double yaw = source.yaw * radians;
        const double pitch = source.pitch * radians;
        result[19] = static_cast<float>((-::sin(yaw) * ::cos(pitch) * relative.x
            - ::sin(pitch) * relative.y + ::cos(yaw) * ::cos(pitch) * relative.z) / distance);
        result[20] = static_cast<float>(wrapObservationAngle(::atan2(-relative.x, relative.z) / radians - source.yaw) / 180.0);
        result[21] = static_cast<float>((::atan2(-relative.y,
            maximum(::hypot(relative.x, relative.z), 0.01)) / radians - source.pitch) / 90.0);
    }
}

MCSIM_HD inline void baseWorldObservation(const PlayerState& source, const PlayerState& target,
                                          BlockWorldView blocks, float* result) {
    constexpr double radians = 0.017453292519943295;
    float rays[8]{};
    for (int ray = 0; ray < 8; ++ray) {
        const double angle = ray * 45.0 * radians;
        rays[ray] = static_cast<float>(blockRayFraction(blocks, source.eyePosition(),
            {-::sin(angle) * 50.0, 0.0, ::cos(angle) * 50.0}));
    }
    baseObservation(source, target, rays, result);
}

inline std::array<float, 22> baseObservation(const PlayerState& source, const PlayerState& target,
                                             const std::array<float, 8>& rays) {
    std::array<float, 22> result{};
    baseObservation(source, target, rays.data(), result.data());
    return result;
}

}