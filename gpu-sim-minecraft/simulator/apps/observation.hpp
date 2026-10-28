#pragma once

#include "configuration.hpp"
#include "minecraft_sim/projectiles.hpp"
#include <cmath>

namespace minecraft_sim::app {

inline float equipmentFraction(double value, double maximum) {
    return static_cast<float>(std::clamp(value / maximum, 0.0, 1.0));
}

inline void equipmentObservation(float* catalogue, const PlayerState& source, const PlayerState& target,
                                 const std::vector<WeaponItem>& own_slots) {
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
        std::fill_n(values, 4, 0.0F);
        if (slot >= static_cast<int>(own_slots.size()) || own_slots[slot] == WeaponItem::unarmed
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

inline void velocityObservation(float* catalogue, const PlayerState& source, const PlayerState& target) {
    catalogue[79] = static_cast<float>(source.velocity.x);
    catalogue[80] = static_cast<float>(source.velocity.y);
    catalogue[81] = static_cast<float>(source.velocity.z);
    catalogue[82] = static_cast<float>(target.velocity.x - source.velocity.x);
    catalogue[83] = static_cast<float>(target.velocity.y - source.velocity.y);
    catalogue[84] = static_cast<float>(target.velocity.z - source.velocity.z);
}

inline void movementObservation(float* catalogue, const PlayerState& source, const PlayerState& target) {
    catalogue[85] = source.on_ground ? 1.0F : 0.0F;
    catalogue[86] = target.on_ground ? 1.0F : 0.0F;
    catalogue[87] = source.sprinting ? 1.0F : 0.0F;
    catalogue[88] = target.sprinting ? 1.0F : 0.0F;
}

// Samples actual bow use and the nearest moving opponent-owned arrow from each actor's feet position.
// Source: minecraft-26.3-src/net/minecraft/world/entity/projectile/Projectile.java, getOwner (379);
// minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java, tick (173-266).
inline void arrowObservation(float* catalogue, const PlayerState& source, const PlayerState& target,
                             const ArrowObservationSummary& summary) {
    catalogue[89] = static_cast<float>(summary.ordinary_arrows) / 64.0F;
    catalogue[90] = source.using_bow ? 1.0F : 0.0F;
    catalogue[91] = source.using_bow ? bowPower(summary.draw_ticks) : 0.0F;
    catalogue[92] = source.weapon == WeaponItem::bow ? 1.0F : 0.0F;
    catalogue[93] = target.weapon == WeaponItem::bow ? 1.0F : 0.0F;
    catalogue[94] = target.using_bow ? 1.0F : 0.0F;
    std::fill_n(catalogue + 95, 7, 0.0F);
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

inline void arrowObservation(float* catalogue, const PlayerState& source, const PlayerState& target,
                             const ProjectileWorldState& projectiles, int source_player) {
    arrowObservation(catalogue, source, target, summarizeArrowObservation(source, projectiles, source_player));
}

inline std::array<float, 22> baseObservation(const PlayerState& source, const PlayerState& target,
                                           const std::array<float, 8>& rays) {
    constexpr double radians = 0.017453292519943295;
    const auto wrap = [](double angle) { return angle - 360.0 * std::floor((angle + 180.0) / 360.0); };
    const Vec3 relative{target.position.x - source.position.x, target.position.y - source.position.y,
                        target.position.z - source.position.z};
    const double distance = std::sqrt(relative.lengthSquared());
    std::array<float, 22> result{};
    result[0] = source.health / 20.0F;
    result[1] = static_cast<float>(wrap(source.yaw) / 180.0);
    result[2] = source.pitch / 90.0F;
    result[3] = target.alive() ? 1.0F : 0.0F;
    result[4] = target.health / 20.0F;
    result[5] = static_cast<float>(relative.x / 10.0);
    result[6] = static_cast<float>(relative.y / 10.0);
    result[7] = static_cast<float>(relative.z / 10.0);
    result[8] = static_cast<float>(wrap(target.yaw) / 180.0);
    result[9] = target.pitch / 90.0F;
    std::copy(rays.begin(), rays.end(), result.begin() + 10);
    result[18] = static_cast<float>(minimum(distance / 10.0, 1.0));
    if (distance > 0.01) {
        const double yaw = source.yaw * radians;
        const double pitch = source.pitch * radians;
        result[19] = static_cast<float>((-std::sin(yaw) * std::cos(pitch) * relative.x
            - std::sin(pitch) * relative.y + std::cos(yaw) * std::cos(pitch) * relative.z) / distance);
        result[20] = static_cast<float>(wrap(std::atan2(-relative.x, relative.z) / radians - source.yaw) / 180.0);
        result[21] = static_cast<float>((std::atan2(-relative.y,
            maximum(std::hypot(relative.x, relative.z), 0.01)) / radians - source.pitch) / 90.0);
    }
    return result;
}

}