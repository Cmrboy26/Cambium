#pragma once

#include "minecraft_sim/core.hpp"

#include <cmath>
#include <type_traits>

namespace minecraft_sim::app {

MCSIM_HD inline double rewardClamp(double value, double lower, double upper) {
    return value < lower ? lower : (upper < value ? upper : value);
}

MCSIM_HD inline double rewardMaximum(double left, double right) {
    return left < right ? right : left;
}

// Sources: minecraft-26.3-src/net/minecraft/world/item/BowItem.java, releaseUsing (26-51);
// minecraft-26.3-src/net/minecraft/world/entity/projectile/Projectile.java, shootFromRotation (183-194);
// minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java, tick (222-266).
MCSIM_HD inline void shapingRewards(const PlayerState& source, const PlayerState& target,
                                    int ordinary_arrows, BlockWorldView blocks,
                                    const float* weights, float* output) {
    constexpr double radians = 0.017453292519943295;
    const Vec3 relative{target.position.x - source.position.x,
                        target.position.y - source.position.y,
                        target.position.z - source.position.z};
    const double distance = ::sqrt(relative.lengthSquared());
    const double pitch_error = ::fabsf(source.pitch);
    double amounts[5]{0.0, rewardClamp((10.0 - distance) / 7.0, 0.0, 1.0),
        rewardMaximum(0.0, 1.0 - pitch_error / 90.0), 0.0, pitch_error > 60.0 ? 1.0 : 0.0};
    const auto source_eye = source.eyePosition();
    const auto target_eye = target.eyePosition();
    Vec3 aim{target_eye.x - source_eye.x, target_eye.y - source_eye.y, target_eye.z - source_eye.z};
    const bool bow_equipped = source.weapon == WeaponItem::bow;
    const bool can_aim = !bow_equipped || ordinary_arrows > 0;
    if ((weights[0] != 0.0F || weights[3] != 0.0F) && can_aim
        && aim.lengthSquared() > 0.0001 && blockRayFraction(blocks, source_eye, aim) >= 1.0) {
        if (bow_equipped) {
            const auto target_bounds = target.bounds();
            const Vec3 target_center{(target_bounds.lower.x + target_bounds.upper.x) * 0.5,
                                     (target_bounds.lower.y + target_bounds.upper.y) * 0.5,
                                     (target_bounds.lower.z + target_bounds.upper.z) * 0.5};
            const Vec3 target_motion{target.velocity.x, target.on_ground ? 0.0 : target.velocity.y, target.velocity.z};
            const Vec3 inherited{source.velocity.x, source.on_ground ? 0.0 : source.velocity.y, source.velocity.z};
            double flight_ticks = rewardClamp(::hypot(aim.x, aim.z) / 3.0, 1.0, 60.0);
            for (int iteration = 0; iteration < 4; ++iteration) {
                const double drag = double(0.99F);
                const double travel = (1.0 - ::pow(drag, flight_ticks)) / (1.0 - drag);
                const double drop = 0.05 * (flight_ticks - travel) / (1.0 - drag);
                aim = {(target_center.x - source_eye.x + target_motion.x * flight_ticks) / travel - inherited.x,
                       (target_center.y - source_eye.y + double(0.1F) + target_motion.y * flight_ticks + drop)
                           / travel - inherited.y,
                       (target_center.z - source_eye.z + target_motion.z * flight_ticks) / travel - inherited.z};
                flight_ticks = rewardClamp(flight_ticks * ::sqrt(aim.lengthSquared()) / 3.0, 1.0, 60.0);
            }
        }
        const double yaw_delta = ::atan2(-aim.x, aim.z) / radians - source.yaw;
        const double yaw_error = ::fabs(yaw_delta - 360.0 * ::floor((yaw_delta + 180.0) / 360.0));
        const double aim_pitch = -::atan2(aim.y, ::hypot(aim.x, aim.z)) / radians;
        const double aim_error = rewardMaximum(yaw_error, ::fabs(aim_pitch - source.pitch));
        amounts[0] = rewardMaximum(0.0, 1.0 - aim_error / 30.0);
        amounts[3] = aim_error < 20.0 ? 1.0 : 0.0;
    }
    for (int term = 0; term < 5; ++term) output[term] = static_cast<float>(amounts[term] * weights[term]);
}

struct DuelRewardSettings {
    float weights[10]{};
    float block_weight = 0.0F;
    float event_weights[2]{};
    float shaping_weights[5]{};
    int max_ticks = 0;
};

struct DuelRewardState {
    bool dealt_damage[2]{};
};

struct DuelRewardStep {
    float rewards[2]{};
    double contributions[2][18]{};
    bool lost[2]{};
    bool terminated = false;
    bool truncated = false;
};

static_assert(std::is_trivially_copyable_v<DuelRewardSettings>);
static_assert(std::is_trivially_copyable_v<DuelRewardState>);
static_assert(std::is_trivially_copyable_v<DuelRewardStep>);

MCSIM_HD inline bool duelOutside(const PlayerState& player, BlockWorldView bounds) {
    const auto& position = player.position;
    return position.x < bounds.origin_x || position.x >= bounds.origin_x + bounds.size_x
        || position.z < bounds.origin_z || position.z >= bounds.origin_z + bounds.size_z
        || position.y < bounds.origin_y - 4;
}

MCSIM_HD inline void addDuelReward(DuelRewardStep& result, int player, int term, float amount) {
    result.rewards[player] += amount;
    result.contributions[player][term] += amount;
}

MCSIM_HD inline DuelRewardStep sharedRewardEvents(const PlayerState* players, const WorldState& world,
                                                  const DuelRewardSettings& settings, DuelRewardState& state) {
    DuelRewardStep result;
    for (int player = 0; player < 2; ++player) {
        const auto& actor = players[player];
        const auto& opponent = players[1 - player];
        const float health_dealt = actor.last_attack.health_lost + world.projectile_health_dealt[player];
        const float health_taken = opponent.last_attack.health_lost + world.projectile_health_taken[player];
        if (health_dealt > 0.0F) {
            addDuelReward(result, player, 0, settings.weights[0] * (health_dealt / 20.0F));
            state.dealt_damage[player] = true;
        }
        if (health_taken > 0.0F) addDuelReward(result, player, 1, settings.weights[1] * (health_taken / 20.0F));
        addDuelReward(result, player, 9, settings.weights[9] * world.projectile_arrow_progress[player]);
        if (actor.last_attack.attempted && actor.last_attack.target < 0) {
            addDuelReward(result, player, 6, settings.weights[6]);
            const double speed = actor.last_attack.attack_speed;
            const float penalty = speed > 0.0 ? static_cast<float>(settings.event_weights[1] / speed) : 0.0F;
            addDuelReward(result, player, 12, penalty);
        }
        addDuelReward(result, player, 11, static_cast<float>(settings.event_weights[0] * world.wind_self_launch[player]));
        if (actor.last_attack.shield_disabled) addDuelReward(result, player, 7, settings.weights[7]);
        if (actor.last_attack.blocked_damage > 0.0F && weaponAttributes(actor.weapon).disable_shield_seconds == 0.0F) {
            addDuelReward(result, 1 - player, 10, settings.block_weight);
        }
    }
    return result;
}

MCSIM_HD inline DuelRewardStep duelRewards(const PlayerState* players, const WorldState& world,
                                           BlockWorldView blocks, const ProjectileWorldState& projectiles,
                                           const DuelRewardSettings& settings, DuelRewardState& state) {
    auto result = sharedRewardEvents(players, world, settings, state);
    for (int player = 0; player < 2; ++player) {
        result.lost[player] = !players[player].alive() || duelOutside(players[player], blocks);
    }
    result.terminated = result.lost[0] || result.lost[1];
    result.truncated = !result.terminated && world.tick >= static_cast<std::uint64_t>(settings.max_ticks);
    for (int player = 0; player < 2; ++player) {
        if (!result.terminated) {
            float shaping[5]{};
            shapingRewards(players[player], players[1 - player], projectiles.bows[player].ordinary_arrows,
                           blocks, settings.shaping_weights, shaping);
            for (int term = 0; term < 5; ++term) addDuelReward(result, player, 13 + term, shaping[term]);
        }
        if (result.terminated) {
            addDuelReward(result, player, result.lost[player] ? 3 : 2,
                          result.lost[player] ? settings.weights[3] : settings.weights[2]);
            if (!result.lost[player]) {
                const float speed_bonus = static_cast<float>(settings.weights[8] * rewardClamp(
                    1.0 - static_cast<double>(world.tick) / settings.max_ticks, 0.0, 1.0));
                addDuelReward(result, player, 8, speed_bonus);
            }
        } else if (result.truncated) {
            result.rewards[player] += settings.weights[4] + (state.dealt_damage[player] ? 0.0F : settings.weights[5]);
            result.contributions[player][4] += settings.weights[4];
            if (!state.dealt_damage[player]) result.contributions[player][5] += settings.weights[5];
        }
    }
    return result;
}

}
