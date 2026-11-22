#pragma once

#include "minecraft_sim/geometry.hpp"
#include "minecraft_sim/bridge_match.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

namespace minecraft_sim {

inline constexpr int bridge_local_voxel_count = 5 * 9 * 9;
inline constexpr int bridge_match_feature_count_v1 = 10;
inline constexpr int bridge_match_feature_count_v2 = 17;

MCSIM_HD inline bool bridgeMatchFeaturesV1(const BridgeMatchRules& rules, const BridgeMatchState& state,
                                          int player, float* output) {
    if (!output || (player != 0 && player != 1) || rules.goals_to_win <= 0 || rules.max_ticks <= 0
        || rules.cage_ticks < 0 || rules.arrow_refill_ticks < 0 || state.ticks < 0) return false;
    const auto& kit = state.kits[player];
    const int remaining = state.ticks < rules.max_ticks ? rules.max_ticks - state.ticks : 0;
    output[0] = static_cast<float>(double(state.scores[player]) / rules.goals_to_win);
    output[1] = static_cast<float>(double(state.scores[1 - player]) / rules.goals_to_win);
    output[2] = static_cast<float>(double(remaining) / rules.max_ticks);
    output[3] = static_cast<float>(double(state.cage_remaining) / (rules.cage_ticks > 0 ? rules.cage_ticks : 1));
    output[4] = static_cast<float>(double(kit.blocks) / 64);
    output[5] = static_cast<float>(double(kit.golden_apples) / 64);
    output[6] = static_cast<float>(double(kit.arrows) / 64);
    output[7] = static_cast<float>(double(kit.arrow_refill_elapsed)
                                   / (rules.arrow_refill_ticks > 0 ? rules.arrow_refill_ticks : 1));
    output[8] = rules.arrow_refill_ticks > 0 ? 1.0F : 0.0F;
    output[9] = static_cast<float>(double(kit.selected_slot) / 8);
    return true;
}

MCSIM_HD inline bool bridgeMatchFeaturesV2(const PlayerState& actor, const BridgeMatchRules& rules,
                                          const BridgeMatchState& state, int player, float* output) {
    if (!output || (player != 0 && player != 1)) return false;
    float features[bridge_match_feature_count_v2]{};
    if (!bridgeMatchFeaturesV1(rules, state, player, features)) return false;
    const auto item = itemProgressState(actor, state.kits[player]);
    if (item.eating_remaining < 0 || item.eating_remaining > 32
        || !(item.mining_progress >= 0.0F && item.mining_progress <= 1.0F)
        || item.use_delay < 0 || item.use_delay > 4 || item.mining_delay < 0 || item.mining_delay > 5) return false;
    const auto& goal = rules.goals[1 - player];
    const Vec3 target{(goal.lower.x + goal.upper.x) / 2.0, (goal.lower.y + goal.upper.y) / 2.0,
                      (goal.lower.z + goal.upper.z) / 2.0};
    const Vec3 relative{target.x - actor.position.x, target.y - actor.position.y, target.z - actor.position.z};
    const double horizontal = ::hypot(relative.x, relative.z);
    const double distance = ::hypot(horizontal, relative.y);
    constexpr double double_max = (std::numeric_limits<double>::max)();
    constexpr double degrees = 180.0 / 3.14159265358979323846;
    if (!(distance >= 0.0 && distance <= double_max)
        || !(actor.yaw >= -double_max && actor.yaw <= double_max)
        || !(actor.pitch >= -double_max && actor.pitch <= double_max)) return false;
    const double desired_yaw = ::atan2(-relative.x, relative.z) * degrees;
    const double yaw_delta = desired_yaw - actor.yaw;
    const double yaw_error = horizontal > 0.0 ? yaw_delta - 360.0 * ::floor((yaw_delta + 180.0) / 360.0) : 0.0;
    const double pitch_error = distance > 0.0 ? ::atan2(-relative.y, horizontal) * degrees - actor.pitch : 0.0;
    features[10] = item.eating_remaining > 0 ? static_cast<float>(1.0 - double(item.eating_remaining) / 32.0) : 0.0F;
    features[11] = item.mining_progress;
    features[12] = static_cast<float>(yaw_error / 180.0);
    features[13] = static_cast<float>(pitch_error / 90.0);
    features[14] = static_cast<float>(minimum(distance / 50.0, 1.0));
    features[15] = static_cast<float>(double(item.use_delay) / 4.0);
    features[16] = static_cast<float>(double(item.mining_delay) / 5.0);
    for (int index = 0; index < bridge_match_feature_count_v2; ++index) output[index] = features[index];
    return true;
}

MCSIM_HD inline bool bridgeLocalVoxels(BlockWorldView blocks, Vec3 feet, float yaw,
                                       int actor_team, std::uint8_t* output) {
    constexpr auto int_max = (std::numeric_limits<int>::max)();
    constexpr auto index_max = (std::numeric_limits<std::int64_t>::max)();
    constexpr double double_max = (std::numeric_limits<double>::max)();
    constexpr float float_max = (std::numeric_limits<float>::max)();
    if (!output || !blocks.cells || (actor_team != 0 && actor_team != 1)
        || blocks.size_x <= 0 || blocks.size_y <= 0 || blocks.size_z <= 0
        || std::int64_t(blocks.origin_x) + blocks.size_x > std::int64_t(int_max) + 1
        || std::int64_t(blocks.origin_y) + blocks.size_y > std::int64_t(int_max) + 1
        || std::int64_t(blocks.origin_z) + blocks.size_z > std::int64_t(int_max) + 1
        || std::int64_t(blocks.size_x) > index_max / blocks.size_y
        || std::int64_t(blocks.size_x) * blocks.size_y > index_max / blocks.size_z
        || !(feet.x >= -double_max && feet.x <= double_max)
        || !(feet.y >= -double_max && feet.y <= double_max)
        || !(feet.z >= -double_max && feet.z <= double_max)
        || !(yaw >= -float_max && yaw <= float_max)) return false;

    const double reduced = ::fmod(static_cast<double>(yaw), 360.0);
    const int quarter = reduced >= 0.0
        ? (reduced < 45.0 ? 0 : reduced < 135.0 ? 1 : reduced < 225.0 ? 2 : reduced < 315.0 ? 3 : 0)
        : (reduced >= -45.0 ? 0 : reduced >= -135.0 ? 3 : reduced >= -225.0 ? 2 : reduced >= -315.0 ? 1 : 0);
    const int forward_x[4]{0, -1, 0, 1};
    const int forward_z[4]{1, 0, -1, 0};
    const int right_x[4]{1, 0, -1, 0};
    const int right_z[4]{0, 1, 0, -1};
    const double positions[3]{feet.x, feet.y, feet.z};
    std::int64_t origin[3]{};
    for (int axis = 0; axis < 3; ++axis) {
        origin[axis] = positions[axis] < -4294967300.0 ? -4294967300LL
            : positions[axis] > 4294967300.0 ? 4294967300LL
            : static_cast<std::int64_t>(::floor(positions[axis]));
    }
    for (int up = 0; up < 5; ++up) {
        for (int right = 0; right < 9; ++right) {
            for (int forward = 0; forward < 9; ++forward) {
                const auto x = origin[0] + (right - 4) * right_x[quarter] + (forward - 4) * forward_x[quarter];
                const auto y = origin[1] + up - 2;
                const auto z = origin[2] + (right - 4) * right_z[quarter] + (forward - 4) * forward_z[quarter];
                const int output_index = (up * 9 + right) * 9 + forward;
                if (x < blocks.origin_x || x >= std::int64_t(blocks.origin_x) + blocks.size_x
                    || y < blocks.origin_y || y >= std::int64_t(blocks.origin_y) + blocks.size_y
                    || z < blocks.origin_z || z >= std::int64_t(blocks.origin_z) + blocks.size_z) {
                    output[output_index] = 4;
                    continue;
                }
                const auto index = blocks.cellIndex(static_cast<int>(x), static_cast<int>(y), static_cast<int>(z));
                const auto base = blocks.cells[index];
                const auto placed = blocks.overlay ? blocks.overlay[index] : PlacedBlock::empty;
                if ((base != Block::air && base != Block::solid)
                    || (placed != PlacedBlock::empty && placed != PlacedBlock::team0 && placed != PlacedBlock::team1)
                    || (base == Block::solid && placed != PlacedBlock::empty)) return false;
                output[output_index] = base == Block::solid ? 1 : placed == PlacedBlock::empty ? 0
                    : placed == (actor_team == 0 ? PlacedBlock::team0 : PlacedBlock::team1) ? 2 : 3;
            }
        }
    }
    return true;
}

}