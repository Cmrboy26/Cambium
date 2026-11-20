#pragma once

#include "minecraft_sim/geometry.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

namespace minecraft_sim {

inline constexpr int bridge_local_voxel_count = 5 * 9 * 9;

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