#pragma once

#include "minecraft_sim/geometry.hpp"
#include "minecraft_sim/math.hpp"

#include <limits>

namespace minecraft_sim {

struct BridgeTerrainGeneration {
    int center_x = 0;
    int floor_y = 0;
    int start_z = 0;
    int end_z = 0;
    int segments = 0;
};

MCSIM_HD inline bool generateBridgeTerrain(BlockWorldView blocks, const BridgeTerrainGeneration& generation,
                                           std::uint64_t seed) {
    if (!blocks.overlay || generation.segments < 1 || generation.segments > 16
        || generation.end_z <= generation.start_z
        || std::int64_t(generation.end_z) - generation.start_z > 10000
        || generation.center_x < (std::numeric_limits<int>::min)() + 2
        || generation.center_x > (std::numeric_limits<int>::max)() - 2
        || generation.floor_y == (std::numeric_limits<int>::max)()) return false;
    for (int z = generation.start_z; z < generation.end_z; ++z) {
        const auto center = blocks.cellIndex(generation.center_x, generation.floor_y, z);
        if (center < 0 || blocks.cells[center] != Block::solid) return false;
        for (int offset = -2; offset <= 2; ++offset) {
            if (offset == 0) continue;
            for (int height = 0; height <= 1; ++height) {
                const auto index = blocks.cellIndex(generation.center_x + offset, generation.floor_y + height, z);
                if (index < 0 || blocks.cells[index] != Block::air) return false;
            }
        }
    }
    LegacyRandom random;
    random.setSeed(seed);
    const int span = generation.end_z - generation.start_z;
    for (int segment = 0; segment < generation.segments; ++segment) {
        const int side = random.next(1) == 0 ? -1 : 1;
        const int requested_length = 2 + static_cast<int>(random.next(2));
        const int length = requested_length < span ? requested_length : span;
        const int start = generation.start_z + static_cast<int>(random.next(31) % (span - length + 1));
        const int width = 1 + static_cast<int>(random.next(1));
        const bool rail = random.next(1) != 0;
        const auto team = random.next(1) == 0 ? PlacedBlock::team0 : PlacedBlock::team1;
        for (int z = start; z < start + length; ++z) {
            for (int offset = 1; offset <= width; ++offset) {
                blocks.setPlacedBlock(generation.center_x + side * offset, generation.floor_y, z, team);
            }
            if (rail) blocks.setPlacedBlock(generation.center_x + side * width, generation.floor_y + 1, z, team);
        }
    }
    return true;
}

MCSIM_HD inline std::uint64_t bridgeTerrainSeed(std::uint64_t player_random_state) {
    return player_random_state ^ 0x7465727261696eULL;
}

}