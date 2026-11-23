#pragma once

#include "minecraft_sim/geometry.hpp"
#include "minecraft_sim/math.hpp"

#include <limits>

namespace minecraft_sim {

struct BridgeTerrainPattern {
    int location = 0;
    int type = 0;
    int minimum = 0;
    int maximum = 0;
    int probability = 0;
};

struct BridgeTerrainGeneration {
    int center_x = 0;
    int floor_y = 0;
    int start_z = 0;
    int end_z = 0;
    int segments = 0;
    int bridge_end_x = 0;
    int pattern_count = 0;
    BridgeTerrainPattern patterns[32]{};
};

MCSIM_HD inline int bridgePatternExtent(const BridgeTerrainPattern& pattern, int position, int span,
                                       LegacyRandom& random) {
    if (pattern.type == 0) return pattern.minimum + static_cast<int>(random.next(31) % (pattern.maximum - pattern.minimum + 1));
    if (pattern.type == 1 || span == 1) return pattern.minimum;
    const std::int64_t denominator = span - 1;
    const std::int64_t product = std::int64_t(position) * (denominator - position);
    const std::int64_t divisor = 5 * denominator * denominator - 4 * product;
    const std::int64_t numerator = std::int64_t(pattern.maximum - pattern.minimum) * 16 * product;
    return pattern.minimum + static_cast<int>((numerator + divisor / 2) / divisor);
}

MCSIM_HD inline bool generateBridgePatterns(BlockWorldView blocks, const BridgeTerrainGeneration& generation,
                                            std::uint64_t seed) {
    if (!blocks.overlay || generation.pattern_count < 1 || generation.pattern_count > 32
        || generation.center_x < (std::numeric_limits<int>::min)() + 8
        || generation.bridge_end_x > (std::numeric_limits<int>::max)() - 8
        || generation.floor_y > (std::numeric_limits<int>::max)() - 8
        || generation.bridge_end_x <= generation.center_x
        || std::int64_t(generation.bridge_end_x) - generation.center_x > 64
        || generation.end_z <= generation.start_z
        || std::int64_t(generation.end_z) - generation.start_z > 10000) return false;
    for (int pattern_index = 0; pattern_index < generation.pattern_count; ++pattern_index) {
        const auto& pattern = generation.patterns[pattern_index];
        if (pattern.location < 0 || pattern.location > 1 || pattern.type < 0 || pattern.type > 2
            || pattern.minimum < 0 || pattern.maximum < pattern.minimum || pattern.maximum > 8
            || pattern.probability < 0 || pattern.probability > (1 << 24)
            || (pattern.type == 1 && pattern.minimum != pattern.maximum)) return false;
    }
    for (int z = generation.start_z; z < generation.end_z; ++z) {
        for (int x = generation.center_x; x < generation.bridge_end_x; ++x) {
            const auto floor = blocks.cellIndex(x, generation.floor_y, z);
            if (floor < 0 || blocks.cells[floor] != Block::solid) return false;
        }
        for (int pattern_index = 0; pattern_index < generation.pattern_count; ++pattern_index) {
            const auto& pattern = generation.patterns[pattern_index];
            for (int offset = 1; offset <= pattern.maximum; ++offset) {
                const int start_x = pattern.location == 0 ? generation.center_x : generation.center_x - offset;
                const int end_x = pattern.location == 0 ? generation.bridge_end_x : generation.bridge_end_x + offset;
                for (int x = start_x; x < end_x; ++x) {
                    if (pattern.location == 1 && x != start_x && x != end_x - 1) continue;
                    const int y = generation.floor_y + (pattern.location == 0 ? offset : 0);
                    const auto cell = blocks.cellIndex(x, y, z);
                    if (cell < 0 || blocks.cells[cell] != Block::air) return false;
                }
            }
        }
    }
    LegacyRandom random;
    random.setSeed(seed);
    const int span = generation.end_z - generation.start_z;
    for (int pattern_index = 0; pattern_index < generation.pattern_count; ++pattern_index) {
        const auto& pattern = generation.patterns[pattern_index];
        for (int z = generation.start_z; z < generation.end_z; ++z) {
            const int columns = pattern.location == 0 ? generation.bridge_end_x - generation.center_x : 2;
            for (int column = 0; column < columns; ++column) {
                if (random.next(24) >= static_cast<std::uint32_t>(pattern.probability)) continue;
                const int extent = bridgePatternExtent(pattern, z - generation.start_z, span, random);
                const auto team = random.next(1) == 0 ? PlacedBlock::team0 : PlacedBlock::team1;
                for (int offset = 1; offset <= extent; ++offset) {
                    const int x = pattern.location == 0 ? generation.center_x + column
                        : column == 0 ? generation.center_x - offset : generation.bridge_end_x + offset - 1;
                    const int y = generation.floor_y + (pattern.location == 0 ? offset : 0);
                    blocks.setPlacedBlock(x, y, z, team);
                }
            }
        }
    }
    return true;
}

MCSIM_HD inline bool generateBridgeTerrain(BlockWorldView blocks, const BridgeTerrainGeneration& generation,
                                           std::uint64_t seed) {
    if (generation.pattern_count != 0) return generateBridgePatterns(blocks, generation, seed);
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