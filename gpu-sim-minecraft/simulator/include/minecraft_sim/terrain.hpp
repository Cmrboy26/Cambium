#pragma once

#include "minecraft_sim/geometry.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace minecraft_sim {

class TerrainOverlays {
public:
    TerrainOverlays(BlockWorldView base, std::size_t world_count) {
        if (!base.cells || base.overlay || world_count == 0
            || base.size_x <= 0 || base.size_y <= 0 || base.size_z <= 0
            || std::int64_t(base.origin_x) + base.size_x > (std::numeric_limits<int>::max)()
            || std::int64_t(base.origin_y) + base.size_y > (std::numeric_limits<int>::max)()
            || std::int64_t(base.origin_z) + base.size_z > (std::numeric_limits<int>::max)()) {
            throw std::invalid_argument("Invalid protected terrain base or world count");
        }
        const auto cells_per_world = checkedProduct(checkedProduct(std::size_t(base.size_x),
            std::size_t(base.size_y)), std::size_t(base.size_z));
        if (cells_per_world > std::size_t((std::numeric_limits<std::int64_t>::max)())) {
            throw std::invalid_argument("Terrain index overflow");
        }
        overlays_.resize(checkedProduct(cells_per_world, world_count), PlacedBlock::empty);
        worlds_.resize(world_count, base);
        for (std::size_t index = 0; index < world_count; ++index) {
            worlds_[index].overlay = overlays_.data() + index * cells_per_world;
        }
    }

    TerrainOverlays(const TerrainOverlays&) = delete;
    TerrainOverlays& operator=(const TerrainOverlays&) = delete;
    TerrainOverlays(TerrainOverlays&&) = delete;
    TerrainOverlays& operator=(TerrainOverlays&&) = delete;

    BlockWorldView view(std::size_t world) const { return worlds_.at(world); }
    const BlockWorldView* data() const { return worlds_.data(); }
    std::size_t size() const { return worlds_.size(); }

    void reset(const std::uint8_t* mask) const {
        for (std::size_t world = 0; world < worlds_.size(); ++world) {
            if (!mask || mask[world]) worlds_[world].clearPlacedBlocks();
        }
    }

private:
    static std::size_t checkedProduct(std::size_t left, std::size_t right) {
        if (right != 0 && left > (std::numeric_limits<std::size_t>::max)() / right) {
            throw std::invalid_argument("Terrain allocation size overflow");
        }
        return left * right;
    }

    std::vector<PlacedBlock> overlays_;
    std::vector<BlockWorldView> worlds_;
};

}
