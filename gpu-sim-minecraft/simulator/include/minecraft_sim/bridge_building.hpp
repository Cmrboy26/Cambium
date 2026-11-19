#pragma once

#include "minecraft_sim/core.hpp"

namespace minecraft_sim {

enum class BridgeTargetKind : std::uint8_t { miss, block, entity };

struct BridgeInteractionTarget {
    BridgeTargetKind kind = BridgeTargetKind::miss;
    BlockTarget block;
    int entity = -1;
};

struct BridgeBuildRules {
    bool enabled = false;
    double block_range = 4.5;
    double entity_range = 3.0;
    Aabb zones[8]{};
    int zone_count = 0;
    float hardness = 0.8F;
    float mining_speed = 1.0F;
    bool correct_tool = true;
};

struct BridgeBuildState {
    int use_delay = 0;
    int break_delay = 0;
    bool breaking = false;
    int target_x = 0;
    int target_y = 0;
    int target_z = 0;
    float break_progress = 0.0F;
};

MCSIM_HD inline bool validBridgeBuildRules(const BridgeBuildRules& rules) {
    if (!rules.enabled) return true;
    if (!(rules.block_range > 0.0 && rules.block_range <= 64.0
          && rules.entity_range >= 0.0 && rules.entity_range <= 64.0
          && rules.hardness > 0.0F && rules.hardness <= (std::numeric_limits<float>::max)()
          && rules.mining_speed > 0.0F && rules.mining_speed <= (std::numeric_limits<float>::max)())
        || rules.zone_count < 1 || rules.zone_count > 8) return false;
    for (int index = 0; index < rules.zone_count; ++index) {
        const auto& zone = rules.zones[index];
        for (int axis = 0; axis < 3; ++axis) {
            if (!(component(zone.lower, axis) >= double((std::numeric_limits<int>::min)())
                && component(zone.upper, axis) <= double((std::numeric_limits<int>::max)())
                && component(zone.lower, axis) < component(zone.upper, axis))) return false;
        }
    }
    return true;
}

// Minecraft 26.3: LocalPlayer.pick/filterHitResult (1291-1320), ProjectileUtil.getEntityHitResult (109-145).
MCSIM_HD inline BridgeInteractionTarget bridgeInteractionTarget(WorldView world, int player,
                                                                const PlayerState& source,
                                                                double block_range, double entity_range) {
    BridgeInteractionTarget result;
    if (player < 0 || player >= world.playerCount() || !source.alive()
        || !(block_range > 0.0 && block_range <= 64.0 && entity_range >= 0.0 && entity_range <= 64.0)) return result;
    const double range = maximum(block_range, entity_range);
    const Vec3 origin = source.eyePosition();
    const Vec3 look = source.lookDirection(world.math());
    const Vec3 delta{look.x * range, look.y * range, look.z * range};
    const double length_squared = delta.lengthSquared();
    result.block = targetFullCube(world.blocks(), origin, delta);
    double nearest = result.block.hit ? result.block.fraction : 1.0;
    int entity = -1;
    for (int other = 0; other < world.playerCount(); ++other) {
        if (other == player || !world.player(other).state().alive()) continue;
        double fraction = 1.0;
        if (rayBoxHit(world.player(other).state().bounds(), origin, delta, fraction) && fraction < nearest) {
            nearest = fraction;
            entity = other;
        }
    }
    if (entity >= 0) {
        if (nearest * nearest * length_squared < entity_range * entity_range) {
            result.kind = BridgeTargetKind::entity;
            result.entity = entity;
        }
        return result;
    }
    if (result.block.hit && result.block.fraction * result.block.fraction * length_squared < block_range * block_range) {
        result.kind = BridgeTargetKind::block;
    }
    return result;
}

MCSIM_HD inline bool bridgeBuildZoneContains(const BridgeBuildRules& rules, int x, int y, int z) {
    for (int index = 0; index < rules.zone_count && index < 8; ++index) {
        const auto& zone = rules.zones[index];
        if (double(x) >= zone.lower.x && double(y) >= zone.lower.y && double(z) >= zone.lower.z
            && double(x) + 1.0 <= zone.upper.x && double(y) + 1.0 <= zone.upper.y
            && double(z) + 1.0 <= zone.upper.z) return true;
    }
    return false;
}

// Minecraft 26.3: BlockPlaceContext (25-57), BlockItem.place/canPlace (55-100, 135-138).
MCSIM_HD inline bool placeBridgeBlock(WorldView world, const BridgeInteractionTarget& target,
                                       const BridgeBuildRules& rules, int player, int& blocks_remaining,
                                       TerrainEdit& edit) {
    if (!rules.enabled || target.kind != BridgeTargetKind::block || !target.block.hit || blocks_remaining <= 0
        || player < 0 || player >= 2
        || !world.blocks().solid(target.block.x, target.block.y, target.block.z)) return false;
    std::int64_t x = target.block.x, y = target.block.y, z = target.block.z;
    switch (target.block.face) {
    case BlockFace::down: --y; break;
    case BlockFace::up: ++y; break;
    case BlockFace::north: --z; break;
    case BlockFace::south: ++z; break;
    case BlockFace::west: --x; break;
    case BlockFace::east: ++x; break;
    }
    if (x < (std::numeric_limits<int>::min)() || x > (std::numeric_limits<int>::max)()
        || y < (std::numeric_limits<int>::min)() || y > (std::numeric_limits<int>::max)()
        || z < (std::numeric_limits<int>::min)() || z > (std::numeric_limits<int>::max)()) return false;
    const int block_x = static_cast<int>(x), block_y = static_cast<int>(y), block_z = static_cast<int>(z);
    const auto blocks = world.blocks();
    const auto index = blocks.cellIndex(block_x, block_y, block_z);
    if (index < 0 || !blocks.overlay || blocks.cells[index] != Block::air
        || blocks.overlay[index] != PlacedBlock::empty
        || !bridgeBuildZoneContains(rules, block_x, block_y, block_z)) return false;
    const Aabb cube{{double(x), double(y), double(z)}, {double(x) + 1.0, double(y) + 1.0, double(z) + 1.0}};
    for (int other = 0; other < world.playerCount(); ++other) {
        if (cube.intersects(world.player(other).state().bounds())) return false;
    }
    if (!blocks.setPlacedBlock(block_x, block_y, block_z, player == 0 ? PlacedBlock::team0 : PlacedBlock::team1)) return false;
    edit = {TerrainEditAction::place, block_x, block_y, block_z};
    --blocks_remaining;
    return true;
}

MCSIM_HD inline void cancelBridgeBreaking(BridgeBuildState& state) {
    state.breaking = false;
    state.break_progress = 0.0F;
}

// Minecraft 26.3: MultiPlayerGameMode.startDestroyBlock/continueDestroyBlock (153-285),
// BlockBehaviour.getDestroyProgress (344-351), Player.getDestroySpeed (583-608).
MCSIM_HD inline bool breakBridgeBlock(WorldView world, const BridgeInteractionTarget& target,
                                       const BridgeBuildRules& rules, BridgeBuildState& state, int player) {
    if (!rules.enabled || target.kind != BridgeTargetKind::block || !target.block.hit
        || player < 0 || player >= world.playerCount()) {
        cancelBridgeBreaking(state);
        return false;
    }
    if (state.break_delay > 0) {
        --state.break_delay;
        return false;
    }
    const auto blocks = world.blocks();
    const auto index = blocks.cellIndex(target.block.x, target.block.y, target.block.z);
    if (index < 0 || !blocks.overlay || blocks.cells[index] != Block::air
        || blocks.overlay[index] == PlacedBlock::empty
        || !bridgeBuildZoneContains(rules, target.block.x, target.block.y, target.block.z)
        || !(rules.hardness > 0.0F && rules.mining_speed > 0.0F)) {
        cancelBridgeBreaking(state);
        return false;
    }
    float speed = rules.mining_speed;
    if (!world.player(player).state().on_ground) speed /= 5.0F;
    const float increment = speed / rules.hardness / (rules.correct_tool ? 30.0F : 100.0F);
    const bool same = state.breaking && state.target_x == target.block.x
        && state.target_y == target.block.y && state.target_z == target.block.z;
    if (!same) {
        state.breaking = true;
        state.target_x = target.block.x;
        state.target_y = target.block.y;
        state.target_z = target.block.z;
        state.break_progress = 0.0F;
        if (increment < 1.0F) return false;
    } else {
        state.break_progress += increment;
        if (state.break_progress < 1.0F) return false;
    }
    if (!blocks.setPlacedBlock(target.block.x, target.block.y, target.block.z, PlacedBlock::empty)) {
        cancelBridgeBreaking(state);
        return false;
    }
    cancelBridgeBreaking(state);
    state.break_delay = 5;
    return true;
}

}