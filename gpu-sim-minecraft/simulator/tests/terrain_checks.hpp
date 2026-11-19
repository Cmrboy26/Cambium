#pragma once

#include "minecraft_sim/geometry.hpp"

namespace terrain_checks {

MCSIM_HD inline bool blockTargeting() {
    using namespace minecraft_sim;
    Block cells[27]{};
    PlacedBlock overlay[27]{};
    BlockWorldView blocks{cells, 3, 3, 3, -1, 0, -1, overlay};
    cells[blocks.cellIndex(0, 1, 0)] = Block::solid;
    auto result = targetFullCube(blocks, {-0.5, 1.5, 0.5}, {2.0, 0.0, 0.0});
    if (!result.hit || result.inside || result.x != 0 || result.y != 1 || result.z != 0
        || result.face != BlockFace::west || result.fraction != 0.25) return false;
    result = targetFullCube(blocks, {1.5, 1.5, 0.5}, {-2.0, 0.0, 0.0});
    if (!result.hit || result.face != BlockFace::east || result.fraction != 0.25) return false;
    result = targetFullCube(blocks, {0.5, 0.5, 0.5}, {0.0, 2.0, 0.0});
    if (!result.hit || result.face != BlockFace::down || result.fraction != 0.25) return false;
    result = targetFullCube(blocks, {0.5, 1.5, -0.5}, {0.0, 0.0, 2.0});
    if (!result.hit || result.face != BlockFace::north || result.fraction != 0.25) return false;
    result = targetFullCube(blocks, {0.5, 1.5, 0.5}, {0.0, 0.0, 2.0});
    if (!result.hit || !result.inside || result.face != BlockFace::north || result.fraction != 0.001) return false;
    result = targetFullCube(blocks, {-0.5, 0.5, 0.5}, {2.0, 2.0, 0.0});
    if (!result.hit || result.face != BlockFace::west || result.fraction != 0.25) return false;
    if (targetFullCube(blocks, {-0.5, 1.5, 0.5}, {0.5, 0.0, 0.0}).hit
        || targetFullCube(blocks, {-0.5, 1.5, 0.5}, {0.0, 0.0, 0.0}).hit) return false;
    if (!blocks.setPlacedBlock(-1, 1, 0, PlacedBlock::team0)) return false;
    result = targetFullCube(blocks, {-1.5, 1.5, 0.5}, {3.0, 0.0, 0.0});
    if (!result.hit || result.x != -1 || result.face != BlockFace::west
        || absolute(result.fraction - 1.0 / 6.0) > 1.0E-12) return false;

    Block corner_cells[8]{};
    BlockWorldView corner{corner_cells, 2, 2, 2, 0, 0, 0};
    corner_cells[corner.cellIndex(0, 0, 1)] = Block::solid;
    corner_cells[corner.cellIndex(0, 1, 0)] = Block::solid;
    result = targetFullCube(corner, {0.5, 0.5, 0.5}, {1.0, 1.0, 1.0});
    return result.hit && result.x == 0 && result.y == 0 && result.z == 1
        && result.face == BlockFace::north && result.fraction == 0.5;
}

MCSIM_HD inline bool orderedEditsAndResets(minecraft_sim::BlockWorldView blocks) {
    using namespace minecraft_sim;
    const auto first = blocks.cellIndex(0, 1, 0);
    const auto second = blocks.cellIndex(1, 1, 0);
    if (!blocks.overlay || first < 0 || second < 0) return false;

    TerrainEdit edits[2]{{TerrainEditAction::place, 0, 1, 0}, {TerrainEditAction::place, 0, 1, 0}};
    auto result = applyTerrainEdits(blocks, edits);
    if (!result.applied[0] || result.applied[1] || blocks.overlay[first] != PlacedBlock::team0) return false;

    edits[0] = {};
    edits[1] = {TerrainEditAction::place, 1, 1, 0};
    result = applyTerrainEdits(blocks, edits);
    if (result.applied[0] || !result.applied[1] || blocks.overlay[second] != PlacedBlock::team1) return false;

    edits[0] = {TerrainEditAction::remove, 1, 1, 0};
    edits[1] = {TerrainEditAction::remove, 1, 1, 0};
    result = applyTerrainEdits(blocks, edits);
    if (!result.applied[0] || result.applied[1] || blocks.overlay[second] != PlacedBlock::empty) return false;

    edits[0] = {TerrainEditAction::remove, 0, 1, 0};
    edits[1] = {TerrainEditAction::place, 0, 1, 0};
    result = applyTerrainEdits(blocks, edits);
    if (!result.applied[0] || !result.applied[1] || blocks.overlay[first] != PlacedBlock::team1) return false;

    edits[0] = {TerrainEditAction::remove, 0, 1, 0};
    edits[1] = {TerrainEditAction::remove, 0, 1, 0};
    result = applyTerrainEdits(blocks, edits);
    if (!result.applied[0] || result.applied[1] || blocks.overlay[first] != PlacedBlock::empty) return false;

    edits[0] = {TerrainEditAction::place, 0, 1, 0};
    edits[1] = {TerrainEditAction::remove, 0, 1, 0};
    result = applyTerrainEdits(blocks, edits);
    if (!result.applied[0] || !result.applied[1] || blocks.overlay[first] != PlacedBlock::empty) return false;

    edits[0] = {TerrainEditAction::place, 0, 0, 0};
    edits[1] = {TerrainEditAction::place, 2, 1, 0};
    result = applyTerrainEdits(blocks, edits);
    if (result.applied[0] || result.applied[1] || !blocks.solid(0, 0, 0)) return false;

    edits[0] = {TerrainEditAction::remove, 0, 0, 0};
    edits[1] = {TerrainEditAction::place, -3, 1, 0};
    result = applyTerrainEdits(blocks, edits);
    if (result.applied[0] || result.applied[1] || blocks.solid(-3, 1, 0)) return false;

    auto static_blocks = blocks;
    static_blocks.overlay = nullptr;
    edits[0] = {TerrainEditAction::place, 0, 1, 0};
    edits[1] = {static_cast<TerrainEditAction>(255), 1, 1, 0};
    result = applyTerrainEdits(static_blocks, edits);
    if (result.applied[0] || result.applied[1]) return false;
    result = applyTerrainEdits(blocks, edits);
    if (!result.applied[0] || result.applied[1] || blocks.overlay[second] != PlacedBlock::empty) return false;

    blocks.resetPlacedBlocks(TerrainReset::player_respawn, false);
    if (blocks.overlay[first] != PlacedBlock::team0) return false;
    blocks.resetPlacedBlocks(TerrainReset::round);
    if (blocks.overlay[first] != PlacedBlock::team0) return false;
    blocks.resetPlacedBlocks(TerrainReset::round, false);
    if (blocks.overlay[first] != PlacedBlock::empty || !blocks.solid(0, 0, 0)) return false;
    if (!blocks.setPlacedBlock(1, 1, 0, PlacedBlock::team1)) return false;
    blocks.resetPlacedBlocks(TerrainReset::match);
    return blocks.overlay[second] == PlacedBlock::empty && blocks.solid(0, 0, 0);
}

}