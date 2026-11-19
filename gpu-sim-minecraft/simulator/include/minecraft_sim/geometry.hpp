#pragma once

#include "minecraft_sim/platform.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

namespace minecraft_sim {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    MCSIM_HD double lengthSquared() const { return x * x + y * y + z * z; }
};

MCSIM_HD inline double absolute(double value) { return value < 0.0 ? -value : value; }
MCSIM_HD inline double minimum(double left, double right) { return left < right ? left : right; }
MCSIM_HD inline double maximum(double left, double right) { return left > right ? left : right; }

struct Aabb {
    Vec3 lower;
    Vec3 upper;

    MCSIM_HD Aabb moved(Vec3 delta) const {
        return {{lower.x + delta.x, lower.y + delta.y, lower.z + delta.z},
                {upper.x + delta.x, upper.y + delta.y, upper.z + delta.z}};
    }

    MCSIM_HD Aabb swept(Vec3 delta) const {
        return {{lower.x + minimum(delta.x, 0.0), lower.y + minimum(delta.y, 0.0),
                 lower.z + minimum(delta.z, 0.0)},
                {upper.x + maximum(delta.x, 0.0), upper.y + maximum(delta.y, 0.0),
                 upper.z + maximum(delta.z, 0.0)}};
    }

    MCSIM_HD bool intersects(Aabb other) const {
        return upper.x > other.lower.x && lower.x < other.upper.x
            && upper.y > other.lower.y && lower.y < other.upper.y
            && upper.z > other.lower.z && lower.z < other.upper.z;
    }
};

enum class Block : std::uint8_t { air, solid };
enum class PlacedBlock : std::uint8_t { empty, team0, team1 };
enum class TerrainReset : std::uint8_t { player_respawn, round, match };
enum class TerrainEditAction : std::uint8_t { none, place, remove };

struct TerrainEdit {
    TerrainEditAction action = TerrainEditAction::none;
    int x = 0;
    int y = 0;
    int z = 0;
};

struct TerrainEditResults {
    bool applied[2]{};
};

struct BlockWorldView {
    const Block* cells = nullptr;
    int size_x = 0;
    int size_y = 0;
    int size_z = 0;
    int origin_x = 0;
    int origin_y = 0;
    int origin_z = 0;
    PlacedBlock* overlay = nullptr;

    MCSIM_HD std::int64_t cellIndex(int block_x, int block_y, int block_z) const {
        const auto local_x = std::int64_t(block_x) - origin_x;
        const auto local_y = std::int64_t(block_y) - origin_y;
        const auto local_z = std::int64_t(block_z) - origin_z;
        if (!cells || local_x < 0 || local_y < 0 || local_z < 0
            || local_x >= size_x || local_y >= size_y || local_z >= size_z) {
            return -1;
        }
        return (local_z * size_y + local_y) * size_x + local_x;
    }

    MCSIM_HD bool solid(int block_x, int block_y, int block_z) const {
        const auto index = cellIndex(block_x, block_y, block_z);
        return index >= 0 && (cells[index] == Block::solid
            || (overlay && overlay[index] != PlacedBlock::empty));
    }

    MCSIM_HD bool setPlacedBlock(int block_x, int block_y, int block_z, PlacedBlock value) const {
        const auto index = cellIndex(block_x, block_y, block_z);
        if (index < 0 || !overlay || cells[index] != Block::air
            || (value != PlacedBlock::empty && value != PlacedBlock::team0 && value != PlacedBlock::team1)) {
            return false;
        }
        overlay[index] = value;
        return true;
    }

    MCSIM_HD void clearPlacedBlocks() const {
        if (!overlay) return;
        const auto count = std::int64_t(size_x) * size_y * size_z;
        for (std::int64_t index = 0; index < count; ++index) overlay[index] = PlacedBlock::empty;
    }

    MCSIM_HD void resetPlacedBlocks(TerrainReset reason, bool persist_on_goal = true) const {
        if (reason == TerrainReset::match || (reason == TerrainReset::round && !persist_on_goal)) {
            clearPlacedBlocks();
        }
    }
};

MCSIM_HD inline TerrainEditResults applyTerrainEdits(BlockWorldView blocks, const TerrainEdit (&edits)[2]) {
    TerrainEditResults results{};
    for (int player = 0; player < 2; ++player) {
        const auto& edit = edits[player];
        const auto index = blocks.cellIndex(edit.x, edit.y, edit.z);
        if (index < 0 || !blocks.overlay || blocks.cells[index] != Block::air) continue;
        if (edit.action == TerrainEditAction::place && blocks.overlay[index] == PlacedBlock::empty) {
            blocks.overlay[index] = player == 0 ? PlacedBlock::team0 : PlacedBlock::team1;
            results.applied[player] = true;
        } else if (edit.action == TerrainEditAction::remove && blocks.overlay[index] != PlacedBlock::empty) {
            blocks.overlay[index] = PlacedBlock::empty;
            results.applied[player] = true;
        }
    }
    return results;
}

MCSIM_HD inline double component(Vec3 vector, int axis) {
    return axis == 0 ? vector.x : (axis == 1 ? vector.y : vector.z);
}

MCSIM_HD inline Vec3 axisMovement(int axis, double distance) {
    return {axis == 0 ? distance : 0.0, axis == 1 ? distance : 0.0,
            axis == 2 ? distance : 0.0};
}

MCSIM_HD inline bool rayBoxHit(Aabb box, Vec3 origin, Vec3 delta, double& fraction) {
    if (origin.x >= box.lower.x && origin.x < box.upper.x
        && origin.y >= box.lower.y && origin.y < box.upper.y
        && origin.z >= box.lower.z && origin.z < box.upper.z) {
        fraction = 0.0;
        return true;
    }
    fraction = 1.0;
    bool hit = false;
    for (int axis = 0; axis < 3; ++axis) {
        const double direction = component(delta, axis);
        if (absolute(direction) <= 1.0E-7) {
            continue;
        }
        const double plane = component(direction > 0.0 ? box.lower : box.upper, axis);
        const double candidate = (plane - component(origin, axis)) / direction;
        if (candidate <= 0.0 || candidate >= fraction) {
            continue;
        }
        bool inside = true;
        for (int offset = 1; offset <= 2; ++offset) {
            const int transverse = (axis + offset) % 3;
            const double coordinate = component(origin, transverse) + candidate * component(delta, transverse);
            inside = inside && coordinate > component(box.lower, transverse) - 1.0E-7
                && coordinate < component(box.upper, transverse) + 1.0E-7;
        }
        if (inside) {
            fraction = candidate;
            hit = true;
        }
    }
    return hit;
}

MCSIM_HD inline double clipCube(Aabb moving, Aabb obstacle, int axis, double distance) {
    constexpr double epsilon = 1.0E-7;
    if (absolute(distance) < epsilon) {
        return 0.0;
    }
    for (int offset = 1; offset <= 2; ++offset) {
        const int transverse = (axis + offset) % 3;
        if (component(moving.upper, transverse) <= component(obstacle.lower, transverse) + epsilon
            || component(moving.lower, transverse) >= component(obstacle.upper, transverse) - epsilon) {
            return distance;
        }
    }
    if (distance > 0.0) {
        const double gap = component(obstacle.lower, axis) - component(moving.upper, axis);
        if (gap >= -epsilon) {
            return minimum(distance, gap);
        }
    } else {
        const double gap = component(obstacle.upper, axis) - component(moving.lower, axis);
        if (gap <= epsilon) {
            return maximum(distance, gap);
        }
    }
    return distance;
}

MCSIM_HD inline int boundedCell(double coordinate, int lower, int upper) {
    return static_cast<int>(maximum(lower, minimum(upper, coordinate)));
}

enum class BlockFace : std::uint8_t { down, up, north, south, west, east };

struct BlockTarget {
    bool hit = false;
    bool inside = false;
    int x = 0;
    int y = 0;
    int z = 0;
    BlockFace face = BlockFace::north;
    double fraction = 1.0;
    Vec3 position;
};

MCSIM_HD inline BlockFace oppositeBlockFace(BlockFace face) {
    switch (face) {
    case BlockFace::down: return BlockFace::up;
    case BlockFace::up: return BlockFace::down;
    case BlockFace::north: return BlockFace::south;
    case BlockFace::south: return BlockFace::north;
    case BlockFace::west: return BlockFace::east;
    case BlockFace::east: return BlockFace::west;
    }
    return BlockFace::north;
}

MCSIM_HD inline BlockFace nearestBlockFace(Vec3 delta) {
    const float values[6]{-static_cast<float>(delta.y), static_cast<float>(delta.y),
                          -static_cast<float>(delta.z), static_cast<float>(delta.z),
                          -static_cast<float>(delta.x), static_cast<float>(delta.x)};
    float best = (std::numeric_limits<float>::denorm_min)();
    BlockFace face = BlockFace::north;
    for (int index = 0; index < 6; ++index) {
        if (values[index] > best) {
            best = values[index];
            face = static_cast<BlockFace>(index);
        }
    }
    return face;
}

// Minecraft 26.3: BlockGetter.traverseBlocks (112-175), VoxelShape.clip (159-178),
// AABB.getDirection/clipPoint (339-399), Direction.getApproximateNearest (319-335).
MCSIM_HD inline BlockTarget targetFullCube(BlockWorldView blocks, Vec3 origin, Vec3 delta) {
    BlockTarget miss;
    miss.position = {origin.x + delta.x, origin.y + delta.y, origin.z + delta.z};
    if (!blocks.cells || delta.lengthSquared() < 1.0E-7) return miss;
    const Vec3 from{origin.x - delta.x * 1.0E-7, origin.y - delta.y * 1.0E-7, origin.z - delta.z * 1.0E-7};
    const Vec3 to{miss.position.x + delta.x * 1.0E-7, miss.position.y + delta.y * 1.0E-7,
                  miss.position.z + delta.z * 1.0E-7};
    constexpr double integer_lower = double((std::numeric_limits<int>::min)()) + 1.0;
    constexpr double integer_upper = double((std::numeric_limits<int>::max)()) - 1.0;
    for (int axis = 0; axis < 3; ++axis) {
        if (!(component(from, axis) >= integer_lower && component(from, axis) <= integer_upper
              && component(to, axis) >= integer_lower && component(to, axis) <= integer_upper)) return miss;
    }
    int cell[3]{static_cast<int>(::floor(from.x)), static_cast<int>(::floor(from.y)), static_cast<int>(::floor(from.z))};
    const double traversed[3]{to.x - from.x, to.y - from.y, to.z - from.z};
    int step[3]{};
    double increment[3]{}, next[3]{};
    for (int axis = 0; axis < 3; ++axis) {
        step[axis] = traversed[axis] > 0.0 ? 1 : traversed[axis] < 0.0 ? -1 : 0;
        increment[axis] = step[axis] == 0 ? (std::numeric_limits<double>::max)()
            : double(step[axis]) / traversed[axis];
        const double fraction = component(from, axis) - ::floor(component(from, axis));
        next[axis] = increment[axis] * (step[axis] > 0 ? 1.0 - fraction : fraction);
    }
    for (;;) {
        if (blocks.solid(cell[0], cell[1], cell[2])) {
            const Vec3 test{origin.x + delta.x * 0.001, origin.y + delta.y * 0.001, origin.z + delta.z * 0.001};
            BlockTarget result;
            result.x = cell[0]; result.y = cell[1]; result.z = cell[2];
            if (test.x >= cell[0] && test.x < double(cell[0]) + 1.0
                && test.y >= cell[1] && test.y < double(cell[1]) + 1.0
                && test.z >= cell[2] && test.z < double(cell[2]) + 1.0) {
                result.hit = true;
                result.inside = true;
                result.fraction = 0.001;
                result.face = oppositeBlockFace(nearestBlockFace(delta));
                result.position = test;
                return result;
            }
            double nearest = 1.0;
            for (int axis = 0; axis < 3; ++axis) {
                const double direction = component(delta, axis);
                if (absolute(direction) <= 1.0E-7) continue;
                const double plane = double(cell[axis]) + (direction < 0.0 ? 1.0 : 0.0);
                const double candidate = (plane - component(origin, axis)) / direction;
                if (candidate <= 0.0 || candidate >= nearest) continue;
                bool within = true;
                for (int offset = 1; offset <= 2; ++offset) {
                    const int transverse = (axis + offset) % 3;
                    const double coordinate = component(origin, transverse) + candidate * component(delta, transverse);
                    within = within && coordinate > double(cell[transverse]) - 1.0E-7
                        && coordinate < double(cell[transverse]) + 1.0 + 1.0E-7;
                }
                if (!within) continue;
                nearest = candidate;
                result.hit = true;
                result.face = axis == 0 ? (direction > 0.0 ? BlockFace::west : BlockFace::east)
                    : axis == 1 ? (direction > 0.0 ? BlockFace::down : BlockFace::up)
                    : (direction > 0.0 ? BlockFace::north : BlockFace::south);
            }
            if (result.hit) {
                result.fraction = nearest;
                result.position = {origin.x + nearest * delta.x, origin.y + nearest * delta.y,
                                   origin.z + nearest * delta.z};
                return result;
            }
        }
        if (!(next[0] <= 1.0 || next[1] <= 1.0 || next[2] <= 1.0)) return miss;
        const int axis = next[0] < next[1] ? (next[0] < next[2] ? 0 : 2)
            : (next[1] < next[2] ? 1 : 2);
        cell[axis] += step[axis];
        next[axis] += increment[axis];
    }
}

MCSIM_HD inline double blockRayFraction(BlockWorldView blocks, Vec3 origin, Vec3 delta) {
    double nearest = 1.0;
    const Aabb sweep = Aabb{origin, origin}.swept(delta);
    const int lower_x = boundedCell(::floor(sweep.lower.x), blocks.origin_x, blocks.origin_x + blocks.size_x);
    const int lower_y = boundedCell(::floor(sweep.lower.y), blocks.origin_y, blocks.origin_y + blocks.size_y);
    const int lower_z = boundedCell(::floor(sweep.lower.z), blocks.origin_z, blocks.origin_z + blocks.size_z);
    const int upper_x = boundedCell(::floor(sweep.upper.x) + 1.0, blocks.origin_x, blocks.origin_x + blocks.size_x);
    const int upper_y = boundedCell(::floor(sweep.upper.y) + 1.0, blocks.origin_y, blocks.origin_y + blocks.size_y);
    const int upper_z = boundedCell(::floor(sweep.upper.z) + 1.0, blocks.origin_z, blocks.origin_z + blocks.size_z);
    for (int block_z = lower_z; block_z < upper_z; ++block_z) {
        for (int block_y = lower_y; block_y < upper_y; ++block_y) {
            for (int block_x = lower_x; block_x < upper_x; ++block_x) {
                if (!blocks.solid(block_x, block_y, block_z)) {
                    continue;
                }
                double fraction = 1.0;
                const Aabb box{{double(block_x), double(block_y), double(block_z)},
                               {double(block_x) + 1.0, double(block_y) + 1.0, double(block_z) + 1.0}};
                if (rayBoxHit(box, origin, delta, fraction)) {
                    nearest = minimum(nearest, fraction);
                }
            }
        }
    }
    return nearest;
}

MCSIM_HD inline bool noBlockCollision(BlockWorldView blocks, Aabb box) {
    const int lower_x = boundedCell(::floor(box.lower.x), blocks.origin_x, blocks.origin_x + blocks.size_x);
    const int lower_y = boundedCell(::floor(box.lower.y), blocks.origin_y, blocks.origin_y + blocks.size_y);
    const int lower_z = boundedCell(::floor(box.lower.z), blocks.origin_z, blocks.origin_z + blocks.size_z);
    const int upper_x = boundedCell(::floor(box.upper.x) + 1.0, blocks.origin_x, blocks.origin_x + blocks.size_x);
    const int upper_y = boundedCell(::floor(box.upper.y) + 1.0, blocks.origin_y, blocks.origin_y + blocks.size_y);
    const int upper_z = boundedCell(::floor(box.upper.z) + 1.0, blocks.origin_z, blocks.origin_z + blocks.size_z);
    for (int block_z = lower_z; block_z < upper_z; ++block_z) {
        for (int block_y = lower_y; block_y < upper_y; ++block_y) {
            for (int block_x = lower_x; block_x < upper_x; ++block_x) {
                const Aabb obstacle{{double(block_x), double(block_y), double(block_z)},
                                    {double(block_x) + 1.0, double(block_y) + 1.0, double(block_z) + 1.0}};
                if (blocks.solid(block_x, block_y, block_z) && box.intersects(obstacle)) {
                    return false;
                }
            }
        }
    }
    return true;
}

MCSIM_HD inline double clipAxis(BlockWorldView blocks, Aabb moving, Aabb sweep,
                                int axis, double distance) {
    if (distance == 0.0 || !blocks.cells) {
        return distance;
    }
    const int lower_x = boundedCell(::floor(sweep.lower.x), blocks.origin_x, blocks.origin_x + blocks.size_x);
    const int lower_y = boundedCell(::floor(sweep.lower.y), blocks.origin_y, blocks.origin_y + blocks.size_y);
    const int lower_z = boundedCell(::floor(sweep.lower.z), blocks.origin_z, blocks.origin_z + blocks.size_z);
    const int upper_x = boundedCell(::floor(sweep.upper.x) + 1.0, blocks.origin_x, blocks.origin_x + blocks.size_x);
    const int upper_y = boundedCell(::floor(sweep.upper.y) + 1.0, blocks.origin_y, blocks.origin_y + blocks.size_y);
    const int upper_z = boundedCell(::floor(sweep.upper.z) + 1.0, blocks.origin_z, blocks.origin_z + blocks.size_z);
    for (int block_z = lower_z; block_z < upper_z; ++block_z) {
        for (int block_y = lower_y; block_y < upper_y; ++block_y) {
            for (int block_x = lower_x; block_x < upper_x; ++block_x) {
                const Aabb obstacle{{double(block_x), double(block_y), double(block_z)},
                                    {double(block_x) + 1.0, double(block_y) + 1.0, double(block_z) + 1.0}};
                if (blocks.solid(block_x, block_y, block_z) && sweep.intersects(obstacle)) {
                    distance = clipCube(moving, obstacle, axis, distance);
                }
            }
        }
    }
    return distance;
}

MCSIM_HD inline Vec3 collide(BlockWorldView blocks, Aabb moving, Vec3 requested) {
    const Aabb sweep = moving.swept(requested);
    Vec3 accepted = requested;
    accepted.y = clipAxis(blocks, moving, sweep, 1, requested.y);
    moving = moving.moved({0.0, accepted.y, 0.0});
    if (absolute(requested.x) < absolute(requested.z)) {
        accepted.z = clipAxis(blocks, moving, sweep, 2, requested.z);
        moving = moving.moved({0.0, 0.0, accepted.z});
        accepted.x = clipAxis(blocks, moving, sweep, 0, requested.x);
    } else {
        accepted.x = clipAxis(blocks, moving, sweep, 0, requested.x);
        moving = moving.moved({accepted.x, 0.0, 0.0});
        accepted.z = clipAxis(blocks, moving, sweep, 2, requested.z);
    }
    return accepted;
}

}