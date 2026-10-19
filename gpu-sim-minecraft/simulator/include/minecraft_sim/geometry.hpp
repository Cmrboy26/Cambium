#pragma once

#include "minecraft_sim/platform.hpp"

#include <cmath>
#include <cstdint>

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

struct BlockWorldView {
    const Block* cells = nullptr;
    int size_x = 0;
    int size_y = 0;
    int size_z = 0;
    int origin_x = 0;
    int origin_y = 0;
    int origin_z = 0;

    MCSIM_HD bool solid(int block_x, int block_y, int block_z) const {
        const int local_x = block_x - origin_x;
        const int local_y = block_y - origin_y;
        const int local_z = block_z - origin_z;
        if (!cells || local_x < 0 || local_y < 0 || local_z < 0
            || local_x >= size_x || local_y >= size_y || local_z >= size_z) {
            return false;
        }
        const auto index = (static_cast<std::int64_t>(local_z) * size_y + local_y)
                         * size_x + local_x;
        return cells[index] == Block::solid;
    }
};

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