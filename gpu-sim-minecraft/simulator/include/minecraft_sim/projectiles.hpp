#pragma once

#include "minecraft_sim/core.hpp"

#include <cstdint>
#include <limits>

namespace minecraft_sim {

inline constexpr int projectile_player_count = 2;
inline constexpr int arrow_lifetime_ticks = 1200;
inline constexpr int projectile_arrow_capacity = 128;

enum class BowCommand : std::uint8_t { idle, hold, release, cancel };
enum class BowActionStatus : std::uint8_t {
    idle,
    drawing,
    cancelled,
    no_ammunition,
    not_equipped,
    shield_conflict,
    not_using,
    too_weak,
    capacity_full,
    invalid_storage,
    fired
};

struct BowState {
    int ordinary_arrows = 0;
    int draw_ticks = 0;
    bool using_bow = false;
};

struct ArrowState {
    Vec3 position;
    Vec3 velocity;
    LegacyRandom random;
    int owner = -1;
    int age = 0;
    int in_ground_ticks = 0;
    double launch_target_distance = 0.0;
    bool active = false;
    bool left_owner = false;
    bool in_ground = false;
    bool critical = false;
    bool progress_rewarded = false;
};

struct WindChargeState {
    Vec3 position;
    Vec3 velocity;
    int owner = -1;
    int age = 0;
    int no_deflect_ticks = WIND_CHARGE_NO_DEFLECT_TICKS;
    bool active = false;
    bool left_owner = false;
};

enum class ThrownItem : std::uint8_t { snowball, egg, ender_pearl };

struct ThrownProjectileState {
    Vec3 position;
    Vec3 velocity;
    ThrownItem item = ThrownItem::snowball;
    int owner = -1;
    int age = 0;
    bool active = false;
    bool left_owner = false;
};

struct ProjectileWorldState {
    BowState bows[projectile_player_count]{};
    ArrowState* arrows = nullptr;
    int arrow_capacity = 0;
    WindChargeState* wind_charges = nullptr;
    int wind_charge_capacity = 0;
    ThrownProjectileState* thrown_items = nullptr;
    int thrown_item_capacity = 0;
    LegacyRandom random;
};

struct ArrowObservationSummary {
    int ordinary_arrows = 0;
    int draw_ticks = 0;
    bool found = false;
    Vec3 relative_position;
    Vec3 relative_velocity;
};

MCSIM_HD inline ArrowObservationSummary summarizeArrowObservation(
        const PlayerState& source, const ProjectileWorldState& projectiles, int source_player) {
    ArrowObservationSummary summary;
    const auto& bow = projectiles.bows[source_player];
    summary.ordinary_arrows = bow.ordinary_arrows;
    summary.draw_ticks = bow.draw_ticks;

    int nearest_slot = -1;
    double nearest_distance = 2500.0;
    for (int slot = 0; slot < projectiles.arrow_capacity; ++slot) {
        const auto& arrow = projectiles.arrows[slot];
        if (!arrow.active || arrow.in_ground || arrow.owner != 1 - source_player
            || arrow.velocity.lengthSquared() == 0.0) {
            continue;
        }
        const Vec3 relative{arrow.position.x - source.position.x, arrow.position.y - source.position.y,
                            arrow.position.z - source.position.z};
        const double distance = relative.lengthSquared();
        if (distance <= 2500.0 && (nearest_slot < 0 || distance < nearest_distance)) {
            nearest_slot = slot;
            nearest_distance = distance;
        }
    }
    if (nearest_slot >= 0) {
        const auto& arrow = projectiles.arrows[nearest_slot];
        summary.found = true;
        summary.relative_position = {arrow.position.x - source.position.x,
                                     arrow.position.y - source.position.y,
                                     arrow.position.z - source.position.z};
        summary.relative_velocity = {arrow.velocity.x - source.velocity.x,
                                     arrow.velocity.y - source.velocity.y,
                                     arrow.velocity.z - source.velocity.z};
    }
    return summary;
}

struct BowActionResult {
    BowActionStatus status = BowActionStatus::idle;
    int arrow_slot = -1;
    float power = 0.0F;
};

struct ProjectileTickResult {
    int player_hits = 0;
    int deflections = 0;
    int terrain_hits = 0;
    int expired = 0;
    int capacity_failures = 0;
    float health_dealt[projectile_player_count]{};
    float health_taken[projectile_player_count]{};
    float arrow_progress[projectile_player_count]{};
};

struct ProjectileBlockHit {
    double fraction = 1.0;
    Vec3 normal;
    bool hit = false;
};

// Minecraft 26.3: VoxelShape.java:153-171; AABB.java:362-383; Direction.java:331-345.
MCSIM_HD inline ProjectileBlockHit fullCubeProjectileHit(BlockWorldView blocks, Vec3 origin, Vec3 movement) {
    ProjectileBlockHit nearest;
    if (movement.lengthSquared() < 1.0E-7) return nearest;
    const Aabb sweep = Aabb{origin, origin}.swept(movement);
    const int lower_x = boundedCell(::floor(sweep.lower.x), blocks.origin_x, blocks.origin_x + blocks.size_x);
    const int lower_y = boundedCell(::floor(sweep.lower.y), blocks.origin_y, blocks.origin_y + blocks.size_y);
    const int lower_z = boundedCell(::floor(sweep.lower.z), blocks.origin_z, blocks.origin_z + blocks.size_z);
    const int upper_x = boundedCell(::floor(sweep.upper.x) + 1.0, blocks.origin_x, blocks.origin_x + blocks.size_x);
    const int upper_y = boundedCell(::floor(sweep.upper.y) + 1.0, blocks.origin_y, blocks.origin_y + blocks.size_y);
    const int upper_z = boundedCell(::floor(sweep.upper.z) + 1.0, blocks.origin_z, blocks.origin_z + blocks.size_z);
    const Vec3 test_point{origin.x + movement.x * 0.001, origin.y + movement.y * 0.001,
                          origin.z + movement.z * 0.001};
    for (int z = lower_z; z < upper_z; ++z) {
        for (int y = lower_y; y < upper_y; ++y) {
            for (int x = lower_x; x < upper_x; ++x) {
                if (!blocks.solid(x, y, z)) continue;
                const Aabb box{{double(x), double(y), double(z)}, {double(x + 1), double(y + 1), double(z + 1)}};
                if (test_point.x >= box.lower.x && test_point.x < box.upper.x
                    && test_point.y >= box.lower.y && test_point.y < box.upper.y
                    && test_point.z >= box.lower.z && test_point.z < box.upper.z) {
                    float largest = 0.0F;
                    Vec3 normal{};
                    for (int order = 0; order < 3; ++order) {
                        const int axis = order == 0 ? 1 : order == 1 ? 2 : 0;
                        const float direction = static_cast<float>(component(movement, axis));
                        const float magnitude = direction < 0.0F ? -direction : direction;
                        if (magnitude > largest) {
                            largest = magnitude;
                            normal = axisMovement(axis, direction > 0.0F ? -1.0 : 1.0);
                        }
                    }
                    if (0.001 < nearest.fraction) nearest = {0.001, normal, true};
                    continue;
                }
                for (int axis = 0; axis < 3; ++axis) {
                    const double direction = component(movement, axis);
                    if (absolute(direction) <= 1.0E-7) continue;
                    const double plane = component(direction > 0.0 ? box.lower : box.upper, axis);
                    const double fraction = (plane - component(origin, axis)) / direction;
                    if (fraction <= 0.0 || fraction >= nearest.fraction) continue;
                    bool within = true;
                    for (int offset = 1; offset <= 2; ++offset) {
                        const int transverse = (axis + offset) % 3;
                        const double coordinate = component(origin, transverse) + fraction * component(movement, transverse);
                        within = within && coordinate > component(box.lower, transverse) - 1.0E-7
                            && coordinate < component(box.upper, transverse) + 1.0E-7;
                    }
                    if (within) nearest = {fraction, axisMovement(axis, direction > 0.0 ? -1.0 : 1.0), true};
                }
            }
        }
    }
    return nearest;
}

// Minecraft 26.3: ServerExplosion.java:72-100, getSeenPercent.
MCSIM_HD inline float explosionExposure(BlockWorldView blocks, Vec3 center, Aabb bounds) {
    const double step_x = 1.0 / ((bounds.upper.x - bounds.lower.x) * 2.0 + 1.0);
    const double step_y = 1.0 / ((bounds.upper.y - bounds.lower.y) * 2.0 + 1.0);
    const double step_z = 1.0 / ((bounds.upper.z - bounds.lower.z) * 2.0 + 1.0);
    const double offset_x = (1.0 - ::floor(1.0 / step_x) * step_x) / 2.0;
    const double offset_z = (1.0 - ::floor(1.0 / step_z) * step_z) / 2.0;
    int visible = 0;
    int total = 0;
    for (double x = 0.0; x <= 1.0; x += step_x) {
        for (double y = 0.0; y <= 1.0; y += step_y) {
            for (double z = 0.0; z <= 1.0; z += step_z) {
                const Vec3 sample{bounds.lower.x + (bounds.upper.x - bounds.lower.x) * x + offset_x,
                                  bounds.lower.y + (bounds.upper.y - bounds.lower.y) * y,
                                  bounds.lower.z + (bounds.upper.z - bounds.lower.z) * z + offset_z};
                const Vec3 ray{center.x - sample.x, center.y - sample.y, center.z - sample.z};
                visible += blockRayFraction(blocks, sample, ray) == 1.0 ? 1 : 0;
                ++total;
            }
        }
    }
    return static_cast<float>(visible) / static_cast<float>(total);
}

// Minecraft 26.3: WindCharge.java:26-30, 68-69; ServerExplosion.java:170-208.
MCSIM_HD inline void applyWindExplosion(WorldView world, BlockWorldView blocks, Vec3 center, int owner = -1) {
    const double radius = double(WIND_CHARGE_EXPLOSION_POWER * 2.0F);
    for (int index = 0; index < world.playerCount(); ++index) {
        auto& player = world.player(index).state();
        const Vec3 from_center{player.position.x - center.x, player.position.y - center.y,
                               player.position.z - center.z};
        const double distance = ::sqrt(from_center.lengthSquared());
        if (distance > radius) continue;
        const Vec3 eye = player.eyePosition();
        const Vec3 direction{eye.x - center.x, eye.y - center.y, eye.z - center.z};
        const double length = ::sqrt(direction.lengthSquared());
        const double scale = length < double(1.0E-5F) ? 0.0
            : (1.0 - distance / radius) * double(explosionExposure(blocks, center, player.bounds()))
                * double(1.22F) / length;
        player.velocity.x += direction.x * scale;
        player.velocity.y += direction.y * scale;
        player.velocity.z += direction.z * scale;
        if (index == owner && index < projectile_player_count && direction.y * scale > 0.0) {
            world.mutableState().wind_self_launch[index] += direction.y * scale;
        }
        player.needs_sync = true;
    }
}

// Minecraft 26.3: AbstractHurtingProjectile.java:68-103; AbstractWindCharge.java:118-136.
MCSIM_HD inline bool resolveWindBlockHit(WorldView world, BlockWorldView blocks, WindChargeState& charge) {
    const auto hit = fullCubeProjectileHit(blocks, charge.position, charge.velocity);
    if (!hit.hit) return false;
    charge.position = {charge.position.x + charge.velocity.x * hit.fraction,
                       charge.position.y + charge.velocity.y * hit.fraction,
                       charge.position.z + charge.velocity.z * hit.fraction};
    const Vec3 center{charge.position.x + hit.normal.x * 0.25,
                      charge.position.y + hit.normal.y * 0.25,
                      charge.position.z + hit.normal.z * 0.25};
    applyWindExplosion(world, blocks, center, charge.owner);
    charge.active = false;
    return true;
}

// Mirrors the Java float expression and strict release threshold used by BowItem.
// Source: minecraft-26.3-src/net/minecraft/world/item/BowItem.java, BowItem.getPowerForTime (74-80), releaseUsing (26-51).
MCSIM_HD inline float bowPower(int held_ticks) {
    float power = static_cast<float>(held_ticks) / 20.0F;
    power = (power * power + 2.0F * power) / 3.0F;
    if (power > 1.0F) {
        power = 1.0F;
    }
    return power;
}

// Uses caller-owned LegacyRandom state for bounded critical-damage draws; this is deterministic simulator RNG,
// not a mapping to Minecraft's projectile RandomSource seed.
// Source: minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java, onHitEntity (423-470).
MCSIM_HD inline int projectileRandomInt(LegacyRandom& random, int bound) {
    if (bound <= 0) {
        return 0;
    }
    const std::uint32_t unsigned_bound = static_cast<std::uint32_t>(bound);
    if ((unsigned_bound & (0U - unsigned_bound)) == unsigned_bound) {
        return static_cast<int>((static_cast<std::uint64_t>(unsigned_bound) * random.next(31)) >> 31);
    }
    for (;;) {
        const std::uint32_t bits = random.next(31);
        const std::uint32_t value = bits % unsigned_bound;
        if (static_cast<std::uint64_t>(bits) - value + (unsigned_bound - 1U) < 0x80000000ULL) {
            return static_cast<int>(value);
        }
    }
}

// Advances the same explicit simulator RNG stream used for the visual-only reverse-deflection rotation.
// Source: minecraft-26.3-src/net/minecraft/world/entity/projectile/ProjectileDeflection.java, REVERSE (8-14).
MCSIM_HD inline float projectileRandomFloat(LegacyRandom& random) {
    return static_cast<float>(random.next(24)) * (1.0F / 16777216.0F);
}

// Builds the owner-clearance AABB using the simulator's half-block arrow-box assumption and vanilla's sweep rule.
// Source: minecraft-26.3-src/net/minecraft/world/entity/projectile/Projectile.java, isOutsideOwnerCollisionRange (148-158).
MCSIM_HD inline Aabb arrowSearchArea(Vec3 position, Vec3 movement) {
    constexpr double half_width = double(0.5F) / 2.0;
    constexpr double inflation = 1.0;
    const Aabb arrow_box{{position.x - half_width, position.y - half_width, position.z - half_width},
                         {position.x + half_width, position.y + half_width, position.z + half_width}};
    Aabb search = arrow_box.swept(movement);
    search.lower.x -= inflation;
    search.lower.y -= inflation;
    search.lower.z -= inflation;
    search.upper.x += inflation;
    search.upper.y += inflation;
    search.upper.z += inflation;
    return search;
}

MCSIM_HD inline double distanceSquaredToAabb(Vec3 point, Aabb bounds) {
    const double x = point.x < bounds.lower.x ? bounds.lower.x - point.x
        : point.x > bounds.upper.x ? point.x - bounds.upper.x : 0.0;
    const double y = point.y < bounds.lower.y ? bounds.lower.y - point.y
        : point.y > bounds.upper.y ? point.y - bounds.upper.y : 0.0;
    const double z = point.z < bounds.lower.z ? bounds.lower.z - point.z
        : point.z > bounds.upper.z ? point.z - bounds.upper.z : 0.0;
    return x * x + y * y + z * z;
}

MCSIM_HD inline float arrowTargetProgress(const ArrowState& arrow, const PlayerState& target) {
    if (arrow.launch_target_distance <= 0.0) {
        return 0.0F;
    }
    const double impact_distance = ::sqrt(distanceSquaredToAabb(arrow.position, target.bounds()));
    return static_cast<float>(minimum(1.0, maximum(0.0,
        1.0 - impact_distance / arrow.launch_target_distance)));
}

MCSIM_HD inline void recordArrowLanding(ArrowState& arrow, WorldView world, ProjectileTickResult& result) {
    if (arrow.progress_rewarded) {
        return;
    }
    if (arrow.owner >= 0 && arrow.owner < projectile_player_count) {
        result.arrow_progress[arrow.owner] += arrowTargetProgress(arrow, world.player(1 - arrow.owner).state());
    }
    arrow.progress_rewarded = true;
}

// Clips an arrow centerline to a player box, including vanilla's age-ramped margin and block visibility check.
// Source: minecraft-26.3-src/net/minecraft/world/entity/projectile/ProjectileUtil.java, computeMargin (153),
// getManyEntityHitResult (201-242); AbstractArrow.findHitEntities (278-286).
MCSIM_HD inline bool arrowHitFraction(Aabb target, Vec3 origin, Vec3 movement, double block_fraction,
                                      int arrow_age, BlockWorldView blocks, double& result_fraction) {
    const Vec3 clipped_movement{movement.x * block_fraction, movement.y * block_fraction,
                                movement.z * block_fraction};
    double fraction = 1.0;
    if (rayBoxHit(target, origin, clipped_movement, fraction)) {
        result_fraction = fraction * block_fraction;
        return true;
    }

    float margin = static_cast<float>(arrow_age - 2) / 20.0F;
    margin = static_cast<float>(maximum(0.0, minimum(double(0.3F), double(margin))));
    if (margin <= 0.0F) {
        return false;
    }
    const Aabb expanded{{target.lower.x - margin, target.lower.y - margin, target.lower.z - margin},
                        {target.upper.x + margin, target.upper.y + margin, target.upper.z + margin}};
    if (!rayBoxHit(expanded, origin, clipped_movement, fraction)) {
        return false;
    }

    const Vec3 outside_position{origin.x + clipped_movement.x * fraction,
                                origin.y + clipped_movement.y * fraction,
                                origin.z + clipped_movement.z * fraction};
    const Vec3 center{(target.lower.x + target.upper.x) * 0.5,
                      (target.lower.y + target.upper.y) * 0.5,
                      (target.lower.z + target.upper.z) * 0.5};
    const Vec3 toward_center{center.x - outside_position.x, center.y - outside_position.y,
                             center.z - outside_position.z};
    const double visibility_fraction = blockRayFraction(blocks, outside_position, toward_center);
    const Vec3 visible_delta{toward_center.x * visibility_fraction,
                             toward_center.y * visibility_fraction,
                             toward_center.z * visibility_fraction};
    double surface_fraction = 1.0;
    if (!rayBoxHit(target, outside_position, visible_delta, surface_fraction)) {
        return false;
    }
    result_fraction = fraction * block_fraction;
    return true;
}

// Minecraft 26.3: Projectile.java:135-158; ProjectileUtil.java:54-82, 153-242;
// AbstractHurtingProjectile.java:68-103; AbstractWindCharge.java:88-107, 118-136, 152.
MCSIM_HD inline ProjectileTickResult tickWindCharge(WorldView world, BlockWorldView blocks, WindChargeState& charge) {
    ProjectileTickResult result;
    if (!charge.active) return result;
    const Vec3 start = charge.position;
    const Vec3 movement = charge.velocity;
    const auto block = fullCubeProjectileHit(blocks, start, movement);
    const double block_fraction = block.hit ? block.fraction : 1.0;
    int hit_player = -1;
    double hit_fraction = 1.0;
    double nearest_distance = (std::numeric_limits<double>::max)();
    for (int index = 0; index < world.playerCount(); ++index) {
        const auto player_view = world.player(index);
        const auto& player = player_view.state();
        if (!player.alive() || (index == charge.owner && !charge.left_owner)) continue;
        double fraction = 1.0;
        const Vec3 relative{player.position.x - start.x, player.position.y - start.y, player.position.z - start.z};
        const double distance = relative.lengthSquared();
        if (arrowHitFraction(player.bounds(), start, movement, block_fraction, charge.age, blocks, fraction)
            && distance < nearest_distance) {
            hit_player = index;
            hit_fraction = fraction;
            nearest_distance = distance;
        }
    }
    const double travel = hit_player >= 0 ? hit_fraction : block_fraction;
    const Vec3 end{start.x + movement.x * travel, start.y + movement.y * travel, start.z + movement.z * travel};
    if (!charge.left_owner && charge.owner >= 0 && charge.owner < world.playerCount()) {
        constexpr double half_width = double(0.3125F) / 2.0;
        constexpr double height = double(0.3125F);
        Aabb search{{end.x - half_width, end.y, end.z - half_width},
                    {end.x + half_width, end.y + height, end.z + half_width}};
        search = search.swept(movement);
        search.lower.x -= 1.0; search.lower.y -= 1.0; search.lower.z -= 1.0;
        search.upper.x += 1.0; search.upper.y += 1.0; search.upper.z += 1.0;
        if (!search.intersects(world.player(charge.owner).state().bounds())) charge.left_owner = true;
    }
    ++charge.age;
    if (charge.no_deflect_ticks > 0) --charge.no_deflect_ticks;
    if (hit_player >= 0) {
        charge.position = end;
        auto target = world.player(hit_player);
        const float health_before = target.state().health;
        target.hurt(WIND_CHARGE_DAMAGE, end, 0.0F, false, DamageKind::arrow, movement);
        const float health_lost = health_before - target.state().health;
        if (hit_player < projectile_player_count) result.health_taken[hit_player] += health_lost;
        if (charge.owner >= 0 && charge.owner < projectile_player_count && charge.owner != hit_player) {
            result.health_dealt[charge.owner] += health_lost;
        }
        applyWindExplosion(world, blocks, end, charge.owner);
        charge.active = false;
        ++result.player_hits;
    } else if (block.hit) {
        resolveWindBlockHit(world, blocks, charge);
        ++result.terrain_hits;
    } else {
        charge.position = end;
    }
    return result;
}

// Minecraft 26.3: ThrowableProjectile.java:43-67, 70-88, 104; Projectile.java:135-158;
// Snowball.java:55-70; ThrownEgg.java:54-56; Player.java:727-757. Eggs intentionally do not spawn chickens.
MCSIM_HD inline ProjectileTickResult tickThrownProjectile(WorldView world, BlockWorldView blocks,
                                                            ThrownProjectileState& projectile) {
    ProjectileTickResult result;
    if (!projectile.active) return result;
    if (projectile.item == ThrownItem::ender_pearl && projectile.owner >= 0
        && projectile.owner < world.playerCount() && !world.player(projectile.owner).state().alive()) {
        projectile.active = false;
        ++result.expired;
        return result;
    }
    const Vec3 start = projectile.position;
    projectile.velocity.y -= 0.03;
    projectile.velocity.x *= double(0.99F);
    projectile.velocity.y *= double(0.99F);
    projectile.velocity.z *= double(0.99F);
    const Vec3 movement = projectile.velocity;
    const auto block = fullCubeProjectileHit(blocks, start, movement);
    const double block_fraction = block.hit ? block.fraction : 1.0;
    int hit_player = -1;
    double hit_fraction = 1.0;
    double nearest_distance = (std::numeric_limits<double>::max)();
    for (int index = 0; index < world.playerCount(); ++index) {
        const auto player_view = world.player(index);
        const auto& player = player_view.state();
        if (!player.alive() || (index == projectile.owner && !projectile.left_owner)) continue;
        const Vec3 relative{player.position.x - start.x, player.position.y - start.y, player.position.z - start.z};
        const double distance = relative.lengthSquared();
        double fraction = 1.0;
        if (arrowHitFraction(player.bounds(), start, movement, block_fraction, projectile.age, blocks, fraction)
            && distance < nearest_distance) {
            hit_player = index;
            hit_fraction = fraction;
            nearest_distance = distance;
        }
    }
    const double travel = hit_player >= 0 ? hit_fraction : block_fraction;
    projectile.position = {start.x + movement.x * travel, start.y + movement.y * travel,
                           start.z + movement.z * travel};
    if (!projectile.left_owner && projectile.owner >= 0 && projectile.owner < world.playerCount()) {
        constexpr double half_width = double(0.25F) / 2.0;
        constexpr double height = double(0.25F);
        Aabb search{{projectile.position.x - half_width, projectile.position.y, projectile.position.z - half_width},
                    {projectile.position.x + half_width, projectile.position.y + height, projectile.position.z + half_width}};
        search = search.swept(movement);
        search.lower.x -= 1.0; search.lower.y -= 1.0; search.lower.z -= 1.0;
        search.upper.x += 1.0; search.upper.y += 1.0; search.upper.z += 1.0;
        if (!search.intersects(world.player(projectile.owner).state().bounds())) projectile.left_owner = true;
    }
    ++projectile.age;
    // Minecraft 26.3: Entity.java:587-591; ThrownEnderpearl.java:106-114, 132-136, 179-182.
    if ((hit_player >= 0 || block.hit) && projectile.item == ThrownItem::ender_pearl
        && projectile.owner >= 0 && projectile.owner < world.playerCount()) {
        auto owner = world.player(projectile.owner);
        auto& state = owner.state();
        if (state.alive()) {
            state.position = start;
            state.velocity = {};
            state.fall_distance = 0.0;
            state.impulse_impact_y = 0.0;
            state.impulse_grace_ticks = 0;
            state.has_impulse_impact = false;
            state.needs_sync = true;
            const float health_before = state.health;
            owner.hurt(5.0F, start, 0.0F, true, DamageKind::fall);
            if (projectile.owner < projectile_player_count) result.health_taken[projectile.owner] += health_before - state.health;
        }
    }
    if (hit_player >= 0) {
        projectile.active = false;
        ++result.player_hits;
    } else if (block.hit) {
        projectile.active = false;
        ++result.terrain_hits;
    }
    return result;
}

// Minecraft 26.3: Projectile.java:164-168, 185-196. The RNG stream is simulator-owned.
MCSIM_HD inline Vec3 projectileLaunchVelocity(PlayerState shooter, float launch_speed, MathView math,
                                               LegacyRandom& world_random) {
    const float pitch_radians = shooter.pitch * 0.017453292F;
    const float yaw_radians = shooter.yaw * 0.017453292F;
    const float x_direction = -math.sin(yaw_radians) * math.cos(pitch_radians);
    const float y_direction = -math.sin(pitch_radians);
    const float z_direction = math.cos(yaw_radians) * math.cos(pitch_radians);
    const double length = ::sqrt(double(x_direction) * x_direction + double(y_direction) * y_direction
                                 + double(z_direction) * z_direction);
    const double spread = 0.0172275;
    const double first_x = world_random.nextDouble();
    const double second_x = world_random.nextDouble();
    const double first_y = world_random.nextDouble();
    const double second_y = world_random.nextDouble();
    const double first_z = world_random.nextDouble();
    const double second_z = world_random.nextDouble();
    const double noisy_x = double(x_direction) / length + spread * (first_x - second_x);
    const double noisy_y = double(y_direction) / length + spread * (first_y - second_y);
    const double noisy_z = double(z_direction) / length + spread * (first_z - second_z);
    return {noisy_x * double(launch_speed) + shooter.velocity.x,
            noisy_y * double(launch_speed) + (shooter.on_ground ? 0.0 : shooter.velocity.y),
            noisy_z * double(launch_speed) + shooter.velocity.z};
}

// Minecraft 26.3: BowItem.java:26-51; AbstractArrow.java:116-125.
MCSIM_HD inline ArrowState launchArrow(PlayerState shooter, int owner, float power, MathView math,
                                       LegacyRandom& world_random) {
    ArrowState arrow;
    arrow.active = true;
    arrow.owner = owner;
    arrow.position = {shooter.position.x, shooter.eyePosition().y - double(0.1F), shooter.position.z};
    arrow.critical = power == 1.0F;
    arrow.velocity = projectileLaunchVelocity(shooter, power * 3.0F, math, world_random);
    arrow.random = world_random;
    return arrow;
}

// Minecraft 26.3: WindChargeItem.java:30-36; WindCharge.java:36, 43-44.
MCSIM_HD inline WindChargeState launchWindCharge(PlayerState shooter, int owner, MathView math,
                                                 LegacyRandom& world_random) {
    WindChargeState charge;
    charge.active = true;
    charge.owner = owner;
    charge.position = {shooter.position.x, shooter.eyePosition().y, shooter.position.z};
    charge.velocity = projectileLaunchVelocity(shooter, WIND_CHARGE_SHOOT_POWER, math, world_random);
    return charge;
}

// Minecraft 26.3: SnowballItem.java:26-33; EggItem.java:26-33; ThrowableItemProjectile.java:29-31.
MCSIM_HD inline ThrownProjectileState launchThrownItem(PlayerState shooter, int owner, ThrownItem item,
                                                        MathView math, LegacyRandom& world_random) {
    ThrownProjectileState projectile;
    projectile.active = true;
    projectile.owner = owner;
    projectile.item = item;
    projectile.position = {shooter.position.x, shooter.eyePosition().y - double(0.1F), shooter.position.z};
    projectile.velocity = projectileLaunchVelocity(shooter, 1.5F, math, world_random);
    return projectile;
}

// Applies the explicit start/hold/release/cancel contract and consumes ammunition only after a slot is available.
// Source: minecraft-26.3-src/net/minecraft/world/item/BowItem.java, use (95-106), releaseUsing (26-51);
// minecraft-26.3-src/net/minecraft/world/entity/LivingEntity.java, startUsingItem (3500), releaseUsingItem (3605), stopUsingItem (3618).
MCSIM_HD inline BowActionResult processBowCommand(WorldView world, MathView math, int player_index,
                                                   BowCommand command, ProjectileWorldState& projectile_state) {
    BowActionResult result;
    if (player_index < 0 || player_index >= projectile_player_count) {
        result.status = BowActionStatus::invalid_storage;
        return result;
    }
    auto& bow = projectile_state.bows[player_index];
    auto& player = world.player(player_index).state();
    if (command == BowCommand::idle) {
        if (bow.using_bow && player.using_shield) {
            bow.using_bow = false;
            bow.draw_ticks = 0;
            player.using_bow = false;
            result.status = BowActionStatus::shield_conflict;
            return result;
        }
        player.using_bow = bow.using_bow;
        result.status = bow.using_bow ? BowActionStatus::drawing : BowActionStatus::idle;
        return result;
    }
    if (command == BowCommand::cancel) {
        bow.using_bow = false;
        bow.draw_ticks = 0;
        player.using_bow = false;
        result.status = BowActionStatus::cancelled;
        return result;
    }
    if (command == BowCommand::hold) {
        if (player.weapon != WeaponItem::bow) {
            result.status = BowActionStatus::not_equipped;
            return result;
        }
        if (player.using_shield) {
            bow.using_bow = false;
            bow.draw_ticks = 0;
            player.using_bow = false;
            result.status = BowActionStatus::shield_conflict;
            return result;
        }
        if (!bow.using_bow && bow.ordinary_arrows <= 0) {
            result.status = BowActionStatus::no_ammunition;
            return result;
        }
        bow.using_bow = true;
        player.using_bow = true;
        if (bow.draw_ticks < 72000) {
            ++bow.draw_ticks;
        }
        result.status = BowActionStatus::drawing;
        return result;
    }

    if (!bow.using_bow) {
        result.status = BowActionStatus::not_using;
        return result;
    }
    const int held_ticks = bow.draw_ticks;
    bow.using_bow = false;
    bow.draw_ticks = 0;
    player.using_bow = false;
    if (player.weapon != WeaponItem::bow) {
        result.status = BowActionStatus::cancelled;
        return result;
    }
    if (player.using_shield) {
        result.status = BowActionStatus::shield_conflict;
        return result;
    }
    if (bow.ordinary_arrows <= 0) {
        result.status = BowActionStatus::no_ammunition;
        return result;
    }
    result.power = bowPower(held_ticks);
    if (result.power < 0.1F) {
        result.status = BowActionStatus::too_weak;
        return result;
    }
    if (!projectile_state.arrows || projectile_state.arrow_capacity <= 0) {
        result.status = BowActionStatus::invalid_storage;
        return result;
    }
    for (int slot = 0; slot < projectile_state.arrow_capacity; ++slot) {
        if (!projectile_state.arrows[slot].active) {
            auto arrow = launchArrow(player, player_index, result.power, math, projectile_state.random);
            const auto target_bounds = world.player(1 - player_index).state().bounds();
            arrow.launch_target_distance = ::sqrt(distanceSquaredToAabb(arrow.position, target_bounds));
            projectile_state.arrows[slot] = arrow;
            --bow.ordinary_arrows;
            result.arrow_slot = slot;
            result.status = BowActionStatus::fired;
            return result;
        }
    }
    result.status = BowActionStatus::capacity_full;
    return result;
}

// Advances existing arrows; newly released slots wait until the next projectile phase.
// Source: minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java, tick (173-266),
// tickDespawn (371-376), onHitEntity (423-512), onHitBlock (530-548); Projectile.checkLeftOwner (141-158);
// LivingEntity.hurtServer (1178-1260); CombatRules.getDamageAfterAbsorb (16-35).
MCSIM_HD inline ProjectileTickResult tickProjectiles(WorldView world, BlockWorldView blocks, ProjectileWorldState& state,
                                                     int new_first_slot = -1, int new_second_slot = -1,
                                                     int new_first_thrown = -1, int new_second_thrown = -1,
                                                     int new_first_wind = -1, int new_second_wind = -1) {
    ProjectileTickResult result;
    for (int slot = 0; state.arrows && slot < state.arrow_capacity; ++slot) {
        auto& arrow = state.arrows[slot];
        if (!arrow.active || slot == new_first_slot || slot == new_second_slot) {
            continue;
        }
        if (arrow.in_ground) {
            ++arrow.age;
            ++arrow.in_ground_ticks;
            if (arrow.in_ground_ticks >= arrow_lifetime_ticks) {
                arrow.active = false;
                ++result.expired;
            }
            continue;
        }

        const Vec3 movement = arrow.velocity;
        const Vec3 start = arrow.position;
        if (blockRayFraction(blocks, start, {}) < 1.0) {
            arrow.velocity = {};
            arrow.in_ground = true;
            arrow.in_ground_ticks = 0;
            recordArrowLanding(arrow, world, result);
            ++arrow.age;
            continue;
        }

        if (!arrow.left_owner && arrow.owner >= 0 && arrow.owner < projectile_player_count) {
            const auto owner_bounds = world.player(arrow.owner).state().bounds();
            if (!arrowSearchArea(start, movement).intersects(owner_bounds)) {
                arrow.left_owner = true;
            }
        }

        const double block_fraction = blockRayFraction(blocks, start, movement);
        int hit_player = -1;
        double hit_fraction = 1.0;
        double nearest_player_distance = (std::numeric_limits<double>::max)();
        for (int player_index = 0; player_index < projectile_player_count; ++player_index) {
            auto target_view = world.player(player_index);
            const auto& target = target_view.state();
            if (!target.alive() || (player_index == arrow.owner && !arrow.left_owner)) {
                continue;
            }
            double candidate_fraction = 1.0;
            const Vec3 to_player{target.position.x - start.x, target.position.y - start.y,
                                 target.position.z - start.z};
            const double player_distance = to_player.lengthSquared();
            if (arrowHitFraction(target.bounds(), start, movement, block_fraction, arrow.age, blocks,
                                 candidate_fraction) && player_distance < nearest_player_distance) {
                hit_player = player_index;
                hit_fraction = candidate_fraction;
                nearest_player_distance = player_distance;
            }
        }

        if (hit_player >= 0) {
            arrow.position = {start.x + movement.x * hit_fraction, start.y + movement.y * hit_fraction,
                              start.z + movement.z * hit_fraction};
            if (!arrow.progress_rewarded && arrow.owner >= 0 && arrow.owner < projectile_player_count
                && hit_player != arrow.owner) {
                result.arrow_progress[arrow.owner] += 1.0F;
                arrow.progress_rewarded = true;
            }
            const float speed = static_cast<float>(::sqrt(arrow.velocity.lengthSquared()));
            const double raw_damage = double(speed) * 2.0;
            const double bounded_damage = maximum(0.0, minimum(2147483647.0, raw_damage));
            int damage = static_cast<int>(::ceil(bounded_damage));
            if (arrow.critical) {
                const int bonus = projectileRandomInt(arrow.random, damage / 2 + 2);
                damage = static_cast<int>(minimum(2147483647.0, double(damage) + bonus));
            }
            const float previous_health = world.player(hit_player).state().health;
            const bool accepted = world.player(hit_player).hurt(float(damage), arrow.position, 0.0F, false,
                                                                DamageKind::arrow, arrow.velocity);
            const float health_lost = static_cast<float>(maximum(0.0,
                double(previous_health - world.player(hit_player).state().health)));
            if (health_lost > 0.0F) {
                result.health_taken[hit_player] += health_lost;
                if (arrow.owner >= 0 && arrow.owner < projectile_player_count && arrow.owner != hit_player) {
                    result.health_dealt[arrow.owner] += health_lost;
                }
            }
            if (accepted) {
                arrow.active = false;
                ++result.player_hits;
                continue;
            }
            arrow.velocity.x *= -0.1;
            arrow.velocity.y *= -0.1;
            arrow.velocity.z *= -0.1;
            const float visual_rotation = 170.0F + projectileRandomFloat(arrow.random) * 20.0F;
            (void)visual_rotation;
            ++result.deflections;
            if (arrow.velocity.lengthSquared() < 1.0E-7) {
                arrow.active = false;
                continue;
            }
        } else if (block_fraction < 1.0) {
            arrow.position = {start.x + movement.x * block_fraction, start.y + movement.y * block_fraction,
                              start.z + movement.z * block_fraction};
            const double contact_offset = double(0.05F);
            arrow.position.x -= movement.x > 0.0 ? contact_offset : movement.x < 0.0 ? -contact_offset : 0.0;
            arrow.position.y -= movement.y > 0.0 ? contact_offset : movement.y < 0.0 ? -contact_offset : 0.0;
            arrow.position.z -= movement.z > 0.0 ? contact_offset : movement.z < 0.0 ? -contact_offset : 0.0;
            arrow.velocity = {};
            arrow.in_ground = true;
            arrow.critical = false;
            arrow.in_ground_ticks = 0;
            recordArrowLanding(arrow, world, result);
            ++arrow.age;
            ++result.terrain_hits;
            continue;
        } else {
            arrow.position = {start.x + movement.x, start.y + movement.y, start.z + movement.z};
        }

        arrow.velocity.x *= double(0.99F);
        arrow.velocity.y = arrow.velocity.y * double(0.99F) - 0.05;
        arrow.velocity.z *= double(0.99F);
        ++arrow.age;
    }
    for (int slot = 0; state.wind_charges && slot < state.wind_charge_capacity; ++slot) {
        if (slot == new_first_wind || slot == new_second_wind) continue;
        const auto wind = tickWindCharge(world, blocks, state.wind_charges[slot]);
        result.player_hits += wind.player_hits;
        result.terrain_hits += wind.terrain_hits;
        for (int player = 0; player < projectile_player_count; ++player) {
            result.health_dealt[player] += wind.health_dealt[player];
            result.health_taken[player] += wind.health_taken[player];
        }
    }
    for (int slot = 0; state.thrown_items && slot < state.thrown_item_capacity; ++slot) {
        if (slot == new_first_thrown || slot == new_second_thrown) continue;
        const auto thrown = tickThrownProjectile(world, blocks, state.thrown_items[slot]);
        result.player_hits += thrown.player_hits;
        result.terrain_hits += thrown.terrain_hits;
        result.expired += thrown.expired;
        for (int player = 0; player < projectile_player_count; ++player) {
            result.health_taken[player] += thrown.health_taken[player];
        }
    }
    return result;
}

// Clears exactly one explicitly supplied world's sidecar state while preserving its caller-owned arrow buffer.
// Simulator lifecycle invariant: no global state, player layout, training reset or other world's buffers are touched.
MCSIM_HD inline void resetProjectileWorld(ProjectileWorldState& state, std::uint64_t random_seed) {
    for (int player = 0; player < projectile_player_count; ++player) {
        state.bows[player] = BowState{};
    }
    if (state.arrows && state.arrow_capacity > 0) {
        for (int slot = 0; slot < state.arrow_capacity; ++slot) {
            state.arrows[slot] = ArrowState{};
        }
    }
    if (state.wind_charges && state.wind_charge_capacity > 0) {
        for (int slot = 0; slot < state.wind_charge_capacity; ++slot) {
            state.wind_charges[slot] = WindChargeState{};
        }
    }
    if (state.thrown_items && state.thrown_item_capacity > 0) {
        for (int slot = 0; slot < state.thrown_item_capacity; ++slot) {
            state.thrown_items[slot] = ThrownProjectileState{};
        }
    }
    state.random.setSeed(random_seed);
}

// Resolves use_item and slot cancellation before look/movement, then advances players and pre-existing arrows.
// Source: minecraft-26.3-src/net/minecraft/world/item/BowItem.java, use (95-106), releaseUsing (26-51);
// minecraft-26.3-src/net/minecraft/world/entity/LivingEntity.java, startUsingItem (3500), releaseUsingItem (3605), stopUsingItem (3618).
// Simulator ordering invariant: each world owns one stable sidecar; selective reset clears only that sidecar and its caller-owned arrow slice.
MCSIM_HD inline ProjectileTickResult processWorldTick(WorldView world, BlockWorldView blocks,
                                                       ProjectileWorldState& projectile_state) {
    auto& world_state = world.mutableState();
    for (int player = 0; player < projectile_player_count; ++player) {
        world_state.projectile_health_dealt[player] = 0.0F;
        world_state.projectile_health_taken[player] = 0.0F;
        world_state.projectile_arrow_progress[player] = 0.0F;
        world_state.wind_self_launch[player] = 0.0;
    }

    bool release_suppressed[projectile_player_count]{};
    int released_slots[projectile_player_count]{-1, -1};
    for (int index = 0; index < projectile_player_count; ++index) {
        auto& actor = world.player(index).state();
        const auto& input = world.queuedInput(index);
        const WeaponItem requested_weapon = input.select_weapon ? input.weapon : actor.weapon;
        const bool slot_changed = input.select_weapon && input.selected_slot >= 0
            && input.selected_slot != actor.selected_weapon_slot;

        if ((slot_changed || requested_weapon != WeaponItem::bow || !actor.alive())
            && projectile_state.bows[index].using_bow) {
            processBowCommand(world, world.math(), index, BowCommand::cancel, projectile_state);
        }
        if (input.select_weapon) {
            world.player(index).equipWeapon(input.weapon);
            if (input.selected_slot >= 0) actor.selected_weapon_slot = input.selected_slot;
        }

        if (requested_weapon == WeaponItem::bow && actor.alive()) {
            if (actor.using_shield) {
                actor.using_shield = false;
                actor.shield_use_ticks = 0;
            }
            if (input.use_item) {
                const auto result = processBowCommand(world, world.math(), index, BowCommand::hold, projectile_state);
                actor.using_bow = result.status == BowActionStatus::drawing;
            } else if (projectile_state.bows[index].using_bow) {
                release_suppressed[index] = true;
                released_slots[index] = processBowCommand(world, world.math(), index,
                    BowCommand::release, projectile_state).arrow_slot;
            } else {
                actor.using_bow = false;
            }
        } else {
            actor.using_bow = false;
        }
    }

    world.beginTick(release_suppressed);
    // Minecraft 26.3: Minecraft.java:1791-1805; SnowballItem.java:26-33; EggItem.java:26-33;
    // EnderpearlItem.java:24-32; UseCooldown.java:37-40.
    int thrown_slots[projectile_player_count]{-1, -1};
    int wind_slots[projectile_player_count]{-1, -1};
    int capacity_failures = 0;
    for (int index = 0; index < projectile_player_count; ++index) {
        auto actor = world.player(index);
        auto& state = actor.state();
        const auto& input = world.queuedInput(index);
        const bool snowball = state.weapon == WeaponItem::snowball;
        const bool egg = state.weapon == WeaponItem::egg;
        const bool pearl = state.weapon == WeaponItem::ender_pearl;
        const bool wind = state.weapon == WeaponItem::wind_charge;
        if (!input.use_item || !state.alive() || state.using_shield || state.using_bow
            || state.item_use_delay != 0 || (!snowball && !egg && !pearl && !wind)
            || (pearl && state.ender_pearl_cooldown > 0)
            || (wind && state.wind_charge_cooldown > 0)) continue;
        state.item_use_delay = 4;
        if (wind) {
            if (state.wind_charge_count <= 0) continue;
            for (int slot = 0; projectile_state.wind_charges && slot < projectile_state.wind_charge_capacity; ++slot) {
                if (!projectile_state.wind_charges[slot].active) {
                    projectile_state.wind_charges[slot] = launchWindCharge(state, index, world.math(), projectile_state.random);
                    wind_slots[index] = slot;
                    --state.wind_charge_count;
                    state.wind_charge_cooldown = WIND_CHARGE_COOLDOWN_TICKS;
                    if (state.wind_charge_count == 0) actor.equipWeapon(WeaponItem::unarmed);
                    break;
                }
            }
            if (wind_slots[index] < 0) ++capacity_failures;
            continue;
        }
        int* count = snowball ? &state.snowball_count : egg ? &state.egg_count : &state.ender_pearl_count;
        if (*count <= 0) continue;
        for (int slot = 0; projectile_state.thrown_items && slot < projectile_state.thrown_item_capacity; ++slot) {
            if (!projectile_state.thrown_items[slot].active) {
                projectile_state.thrown_items[slot] = launchThrownItem(state, index,
                    snowball ? ThrownItem::snowball : egg ? ThrownItem::egg : ThrownItem::ender_pearl,
                    world.math(), projectile_state.random);
                thrown_slots[index] = slot;
                --*count;
                // Minecraft 26.3: Items.java:1034; ItemStack.java:400-410.
                if (pearl) state.ender_pearl_cooldown = 20;
                if (*count == 0) actor.equipWeapon(WeaponItem::unarmed);
                break;
            }
        }
        if (thrown_slots[index] < 0) ++capacity_failures;
    }
    world.finishTick();
    auto result = tickProjectiles(world, blocks, projectile_state, released_slots[0], released_slots[1],
                                  thrown_slots[0], thrown_slots[1], wind_slots[0], wind_slots[1]);
    result.capacity_failures = capacity_failures;
    for (int player = 0; player < projectile_player_count; ++player) {
        world_state.projectile_health_dealt[player] = result.health_dealt[player];
        world_state.projectile_health_taken[player] = result.health_taken[player];
        world_state.projectile_arrow_progress[player] = result.arrow_progress[player];
    }
    return result;
}

} // namespace minecraft_sim
