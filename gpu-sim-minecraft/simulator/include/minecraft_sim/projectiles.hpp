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

struct ProjectileWorldState {
    BowState bows[projectile_player_count]{};
    ArrowState* arrows = nullptr;
    int arrow_capacity = 0;
    LegacyRandom random;
};

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
    float health_dealt[projectile_player_count]{};
    float health_taken[projectile_player_count]{};
    float arrow_progress[projectile_player_count]{};
};

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

// Creates one ordinary arrow using BowItem charge, Projectile.shootFromRotation, triangular spread and inherited motion.
// Source: minecraft-26.3-src/net/minecraft/world/item/BowItem.java, releaseUsing (26-51);
// minecraft-26.3-src/net/minecraft/world/entity/projectile/Projectile.java, getMovementToShoot (161-181),
// shootFromRotation (183-194); AbstractArrow constructor (116-125).
MCSIM_HD inline ArrowState launchArrow(PlayerState shooter, int owner, float power, MathView math,
                                       LegacyRandom& world_random) {
    ArrowState arrow;
    arrow.active = true;
    arrow.owner = owner;
    arrow.position = {shooter.position.x, shooter.eyePosition().y - double(0.1F), shooter.position.z};
    arrow.critical = power == 1.0F;
    const float launch_speed = power * 3.0F;

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
    arrow.velocity = {noisy_x * double(launch_speed), noisy_y * double(launch_speed), noisy_z * double(launch_speed)};
    arrow.velocity.x += shooter.velocity.x;
    arrow.velocity.y += shooter.on_ground ? 0.0 : shooter.velocity.y;
    arrow.velocity.z += shooter.velocity.z;
    arrow.random = world_random;
    return arrow;
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
                                                     int new_first_slot = -1, int new_second_slot = -1) {
    ProjectileTickResult result;
    if (!state.arrows || state.arrow_capacity <= 0) {
        return result;
    }
    for (int slot = 0; slot < state.arrow_capacity; ++slot) {
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

    world.processTick(release_suppressed);
    const auto result = tickProjectiles(world, blocks, projectile_state, released_slots[0], released_slots[1]);
    for (int player = 0; player < projectile_player_count; ++player) {
        world_state.projectile_health_dealt[player] = result.health_dealt[player];
        world_state.projectile_health_taken[player] = result.health_taken[player];
        world_state.projectile_arrow_progress[player] = result.arrow_progress[player];
    }
    return result;
}

} // namespace minecraft_sim
