#pragma once

#include "minecraft_sim/equipment.hpp"
#include "minecraft_sim/geometry.hpp"
#include "minecraft_sim/math.hpp"

#include <cstdint>

namespace minecraft_sim {

struct Controls {
    bool forward = false;
    bool backward = false;
    bool left = false;
    bool right = false;
    bool jump = false;
    float yaw_delta = 0.0F;
    float pitch_delta = 0.0F;
    bool sprint = false;
    bool sneak = false;
    bool attack = false;
    bool use_item = false;
    bool select_weapon = false;
    WeaponItem weapon = WeaponItem::diamond_sword;
};

enum class PlayerPose : std::uint8_t { standing, crouching, crawling };

struct AttackOutcome {
    int target = -1;
    bool attempted = false;
    bool accepted = false;
    bool critical = false;
    bool sprint_bonus = false;
    bool sweep = false;
    bool hit_block = false;
    int sweep_hits = 0;
    float charge = 0.0F;
    float incoming_damage = 0.0F;
    float health_lost = 0.0F;
    float blocked_damage = 0.0F;
    bool shield_disabled = false;
};

struct PlayerState {
    Vec3 position;
    Vec3 velocity;
    float yaw = 0.0F;
    float pitch = 0.0F;
    bool on_ground = false;
    bool horizontal_collision = false;
    bool vertical_collision = false;
    int jump_delay = 0;
    std::uint64_t tick = 0;
    bool sprinting = false;
    bool minor_horizontal_collision = false;
    int food_level = 20;
    float food_saturation = 5.0F;
    float food_exhaustion = 0.0F;
    int food_tick_timer = 0;
    bool mobility_restricted = false;
    double movement_speed = 0.1;
    bool may_fly = false;
    bool crouching = false;
    bool previous_forward = false;
    bool previous_sneak = false;
    int sprint_trigger_time = 0;
    int sprint_window = 7;
    double fall_distance = 0.0;
    PlayerPose pose = PlayerPose::standing;
    double knockback_resistance = 0.0;
    bool pushable = true;
    bool no_physics = false;
    bool needs_sync = false;
    LegacyRandom random;
    float health = 20.0F;
    float max_health = 20.0F;
    float absorption = 0.0F;
    float armor = 0.0F;
    float armor_toughness = 0.0F;
    ArmorMaterial armor_items[4]{};
    double armor_knockback_resistance = 0.0;
    float last_hurt = 0.0F;
    int damage_cooldown = 0;
    int hurt_time = 0;
    int hurt_duration = 0;
    int death_time = 0;
    bool invulnerable = false;
    int attack_strength_ticks = 0;
    int attack_miss_time = 0;
    WeaponItem weapon = WeaponItem::diamond_sword;
    double attack_damage = 7.0;
    double attack_speed = 4.0 + double(-2.4F);
    double attack_knockback = 0.0;
    bool shield_equipped = false;
    bool using_shield = false;
    int shield_use_ticks = 0;
    int shield_cooldown = 0;
    int item_use_delay = 0;
    bool attack_suppressed = false;
    float last_blocked_damage = 0.0F;
    bool last_shield_disabled = false;
    AttackOutcome last_attack;

    MCSIM_HD bool alive() const { return health > 0.0F; }
    MCSIM_HD bool blocking() const {
        return alive() && shield_equipped && using_shield && shield_cooldown == 0 && shield_use_ticks >= 5;
    }

    MCSIM_HD float attackCharge() const {
        const float delay = static_cast<float>(1.0 / attack_speed * 20.0);
        const float charge = (static_cast<float>(attack_strength_ticks) + 0.5F) / delay;
        return static_cast<float>(maximum(0.0, minimum(1.0, charge)));
    }

    MCSIM_HD Vec3 eyePosition() const {
        const float height = pose == PlayerPose::standing ? 1.62F
            : pose == PlayerPose::crouching ? 1.27F : 0.4F;
        return {position.x, position.y + double(height), position.z};
    }

    MCSIM_HD Vec3 lookDirection(MathView math) const {
        const float pitch_angle = pitch * 0.017453292F;
        const float yaw_angle = -yaw * 0.017453292F;
        const float pitch_cosine = math.cos(pitch_angle);
        return {double(math.sin(yaw_angle) * pitch_cosine), double(-math.sin(pitch_angle)),
                double(math.cos(yaw_angle) * pitch_cosine)};
    }

    MCSIM_HD Aabb bounds(PlayerPose requested_pose) const {
        const double half_width = double(0.6F) / 2.0;
        const double height = double(requested_pose == PlayerPose::standing ? 1.8F
            : requested_pose == PlayerPose::crouching ? 1.5F : 0.6F);
        return {{position.x - half_width, position.y, position.z - half_width},
                {position.x + half_width, position.y + height, position.z + half_width}};
    }

    MCSIM_HD Aabb bounds() const { return bounds(pose); }
};

enum class Difficulty : std::uint8_t { peaceful, easy, normal, hard };
enum class DamageKind : std::uint8_t { melee, starvation };

struct WorldState {
    std::uint64_t tick = 0;
    bool natural_regeneration = false;
    bool simulate_food = false;
    Difficulty difficulty = Difficulty::normal;
};

class PlayerView {
public:
    MCSIM_HD PlayerView(PlayerState* state, Controls* queued, Controls* held, MathView math = {}, WorldState* world = nullptr)
        : state_(state), queued_(queued), held_(held), math_(math), world_(world) {}

    MCSIM_HD void setInputs(Controls inputs) { *queued_ = inputs; }
    MCSIM_HD PlayerState& state() const { return *state_; }
    MCSIM_HD const Controls& inputs() const { return *held_; }

    MCSIM_HD void heal(float amount) {
        if (state_->alive() && amount > 0.0F) {
            state_->health = static_cast<float>(minimum(state_->max_health, state_->health + amount));
        }
    }

    MCSIM_HD void addExhaustion(float amount) {
        if (world_ && (world_->simulate_food || world_->natural_regeneration) && !state_->invulnerable) {
            state_->food_exhaustion = static_cast<float>(minimum(state_->food_exhaustion + amount, 40.0F));
        }
    }

    MCSIM_HD void tickFood() {
        if (!world_ || !(world_->simulate_food || world_->natural_regeneration) || !state_->alive()) {
            return;
        }
        if (state_->food_exhaustion > 4.0F) {
            state_->food_exhaustion -= 4.0F;
            if (state_->food_saturation > 0.0F) {
                state_->food_saturation = static_cast<float>(maximum(state_->food_saturation - 1.0F, 0.0F));
            } else if (world_->difficulty != Difficulty::peaceful) {
                state_->food_level = state_->food_level > 0 ? state_->food_level - 1 : 0;
            }
        }
        const bool is_hurt = state_->health < state_->max_health;
        if (world_->natural_regeneration && state_->food_saturation > 0.0F && is_hurt && state_->food_level >= 20) {
            if (++state_->food_tick_timer >= 10) {
                const float spent = static_cast<float>(minimum(state_->food_saturation, 6.0F));
                heal(spent / 6.0F);
                state_->food_exhaustion = static_cast<float>(minimum(state_->food_exhaustion + spent, 40.0F));
                state_->food_tick_timer = 0;
            }
        } else if (world_->natural_regeneration && state_->food_level >= 18 && is_hurt) {
            if (++state_->food_tick_timer >= 80) {
                heal(1.0F);
                state_->food_exhaustion = static_cast<float>(minimum(state_->food_exhaustion + 6.0F, 40.0F));
                state_->food_tick_timer = 0;
            }
        } else if (state_->food_level <= 0) {
            if (++state_->food_tick_timer >= 80) {
                if (state_->health > 10.0F || world_->difficulty == Difficulty::hard
                    || (state_->health > 1.0F && world_->difficulty == Difficulty::normal)) {
                    hurt(1.0F, state_->position, 0.0F, true, DamageKind::starvation);
                }
                state_->food_tick_timer = 0;
            }
        } else {
            state_->food_tick_timer = 0;
        }
    }

    MCSIM_HD void equipWeapon(WeaponItem item) {
        if (state_->weapon == item) {
            return;
        }
        const auto attributes = weaponAttributes(item);
        state_->weapon = item;
        state_->attack_damage = attributes.damage;
        state_->attack_speed = attributes.speed;
        state_->attack_strength_ticks = 0;
    }

    MCSIM_HD void equipArmor(ArmorSlot slot, ArmorMaterial material) {
        state_->armor_items[static_cast<int>(slot)] = material;
        state_->armor = 0.0F;
        state_->armor_toughness = 0.0F;
        state_->armor_knockback_resistance = 0.0;
        for (int index = 0; index < 4; ++index) {
            const auto attributes = armorAttributes(state_->armor_items[index], static_cast<ArmorSlot>(index));
            state_->armor += attributes.defense;
            state_->armor_toughness += attributes.toughness;
            state_->armor_knockback_resistance += attributes.knockback_resistance;
        }
    }

    MCSIM_HD void equipShield(bool equipped) {
        state_->shield_equipped = equipped;
        if (!equipped) {
            state_->using_shield = false;
            state_->shield_use_ticks = 0;
        }
    }

    MCSIM_HD bool blocksMeleeFrom(Vec3 source_position) const {
        if (!state_->blocking()) {
            return false;
        }
        const double direction_x = source_position.x - state_->position.x;
        const double direction_z = source_position.z - state_->position.z;
        const double length = ::sqrt(direction_x * direction_x + direction_z * direction_z);
        const float angle = -state_->yaw * 0.017453292F;
        const double dot = length < 1.0E-4 ? 0.0
            : (direction_x / length) * math_.sin(angle) + (direction_z / length) * math_.cos(angle);
        return !(::acos(dot) > double(0.017453292F * 90.0F));
    }

    MCSIM_HD bool hurt(float damage, Vec3 source_position, float disable_shield_seconds = 0.0F,
                       bool bypass_shield = false, DamageKind kind = DamageKind::melee) {
        state_->last_blocked_damage = 0.0F;
        state_->last_shield_disabled = false;
        if (state_->invulnerable || !state_->alive() || damage == 0.0F) {
            return false;
        }
        if (damage < 0.0F) {
            damage = 0.0F;
        }
        if (damage != damage || damage > (std::numeric_limits<float>::max)()) {
            damage = (std::numeric_limits<float>::max)();
        }
        const bool blocked = damage > 0.0F && kind == DamageKind::melee
            && !bypass_shield && blocksMeleeFrom(source_position);
        if (blocked) {
            state_->last_blocked_damage = damage;
            damage = 0.0F;
            const int cooldown = static_cast<int>(::floor(double(disable_shield_seconds * 20.0F) + 0.5));
            if (cooldown > 0) {
                state_->shield_cooldown = cooldown;
                state_->using_shield = false;
                state_->shield_use_ticks = 0;
                state_->last_shield_disabled = true;
            }
        }
        const bool full_hit = state_->damage_cooldown <= 10;
        float applied_damage = damage;
        if (!full_hit) {
            if (damage <= state_->last_hurt) {
                return false;
            }
            applied_damage = damage - state_->last_hurt;
        }
        state_->last_hurt = damage;
        if (full_hit) {
            state_->damage_cooldown = 20;
            state_->hurt_duration = 10;
            state_->hurt_time = state_->hurt_duration;
        }
        if (kind != DamageKind::starvation) {
            const float toughness = 2.0F + state_->armor_toughness / 4.0F;
            const float effective_armor = static_cast<float>(maximum(state_->armor * 0.2F,
                minimum(20.0F, state_->armor - applied_damage / toughness)));
            applied_damage *= 1.0F - effective_armor / 25.0F;
        }
        const float health_damage = static_cast<float>(maximum(applied_damage - state_->absorption, 0.0F));
        state_->absorption -= applied_damage - health_damage;
        state_->health = static_cast<float>(maximum(0.0F, minimum(state_->max_health, state_->health - health_damage)));
        if (health_damage > 0.0F && kind == DamageKind::melee) {
            addExhaustion(0.1F);
        }
        if (full_hit && !blocked) {
            state_->needs_sync = true;
            if (kind != DamageKind::starvation) {
                knockback(double(0.4F), source_position.x - state_->position.x, source_position.z - state_->position.z);
            }
        }
        if (!state_->alive()) {
            state_->using_shield = false;
            state_->shield_use_ticks = 0;
        }
        return !blocked;
    }

    MCSIM_HD void resolveMelee(PlayerView victim, MathView math, int target_index = -1) {
        auto& outcome = state_->last_attack;
        outcome = {};
        if (!state_->alive() || !victim.state_->alive() || state_ == victim.state_ || state_->using_shield) {
            return;
        }
        outcome.target = target_index;
        outcome.attempted = true;
        outcome.charge = state_->attackCharge();
        const bool full_strength = outcome.charge > 0.9F;
        outcome.sprint_bonus = state_->sprinting && full_strength;
        outcome.critical = full_strength && state_->fall_distance > 0.0 && !state_->on_ground
            && !state_->mobility_restricted && !state_->sprinting;
        const double sweep_speed = static_cast<float>(state_->movement_speed) * 2.5;
        outcome.sweep = weaponAttributes(state_->weapon).sweep && full_strength
            && !outcome.critical && !outcome.sprint_bonus && state_->on_ground
            && state_->velocity.x * state_->velocity.x + state_->velocity.z * state_->velocity.z < sweep_speed * sweep_speed;
        float damage = static_cast<float>(state_->attack_damage);
        damage *= 0.2F + outcome.charge * outcome.charge * 0.8F;
        state_->attack_strength_ticks = 0;
        if (damage <= 0.0F) {
            return;
        }
        if (outcome.critical) {
            damage *= 1.5F;
        }
        outcome.incoming_damage = damage;
        const float old_health = victim.state_->health;
        outcome.accepted = victim.hurt(damage, state_->position, weaponAttributes(state_->weapon).disable_shield_seconds);
        outcome.health_lost = old_health - victim.state_->health;
        outcome.blocked_damage = victim.state_->last_blocked_damage;
        outcome.shield_disabled = victim.state_->last_shield_disabled;
        if (!outcome.accepted) {
            return;
        }
        addExhaustion(0.1F);
        const float extra_power = static_cast<float>(state_->attack_knockback) / 2.0F
            + (outcome.sprint_bonus ? 0.5F : 0.0F);
        if (extra_power > 0.0F) {
            const float angle = state_->yaw * 0.017453292F;
            victim.knockback(double(extra_power), math.sin(angle), -math.cos(angle));
            state_->velocity.x *= 0.6;
            state_->velocity.z *= 0.6;
            state_->sprinting = false;
        }
    }

    MCSIM_HD void knockback(double power, double direction_x, double direction_z) {
        power *= 1.0 - minimum(1.0, state_->knockback_resistance + state_->armor_knockback_resistance);
        if (power <= 0.0) {
            return;
        }
        state_->needs_sync = true;
        while (direction_x * direction_x + direction_z * direction_z < double(1.0E-5F)) {
            const double first_x = state_->random.nextDouble();
            const double second_x = state_->random.nextDouble();
            const double first_z = state_->random.nextDouble();
            const double second_z = state_->random.nextDouble();
            direction_x = (first_x - second_x) * 0.01;
            direction_z = (first_z - second_z) * 0.01;
        }
        const double length = ::sqrt(direction_x * direction_x + direction_z * direction_z);
        state_->velocity.x = state_->velocity.x / 2.0 - (direction_x / length) * power;
        state_->velocity.z = state_->velocity.z / 2.0 - (direction_z / length) * power;
        if (state_->on_ground) {
            state_->velocity.y = minimum(0.4, state_->velocity.y / 2.0 + power);
        }
    }

    MCSIM_HD void push(PlayerState& other) {
        if (state_->no_physics || other.no_physics || state_ == &other) {
            return;
        }
        double impulse_x = other.position.x - state_->position.x;
        double impulse_z = other.position.z - state_->position.z;
        const double largest = maximum(absolute(impulse_x), absolute(impulse_z));
        if (largest < double(0.01F)) {
            return;
        }
        const double length = ::sqrt(largest);
        impulse_x /= length;
        impulse_z /= length;
        const double scale = minimum(1.0, 1.0 / length);
        impulse_x *= scale;
        impulse_z *= scale;
        impulse_x *= double(0.05F);
        impulse_z *= double(0.05F);
        if (state_->pushable) {
            state_->velocity.x -= impulse_x;
            state_->velocity.z -= impulse_z;
            state_->needs_sync = true;
        }
        if (other.pushable) {
            other.velocity.x += impulse_x;
            other.velocity.z += impulse_z;
            other.needs_sync = true;
        }
    }

    MCSIM_HD void prepareTick() {
        if (held_->select_weapon) {
            equipWeapon(held_->weapon);
        }
        state_->last_attack = {};
        state_->last_blocked_damage = 0.0F;
        state_->last_shield_disabled = false;
        state_->attack_suppressed = state_->using_shield;
        if (state_->item_use_delay > 0) {
            --state_->item_use_delay;
        }
        if (!held_->use_item || !state_->shield_equipped || !state_->alive()) {
            state_->using_shield = false;
            state_->shield_use_ticks = 0;
        }
        if (state_->damage_cooldown > 0) {
            --state_->damage_cooldown;
        }
        if (state_->hurt_time > 0) {
            --state_->hurt_time;
        }
        if (state_->attack_miss_time > 0) {
            --state_->attack_miss_time;
        }
        if (!held_->attack) {
            state_->attack_miss_time = 0;
        }
        if (!state_->alive() && state_->death_time < 20) {
            ++state_->death_time;
        }
        state_->yaw += held_->yaw_delta;
        state_->pitch += held_->pitch_delta;
        state_->pitch = static_cast<float>(maximum(-90.0, minimum(90.0, state_->pitch)));
    }

    MCSIM_HD void tick(BlockWorldView blocks, MathView math, PlayerState* players = nullptr,
                       int player_count = 0, bool prepared = false) {
        if (!prepared) {
            prepareTick();
        }
        if (held_->use_item && !state_->using_shield && state_->item_use_delay == 0 && state_->alive()) {
            state_->item_use_delay = 4;
            if (state_->shield_equipped && state_->shield_cooldown == 0) {
                state_->using_shield = true;
                state_->shield_use_ticks = 0;
            }
        }
        updateLocalMovementState(blocks);
        updateTickState();
        if (world_ && world_->natural_regeneration && world_->difficulty == Difficulty::peaceful && state_->alive()) {
            if (state_->tick % 20 == 0) {
                heal(1.0F);
                if (state_->food_saturation < 20.0F) state_->food_saturation += 1.0F;
            }
            if (state_->tick % 10 == 0 && state_->food_level < 20) ++state_->food_level;
        }
        const Vec3 movement_input = processMovementInputs(math);
        const Vec3 previous_position = state_->position;
        travel(blocks, math, movement_input);
        if (state_->on_ground && state_->sprinting) {
            const double distance_x = state_->position.x - previous_position.x;
            const double distance_z = state_->position.z - previous_position.z;
            const float centimeters = static_cast<float>(::sqrt(distance_x * distance_x + distance_z * distance_z)) * 100.0F;
            const int rounded_distance = static_cast<int>(::floor(double(centimeters) + 0.5));
            if (rounded_distance > 0) addExhaustion(0.1F * rounded_distance * 0.01F);
        }
        for (int index = 0; index < player_count; ++index) {
            auto& other = players[index];
            if (&other != state_ && state_->alive() && other.alive() && other.pushable
                && state_->bounds().intersects(other.bounds())) {
                push(other);
            }
        }
        updatePostMovementState(blocks);
        tickFood();
    }

private:
    MCSIM_HD bool fitsPose(BlockWorldView blocks, PlayerPose pose) const {
        auto box = state_->bounds(pose);
        box.lower.x += 1.0E-7;
        box.lower.y += 1.0E-7;
        box.lower.z += 1.0E-7;
        box.upper.x -= 1.0E-7;
        box.upper.y -= 1.0E-7;
        box.upper.z -= 1.0E-7;
        return noBlockCollision(blocks, box);
    }

    MCSIM_HD void updateLocalMovementState(BlockWorldView blocks) {
        if (state_->sprint_trigger_time > 0) {
            --state_->sprint_trigger_time;
        }
        state_->crouching = fitsPose(blocks, PlayerPose::crouching)
            && (state_->previous_sneak || !fitsPose(blocks, PlayerPose::standing));
        const bool forward = held_->forward && !held_->backward;
        const bool possible = state_->alive() && (state_->food_level > 6 || state_->may_fly) && !state_->mobility_restricted;
        if (state_->previous_sneak || held_->backward || state_->using_shield) {
            state_->sprint_trigger_time = 0;
        }
        if (!state_->sprinting && forward && possible && !state_->crouching && !state_->using_shield
            && state_->pose != PlayerPose::crawling) {
            if (!state_->previous_forward) {
                if (state_->sprint_trigger_time > 0) {
                    state_->sprinting = true;
                } else {
                    state_->sprint_trigger_time = state_->sprint_window;
                }
            }
            if (held_->sprint) {
                state_->sprinting = true;
            }
        }
        if (!possible || !forward || (state_->horizontal_collision && !state_->minor_horizontal_collision)) {
            state_->sprinting = false;
        }
        state_->previous_forward = forward;
        state_->previous_sneak = held_->sneak;
    }

    MCSIM_HD void updateTickState() {
        ++state_->tick;
        if (state_->jump_delay > 0) {
            --state_->jump_delay;
        }
        if (state_->velocity.x * state_->velocity.x + state_->velocity.z * state_->velocity.z < 9.0E-6) {
            state_->velocity.x = 0.0;
            state_->velocity.z = 0.0;
        }
        if (absolute(state_->velocity.y) < 0.003) {
            state_->velocity.y = 0.0;
        }
    }

    MCSIM_HD Vec3 processMovementInputs(MathView math) {
        if (!state_->alive()) {
            state_->jump_delay = 0;
            state_->sprinting = false;
            return {};
        }
        float strafe = float(held_->left) - float(held_->right);
        float forward = float(held_->forward) - float(held_->backward);
        const float raw_length = static_cast<float>(::sqrt(double(strafe * strafe + forward * forward)));
        if (raw_length >= 1.0E-4F) {
            strafe = (strafe / raw_length) * 0.98F;
            forward = (forward / raw_length) * 0.98F;
            if (state_->using_shield) {
                strafe *= 0.2F;
                forward *= 0.2F;
            }
            if (state_->crouching || state_->pose == PlayerPose::crawling) {
                strafe *= 0.3F;
                forward *= 0.3F;
            }
            const float length = static_cast<float>(::sqrt(double(strafe * strafe + forward * forward)));
            const float inverse_length = 1.0F / length;
            strafe *= inverse_length;
            forward *= inverse_length;
            const float strafe_absolute = static_cast<float>(absolute(strafe));
            const float forward_absolute = static_cast<float>(absolute(forward));
            const float tangent = forward_absolute > strafe_absolute
                ? strafe_absolute / forward_absolute : forward_absolute / strafe_absolute;
            const float square_distance = static_cast<float>(::sqrt(double(1.0F + tangent * tangent)));
            const float modified_length = static_cast<float>(minimum(length * square_distance, 1.0F));
            strafe *= modified_length;
            forward *= modified_length;
        }
        if (!held_->jump) {
            state_->jump_delay = 0;
        } else if (state_->on_ground && state_->jump_delay == 0) {
            addExhaustion(state_->sprinting ? 0.2F : 0.05F);
            state_->velocity.y = maximum(double(0.42F), state_->velocity.y);
            if (state_->sprinting) {
                const float angle = state_->yaw * 0.017453292F;
                state_->velocity.x += -math.sin(angle) * 0.2;
                state_->velocity.z += math.cos(angle) * 0.2;
            }
            state_->jump_delay = 10;
        }
        return {double(strafe), 0.0, double(forward)};
    }

    MCSIM_HD void moveRelative(MathView math, Vec3 input, float acceleration) {
        const double length_squared = input.lengthSquared();
        if (length_squared < 1.0E-7) {
            return;
        }
        if (length_squared > 1.0) {
            const double length = ::sqrt(length_squared);
            input = {input.x / length, input.y / length, input.z / length};
        }
        input = {input.x * acceleration, input.y * acceleration, input.z * acceleration};
        const float radians = state_->yaw * 0.017453292F;
        const float sine = math.sin(radians);
        const float cosine = math.cos(radians);
        state_->velocity.x += input.x * cosine - input.z * sine;
        state_->velocity.y += input.y;
        state_->velocity.z += input.z * cosine + input.x * sine;
    }

    MCSIM_HD bool canFallAtLeast(BlockWorldView blocks, double delta_x, double delta_z, double height) const {
        const auto box = state_->bounds();
        return noBlockCollision(blocks,
            {{box.lower.x + 1.0E-7 + delta_x, box.lower.y - height - 1.0E-7, box.lower.z + 1.0E-7 + delta_z},
             {box.upper.x - 1.0E-7 + delta_x, box.lower.y, box.upper.z - 1.0E-7 + delta_z}});
    }

    MCSIM_HD Vec3 backOffFromEdge(BlockWorldView blocks, Vec3 movement) const {
        const double step_height = double(0.6F);
        const bool above_ground = state_->on_ground || (state_->fall_distance < step_height
            && !canFallAtLeast(blocks, 0.0, 0.0, step_height - state_->fall_distance));
        if (!state_->alive() || !held_->sneak || movement.y > 0.0 || !above_ground) {
            return movement;
        }
        const double step_x = movement.x < 0.0 ? -0.05 : 0.05;
        const double step_z = movement.z < 0.0 ? -0.05 : 0.05;
        while (movement.x != 0.0 && canFallAtLeast(blocks, movement.x, 0.0, step_height)) {
            movement.x = absolute(movement.x) <= 0.05 ? 0.0 : movement.x - step_x;
        }
        while (movement.z != 0.0 && canFallAtLeast(blocks, 0.0, movement.z, step_height)) {
            movement.z = absolute(movement.z) <= 0.05 ? 0.0 : movement.z - step_z;
        }
        while (movement.x != 0.0 && movement.z != 0.0
               && canFallAtLeast(blocks, movement.x, movement.z, step_height)) {
            movement.x = absolute(movement.x) <= 0.05 ? 0.0 : movement.x - step_x;
            movement.z = absolute(movement.z) <= 0.05 ? 0.0 : movement.z - step_z;
        }
        return movement;
    }

    MCSIM_HD void moveAndCollide(BlockWorldView blocks, MathView math, Vec3 input) {
        const Aabb bounds = state_->bounds();
        const Vec3 requested = backOffFromEdge(blocks, state_->velocity);
        const Vec3 accepted = collide(blocks, bounds, requested);
        const double accepted_squared = accepted.lengthSquared();
        if (accepted_squared > 1.0E-7 || requested.lengthSquared() - accepted_squared < 1.0E-7) {
            state_->position.x += accepted.x;
            state_->position.y += accepted.y;
            state_->position.z += accepted.z;
        }
        const bool collision_x = absolute(requested.x - accepted.x) >= double(1.0E-5F);
        const bool collision_z = absolute(requested.z - accepted.z) >= double(1.0E-5F);
        state_->horizontal_collision = collision_x || collision_z;
        state_->minor_horizontal_collision = false;
        if (state_->horizontal_collision) {
            const float angle = state_->yaw * 0.017453292F;
            const double desired_x = input.x * math.cos(angle) - input.z * math.sin(angle);
            const double desired_z = input.z * math.cos(angle) + input.x * math.sin(angle);
            const double desired_squared = desired_x * desired_x + desired_z * desired_z;
            const double accepted_horizontal = accepted.x * accepted.x + accepted.z * accepted.z;
            if (desired_squared >= double(1.0E-5F) && accepted_horizontal >= double(1.0E-5F)) {
                const double cosine = (desired_x * accepted.x + desired_z * accepted.z)
                    / ::sqrt(desired_squared * accepted_horizontal);
                state_->minor_horizontal_collision = ::acos(cosine) < double(0.13962634F);
            }
        }
        state_->vertical_collision = requested.y != accepted.y;
        state_->on_ground = state_->vertical_collision && requested.y < 0.0;
        if (state_->on_ground) {
            state_->fall_distance = 0.0;
        } else if (accepted.y < 0.0) {
            state_->fall_distance -= accepted.y;
        }
        if (collision_x) {
            state_->velocity.x = -state_->velocity.x * 0.0;
        }
        if (collision_z) {
            state_->velocity.z = -state_->velocity.z * 0.0;
        }
        if (state_->vertical_collision) {
            state_->velocity.y = (0.0 - state_->velocity.y) * 0.0;
        }
    }

    MCSIM_HD void travel(BlockWorldView blocks, MathView math, Vec3 input) {
        const float block_friction = state_->on_ground ? 0.6F : 1.0F;
        const float speed = static_cast<float>(state_->movement_speed
            * (state_->sprinting ? 1.0 + double(0.3F) : 1.0));
        const float acceleration = state_->on_ground
            ? (block_friction > 0.6 ? speed * (0.21600002F / (block_friction * block_friction * block_friction)) : speed)
            : (state_->sprinting ? 0.025999999F : 0.02F);
        moveRelative(math, input, acceleration);
        moveAndCollide(blocks, math, input);
        const float horizontal_drag = block_friction * 0.91F;
        state_->velocity.x *= horizontal_drag;
        state_->velocity.y = (state_->velocity.y - 0.08) * double(0.98F);
        state_->velocity.z *= horizontal_drag;
    }

    MCSIM_HD void updatePostMovementState(BlockWorldView blocks) {
        if (state_->using_shield && state_->shield_use_ticks < (std::numeric_limits<int>::max)()) {
            ++state_->shield_use_ticks;
            if (state_->shield_use_ticks >= 72000) {
                state_->using_shield = false;
                state_->shield_use_ticks = 0;
            }
        }
        if (state_->shield_cooldown > 0) {
            --state_->shield_cooldown;
        }
        if (state_->attack_strength_ticks < (std::numeric_limits<int>::max)()) {
            ++state_->attack_strength_ticks;
        }
        state_->position.x = maximum(-29999999.0, minimum(29999999.0, state_->position.x));
        state_->position.z = maximum(-29999999.0, minimum(29999999.0, state_->position.z));
        if (fitsPose(blocks, PlayerPose::crawling)) {
            const auto desired = held_->sneak ? PlayerPose::crouching : PlayerPose::standing;
            state_->pose = fitsPose(blocks, desired) ? desired
                : fitsPose(blocks, PlayerPose::crouching) ? PlayerPose::crouching : PlayerPose::crawling;
        }
    }

    PlayerState* state_;
    Controls* queued_;
    Controls* held_;
    MathView math_;
    WorldState* world_;
};

struct BatchView;

class WorldView {
public:
    MCSIM_HD WorldView(PlayerState* players, Controls* queued, Controls* held,
                                            WorldState* state, int player_count, BlockWorldView blocks, MathView math)
        : players_(players), queued_(queued), held_(held), state_(state),
                    player_count_(player_count), blocks_(blocks), math_(math) {}

    MCSIM_HD PlayerView player(int index) const {
        return {players_ + index, queued_ + index, held_ + index, math_, state_};
    }

    MCSIM_HD int pickAttackTarget(int attacker, bool* hit_block = nullptr) const {
        if (hit_block) {
            *hit_block = false;
        }
        const auto& source = players_[attacker];
        if (source.health <= 0.0F) {
            return -1;
        }
        const Vec3 origin = source.eyePosition();
        const Vec3 look = source.lookDirection(math_);
        const double projection = source.velocity.x * look.x + source.velocity.y * look.y + source.velocity.z * look.z;
        const double reach = 3.0 + maximum(0.0, projection);
        Vec3 delta{look.x * reach, look.y * reach, look.z * reach};
        const double block_fraction = blockRayFraction(blocks_, origin, delta);
        delta = {delta.x * block_fraction, delta.y * block_fraction, delta.z * block_fraction};
        double nearest = 2.0;
        int target = -1;
        for (int index = 0; index < player_count_; ++index) {
            if (index == attacker || players_[index].health <= 0.0F) {
                continue;
            }
            double fraction = 1.0;
            if (rayBoxHit(players_[index].bounds(), origin, delta, fraction) && fraction < nearest) {
                nearest = fraction;
                target = index;
            }
        }
        if (hit_block) {
            *hit_block = target < 0 && block_fraction < 1.0;
        }
        return target;
    }

    MCSIM_HD const WorldState& state() const { return *state_; }

    MCSIM_HD void processTick() {
        latchInputs();
        for (int index = 0; index < player_count_; ++index) {
            player(index).prepareTick();
        }
        for (int index = 0; index < player_count_; ++index) {
            auto& state = players_[index];
            if (held_[index].attack && state.alive() && state.attack_miss_time == 0 && !state.attack_suppressed) {
                state.last_attack.attempted = true;
                state.last_attack.target = pickAttackTarget(index, &state.last_attack.hit_block);
            }
        }
        for (int index = 0; index < player_count_; ++index) {
            auto& state = players_[index];
            if (!state.last_attack.attempted || !state.alive()) {
                continue;
            }
            const int target = state.last_attack.target;
            if (target >= 0) {
                player(index).resolveMelee(player(target), math_, target);
                if (state.last_attack.accepted && state.last_attack.sweep) {
                    sweepAttack(index, target);
                }
            } else if (!state.last_attack.hit_block) {
                state.last_attack.charge = state.attackCharge();
                state.attack_strength_ticks = 0;
                state.attack_miss_time = 10;
            }
        }
        for (int index = 0; index < player_count_; ++index) {
            player(index).tick(blocks_, math_, players_, player_count_, true);
        }
        ++state_->tick;
    }

private:
    MCSIM_HD void sweepAttack(int attacker, int primary) {
        const auto primary_box = players_[primary].bounds();
        const Aabb area{{primary_box.lower.x - 1.0, primary_box.lower.y - 0.25, primary_box.lower.z - 1.0},
                        {primary_box.upper.x + 1.0, primary_box.upper.y + 0.25, primary_box.upper.z + 1.0}};
        const auto& source = players_[attacker];
        const float angle = source.yaw * 0.017453292F;
        for (int index = 0; index < player_count_; ++index) {
            const auto& candidate = players_[index];
            const Vec3 difference{candidate.position.x - source.position.x, candidate.position.y - source.position.y,
                                  candidate.position.z - source.position.z};
            if (index != attacker && index != primary && candidate.alive() && area.intersects(candidate.bounds())
                && difference.lengthSquared() < 9.0 && player(index).hurt(1.0F, source.position)) {
                player(index).knockback(double(0.4F), math_.sin(angle), -math_.cos(angle));
                ++players_[attacker].last_attack.sweep_hits;
            }
        }
    }

    MCSIM_HD void latchInputs() {
        for (int index = 0; index < player_count_; ++index) {
            held_[index] = queued_[index];
            queued_[index].yaw_delta = 0.0F;
            queued_[index].pitch_delta = 0.0F;
            queued_[index].attack = false;
        }
    }

    PlayerState* players_;
    Controls* queued_;
    Controls* held_;
    WorldState* state_;
    int player_count_;
    BlockWorldView blocks_;
    MathView math_;
};

struct BatchView {
    PlayerState* players = nullptr;
    Controls* queued_controls = nullptr;
    Controls* held_controls = nullptr;
    WorldState* worlds = nullptr;
    int environment_count = 0;
    int players_per_environment = 0;
    const BlockWorldView* block_worlds = nullptr;
    MathView math;

    MCSIM_HD WorldView world(int index) const {
        const int offset = index * players_per_environment;
        return {players + offset, queued_controls + offset,
            held_controls + offset, worlds + index, players_per_environment,
            block_worlds ? block_worlds[index] : BlockWorldView{}, math};
    }
};

}