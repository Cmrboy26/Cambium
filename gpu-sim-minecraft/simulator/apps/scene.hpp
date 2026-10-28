#pragma once

#include "configuration.hpp"
#include "minecraft_sim/projectiles.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

namespace minecraft_sim::app {

struct Scene {
    struct PlayerDefinitions {
        std::array<PlayerState, 2> players{};
        std::array<std::vector<WeaponItem>, 2> weapon_slots;
        nlohmann::json resolved;
    };

    nlohmann::json definition;
    std::array<PlayerState, 2> initial_players{};
    std::array<std::vector<WeaponItem>, 2> weapon_slots;
    std::array<PlayerState, 1> players{};
    std::array<Controls, 1> queued{};
    std::array<Controls, 1> held{};
    std::array<WorldState, 1> worlds{};
    std::vector<ArrowState> arrow_storage;
    std::vector<ThrownProjectileState> thrown_storage;
    std::vector<WindChargeState> wind_storage;
    std::array<ProjectileWorldState, 1> projectile_states{};
    std::vector<Block> cells;
    SineTable sine = makeSineTable();
    BlockWorldView blocks;

    static nlohmann::json readEnvironment(const std::filesystem::path& path) {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("Cannot open environment: " + path.string());
        }
        return nlohmann::json::parse(input);
    }

    static const nlohmann::json& defaultEnvironment() {
        static const auto cached = readEnvironment(MCSIM_DEFAULT_ENVIRONMENT_PATH);
        return cached;
    }

    explicit Scene(const std::filesystem::path& path = {})
        : Scene(path.empty() ? defaultEnvironment() : readEnvironment(path)) {}

    explicit Scene(nlohmann::json environment) : definition(std::move(environment)) {
        if (definition.at("format") != "minecraft_sim_environment" || definition.at("version") != 1
            || definition.at("minecraft") != "26.3"
            || (definition.value("profile", "configured") != "unarmored_diamond_sword"
                && definition.value("profile", "configured") != "unarmored_sword_axe_shield"
                && definition.value("profile", "configured") != "configured")) {
            throw std::invalid_argument("Unsupported environment format/version/build/profile");
        }
        for (const auto& capability : definition.value("requires", nlohmann::json::array())) {
            if (capability != "movement" && capability != "melee" && capability != "equipment"
                && capability != "shield" && capability != "armor" && capability != "bow") {
                throw std::invalid_argument("Unsupported scenario capability: " + capability.dump());
            }
        }
        const auto rules = definition.value("rules", nlohmann::json::object());
        if (!rules.is_object()) throw std::invalid_argument("Scenario rules must be an object");
        for (const auto& rule : rules.items()) {
            if (rule.key() == "simulate_fall_damage" && rule.value().is_boolean()) {
                worlds[0].simulate_fall_damage = rule.value().get<bool>();
            } else if ((rule.key() != "simulate_food" && rule.key() != "natural_regeneration") || rule.value() != false) {
                throw std::invalid_argument("Scenario bridge requires frozen food and no natural regeneration");
            }
        }
        const auto integer_vector = [](const nlohmann::json& value) {
            std::array<int, 3> result{};
            if (!value.is_array() || value.size() != 3) {
                throw std::invalid_argument("Environment coordinates need three integers");
            }
            for (int axis = 0; axis < 3; ++axis) {
                if (!value[axis].is_number_integer() || value[axis] < -4096 || value[axis] > 4096) {
                    throw std::invalid_argument("Environment coordinates must be integers in -4096..4096");
                }
                result[axis] = value[axis].get<int>();
            }
            return result;
        };
        const auto origin = integer_vector(definition.at("bounds").at("min"));
        const auto size = integer_vector(definition.at("bounds").at("size"));
        std::size_t volume = 1;
        for (int extent : size) {
            if (extent < 1 || extent > 128) {
                throw std::invalid_argument("Environment dimensions must be in 1..128");
            }
            volume *= extent;
        }
        if (volume > 32768) {
            throw std::invalid_argument("Environment exceeds 32768 cells");
        }
        cells.resize(volume, Block::air);
        blocks = {cells.data(), size[0], size[1], size[2], origin[0], origin[1], origin[2]};
        const auto& fills = definition.at("blocks");
        if (!fills.is_array() || fills.size() > 256) {
            throw std::invalid_argument("Environment blocks must be at most 256 fill regions");
        }
        for (const auto& fill : fills) {
            const auto lower = integer_vector(fill.at("from"));
            const auto upper = integer_vector(fill.at("to"));
            const auto block_name = fill.at("block").get<std::string>();
            if (block_name != "minecraft:air" && block_name != "minecraft:stone") {
                throw std::invalid_argument("Only minecraft:air and minecraft:stone are supported");
            }
            for (int axis = 0; axis < 3; ++axis) {
                if (lower[axis] < origin[axis] || lower[axis] > upper[axis]
                    || upper[axis] >= origin[axis] + size[axis]) {
                    throw std::invalid_argument("Environment fill is outside bounds");
                }
            }
            for (int block_z = lower[2]; block_z <= upper[2]; ++block_z) {
                for (int block_y = lower[1]; block_y <= upper[1]; ++block_y) {
                    for (int block_x = lower[0]; block_x <= upper[0]; ++block_x) {
                        setBlock(block_x, block_y, block_z, block_name == "minecraft:air" ? Block::air : Block::solid);
                    }
                }
            }
        }
        const auto parsed = parsePlayers(definition.at("players"));
        initial_players = parsed.players;
        weapon_slots = parsed.weapon_slots;
        players[0] = initial_players[0];
        arrow_storage.resize(projectile_arrow_capacity);
        projectile_states[0].arrows = arrow_storage.data();
        projectile_states[0].arrow_capacity = projectile_arrow_capacity;
        for (const auto& player : initial_players) {
            if (player.snowball_count || player.egg_count || player.ender_pearl_count) {
                thrown_storage.resize(96);
                projectile_states[0].thrown_items = thrown_storage.data();
                projectile_states[0].thrown_item_capacity = static_cast<int>(thrown_storage.size());
            }
            if (player.wind_charge_count) {
                wind_storage.resize(128);
                projectile_states[0].wind_charges = wind_storage.data();
                projectile_states[0].wind_charge_capacity = static_cast<int>(wind_storage.size());
            }
        }
        resetProjectileWorld(projectile_states[0], 1);
        for (int player = 0; player < 2; ++player) {
            projectile_states[0].bows[player].ordinary_arrows = initial_players[player].ordinary_arrow_count;
        }
    }

    PlayerDefinitions parsePlayers(const nlohmann::json& spawns, bool concrete = false) const {
        if (!spawns.is_array() || spawns.size() != 2) {
            throw std::invalid_argument("Environment needs two initial players");
        }
        const auto vector = [](const nlohmann::json& value, double limit) {
            if (!value.is_array() || value.size() != 3) {
                throw std::invalid_argument("Environment state vectors need three numbers");
            }
            for (const auto& number : value) {
                if (!number.is_number()) throw std::invalid_argument("Environment state vectors must be numeric");
            }
            Vec3 result{value[0].get<double>(), value[1].get<double>(), value[2].get<double>()};
            for (int axis = 0; axis < 3; ++axis) {
                if (!std::isfinite(component(result, axis)) || std::abs(component(result, axis)) > limit) {
                    throw std::invalid_argument("Invalid environment state vector");
                }
            }
            return result;
        };
        PlayerDefinitions parsed;
        parsed.resolved = nlohmann::json::array();
        for (int index = 0; index < 2; ++index) {
            auto& state = parsed.players[index];
            const auto& spawn = spawns[index];
            if (!spawn.is_object()) throw std::invalid_argument("Initial player must be an object");
            if (concrete) {
                for (const auto& field : spawn.items()) {
                    if (field.key() != "position" && field.key() != "velocity" && field.key() != "yaw"
                        && field.key() != "pitch" && field.key() != "on_ground" && field.key() != "health"
                        && field.key() != "absorption" && field.key() != "attack_strength_ticks" && field.key() != "loadout") {
                        throw std::invalid_argument("Unsupported initial player field: " + field.key());
                    }
                }
                if (!spawn.contains("loadout") || !spawn.at("loadout").is_object()
                    || !spawn.at("loadout").contains("weapons")) {
                    throw std::invalid_argument("Concrete player requires a weapon loadout");
                }
            }
            state.position = vector(spawn.at("position"), 8192);
            state.velocity = vector(spawn.at("velocity"), 1000);
            if (!spawn.at("yaw").is_number() || !spawn.at("pitch").is_number()
                || (spawn.contains("health") && !spawn.at("health").is_number())
                || (spawn.contains("absorption") && !spawn.at("absorption").is_number())
                || (spawn.contains("attack_strength_ticks") && (!spawn.at("attack_strength_ticks").is_number_integer()
                    || spawn.at("attack_strength_ticks") < 0
                    || spawn.at("attack_strength_ticks") > std::numeric_limits<int>::max()))) {
                throw std::invalid_argument("Initial player scalar fields have invalid types");
            }
            state.yaw = spawn.at("yaw").get<float>();
            state.pitch = spawn.at("pitch").get<float>();
            state.on_ground = spawn.at("on_ground").get<bool>();
            state.health = spawn.value("health", 20.0F);
            state.absorption = spawn.value("absorption", 0.0F);
            state.attack_strength_ticks = spawn.value("attack_strength_ticks", 0);
            nlohmann::json fallback = {{"weapons", {"minecraft:diamond_sword"}}};
            if (definition.value("profile", "configured") == "unarmored_sword_axe_shield") {
                fallback = {{"weapons", {"minecraft:diamond_sword", "minecraft:diamond_axe"}}, {"offhand", "minecraft:shield"}};
            }
            parsed.weapon_slots[index] = applyLoadout(state, spawn.value("loadout", fallback));
            if (!std::isfinite(state.yaw) || !std::isfinite(state.pitch) || std::abs(state.pitch) > 90
                || !std::isfinite(state.health) || state.health <= 0 || state.health > 20
                || !std::isfinite(state.absorption) || state.absorption < 0 || state.attack_strength_ticks < 0) {
                throw std::invalid_argument("Invalid environment initial player state");
            }
            for (int axis = 0; axis < 3; ++axis) {
                const int origin = axis == 0 ? blocks.origin_x : axis == 1 ? blocks.origin_y : blocks.origin_z;
                const int size = axis == 0 ? blocks.size_x : axis == 1 ? blocks.size_y : blocks.size_z;
                if (component(state.position, axis) < origin
                    || component(state.position, axis) >= origin + size) {
                    throw std::invalid_argument("Initial player position is outside environment bounds");
                }
            }
            if (concrete && !noBlockCollision(blocks, state.bounds())) {
                throw std::invalid_argument("Initial player intersects terrain");
            }
            parsed.resolved.push_back({
                {"position", {state.position.x, state.position.y, state.position.z}},
                {"velocity", {state.velocity.x, state.velocity.y, state.velocity.z}},
                {"yaw", state.yaw}, {"pitch", state.pitch}, {"on_ground", state.on_ground},
                {"health", state.health}, {"absorption", state.absorption},
                {"attack_strength_ticks", state.attack_strength_ticks},
                {"loadout", describeLoadout(state, parsed.weapon_slots[index],
                    spawn.value("loadout", nlohmann::json::object()).value("selected", 0))}});
        }
        if (concrete && parsed.players[0].bounds().intersects(parsed.players[1].bounds())) {
            throw std::invalid_argument("Initial players intersect each other");
        }
        return parsed;
    }

    nlohmann::json resolvedDefinition() const {
        auto resolved = definition;
        resolved["profile"] = "configured";
        resolved.erase("policy");
        resolved["requires"] = {"movement", "melee", "equipment", "shield", "armor"};
        resolved["rules"] = {{"simulate_food", false}, {"natural_regeneration", false}};
        if (worlds[0].simulate_fall_damage) resolved["rules"]["simulate_fall_damage"] = true;
        const auto parsed = parsePlayers(definition.at("players"));
        resolved["players"] = parsed.resolved;
        for (const auto& slots : parsed.weapon_slots) {
            if (std::find(slots.begin(), slots.end(), WeaponItem::bow) != slots.end()) {
                resolved["requires"].push_back("bow");
                break;
            }
        }
        return resolved;
    }

    void setBlock(int block_x, int block_y, int block_z, Block block = Block::solid) {
        if (block_x < blocks.origin_x || block_x >= blocks.origin_x + blocks.size_x
            || block_y < blocks.origin_y || block_y >= blocks.origin_y + blocks.size_y
            || block_z < blocks.origin_z || block_z >= blocks.origin_z + blocks.size_z) {
            throw std::out_of_range("Block outside environment bounds");
        }
        cells[((block_z - blocks.origin_z) * blocks.size_y + block_y - blocks.origin_y)
            * blocks.size_x + block_x - blocks.origin_x] = block;
    }

    BatchView view() {
        projectile_states[0].arrows = arrow_storage.data();
        return {players.data(), queued.data(), held.data(), worlds.data(),
                1, 1, &blocks, {sine.data()}, nullptr};
    }
};

}