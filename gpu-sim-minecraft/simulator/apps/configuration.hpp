#pragma once

#include "minecraft_sim/core.hpp"
#include "policy_catalogue.generated.hpp"
#include <algorithm>
#include <array>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace minecraft_sim::app {

inline constexpr std::array<const char*, 15> weapon_names{
    "minecraft:air", "minecraft:wooden_sword", "minecraft:stone_sword", "minecraft:copper_sword",
    "minecraft:golden_sword", "minecraft:iron_sword", "minecraft:diamond_sword", "minecraft:netherite_sword",
    "minecraft:wooden_axe", "minecraft:stone_axe", "minecraft:copper_axe", "minecraft:golden_axe",
    "minecraft:iron_axe", "minecraft:diamond_axe", "minecraft:netherite_axe"};
inline constexpr auto armor_slots = armor_slot_names;
inline constexpr std::array<const char*, 9> armor_materials{
    "none", "leather", "copper", "chainmail", "iron", "golden", "diamond", "netherite", "turtle"};

inline const char* weaponName(WeaponItem item) { return weapon_names.at(static_cast<std::size_t>(item)); }

inline WeaponItem parseWeapon(const std::string& name) {
    for (std::size_t index = 0; index < weapon_names.size(); ++index) {
        if (name == weapon_names[index]) return static_cast<WeaponItem>(index);
    }
    throw std::invalid_argument("Unsupported weapon item: " + name);
}

inline std::string armorName(ArmorMaterial material, int slot) {
    if (material == ArmorMaterial::none) return "minecraft:air";
    return "minecraft:" + std::string(armor_materials.at(static_cast<std::size_t>(material))) + "_" + armor_slots.at(slot);
}

inline std::vector<WeaponItem> applyLoadout(PlayerState& state, const nlohmann::json& loadout) {
    if (!loadout.is_object()) throw std::invalid_argument("Loadout must be an object");
    for (const auto& entry : loadout.items()) {
        if (entry.key() != "weapons" && entry.key() != "selected" && entry.key() != "offhand" && entry.key() != "armor") {
            throw std::invalid_argument("Unsupported loadout field: " + entry.key());
        }
    }
    const auto weapons = loadout.value("weapons", nlohmann::json::array({"minecraft:diamond_sword"}));
    if (!weapons.is_array() || weapons.empty() || weapons.size() > 9) {
        throw std::invalid_argument("Loadout needs 1..9 weapon slots");
    }
    std::vector<WeaponItem> slots;
    for (const auto& item : weapons) {
        const auto weapon = parseWeapon(item.get<std::string>());
        if (std::find(slots.begin(), slots.end(), weapon) != slots.end()) {
            throw std::invalid_argument("Weapon slots must be distinct");
        }
        slots.push_back(weapon);
    }
    if (loadout.contains("selected") && !loadout.at("selected").is_number_integer()) {
        throw std::invalid_argument("Selected weapon must be an integer slot");
    }
    const auto selected = loadout.value("selected", 0);
    if (selected < 0 || selected >= static_cast<int>(slots.size())) throw std::invalid_argument("Selected weapon outside loadout");
    const auto offhand = loadout.value("offhand", std::string("minecraft:air"));
    if (offhand != "minecraft:air" && offhand != "minecraft:shield") throw std::invalid_argument("Unsupported offhand item: " + offhand);
    state.weapon = slots[selected];
    const auto attributes = weaponAttributes(state.weapon);
    state.attack_damage = attributes.damage;
    state.attack_speed = attributes.speed;
    state.shield_equipped = offhand == "minecraft:shield";
    Controls queued, held;
    PlayerView player(&state, &queued, &held);
    const auto armor = loadout.value("armor", nlohmann::json::object());
    if (!armor.is_object()) throw std::invalid_argument("Armor must be an object keyed by slot");
    for (const auto& entry : armor.items()) {
        int slot = -1;
        for (int index = 0; index < 4; ++index) if (entry.key() == armor_slots[index]) slot = index;
        if (slot < 0) throw std::invalid_argument("Unsupported armor slot: " + entry.key());
        const auto name = entry.value().get<std::string>();
        bool found = false;
        for (std::size_t material = 0; material < armor_materials.size(); ++material) {
            if (material == static_cast<std::size_t>(ArmorMaterial::turtle) && slot != 3) continue;
            if (name == armorName(static_cast<ArmorMaterial>(material), slot)) {
                player.equipArmor(static_cast<ArmorSlot>(slot), static_cast<ArmorMaterial>(material));
                found = true;
            }
        }
        if (!found) throw std::invalid_argument("Unsupported armor item: " + name);
    }
    return slots;
}

inline nlohmann::json describeLoadout(const PlayerState& state, const std::vector<WeaponItem>& slots) {
    auto weapons = nlohmann::json::array();
    int selected = 0;
    for (std::size_t index = 0; index < slots.size(); ++index) {
        weapons.push_back(weaponName(slots[index]));
        if (slots[index] == state.weapon) selected = static_cast<int>(index);
    }
    auto armor = nlohmann::json::object();
    for (int slot = 0; slot < 4; ++slot) armor[armor_slots[slot]] = armorName(state.armor_items[slot], slot);
    return {{"weapons", weapons}, {"selected", selected}, {"offhand", state.shield_equipped ? "minecraft:shield" : "minecraft:air"}, {"armor", armor}};
}

template<std::size_t Size>
inline int nameIndex(const std::array<const char*, Size>& names, const std::string& name) {
    for (std::size_t index = 0; index < Size; ++index) if (name == names[index]) return static_cast<int>(index);
    throw std::invalid_argument("Unsupported contract primitive: " + name);
}

struct PolicyLayout {
    struct Head { int kind; std::vector<float> values; };
    std::vector<int> observations;
    std::vector<float> scales;
    std::vector<Head> actions;

    PolicyLayout(const nlohmann::json& value, const std::array<std::vector<WeaponItem>, 2>& slots) {
        if (value.at("format") != "minecraft_sim_policy" || value.at("version") != 1
            || value.at("minecraft") != "26.3" || value.at("ticks_per_action") != 1) {
            throw std::invalid_argument("Unsupported policy contract format/version/build/tick rate");
        }
        for (const auto& capability : value.value("requires", nlohmann::json::array())) {
            const auto name = capability.get<std::string>();
            if (std::find(capability_names.begin(), capability_names.end(), name) == capability_names.end()) {
                throw std::invalid_argument("Unsupported required capability: " + name);
            }
        }
        const auto& features = value.at("observations");
        if (!features.is_array() || features.empty() || features.size() > policy_max_observations) throw std::invalid_argument("Invalid observation count");
        for (const auto& feature : features) {
            observations.push_back(nameIndex(feature_names, feature.at("feature").get<std::string>()));
            const auto scale = feature.value("scale", 1.0F);
            if (!std::isfinite(scale) || std::abs(scale) > policy_max_scale) throw std::invalid_argument("Invalid observation scale");
            scales.push_back(scale);
        }
        const auto& heads = value.at("actions");
        if (!heads.is_array() || heads.empty() || heads.size() > action_names.size()) throw std::invalid_argument("Invalid action head count");
        std::array<bool, action_names.size()> seen{};
        for (const auto& descriptor : heads) {
            Head head{nameIndex(action_names, descriptor.at("name").get<std::string>()), {}};
            if (seen[head.kind]) throw std::invalid_argument("Duplicate action primitive");
            seen[head.kind] = true;
            const auto& choices = descriptor.at("values");
            if (!choices.is_array() || choices.empty() || choices.size() > policy_max_choices) throw std::invalid_argument("Invalid action choice count");
            const std::string kind = action_kinds[head.kind];
            for (const auto& choice : choices) {
                float number = 0;
                if (kind == "movement") {
                    number = static_cast<float>(nameIndex(movement_names, choice.get<std::string>()));
                } else if (kind == "look") {
                    if (!choice.is_number()) throw std::invalid_argument("Look choices must be numeric");
                    number = choice.get<float>();
                    if (!std::isfinite(number) || std::abs(number) > policy_max_look_delta) throw std::invalid_argument("Invalid look delta");
                } else if (kind == "weapon") {
                    if (!choice.is_number_integer()) throw std::invalid_argument("Weapon choices must be slot indices");
                    const int slot = choice.get<int>();
                    if (slot < 0 || slot >= static_cast<int>(slots[0].size()) || slot >= static_cast<int>(slots[1].size())) {
                        throw std::invalid_argument("Weapon action outside a player's loadout");
                    }
                    number = static_cast<float>(slot);
                } else {
                    if (!choice.is_boolean()) throw std::invalid_argument("Binary controls need boolean choices");
                    number = choice.get<bool>() ? 1.0F : 0.0F;
                }
                if (std::find(head.values.begin(), head.values.end(), number) != head.values.end()) throw std::invalid_argument("Duplicate action choice");
                head.values.push_back(number);
            }
            actions.push_back(std::move(head));
        }
    }

    Controls decode(const std::int32_t* indices, const std::vector<WeaponItem>& slots) const {
        constexpr int forward[] = {0, 1, 1, 0, -1, -1, -1, 0, 1};
        constexpr int strafe[] = {0, 0, -1, -1, -1, 0, 1, 1, 1};
        Controls result;
        for (std::size_t index = 0; index < actions.size(); ++index) {
            const auto& head = actions[index];
            const auto value = head.values[indices[index]];
            switch (head.kind) {
            case 0: {
                const auto move = static_cast<int>(value);
                result.forward = forward[move] > 0; result.backward = forward[move] < 0;
                result.left = strafe[move] > 0; result.right = strafe[move] < 0;
                break;
            }
            case 1: result.jump = value != 0; break;
            case 2: result.sprint = value != 0; break;
            case 3: result.attack = value != 0; break;
            case 4: result.yaw_delta = value; break;
            case 5: result.pitch_delta = value; break;
            case 6: result.select_weapon = true; result.weapon = slots[static_cast<int>(value)]; break;
            case 7: result.use_item = value != 0; break;
            case 8: result.sneak = value != 0; break;
            }
        }
        return result;
    }
};

}