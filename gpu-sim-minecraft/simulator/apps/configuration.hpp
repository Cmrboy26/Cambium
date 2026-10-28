#pragma once

#include "minecraft_sim/core.hpp"
#include "policy_catalogue.generated.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace minecraft_sim::app {

inline constexpr std::array<const char*, 21> weapon_names{
    "minecraft:air", "minecraft:wooden_sword", "minecraft:stone_sword", "minecraft:copper_sword",
    "minecraft:golden_sword", "minecraft:iron_sword", "minecraft:diamond_sword", "minecraft:netherite_sword",
    "minecraft:wooden_axe", "minecraft:stone_axe", "minecraft:copper_axe", "minecraft:golden_axe",
    "minecraft:iron_axe", "minecraft:diamond_axe", "minecraft:netherite_axe", "minecraft:bow", "minecraft:mace",
    "minecraft:snowball", "minecraft:egg", "minecraft:ender_pearl", "minecraft:wind_charge"};
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
        if (entry.key() != "weapons" && entry.key() != "selected" && entry.key() != "offhand"
            && entry.key() != "armor" && entry.key() != "arrows"
            && entry.key() != "snowballs" && entry.key() != "eggs" && entry.key() != "ender_pearls"
            && entry.key() != "wind_charges") {
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
        if (weapon != WeaponItem::unarmed && std::find(slots.begin(), slots.end(), weapon) != slots.end()) {
            throw std::invalid_argument("Nonempty weapon slots must be distinct");
        }
        slots.push_back(weapon);
    }
    if (loadout.contains("selected") && !loadout.at("selected").is_number_integer()) {
        throw std::invalid_argument("Selected weapon must be an integer slot");
    }
    if (loadout.contains("selected") && (loadout.at("selected") < 0 || loadout.at("selected") >= slots.size())) {
        throw std::invalid_argument("Selected weapon outside loadout");
    }
    const auto selected = loadout.value("selected", 0);
    if (selected < 0 || selected >= static_cast<int>(slots.size())) throw std::invalid_argument("Selected weapon outside loadout");
    if (loadout.contains("arrows") && !loadout.at("arrows").is_number_integer()) {
        throw std::invalid_argument("Ordinary arrow count must be an integer");
    }
    const int arrow_count = loadout.value("arrows", 0);
    if (arrow_count < 0 || arrow_count > 64) throw std::invalid_argument("Ordinary arrow count outside 0..64");
    if (arrow_count > 0 && std::find(slots.begin(), slots.end(), WeaponItem::bow) == slots.end()) {
        throw std::invalid_argument("Ordinary arrows require a bow in the loadout");
    }
    state.selected_weapon_slot = selected;
    state.ordinary_arrow_count = arrow_count;
    const auto stackCount = [&](const char* field, WeaponItem item, int maximum = 16) {
        const bool present = std::find(slots.begin(), slots.end(), item) != slots.end();
        if (loadout.contains(field) && !loadout.at(field).is_number_integer()) {
            throw std::invalid_argument(std::string(field) + " count must be an integer");
        }
        const int count = loadout.value(field, present ? 1 : 0);
        if ((present && (count < 1 || count > maximum)) || (!present && count != 0)) {
            throw std::invalid_argument(std::string(field) + " requires a matching stack of 1.." + std::to_string(maximum) + " items");
        }
        return count;
    };
    state.snowball_count = stackCount("snowballs", WeaponItem::snowball);
    state.egg_count = stackCount("eggs", WeaponItem::egg);
    state.ender_pearl_count = stackCount("ender_pearls", WeaponItem::ender_pearl);
    state.wind_charge_count = stackCount("wind_charges", WeaponItem::wind_charge, 64);
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

inline nlohmann::json describeLoadout(const PlayerState& state, const std::vector<WeaponItem>& slots, int selected = -1) {
    auto weapons = nlohmann::json::array();
    if (selected < 0) selected = static_cast<int>(std::distance(slots.begin(), std::find(slots.begin(), slots.end(), state.weapon)));
    for (std::size_t index = 0; index < slots.size(); ++index) {
        weapons.push_back(weaponName(slots[index]));
    }
    if (selected >= static_cast<int>(slots.size()) || slots[selected] != state.weapon) {
        throw std::invalid_argument("Selected slot disagrees with equipped weapon");
    }
    auto armor = nlohmann::json::object();
    for (int slot = 0; slot < 4; ++slot) armor[armor_slots[slot]] = armorName(state.armor_items[slot], slot);
    nlohmann::json result = {{"weapons", weapons}, {"selected", selected}, {"offhand", state.shield_equipped ? "minecraft:shield" : "minecraft:air"},
        {"armor", armor}, {"arrows", state.ordinary_arrow_count}};
    if (std::find(slots.begin(), slots.end(), WeaponItem::snowball) != slots.end()) result["snowballs"] = state.snowball_count;
    if (std::find(slots.begin(), slots.end(), WeaponItem::egg) != slots.end()) result["eggs"] = state.egg_count;
    if (std::find(slots.begin(), slots.end(), WeaponItem::ender_pearl) != slots.end()) result["ender_pearls"] = state.ender_pearl_count;
    if (std::find(slots.begin(), slots.end(), WeaponItem::wind_charge) != slots.end()) result["wind_charges"] = state.wind_charge_count;
    return result;
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
    std::vector<float> categories;
    std::vector<float> lower_bounds;
    std::vector<float> upper_bounds;
    std::vector<Head> actions;

    PolicyLayout(const nlohmann::json& value, const std::array<std::vector<WeaponItem>, 2>& slots) {
        if (value.at("format") != "minecraft_sim_policy" || value.at("version") != 1
            || value.at("minecraft") != "26.3" || value.at("ticks_per_action") != 1) {
            throw std::invalid_argument("Unsupported policy contract format/version/build/tick rate");
        }
        bool requires_bow = false;
        for (const auto& capability : value.value("requires", nlohmann::json::array())) {
            const auto name = capability.get<std::string>();
            requires_bow = requires_bow || name == "bow";
            if (std::find(capability_names.begin(), capability_names.end(), name) == capability_names.end()) {
                throw std::invalid_argument("Unsupported required capability: " + name);
            }
        }
        const auto& features = value.at("observations");
        if (!features.is_array() || features.empty() || features.size() > policy_max_observations) throw std::invalid_argument("Invalid observation count");
        for (const auto& feature : features) {
            for (const auto& entry : feature.items()) {
                if (entry.key() != "feature" && entry.key() != "scale" && entry.key() != "encoding"
                    && entry.key() != "categories" && entry.key() != "clip") throw std::invalid_argument("Unsupported observation field");
            }
            const auto name = feature.at("feature").get<std::string>();
            const auto source = nameIndex(feature_names, name);
            if (source >= 89 && !requires_bow) throw std::invalid_argument("Bow observations require the bow capability");
            if (feature.contains("scale") && !feature.at("scale").is_number()) throw std::invalid_argument("Invalid observation scale");
            const auto scale = feature.value("scale", 1.0F);
            if (!std::isfinite(scale) || std::abs(scale) > policy_max_scale) throw std::invalid_argument("Invalid observation scale");
            const auto encoding = feature.value("encoding", std::string("scalar"));
            float lower = -std::numeric_limits<float>::infinity();
            float upper = std::numeric_limits<float>::infinity();
            if (feature.contains("clip")) {
                const auto& bounds = feature.at("clip");
                if (encoding != "scalar" || !bounds.is_array() || bounds.size() != 2
                    || !bounds[0].is_number() || !bounds[1].is_number()) throw std::invalid_argument("Invalid observation clip");
                if (bounds[0].get<double>() >= bounds[1].get<double>()) throw std::invalid_argument("Invalid observation clip bounds");
                lower = bounds[0].get<float>();
                upper = bounds[1].get<float>();
                if (!std::isfinite(lower) || !std::isfinite(upper)) throw std::invalid_argument("Invalid observation clip bounds");
            }
            const auto append = [&](float category) {
                observations.push_back(source);
                scales.push_back(scale);
                categories.push_back(category);
                lower_bounds.push_back(lower);
                upper_bounds.push_back(upper);
            };
            if (encoding == "scalar") {
                if (feature.contains("categories")) throw std::invalid_argument("Scalar observations cannot have categories");
                append(std::numeric_limits<float>::quiet_NaN());
            } else if (encoding == "one_hot") {
                const bool slot = name == "self.weapon_slot" || name == "opponent.weapon_slot";
                const bool binary = name.ends_with(".alive") || name.ends_with(".weapon_is_axe") || name.ends_with(".shield_using")
                    || name.ends_with(".shield_disabled") || name.ends_with(".shield_disabled_observed") || name.ends_with(".has_shield")
                    || name.ends_with(".present") || name.ends_with(".is_axe") || name.ends_with(".on_ground")
                    || name.ends_with(".sprinting") || name.ends_with(".bow_using") || name.ends_with(".bow_equipped");
                const auto& choices = feature.at("categories");
                if ((!slot && !binary) || !choices.is_array() || choices.empty()) throw std::invalid_argument("Invalid one_hot observation");
                std::vector<int> seen;
                for (const auto& choice : choices) {
                    if (!choice.is_number_integer() || choice < 0 || choice >= (slot ? policy_max_weapon_slots : 2)) {
                        throw std::invalid_argument("Invalid observation category");
                    }
                    const int category = choice.get<int>();
                    if (std::find(seen.begin(), seen.end(), category) != seen.end()) throw std::invalid_argument("Duplicate observation category");
                    seen.push_back(category);
                    append(static_cast<float>(category));
                }
            } else {
                throw std::invalid_argument("Unsupported observation encoding");
            }
        }
        if (observations.size() > policy_max_observations) throw std::invalid_argument("Expanded observation count exceeds limit");
        if (value.contains("observation_dim") && (!value.at("observation_dim").is_number_integer()
            || value.at("observation_dim") != observations.size())) throw std::invalid_argument("Observation dimension disagrees with descriptors");
        const auto& heads = value.at("actions");
        if (!heads.is_array() || heads.empty() || heads.size() > action_names.size()) throw std::invalid_argument("Invalid action head count");
        std::array<bool, action_names.size()> seen{};
        bool has_shield_action = false;
        bool has_use_item_action = false;
        for (const auto& descriptor : heads) {
            Head head{nameIndex(action_names, descriptor.at("name").get<std::string>()), {}};
            if (seen[head.kind]) throw std::invalid_argument("Duplicate action primitive");
            seen[head.kind] = true;
            has_shield_action = has_shield_action || std::string(action_names[head.kind]) == "shield";
            has_use_item_action = has_use_item_action || std::string(action_names[head.kind]) == "use_item";
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
        if (has_shield_action && has_use_item_action) {
            throw std::invalid_argument("Policy cannot declare both shield and use_item action aliases");
        }
    }

    void project(const float* catalogue, float* output) const {
        for (std::size_t index = 0; index < observations.size(); ++index) {
            const float value = catalogue[observations[index]];
            output[index] = (std::isnan(categories[index]) ? std::clamp(value, lower_bounds[index], upper_bounds[index])
                : (value == categories[index] ? 1.0F : 0.0F)) * scales[index];
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
            case 6:
                result.select_weapon = true;
                result.weapon = slots[static_cast<int>(value)];
                result.selected_slot = static_cast<int>(value);
                break;
            case 7: result.use_item = value != 0; break;
            case 8: result.sneak = value != 0; break;
            case 9: result.use_item = value != 0; break;
            }
        }
        return result;
    }
};

}