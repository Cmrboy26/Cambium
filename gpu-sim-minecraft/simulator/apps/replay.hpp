#pragma once

#include "configuration.hpp"
#include <nlohmann/json.hpp>

namespace minecraft_sim::app {

inline nlohmann::json replayFrame(std::uint64_t tick, const PlayerState* players, const Controls* actions,
                                  bool equipment = false, const int* selected_slots = nullptr) {
    auto actors = nlohmann::json::array();
    for (int index = 0; index < 2; ++index) {
        const auto& state = players[index];
        const auto& input = actions[index];
        const auto& attack = state.last_attack;
        actors.push_back({{"id", index},
            {"position", {state.position.x, state.position.y, state.position.z}},
            {"velocity", {state.velocity.x, state.velocity.y, state.velocity.z}},
            {"yaw", state.yaw}, {"pitch", state.pitch}, {"on_ground", state.on_ground},
            {"sprinting", state.sprinting}, {"pose", static_cast<int>(state.pose)},
            {"health", state.health}, {"absorption", state.absorption},
            {"damage_cooldown", state.damage_cooldown}, {"hurt_time", state.hurt_time},
            {"attack_strength_ticks", state.attack_strength_ticks},
            {"controls", {{"forward", input.forward}, {"backward", input.backward},
                {"left", input.left}, {"right", input.right}, {"jump", input.jump},
                {"sprint", input.sprint}, {"sneak", input.sneak}, {"attack", input.attack},
                {"yaw_delta", input.yaw_delta}, {"pitch_delta", input.pitch_delta}}},
            {"outcome", {{"target", attack.target}, {"attempted", attack.attempted},
                {"accepted", attack.accepted}, {"critical", attack.critical},
                {"sprint_bonus", attack.sprint_bonus}, {"sweep", attack.sweep},
                {"damage", attack.incoming_damage}, {"health_lost", attack.health_lost}}}});
        if (equipment) {
            auto& actor = actors.back();
            actor.update({{"weapon", weaponName(state.weapon)},
                {"shield_equipped", state.shield_equipped}, {"using_shield", state.using_shield},
                {"shield_use_ticks", state.shield_use_ticks}, {"shield_cooldown", state.shield_cooldown},
                {"item_use_delay", state.item_use_delay}, {"attack_suppressed", state.attack_suppressed}});
            actor["controls"].update({{"select_weapon", input.select_weapon},
                {"weapon", weaponName(input.weapon)},
                {"use_item", input.use_item}});
            if (selected_slots) {
                actor["weapon_slot"] = selected_slots[index];
                actor["controls"]["select_slot"] = input.select_weapon ? selected_slots[index] : -1;
            }
            actor["outcome"].update({{"blocked_damage", attack.blocked_damage}, {"shield_disabled", attack.shield_disabled}});
            auto armor = nlohmann::json::object();
            for (int slot = 0; slot < 4; ++slot) armor[armor_slots[slot]] = armorName(state.armor_items[slot], slot);
            actor["armor"] = armor;
        }
    }
    return {{"tick", tick}, {"actors", actors}};
}

}