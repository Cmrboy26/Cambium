#pragma once
#include <array>

namespace minecraft_sim::app {
inline constexpr const char* catalogue_hash = "7655aebe65f76f9d2096567fc7aad0f881133029eb8e58bd591285379aa42122";
inline constexpr std::array<const char*, 33> feature_names{"self.health_fraction", "self.yaw_fraction", "self.pitch_fraction", "opponent.alive", "opponent.health_fraction", "relative.x", "relative.y", "relative.z", "opponent.yaw_fraction", "opponent.pitch_fraction", "arena.ray_world_0", "arena.ray_world_45", "arena.ray_world_90", "arena.ray_world_135", "arena.ray_world_180", "arena.ray_world_225", "arena.ray_world_270", "arena.ray_world_315", "target.distance_fraction", "target.aim_dot", "target.yaw_error", "target.pitch_error", "self.weapon_is_axe", "opponent.weapon_is_axe", "self.shield_using", "opponent.shield_using", "self.shield_disabled", "opponent.shield_disabled_observed", "self.shield_cooldown_fraction", "self.attack_charge", "self.shield_raise_fraction", "self.weapon_slot", "opponent.weapon_slot"};
inline constexpr std::array<const char*, 9> action_names{"movement", "jump", "sprint", "attack", "yaw", "pitch", "weapon", "shield", "sneak"};
inline constexpr std::array<const char*, 9> action_kinds{"movement", "boolean", "boolean", "boolean", "look", "look", "weapon", "boolean", "boolean"};
inline constexpr std::array<const char*, 9> movement_names{"neutral", "forward", "forward_right", "right", "backward_right", "backward", "backward_left", "left", "forward_left"};
inline constexpr std::array<const char*, 5> capability_names{"movement", "melee", "equipment", "shield", "armor"};
inline constexpr std::array<const char*, 4> armor_slot_names{"boots", "leggings", "chestplate", "helmet"};
inline constexpr int policy_max_observations = 128;
inline constexpr int policy_max_choices = 64;
inline constexpr int policy_max_scale = 1000;
inline constexpr int policy_max_look_delta = 180;
inline constexpr int policy_max_weapon_slots = 9;
}
