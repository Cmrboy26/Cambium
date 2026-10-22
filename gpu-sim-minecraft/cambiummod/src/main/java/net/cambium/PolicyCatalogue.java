package net.cambium;

import java.util.List;

final class PolicyCatalogue {
	static final String HASH = "7ba67f599d9f39e8738f019a05513a4e829f9b101448c54aca88c73a5d3bc91a";
	static final List<String> FEATURE_NAMES = List.of("self.health_fraction", "self.yaw_fraction", "self.pitch_fraction", "opponent.alive", "opponent.health_fraction", "relative.x", "relative.y", "relative.z", "opponent.yaw_fraction", "opponent.pitch_fraction", "arena.ray_world_0", "arena.ray_world_45", "arena.ray_world_90", "arena.ray_world_135", "arena.ray_world_180", "arena.ray_world_225", "arena.ray_world_270", "arena.ray_world_315", "target.distance_fraction", "target.aim_dot", "target.yaw_error", "target.pitch_error", "self.weapon_is_axe", "opponent.weapon_is_axe", "self.shield_using", "opponent.shield_using", "self.shield_disabled", "opponent.shield_disabled_observed", "self.shield_cooldown_fraction", "self.attack_charge", "self.shield_raise_fraction", "self.weapon_slot", "opponent.weapon_slot", "self.has_shield", "opponent.has_shield", "self.weapon_damage_fraction", "opponent.weapon_damage_fraction", "self.weapon_speed_fraction", "opponent.weapon_speed_fraction", "self.armor_defense_fraction", "opponent.armor_defense_fraction", "self.armor_toughness_fraction", "opponent.armor_toughness_fraction", "self.weapon_slot_0.present", "self.weapon_slot_0.is_axe", "self.weapon_slot_0.damage_fraction", "self.weapon_slot_0.speed_fraction", "self.weapon_slot_1.present", "self.weapon_slot_1.is_axe", "self.weapon_slot_1.damage_fraction", "self.weapon_slot_1.speed_fraction", "self.weapon_slot_2.present", "self.weapon_slot_2.is_axe", "self.weapon_slot_2.damage_fraction", "self.weapon_slot_2.speed_fraction", "self.weapon_slot_3.present", "self.weapon_slot_3.is_axe", "self.weapon_slot_3.damage_fraction", "self.weapon_slot_3.speed_fraction", "self.weapon_slot_4.present", "self.weapon_slot_4.is_axe", "self.weapon_slot_4.damage_fraction", "self.weapon_slot_4.speed_fraction", "self.weapon_slot_5.present", "self.weapon_slot_5.is_axe", "self.weapon_slot_5.damage_fraction", "self.weapon_slot_5.speed_fraction", "self.weapon_slot_6.present", "self.weapon_slot_6.is_axe", "self.weapon_slot_6.damage_fraction", "self.weapon_slot_6.speed_fraction", "self.weapon_slot_7.present", "self.weapon_slot_7.is_axe", "self.weapon_slot_7.damage_fraction", "self.weapon_slot_7.speed_fraction", "self.weapon_slot_8.present", "self.weapon_slot_8.is_axe", "self.weapon_slot_8.damage_fraction", "self.weapon_slot_8.speed_fraction", "self.velocity.x", "self.velocity.y", "self.velocity.z", "relative.velocity.x", "relative.velocity.y", "relative.velocity.z", "self.on_ground", "opponent.on_ground", "self.sprinting", "opponent.sprinting");
	static final List<String> ACTION_NAMES = List.of("movement", "jump", "sprint", "attack", "yaw", "pitch", "weapon", "shield", "sneak");
	static final List<String> ACTION_KINDS = List.of("movement", "boolean", "boolean", "boolean", "look", "look", "weapon", "boolean", "boolean");
	static final List<String> MOVEMENT_NAMES = List.of("neutral", "forward", "forward_right", "right", "backward_right", "backward", "backward_left", "left", "forward_left");
	static final List<String> CAPABILITY_NAMES = List.of("movement", "melee", "equipment", "shield", "armor");
	static final List<String> ARMOR_SLOT_NAMES = List.of("boots", "leggings", "chestplate", "helmet");
	static final int MAX_OBSERVATIONS = 128;
	static final int MAX_CHOICES = 64;
	static final int MAX_SCALE = 1000;
	static final int MAX_LOOK_DELTA = 180;
	static final int MAX_WEAPON_SLOTS = 9;
	private PolicyCatalogue() {}
}
