package net.cambium;

import java.util.List;

final class PolicyCatalogue {
	static final String HASH = "7655aebe65f76f9d2096567fc7aad0f881133029eb8e58bd591285379aa42122";
	static final List<String> FEATURE_NAMES = List.of("self.health_fraction", "self.yaw_fraction", "self.pitch_fraction", "opponent.alive", "opponent.health_fraction", "relative.x", "relative.y", "relative.z", "opponent.yaw_fraction", "opponent.pitch_fraction", "arena.ray_world_0", "arena.ray_world_45", "arena.ray_world_90", "arena.ray_world_135", "arena.ray_world_180", "arena.ray_world_225", "arena.ray_world_270", "arena.ray_world_315", "target.distance_fraction", "target.aim_dot", "target.yaw_error", "target.pitch_error", "self.weapon_is_axe", "opponent.weapon_is_axe", "self.shield_using", "opponent.shield_using", "self.shield_disabled", "opponent.shield_disabled_observed", "self.shield_cooldown_fraction", "self.attack_charge", "self.shield_raise_fraction", "self.weapon_slot", "opponent.weapon_slot");
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
