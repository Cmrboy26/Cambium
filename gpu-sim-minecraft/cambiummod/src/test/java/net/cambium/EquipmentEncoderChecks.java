package net.cambium;

import java.util.Arrays;
import net.cambium.ObservationEncoder.Equipment;
import net.cambium.ObservationEncoder.Weapon;

public final class EquipmentEncoderChecks {
	public static void main(String[] args) {
		check();
	}

	static void check() {
		if (PolicyCatalogue.FEATURE_NAMES.size() != 89
				|| !PolicyCatalogue.FEATURE_NAMES.contains("opponent.weapon_slot")
				|| PolicyCatalogue.FEATURE_NAMES.stream().anyMatch(name -> name.startsWith("opponent.weapon_slot_"))) {
			throw new AssertionError("Catalogue must preserve legacy features but exclude hidden opponent hotbar metadata");
		}
		float[] catalogue = new float[PolicyCatalogue.FEATURE_NAMES.size()];
		Arrays.fill(catalogue, -1);
		var sword = new Weapon(true, false, 4, 4.0 + (double) -2.4F);
		var axe = new Weapon(true, true, 10, 1);
		var own = new Equipment(true, 10, 1, 20, 12);
		var other = new Equipment(false, 7, 4.0 + (double) -2.4F, 8, 2);
		ObservationEncoder.equipment(catalogue, own, other,
				new Weapon[]{sword, axe, new Weapon(false, false, 1, 4)});
		near(catalogue, "self.has_shield", 1);
		near(catalogue, "opponent.has_shield", 0);
		near(catalogue, "self.weapon_damage_fraction", 1);
		near(catalogue, "self.weapon_speed_fraction", 0.25);
		near(catalogue, "opponent.weapon_damage_fraction", 0.7);
		near(catalogue, "opponent.weapon_speed_fraction", 0.4);
		near(catalogue, "self.armor_defense_fraction", 1);
		near(catalogue, "self.armor_toughness_fraction", 1);
		near(catalogue, "opponent.armor_defense_fraction", 0.4);
		near(catalogue, "opponent.armor_toughness_fraction", 1.0 / 6);
		near(catalogue, "self.weapon_slot_0.present", 1);
		near(catalogue, "self.weapon_slot_0.is_axe", 0);
		near(catalogue, "self.weapon_slot_0.damage_fraction", 0.4);
		near(catalogue, "self.weapon_slot_0.speed_fraction", 0.4);
		near(catalogue, "self.weapon_slot_1.is_axe", 1);
		for (int slot = 2; slot < 9; slot++) {
			for (String field : new String[]{"present", "is_axe", "damage_fraction", "speed_fraction"}) {
				near(catalogue, "self.weapon_slot_" + slot + "." + field, 0);
			}
		}
		Weapon[] nine = new Weapon[9];
		nine[8] = new Weapon(true, false, 8, 4.0 + (double) -2.4F);
		ObservationEncoder.equipment(catalogue, new Equipment(false, 100, 100, 100, 100),
				new Equipment(false, -1, -1, -1, -1), nine);
		for (String field : new String[]{"weapon_damage_fraction", "weapon_speed_fraction", "armor_defense_fraction", "armor_toughness_fraction"}) {
			near(catalogue, "self." + field, 1);
			near(catalogue, "opponent." + field, 0);
		}
		near(catalogue, "self.weapon_slot_8.damage_fraction", 0.8);
		var unarmed = new Equipment(false, 1, 4, 0, 0);
		ObservationEncoder.equipment(catalogue, unarmed, unarmed, new Weapon[0]);
		near(catalogue, "self.weapon_damage_fraction", 0.1);
		near(catalogue, "self.weapon_speed_fraction", 1);
		for (int index = 0; index < 33; index++) {
			if (catalogue[index] != -1) throw new AssertionError("Legacy catalogue changed at " + index);
		}
		System.out.println("PASS: 89-feature catalogue, equipment normalization, own hotbar, empty slots and legacy isolation");
	}

	private static void near(float[] catalogue, String name, double expected) {
		int index = PolicyCatalogue.FEATURE_NAMES.indexOf(name);
		if (index < 0 || Math.abs(catalogue[index] - expected) > 1e-6) {
			throw new AssertionError(name + " expected " + expected + ", got " + (index < 0 ? "missing" : catalogue[index]));
		}
	}
}
