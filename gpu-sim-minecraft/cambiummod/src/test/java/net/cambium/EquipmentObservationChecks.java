package net.cambium;

import net.minecraft.SharedConstants;
import net.minecraft.server.Bootstrap;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;

public final class EquipmentObservationChecks {
	public static void main(String[] args) {
		check();
	}

	static void check() {
		EquipmentEncoderChecks.check();
		SharedConstants.tryDetectVersion();
		Bootstrap.bootStrap();
		checkWeapon(new ItemStack(Items.WOODEN_SWORD), true, false, 4, 4.0 + (double) -2.4F);
		checkWeapon(new ItemStack(Items.IRON_AXE), true, true, 9, 4.0 + (double) -3.1F);
		checkWeapon(new ItemStack(Items.NETHERITE_AXE), true, true, 10, 1);
		checkWeapon(ItemStack.EMPTY, false, false, 0, 0);
		checkWeapon(new ItemStack(Items.STONE), false, false, 0, 0);
		System.out.println("PASS: live equipment normalization, own hotbar, legacy isolation and vanilla item attributes");
	}

	private static void checkWeapon(ItemStack stack, boolean present, boolean axe, double damage, double speed) {
		var actual = LivePolicy.weapon(stack);
		if (actual.present() != present || actual.axe() != axe || Math.abs(actual.damage() - damage) > 1e-6
				|| Math.abs(actual.speed() - speed) > 1e-6) throw new AssertionError("Incorrect vanilla metadata for " + stack);
	}
}
