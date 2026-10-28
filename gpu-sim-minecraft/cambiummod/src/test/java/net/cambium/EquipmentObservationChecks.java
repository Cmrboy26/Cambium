package net.cambium;

import net.minecraft.SharedConstants;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.data.registries.VanillaRegistries;
import net.minecraft.server.Bootstrap;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.phys.Vec3;
import net.minecraft.world.phys.shapes.Shapes;

public final class EquipmentObservationChecks {
	public static void main(String[] args) {
		check();
	}

	static void check() {
		EquipmentEncoderChecks.check();
		SharedConstants.tryDetectVersion();
		Bootstrap.bootStrap();
		for (var components : BuiltInRegistries.DATA_COMPONENT_INITIALIZERS.build(VanillaRegistries.createWorldLookup())) {
			components.apply();
		}
		checkWeapon(new ItemStack(Items.WOODEN_SWORD), true, false, 4, 4.0 + (double) -2.4F);
		checkWeapon(new ItemStack(Items.IRON_AXE), true, true, 9, 4.0 + (double) -3.1F);
		checkWeapon(new ItemStack(Items.NETHERITE_AXE), true, true, 10, 1);
		checkWeapon(new ItemStack(Items.MACE), true, false, 6, 4.0 + (double) -3.4F);
		checkWeapon(new ItemStack(Items.BOW), true, false, 1, 4);
		checkWeapon(new ItemStack(Items.SNOWBALL), true, false, 1, 4);
		checkWeapon(new ItemStack(Items.EGG), true, false, 1, 4);
		checkWeapon(new ItemStack(Items.ENDER_PEARL), true, false, 1, 4);
		if (new ItemStack(Items.SNOWBALL).getMaxStackSize() != 16 || new ItemStack(Items.EGG).getMaxStackSize() != 16
				|| new ItemStack(Items.ENDER_PEARL).getMaxStackSize() != 16) {
			throw new AssertionError("Throwable loadout stack limits must match Minecraft");
		}
		if (!SimulationContract.weaponItem("minecraft:mace") || !SimulationContract.weaponChoices().contains("minecraft:mace")) {
			throw new AssertionError("Mace must be available to configured loadouts");
		}
		checkWeapon(ItemStack.EMPTY, false, false, 0, 0);
		checkWeapon(new ItemStack(Items.STONE), false, false, 0, 0);
		checkProjectileBlockHits();
		System.out.println("PASS: live equipment normalization, own hotbar, legacy isolation and vanilla item attributes");
	}

	private static void checkProjectileBlockHits() {
		var cube = Shapes.block();
		var position = new BlockPos(2, 2, 2);
		var west = cube.clip(new Vec3(0, 2.5, 2.5), new Vec3(4, 2.5, 2.5), position);
		if (west == null || west.getDirection() != Direction.WEST || west.getLocation().x != 2.0) {
			throw new AssertionError("Unexpected vanilla west-face collision");
		}
		var tie = cube.clip(new Vec3(0, 0, 2.5), new Vec3(4, 4, 2.5), position);
		if (tie == null || tie.getDirection() != Direction.WEST) {
			throw new AssertionError("Unexpected vanilla entry-face tie");
		}
		var inside = cube.clip(new Vec3(2.5, 2.5, 2.5), new Vec3(3.5, 3.5, 2.5), position);
		if (inside == null || inside.getDirection() != Direction.DOWN || inside.getLocation().x != 2.501) {
			throw new AssertionError("Unexpected vanilla inside-block collision");
		}
		if (cube.clip(new Vec3(2.5, 2.5, 2.5), new Vec3(2.5001, 2.5, 2.5), position) != null
				|| cube.clip(new Vec3(0, 2.5, 2.5), new Vec3(2, 2.5, 2.5), position) != null) {
			throw new AssertionError("Unexpected vanilla short or endpoint collision");
		}
	}

	private static void checkWeapon(ItemStack stack, boolean present, boolean axe, double damage, double speed) {
		var actual = LivePolicy.weapon(stack);
		if (actual.present() != present || actual.axe() != axe || Math.abs(actual.damage() - damage) > 1e-6
				|| Math.abs(actual.speed() - speed) > 1e-6) throw new AssertionError("Incorrect vanilla metadata for " + stack);
	}
}
