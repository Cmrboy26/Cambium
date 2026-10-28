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
import com.google.gson.JsonArray;
import com.google.gson.JsonParser;

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
		checkClientObservations();
		checkProjectileBlockHits();
		System.out.println("PASS: live equipment normalization, own hotbar, legacy isolation and vanilla item attributes");
	}

	private static void checkClientObservations() {
		if (ClientObservation.FEATURES.size() != 100
				|| ClientObservation.FEATURES.contains("opponent.weapon_slot")
				|| ClientObservation.FEATURES.contains("opponent.shield_disabled_observed")) {
			throw new AssertionError("Only the opponent's hidden slot and unobserved shield disable must be rejected");
		}
		var descriptor = JsonParser.parseString("""
				{"format":"minecraft_sim_policy","version":1,"minecraft":"26.3","ticks_per_action":1,
				 "observations":[],"actions":[{"name":"jump","values":[false,true]}]}
				""").getAsJsonObject();
		JsonArray observations = descriptor.getAsJsonArray("observations");
		for (String feature : SimulationContract.FEATURES.subList(22, 79)) {
			var entry = new com.google.gson.JsonObject();
			entry.addProperty("feature", feature);
			observations.add(entry);
			boolean unsupported = feature.equals("opponent.weapon_slot") || feature.equals("opponent.shield_disabled_observed");
			try {
				new SimulationContract(descriptor).requireSupported(ClientObservation.FEATURES, ClientObservation.ACTIONS);
				if (unsupported) throw new AssertionError("Unsupported feature accepted: " + feature);
			} catch (IllegalArgumentException error) {
				if (!unsupported || !error.getMessage().contains(feature)) throw error;
			}
			observations.remove(0);
		}
		float[] catalogue = new float[SimulationContract.FEATURES.size()];
		java.util.Arrays.fill(catalogue, -1);
		ClientObservation.combat(catalogue, true, false, true, false, 50, 0.75F, 3, 8);
		if (catalogue[22] != 1 || catalogue[23] != 0 || catalogue[24] != 1 || catalogue[25] != 0
				|| catalogue[26] != 1 || catalogue[27] != -1 || catalogue[28] != 0.5F || catalogue[29] != 0.75F
				|| catalogue[30] != 0.6F || catalogue[31] != 8 || catalogue[32] != -1) {
			throw new AssertionError("Client combat observation indices or hidden fields are incorrect");
		}
		ClientObservation.combat(catalogue, false, true, false, true, 0, 1, 10, 0);
		if (catalogue[22] != 0 || catalogue[23] != 1 || catalogue[24] != 0 || catalogue[25] != 1
				|| catalogue[26] != 0 || catalogue[28] != 0 || catalogue[29] != 1 || catalogue[30] != 0 || catalogue[31] != 0) {
			throw new AssertionError("Client combat observations must reset between samples");
		}
		ClientObservation.combat(catalogue, false, false, true, true, 100, 0, 10, 1);
		if (catalogue[28] != 1 || catalogue[30] != 1) throw new AssertionError("Combat fractions must cap at one");
		if (Math.abs(ClientObservation.visibleDamage(ItemStack.EMPTY) - 1) > 1e-6
				|| Math.abs(ClientObservation.visibleDamage(new ItemStack(Items.IRON_AXE)) - 9) > 1e-6
				|| Math.abs(ClientObservation.visibleDamage(new ItemStack(Items.WOODEN_SWORD)) - 4) > 1e-6) {
			throw new AssertionError("Visible item damage must include vanilla main-hand modifiers");
		}
		observations.add(JsonParser.parseString("{\"feature\":\"self.weapon_slot\",\"encoding\":\"one_hot\",\"categories\":[1,0,8]}"));
		observations.add(JsonParser.parseString("{\"feature\":\"self.shield_raise_fraction\",\"scale\":0.5}"));
		var contract = new SimulationContract(descriptor);
		contract.requireSupported(ClientObservation.FEATURES, ClientObservation.ACTIONS);
		float[] projected = contract.project(catalogue);
		if (!java.util.Arrays.equals(projected, new float[]{1, 0, 0, 0.5F})) {
			throw new AssertionError("Client combat observations must support reordered one-hot and scaled projection");
		}
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
