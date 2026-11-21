package net.cambium;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import net.minecraft.core.component.DataComponents;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.Identifier;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.util.Unit;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;

final class BridgeKit {
	private static final Set<String> ITEMS = Set.of("empty", "sword", "bow", "pickaxe", "blocks", "golden_apple");
	private final List<String> slots;
	private final int blocks;
	private final int apples;

	BridgeKit(JsonObject definition, JsonObject loadout) {
		if (definition == null || loadout == null || !definition.keySet().equals(Set.of("slots", "blocks", "golden_apples"))) {
			throw new IllegalArgumentException("Invalid Bridge kit");
		}
		JsonArray entries = definition.getAsJsonArray("slots");
		JsonArray weapons = loadout.getAsJsonArray("weapons");
		if (entries == null || weapons == null || entries.isEmpty() || entries.size() > 9 || entries.size() != weapons.size()) {
			throw new IllegalArgumentException("Bridge kit must match its loadout");
		}
		var parsed = new ArrayList<String>();
		var seen = new HashSet<String>();
		for (int index = 0; index < entries.size(); index++) {
			JsonElement value = entries.get(index);
			if (!value.isJsonPrimitive() || !value.getAsJsonPrimitive().isString()) throw new IllegalArgumentException("Invalid Bridge kit slot");
			String item = value.getAsString();
			String weapon = weapons.get(index).getAsString();
			if (!ITEMS.contains(item) || !item.equals("empty") && !seen.add(item)
					|| item.equals("sword") && !weapon.matches("minecraft:(wooden|stone|iron|golden|diamond|netherite)_sword")
					|| item.equals("bow") && !weapon.equals("minecraft:bow")
					|| !item.equals("sword") && !item.equals("bow") && !weapon.equals("minecraft:air")) {
				throw new IllegalArgumentException("Bridge kit slot disagrees with loadout");
			}
			parsed.add(item);
		}
		blocks = SimulationContract.integer(definition.get("blocks"));
		apples = SimulationContract.integer(definition.get("golden_apples"));
		if (blocks < 0 || blocks > 64 || apples < 0 || apples > 64
				|| blocks > 0 && !seen.contains("blocks") || apples > 0 && !seen.contains("golden_apple")) {
			throw new IllegalArgumentException("Invalid Bridge kit counts");
		}
		slots = List.copyOf(parsed);
	}

	String slot(int index) { return index >= 0 && index < slots.size() ? slots.get(index) : "empty"; }
	int index(String item) { return slots.indexOf(item); }

	void equip(ServerPlayer player, int actor) {
		if (actor < 0 || actor > 1) throw new IllegalArgumentException("Invalid Bridge actor");
		for (int index = 0; index < slots.size(); index++) {
			ItemStack stack = switch (slots.get(index)) {
				case "pickaxe" -> new ItemStack(Items.DIAMOND_PICKAXE);
				case "blocks" -> blocks == 0 ? ItemStack.EMPTY : new ItemStack(BuiltInRegistries.ITEM.getValue(
						Identifier.parse(actor == 0 ? "minecraft:blue_wool" : "minecraft:red_wool")), blocks);
				case "golden_apple" -> apples == 0 ? ItemStack.EMPTY : new ItemStack(Items.GOLDEN_APPLE, apples);
				default -> null;
			};
			if (stack == null) continue;
			if (!stack.isEmpty()) stack.set(DataComponents.UNBREAKABLE, Unit.INSTANCE);
			player.getInventory().setItem(index, stack);
		}
		player.inventoryMenu.broadcastChanges();
	}
}