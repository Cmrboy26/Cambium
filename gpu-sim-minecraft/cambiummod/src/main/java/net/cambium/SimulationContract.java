package net.cambium;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

final class SimulationContract {

	static final List<String> FEATURES = PolicyCatalogue.FEATURE_NAMES;
	static final List<String> MOVEMENTS = PolicyCatalogue.MOVEMENT_NAMES;
	static final Set<String> CAPABILITIES = Set.copyOf(PolicyCatalogue.CAPABILITY_NAMES);
	static final List<String> ARMOR_SLOTS = PolicyCatalogue.ARMOR_SLOT_NAMES;
	private static final Set<String> HEADS = Set.copyOf(PolicyCatalogue.ACTION_NAMES);
	private static final Set<String> WEAPONS = weaponItems();
	private final List<Observation> observations = new ArrayList<>();
	private int observationCount;
	private final List<Head> heads = new ArrayList<>();
	record Head(String name, JsonArray values) {}
	record Observation(int feature, float scale, int[] categories, float lower, float upper) {}

	SimulationContract(JsonObject descriptor) {
		if (!descriptor.get("format").getAsString().equals("minecraft_sim_policy") || integer(descriptor.get("version")) != 1
				|| !descriptor.get("minecraft").getAsString().equals("26.3") || integer(descriptor.get("ticks_per_action")) != 1) {
			throw new IllegalArgumentException("Unsupported policy contract format/version/build/tick rate");
		}
		capabilities(descriptor);
		JsonArray entries = descriptor.getAsJsonArray("observations");
		if (entries.size() < 1 || entries.size() > PolicyCatalogue.MAX_OBSERVATIONS) throw new IllegalArgumentException("Invalid observation count");
		for (JsonElement element : entries) {
			JsonObject observation = element.getAsJsonObject();
			for (String key : observation.keySet()) if (!Set.of("feature", "scale", "encoding", "categories", "clip").contains(key)) {
				throw new IllegalArgumentException("Unsupported observation field: " + key);
			}
			String name = observation.get("feature").getAsString();
			int feature = FEATURES.indexOf(name);
			if (observation.has("scale") && !observation.get("scale").getAsJsonPrimitive().isNumber()) {
				throw new IllegalArgumentException("Invalid observation scale");
			}
			float scale = observation.has("scale") ? observation.get("scale").getAsFloat() : 1;
			if (feature < 0 || !Float.isFinite(scale) || Math.abs(scale) > PolicyCatalogue.MAX_SCALE) {
				throw new IllegalArgumentException("Unsupported observation primitive or scale: " + observation);
			}
			String encoding = observation.has("encoding") ? observation.get("encoding").getAsString() : "scalar";
			float lower = Float.NEGATIVE_INFINITY;
			float upper = Float.POSITIVE_INFINITY;
			if (observation.has("clip")) {
				JsonElement clip = observation.get("clip");
				if (!encoding.equals("scalar") || !clip.isJsonArray() || clip.getAsJsonArray().size() != 2) {
					throw new IllegalArgumentException("Invalid observation clip");
				}
				JsonArray bounds = clip.getAsJsonArray();
				for (JsonElement bound : bounds) if (!bound.isJsonPrimitive() || !bound.getAsJsonPrimitive().isNumber()) {
					throw new IllegalArgumentException("Clip bounds must be numeric");
				}
				if (bounds.get(0).getAsDouble() >= bounds.get(1).getAsDouble()) throw new IllegalArgumentException("Invalid clip bounds");
				lower = bounds.get(0).getAsFloat();
				upper = bounds.get(1).getAsFloat();
				if (!Float.isFinite(lower) || !Float.isFinite(upper)) throw new IllegalArgumentException("Invalid clip bounds");
			}
			int[] categories = null;
			if (encoding.equals("scalar")) {
				if (observation.has("categories")) throw new IllegalArgumentException("Scalar observations cannot have categories");
			} else if (encoding.equals("one_hot")) {
				boolean slot = name.equals("self.weapon_slot") || name.equals("opponent.weapon_slot");
				boolean binary = name.endsWith(".alive") || name.endsWith(".weapon_is_axe") || name.endsWith(".shield_using")
						|| name.endsWith(".shield_disabled") || name.endsWith(".shield_disabled_observed") || name.endsWith(".has_shield")
						|| name.endsWith(".present") || name.endsWith(".is_axe") || name.endsWith(".on_ground") || name.endsWith(".sprinting");
				if ((!slot && !binary) || !observation.has("categories") || !observation.get("categories").isJsonArray()
						|| observation.getAsJsonArray("categories").isEmpty()) throw new IllegalArgumentException("Invalid one_hot observation");
				JsonArray choices = observation.getAsJsonArray("categories");
				categories = new int[choices.size()];
				Set<Integer> seen = new HashSet<>();
				for (int index = 0; index < categories.length; index++) {
					JsonElement choice = choices.get(index);
					if (!choice.isJsonPrimitive() || !choice.getAsJsonPrimitive().isNumber()
							|| !choice.getAsString().matches("-?[0-9]+")) throw new IllegalArgumentException("Categories must be integer literals");
					int category = integer(choices.get(index));
					if (category < 0 || category >= (slot ? PolicyCatalogue.MAX_WEAPON_SLOTS : 2) || !seen.add(category)) {
						throw new IllegalArgumentException("Invalid/duplicate observation category");
					}
					categories[index] = category;
				}
			} else {
				throw new IllegalArgumentException("Unsupported observation encoding: " + encoding);
			}
			observations.add(new Observation(feature, scale, categories, lower, upper));
			observationCount += categories == null ? 1 : categories.length;
		}
		if (observationCount > PolicyCatalogue.MAX_OBSERVATIONS) throw new IllegalArgumentException("Expanded observation count exceeds limit");
		JsonArray actions = descriptor.getAsJsonArray("actions");
		if (actions.size() < 1 || actions.size() > HEADS.size()) throw new IllegalArgumentException("Invalid action head count");
		Set<String> seen = new HashSet<>();
		for (JsonElement element : actions) {
			JsonObject action = element.getAsJsonObject();
			String name = action.get("name").getAsString();
			JsonArray values = action.getAsJsonArray("values");
			if (!HEADS.contains(name) || !seen.add(name) || values.size() < 1 || values.size() > PolicyCatalogue.MAX_CHOICES) {
				throw new IllegalArgumentException("Unsupported/duplicate action primitive or choice count: " + name);
			}
			String kind = PolicyCatalogue.ACTION_KINDS.get(PolicyCatalogue.ACTION_NAMES.indexOf(name));
			Set<String> choices = new HashSet<>();
			for (JsonElement value : values) {
				if (!choices.add(value.toString())) throw new IllegalArgumentException("Duplicate action choice");
				if (kind.equals("movement")) {
					if (!MOVEMENTS.contains(value.getAsString())) throw new IllegalArgumentException("Unsupported movement choice");
				} else if (kind.equals("look")) {
					float delta = value.getAsFloat();
					if (!value.getAsJsonPrimitive().isNumber() || !Float.isFinite(delta) || Math.abs(delta) > PolicyCatalogue.MAX_LOOK_DELTA) {
						throw new IllegalArgumentException("Invalid look delta");
					}
				} else if (kind.equals("weapon")) {
					if (integer(value) < 0 || integer(value) >= PolicyCatalogue.MAX_WEAPON_SLOTS) throw new IllegalArgumentException("Invalid weapon slot choice");
				} else if (!value.getAsJsonPrimitive().isBoolean()) {
					throw new IllegalArgumentException("Binary controls need boolean choices");
				}
			}
			heads.add(new Head(name, values.deepCopy()));
		}
		if (descriptor.has("observation_dim") && integer(descriptor.get("observation_dim")) != observationCount) {
			throw new IllegalArgumentException("Observation dimension disagrees with descriptors");
		}
		if (descriptor.has("action_names") || descriptor.has("action_sizes")) {
			if (descriptor.getAsJsonArray("action_names").size() != heads.size() || descriptor.getAsJsonArray("action_sizes").size() != heads.size()) {
				throw new IllegalArgumentException("Action dimensions disagree with descriptors");
			}
			for (int index = 0; index < heads.size(); index++) {
				if (!descriptor.getAsJsonArray("action_names").get(index).getAsString().equals(heads.get(index).name)
						|| integer(descriptor.getAsJsonArray("action_sizes").get(index)) != heads.get(index).values.size()) {
					throw new IllegalArgumentException("Action metadata disagrees with descriptors");
				}
			}
		}
	}

	int observationCount() { return observationCount; }

	float[] project(float[] catalogue) {
		if (catalogue.length != FEATURES.size()) throw new IllegalArgumentException("Incomplete observation catalogue");
		float[] result = new float[observationCount];
		int offset = 0;
		for (Observation observation : observations) {
			float value = catalogue[observation.feature];
			if (observation.categories == null) {
				result[offset++] = Math.clamp(value, observation.lower, observation.upper) * observation.scale;
			} else {
				for (int category : observation.categories) result[offset++] = (value == category ? 1 : 0) * observation.scale;
			}
		}
		for (float value : result) if (!Float.isFinite(value)) throw new IllegalArgumentException("Non-finite observation");
		return result;
	}

	void validateLoadouts(List<JsonObject> loadouts) {
		for (Head head : heads) if (head.name.equals("weapon")) {
			for (JsonElement choice : head.values) for (JsonObject loadout : loadouts) {
				if (integer(choice) >= loadout.getAsJsonArray("weapons").size()) throw new IllegalArgumentException("Weapon choice outside a player's loadout");
			}
		}
	}

	ReplayTrace.Controls decode(JsonArray indices, JsonObject loadout) {
		if (indices.size() != heads.size()) throw new IllegalArgumentException("Action head count does not match policy contract");
		boolean forward = false, backward = false, left = false, right = false, jump = false, sprint = false;
		boolean sneak = false, attack = false, select = false, use = false;
		float yaw = 0, pitch = 0;
		int slot = -1;
		String weapon = "minecraft:diamond_sword";
		for (int index = 0; index < heads.size(); index++) {
			Head head = heads.get(index);
			int selected = integer(indices.get(index));
			if (selected < 0 || selected >= head.values.size()) throw new IllegalArgumentException("Policy action outside head range");
			JsonElement value = head.values.get(selected);
			switch (head.name) {
				case "movement" -> {
					int move = MOVEMENTS.indexOf(value.getAsString());
					forward = move == 1 || move == 2 || move == 8;
					backward = move == 4 || move == 5 || move == 6;
					right = move == 2 || move == 3 || move == 4;
					left = move == 6 || move == 7 || move == 8;
				}
				case "jump" -> jump = value.getAsBoolean();
				case "sprint" -> sprint = value.getAsBoolean();
				case "sneak" -> sneak = value.getAsBoolean();
				case "attack" -> attack = value.getAsBoolean();
				case "shield" -> use = value.getAsBoolean();
				case "yaw" -> yaw = value.getAsFloat();
				case "pitch" -> pitch = value.getAsFloat();
				case "weapon" -> { select = true; slot = integer(value); weapon = loadout.getAsJsonArray("weapons").get(slot).getAsString(); }
				default -> throw new IllegalArgumentException("Unsupported control primitive");
			}
		}
		return new ReplayTrace.Controls(forward, backward, left, right, jump, sprint, sneak, attack, yaw, pitch, select, weapon, use, slot);
	}

	static int selectedSlot(JsonObject loadout, String weapon, int requested, int current) {
		JsonArray weapons = loadout.getAsJsonArray("weapons");
		if (requested >= 0) {
			if (requested >= weapons.size() || !weapons.get(requested).getAsString().equals(weapon)) {
				throw new IllegalArgumentException("Selected slot disagrees with scenario weapon");
			}
			return requested;
		}
		if (current >= 0 && current < weapons.size() && weapons.get(current).getAsString().equals(weapon)) return current;
		for (int slot = 0; slot < weapons.size(); slot++) if (weapons.get(slot).getAsString().equals(weapon)) return slot;
		throw new IllegalArgumentException("Selected weapon outside scenario loadout");
	}

	static void capabilities(JsonObject descriptor) {
		if (descriptor.has("requires")) for (JsonElement value : descriptor.getAsJsonArray("requires")) {
			if (!CAPABILITIES.contains(value.getAsString())) throw new IllegalArgumentException("Unsupported required capability: " + value.getAsString());
		}
	}

	static JsonObject loadout(JsonObject spawn, String profile) {
		JsonObject result = spawn.has("loadout") ? spawn.getAsJsonObject("loadout").deepCopy() : new JsonObject();
		for (String key : result.keySet()) if (!Set.of("weapons", "selected", "offhand", "armor").contains(key)) {
			throw new IllegalArgumentException("Unsupported loadout field: " + key);
		}
		boolean equippedDefault = !spawn.has("loadout") && profile.equals("unarmored_sword_axe_shield");
		if (!result.has("weapons")) result.add("weapons", ReplayAnalysis.JSON.toJsonTree(equippedDefault
				? new String[]{"minecraft:diamond_sword", "minecraft:diamond_axe"} : new String[]{"minecraft:diamond_sword"}));
		JsonArray weapons = result.getAsJsonArray("weapons");
		if (weapons.size() < 1 || weapons.size() > 9) throw new IllegalArgumentException("Loadout needs 1..9 weapon slots");
		Set<String> seen = new HashSet<>();
		for (JsonElement item : weapons) if (!WEAPONS.contains(item.getAsString())
				|| !item.getAsString().equals("minecraft:air") && !seen.add(item.getAsString())) {
			throw new IllegalArgumentException("Unsupported or duplicate weapon item: " + item);
		}
		int selected = result.has("selected") ? integer(result.get("selected")) : 0;
		if (selected < 0 || selected >= weapons.size()) throw new IllegalArgumentException("Selected weapon outside loadout");
		result.addProperty("selected", selected);
		String offhand = result.has("offhand") ? result.get("offhand").getAsString() : equippedDefault ? "minecraft:shield" : "minecraft:air";
		if (!offhand.equals("minecraft:shield") && !offhand.equals("minecraft:air")) throw new IllegalArgumentException("Unsupported offhand item");
		result.addProperty("offhand", offhand);
		JsonObject armor = result.has("armor") ? result.getAsJsonObject("armor") : new JsonObject();
		for (String key : armor.keySet()) if (!ARMOR_SLOTS.contains(key)) throw new IllegalArgumentException("Unsupported armor slot");
		for (String slot : ARMOR_SLOTS) {
			String item = armor.has(slot) ? armor.get(slot).getAsString() : "minecraft:air";
			if (!armorItem(item, slot)) throw new IllegalArgumentException("Unsupported armor item: " + item);
			armor.addProperty(slot, item);
		}
		result.add("armor", armor);
		return result;
	}

	static boolean armorItem(String item, String slot) {
		if (item.equals("minecraft:air") || item.equals("minecraft:turtle_helmet") && slot.equals("helmet")) return true;
		for (String material : List.of("leather", "copper", "chainmail", "iron", "golden", "diamond", "netherite")) {
			if (item.equals("minecraft:" + material + "_" + slot)) return true;
		}
		return false;
	}

	static boolean weaponItem(String item) { return WEAPONS.contains(item); }

	static List<String> weaponChoices() {
		List<String> items = new ArrayList<>(List.of("minecraft:air", "minecraft:wooden_sword", "minecraft:wooden_axe",
				"minecraft:stone_sword", "minecraft:stone_axe", "minecraft:copper_sword", "minecraft:copper_axe",
				"minecraft:golden_sword", "minecraft:golden_axe", "minecraft:iron_sword", "minecraft:iron_axe",
				"minecraft:diamond_sword", "minecraft:diamond_axe", "minecraft:netherite_sword", "minecraft:netherite_axe"));
		return items.stream().filter(WEAPONS::contains).toList();
	}

	static List<String> armorChoices(String slot) {
		List<String> candidates = new ArrayList<>(List.of("minecraft:air", "minecraft:turtle_helmet"));
		for (String material : List.of("leather", "copper", "chainmail", "iron", "golden", "diamond", "netherite")) {
			candidates.add("minecraft:" + material + "_" + slot);
		}
		return candidates.stream().filter(item -> armorItem(item, slot)).toList();
	}

	private static Set<String> weaponItems() {
		Set<String> result = new HashSet<>(Set.of("minecraft:air"));
		for (String material : List.of("wooden", "stone", "copper", "golden", "iron", "diamond", "netherite")) {
			for (String kind : List.of("sword", "axe")) result.add("minecraft:" + material + "_" + kind);
		}
		return Set.copyOf(result);
	}

	static int integer(JsonElement value) {
		if (value == null || !value.isJsonPrimitive() || !value.getAsJsonPrimitive().isNumber()) throw new IllegalArgumentException("Expected an integer");
		double number = value.getAsDouble();
		if (!Double.isFinite(number) || number != Math.rint(number) || number < Integer.MIN_VALUE || number > Integer.MAX_VALUE) {
			throw new IllegalArgumentException("Invalid integer value");
		}
		return (int) number;
	}
}