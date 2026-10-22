package net.cambium;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;

final class DuelLibrary {

	final Path root;
	final List<JsonObject> policies = new ArrayList<>();
	final List<JsonObject> environments = new ArrayList<>();
	final List<JsonObject> runs = new ArrayList<>();
	private JsonObject index;

	DuelLibrary(Path game) throws IOException {
		root = game.toAbsolutePath().normalize().resolve("sim-rosie-cache");
		Path path = root.resolve("library.json");
		index = Files.isRegularFile(path) ? JsonParser.parseString(Files.readString(path)).getAsJsonObject() : new JsonObject();
		if (index.has("version") && index.get("version").getAsInt() != 1) throw new IOException("Unsupported local library version");
		index.addProperty("version", 1);
		for (String category : List.of("policies", "environments")) {
			if (!index.has(category)) index.add(category, new JsonArray());
			for (var value : index.getAsJsonArray(category)) {
				JsonObject entry = value.getAsJsonObject();
				if (!entry.has("id") || !entry.has("name") || !entry.has("contract_hash")) throw new IOException("Invalid local library entry");
				(category.equals("policies") ? policies : environments).add(entry);
			}
		}
		if (!index.has("runs")) index.add("runs", new JsonArray());
		for (var value : index.getAsJsonArray("runs")) runs.add(value.getAsJsonObject());
		for (JsonObject policy : policies) {
			if (!policy.has("run")) continue;
			String identity = policy.get("run").getAsString();
			if (find(runs, identity) == null) {
				JsonObject run = new JsonObject();
				run.addProperty("id", identity);
				run.addProperty("name", identity);
				runs.add(run);
				index.getAsJsonArray("runs").add(run);
			}
		}
	}

	List<JsonObject> versions(String run) {
		return policies.stream().filter(entry -> entry.has("run") && entry.get("run").getAsString().equals(run))
				.sorted(java.util.Comparator.<JsonObject, Boolean>comparing(entry -> !entry.get("id").getAsString().endsWith(":latest.pt"))
						.thenComparing(java.util.Comparator.comparingLong((JsonObject entry) -> entry.has("update") ? entry.get("update").getAsLong() : 0).reversed()))
				.toList();
	}

	List<JsonObject> environmentChoices(String selected) {
		var choices = new java.util.LinkedHashMap<JsonObject, JsonObject>();
		for (JsonObject entry : environments) {
			JsonObject definition = entry.getAsJsonObject("environment").deepCopy();
			definition.remove("name");
			JsonObject key = new JsonObject();
			key.add("contract_hash", entry.get("contract_hash"));
			key.add("environment", definition);
			if (!choices.containsKey(key) || hasSource(entry, selected)) {
				JsonObject choice = entry.deepCopy();
				JsonObject environment = entry.getAsJsonObject("environment");
				if (environment.has("name") && !environment.get("name").getAsString().isBlank()) {
					choice.addProperty("name", environment.get("name").getAsString());
				}
				if (hasSource(entry, selected)) choice.addProperty("id", selected);
				choices.put(key, choice);
			}
		}
		var names = choices.values().stream().collect(java.util.stream.Collectors.groupingBy(
				choice -> choice.get("name").getAsString(), java.util.stream.Collectors.counting()));
		for (JsonObject choice : choices.values()) {
			String name = choice.get("name").getAsString();
			if (names.get(name) > 1) choice.addProperty("name", name + " / "
					+ sourceName(choice, choice.get("id").getAsString()));
		}
		return List.copyOf(choices.values());
	}

	List<JsonObject> environmentChoices(String selected, List<String> contractHashes) {
		return environmentChoices(selected).stream().filter(choice -> contractHashes.stream()
				.allMatch(hash -> choice.get("contract_hash").getAsString().equals(hash))).toList();
	}

	private static boolean hasSource(JsonObject entry, String identity) {
		if (entry.get("id").getAsString().equals(identity)) return true;
		if (!entry.has("sources")) return false;
		for (var source : entry.getAsJsonArray("sources")) {
			if (source.getAsJsonObject().get("id").getAsString().equals(identity)) return true;
		}
		return false;
	}

	private String sourceName(JsonObject entry, String identity) {
		if (entry.has("sources")) {
			for (var source : entry.getAsJsonArray("sources")) {
				JsonObject value = source.getAsJsonObject();
				if (value.get("id").getAsString().equals(identity)) return value.get("name").getAsString();
			}
		}
		JsonObject source = find(environments, identity);
		return source == null ? entry.get("name").getAsString() : source.get("name").getAsString();
	}

	static String versionName(JsonObject entry) {
		String identity = entry.get("id").getAsString();
		String version = identity.endsWith(":latest.pt") ? "Latest" : identity.substring(identity.lastIndexOf('/') + 1).replace(".pt", "");
		return version + " / update " + (entry.has("update") ? entry.get("update").getAsLong() : 0) + " / " + entry.get("name").getAsString();
	}

	JsonObject find(List<JsonObject> entries, String identity) {
		return entries.stream().filter(entry -> entry.get("id").getAsString().equals(identity)).findFirst().orElse(null);
	}

	JsonObject policyFighter(JsonObject entry, JsonObject environment) throws IOException {
		if (entry == null || environment == null) throw new IOException("Choose a local policy and environment first");
		if (!entry.get("contract_hash").equals(environment.get("contract_hash"))) throw new IOException("Policy layout does not match this environment");
		return policyFighter(entry, environment.get("contract_hash").getAsString());
	}

	JsonObject policyFighter(JsonObject entry, String contractHash) throws IOException {
		if (entry == null) throw new IOException("Choose a local policy first");
		if (!entry.get("contract_hash").getAsString().equals(contractHash)) throw new IOException("Policy layouts must match");
		Path path = Path.of(entry.get("path").getAsString()).toAbsolutePath().normalize();
		if (!Files.isRegularFile(path) || !path.toRealPath().startsWith(root.toRealPath())) throw new IOException("Policy is missing from the local cache; sync it again");
		JsonObject fighter = new JsonObject();
		fighter.addProperty("type", "checkpoint");
		fighter.addProperty("name", entry.get("name").getAsString());
		fighter.addProperty("path", path.toString());
		return fighter;
	}

	JsonObject worldPolicy(JsonObject first, JsonObject second) throws IOException {
		JsonObject anchor = worldPolicyEntry(first, second);
		return anchor.getAsJsonObject("policy").deepCopy();
	}

	String worldPolicyHash(JsonObject first, JsonObject second) throws IOException {
		return worldPolicyEntry(first, second).get("contract_hash").getAsString();
	}

	private JsonObject worldPolicyEntry(JsonObject first, JsonObject second) throws IOException {
		if (first != null && second != null && !first.get("contract_hash").equals(second.get("contract_hash"))) {
			throw new IOException("World-duel policies must use the same layout");
		}
		JsonObject anchor = first != null ? first : second;
		if (anchor == null) anchor = !policies.isEmpty() ? policies.getFirst() : environments.isEmpty() ? null : environments.getFirst();
		if (anchor == null) throw new IOException("Sync a policy layout before starting a World duel");
		if (!anchor.has("policy")) {
			String hash = anchor.get("contract_hash").getAsString();
			anchor = java.util.stream.Stream.concat(policies.stream(), environments.stream())
					.filter(entry -> entry.has("policy") && entry.get("contract_hash").getAsString().equals(hash)).findFirst().orElse(null);
		}
		if (anchor == null || !anchor.has("policy")) throw new IOException("Sync a policy layout before starting a World duel");
		return anchor;
	}

	static JsonObject scriptedFighter(String controller, JsonObject options) {
		String name = switch (controller) {
			case "basic_melee" -> "Basic melee";
			case "stationary_melee" -> "Stationary melee";
			case "tactical_melee" -> "Tactical melee";
			case "random_walk" -> "Random walk";
			case "stationary_target" -> "Stationary target";
			default -> throw new IllegalArgumentException("Unknown scripted fighter");
		};
		JsonObject fighter = new JsonObject();
		fighter.addProperty("type", "scripted");
		fighter.addProperty("controller", controller);
		fighter.addProperty("name", name);
		if (!controller.equals("stationary_target")) fighter.add("options", options.deepCopy());
		return fighter;
	}

	static String scriptedController(String mode) {
		return switch (mode) {
			case "Basic melee" -> "basic_melee";
			case "Stationary melee" -> "stationary_melee";
			case "Tactical melee" -> "tactical_melee";
			case "Random walk" -> "random_walk";
			case "Stationary target" -> "stationary_target";
			default -> throw new IllegalArgumentException("Unknown scripted fighter mode: " + mode);
		};
	}

	Path saveDuel(JsonObject environment, JsonObject first, JsonObject second) throws IOException {
		return saveDuel(environment, first, second, 0);
	}

	Path saveDuel(JsonObject environment, JsonObject first, JsonObject second, int maxTicks) throws IOException {
		if (environment == null) throw new IOException("Sync and choose an environment first");
		if (maxTicks < 0 || maxTicks > 1000000) throw new IOException("Invalid duel duration");
		JsonObject configuration = new JsonObject();
		configuration.addProperty("format", "minecraft_sim_duel");
		configuration.addProperty("version", 1);
		if (maxTicks > 0) configuration.addProperty("max_ticks", maxTicks);
		configuration.add("policy", environment.get("policy").deepCopy());
		configuration.add("environment", environment.get("environment").deepCopy());
		JsonArray fighters = new JsonArray();
		fighters.add(first.deepCopy());
		fighters.add(second.deepCopy());
		configuration.add("fighters", fighters);
		write(root.resolve("duel.json"), configuration);
		return root.resolve("duel.json");
	}

	Path saveWorldDuel(JsonObject policy, JsonObject first, JsonObject second, JsonObject firstLoadout,
			JsonObject secondLoadout, int maxTicks) throws IOException {
		if (policy == null) throw new IOException("Sync a policy layout before starting a World duel");
		if (maxTicks < 1 || maxTicks > 1000000) throw new IOException("World duels require an explicit duration from 1 to 1000000 ticks");
		JsonObject[] loadouts = {normalizeLoadout(firstLoadout), normalizeLoadout(secondLoadout)};
		new SimulationContract(policy).validateLoadouts(List.of(loadouts));
		JsonObject configuration = new JsonObject();
		configuration.addProperty("format", "minecraft_sim_duel");
		configuration.addProperty("version", 1);
		configuration.addProperty("max_ticks", maxTicks);
		configuration.add("policy", policy.deepCopy());
		configuration.add("environment", worldScenario(loadouts));
		JsonArray fighters = new JsonArray();
		fighters.add(first.deepCopy());
		fighters.add(second.deepCopy());
		configuration.add("fighters", fighters);
		write(root.resolve("duel.json"), configuration);
		return root.resolve("duel.json");
	}

	static JsonObject defaultWorldLoadout() {
		JsonObject loadout = new JsonObject();
		JsonArray weapons = new JsonArray();
		weapons.add("minecraft:diamond_sword");
		for (int index = 1; index < 9; index++) weapons.add("minecraft:air");
		loadout.add("weapons", weapons);
		loadout.addProperty("selected", 0);
		loadout.addProperty("offhand", "minecraft:air");
		JsonObject armor = new JsonObject();
		for (String slot : SimulationContract.ARMOR_SLOTS) armor.addProperty(slot, "minecraft:air");
		loadout.add("armor", armor);
		return loadout;
	}

	static JsonObject normalizeLoadout(JsonObject loadout) {
		JsonObject player = new JsonObject();
		player.add("loadout", loadout.deepCopy());
		return SimulationContract.loadout(player, "configured");
	}

	private static JsonObject worldScenario(JsonObject[] loadouts) {
		JsonObject scenario = new JsonObject();
		scenario.addProperty("format", "minecraft_sim_environment");
		scenario.addProperty("version", 1);
		scenario.addProperty("minecraft", "26.3");
		scenario.addProperty("profile", "configured");
		JsonObject bounds = new JsonObject();
		bounds.add("min", new com.google.gson.Gson().toJsonTree(new int[]{0, 0, 0}));
		bounds.add("size", new com.google.gson.Gson().toJsonTree(new int[]{1, 1, 1}));
		scenario.add("bounds", bounds);
		scenario.add("blocks", new JsonArray());
		JsonObject rules = new JsonObject();
		rules.addProperty("simulate_food", false);
		rules.addProperty("natural_regeneration", false);
		scenario.add("rules", rules);
		JsonArray players = new JsonArray();
		for (int index = 0; index < 2; index++) {
			JsonObject player = new JsonObject();
			player.add("position", new com.google.gson.Gson().toJsonTree(new double[]{0.5, 1, 0.5}));
			player.add("velocity", new com.google.gson.Gson().toJsonTree(new double[]{0, 0, 0}));
			player.addProperty("yaw", 0);
			player.addProperty("pitch", 0);
			player.addProperty("on_ground", false);
			player.addProperty("health", 20);
			player.addProperty("absorption", 0);
			player.addProperty("attack_strength_ticks", 20);
			player.add("loadout", loadouts[index].deepCopy());
			players.add(player);
		}
		scenario.add("players", players);
		return scenario;
	}

	static int durationTicks(String seconds) {
		double duration = Double.parseDouble(seconds);
		if (!Double.isFinite(duration) || duration <= 0 || duration > 50000) throw new IllegalArgumentException("Use a duration above 0 and up to 50000 seconds");
		return Math.max(1, (int) Math.round(duration * 20));
	}

	void rename(JsonObject entry, String name) throws IOException {
		name = name.strip();
		if (name.isEmpty() || name.length() > 80) throw new IOException("Use a name of 1 to 80 characters");
		entry.addProperty("name", name);
		write(root.resolve("library.json"), index);
	}

	static void write(Path path, JsonObject value) throws IOException {
		Files.createDirectories(path.getParent());
		Path temporary = Files.createTempFile(path.getParent(), "cambium-", ".tmp");
		try {
			Files.writeString(temporary, new com.google.gson.GsonBuilder().setPrettyPrinting().create().toJson(value) + System.lineSeparator());
			Files.move(temporary, path, StandardCopyOption.REPLACE_EXISTING);
		} finally {
			Files.deleteIfExists(temporary);
		}
	}
}