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
	}

	JsonObject find(List<JsonObject> entries, String identity) {
		return entries.stream().filter(entry -> entry.get("id").getAsString().equals(identity)).findFirst().orElse(null);
	}

	JsonObject policyFighter(JsonObject entry, JsonObject environment) throws IOException {
		if (entry == null || environment == null) throw new IOException("Choose a local policy and environment first");
		if (!entry.get("contract_hash").equals(environment.get("contract_hash"))) throw new IOException("Policy layout does not match this environment");
		Path path = Path.of(entry.get("path").getAsString()).toAbsolutePath().normalize();
		if (!Files.isRegularFile(path) || !path.toRealPath().startsWith(root.toRealPath())) throw new IOException("Policy is missing from the local cache; sync it again");
		JsonObject fighter = new JsonObject();
		fighter.addProperty("type", "checkpoint");
		fighter.addProperty("name", entry.get("name").getAsString());
		fighter.addProperty("path", path.toString());
		return fighter;
	}

	static JsonObject scriptedFighter(String controller, JsonObject options) {
		if (!List.of("basic_melee", "stationary_target").contains(controller)) throw new IllegalArgumentException("Unknown scripted fighter");
		JsonObject fighter = new JsonObject();
		fighter.addProperty("type", "scripted");
		fighter.addProperty("controller", controller);
		fighter.addProperty("name", controller.equals("basic_melee") ? "Basic melee" : "Stationary target");
		if (controller.equals("basic_melee")) fighter.add("options", options.deepCopy());
		return fighter;
	}

	Path saveDuel(JsonObject environment, JsonObject first, JsonObject second) throws IOException {
		if (environment == null) throw new IOException("Sync and choose an environment first");
		JsonObject configuration = new JsonObject();
		configuration.addProperty("format", "minecraft_sim_duel");
		configuration.addProperty("version", 1);
		configuration.add("policy", environment.get("policy").deepCopy());
		configuration.add("environment", environment.get("environment").deepCopy());
		JsonArray fighters = new JsonArray();
		fighters.add(first.deepCopy());
		fighters.add(second.deepCopy());
		configuration.add("fighters", fighters);
		write(root.resolve("duel.json"), configuration);
		return root.resolve("duel.json");
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