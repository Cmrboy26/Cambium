package net.cambium;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.UUID;

final class ReplayAnalysis {

	static final Gson JSON = new GsonBuilder().serializeNulls().create();
	private static final String[] STATE_FIELDS = {"position", "velocity", "yaw", "pitch", "on_ground",
			"sprinting", "pose", "health", "absorption", "damage_cooldown", "hurt_time", "attack_strength_ticks",
			"weapon", "shield_equipped", "using_shield", "shield_use_ticks", "shield_cooldown", "item_use_delay", "attack_suppressed", "armor"};
	private static final String[] OUTCOME_FIELDS = {"target", "attempted", "accepted", "damage", "health_lost"};
	final Path detailsPath;
	final Path summaryPath;
	private final BufferedWriter writer;
	private final JsonObject metadata;
	private final Map<String, Statistics> statistics = new LinkedHashMap<>();
	private final JsonObject firstByCategory = new JsonObject();
	private JsonObject firstDifference;
	private int ticks;
	private int actorTicks;
	private int divergentActorTicks;
	private boolean finished;

	ReplayAnalysis(Path folder, JsonObject metadata) throws IOException {
		Files.createDirectories(folder);
		String stamp = DateTimeFormatter.ofPattern("uuuuMMdd-HHmmss-SSS").withZone(ZoneOffset.UTC).format(Instant.now());
		String name = "analysis-" + stamp + "-" + UUID.randomUUID();
		detailsPath = folder.resolve(name + ".jsonl");
		summaryPath = folder.resolve(name + ".summary.json");
		this.metadata = metadata.deepCopy();
		writer = Files.newBufferedWriter(detailsPath, StandardCharsets.UTF_8, StandardOpenOption.CREATE_NEW);
		JsonObject header = new JsonObject();
		header.addProperty("format", "minecraft_sim_analysis");
		header.addProperty("version", 1);
		header.addProperty("type", "header");
		header.addProperty("created_utc", Instant.now().toString());
		header.add("metadata", this.metadata);
		JsonObject tolerances = new JsonObject();
		tolerances.addProperty("position_velocity", 1.0E-6);
		tolerances.addProperty("health_damage_angles", 1.0E-5);
		tolerances.addProperty("flags_counters_targets", 0);
		header.add("absolute_tolerances", tolerances);
		try {
			write(header);
			writer.flush();
		} catch (IOException error) {
			writer.close();
			throw error;
		}
	}

	void record(int tick, JsonArray expected, JsonArray actual, JsonObject phases) throws IOException {
		if (finished || tick != ticks + 1 || expected.size() != 2 || actual.size() != 2) {
			throw new IllegalArgumentException("Analysis requires consecutive input ticks and two actors");
		}
		JsonArray differences = new JsonArray();
		for (int actor = 0; actor < 2; actor++) {
			int before = differences.size();
			JsonObject reference = expected.get(actor).getAsJsonObject();
			JsonObject observed = actual.get(actor).getAsJsonObject();
			for (String field : STATE_FIELDS) {
				if (reference.has(field)) compare(tick, actor, field, reference.get(field), observed.get(field), differences);
			}
			if (reference.has("outcome")) {
				for (String field : OUTCOME_FIELDS) {
					JsonObject outcome = reference.getAsJsonObject("outcome");
					if (outcome.has(field)) compare(tick, actor, "outcome." + field, outcome.get(field),
							observed.getAsJsonObject("outcome").get(field), differences);
				}
			}
			for (String vector : new String[]{"position", "velocity"}) {
				if (!reference.has(vector)) continue;
				JsonArray referenceVector = reference.getAsJsonArray(vector);
				JsonArray observedVector = observed.getAsJsonArray(vector);
				double squared = 0;
				for (int axis = 0; axis < 3; axis++) {
					double delta = observedVector.get(axis).getAsDouble() - referenceVector.get(axis).getAsDouble();
					squared += delta * delta;
				}
				compare(tick, actor, vector + "_error", JSON.toJsonTree(0), JSON.toJsonTree(Math.sqrt(squared)), differences);
			}
			actorTicks++;
			if (differences.size() != before) divergentActorTicks++;
		}
		JsonObject row = new JsonObject();
		row.addProperty("type", "tick");
		row.addProperty("tick", tick);
		row.add("expected", expected);
		row.add("actual", actual);
		row.add("phases", phases);
		row.add("differences", differences);
		write(row);
		ticks = tick;
		if (tick % 20 == 0) writer.flush();
	}

	private void compare(int tick, int actor, String field, JsonElement expected, JsonElement actual, JsonArray differences) {
		if (actual == null || actual.isJsonNull()) throw new IllegalArgumentException("Missing observed " + field);
		if (expected.isJsonObject()) {
			for (var entry : expected.getAsJsonObject().entrySet()) {
				compare(tick, actor, field + "." + entry.getKey(), entry.getValue(), actual.getAsJsonObject().get(entry.getKey()), differences);
			}
			return;
		}
		if (expected.isJsonArray()) {
			for (int axis = 0; axis < expected.getAsJsonArray().size(); axis++) {
				compare(tick, actor, field + "." + "xyz".charAt(axis), expected.getAsJsonArray().get(axis),
						actual.getAsJsonArray().get(axis), differences);
			}
			return;
		}
		boolean numeric = expected.getAsJsonPrimitive().isNumber();
		double error = numeric ? Math.abs(actual.getAsDouble() - expected.getAsDouble()) : expected.equals(actual) ? 0 : 1;
		double tolerance = field.startsWith("position") || field.startsWith("velocity") ? 1.0E-6
				: field.equals("yaw") || field.equals("pitch") || field.equals("health") || field.equals("absorption")
				|| field.equals("outcome.damage") || field.equals("outcome.health_lost") ? 1.0E-5 : 0;
		boolean mismatch = error > tolerance;
		statistics.computeIfAbsent(actor + "." + field, key -> new Statistics()).add(tick, error, mismatch);
		if (!mismatch) return;
		JsonObject difference = new JsonObject();
		difference.addProperty("tick", tick);
		difference.addProperty("actor", actor);
		difference.addProperty("field", field);
		difference.add("expected", expected);
		difference.add("actual", actual);
		difference.addProperty("absolute_error", error);
		difference.addProperty("tolerance", tolerance);
		String category = field.equals("outcome.target") || field.equals("outcome.attempted") ? "targeting"
				: field.startsWith("position") || field.startsWith("velocity") ? "movement"
				: field.startsWith("outcome.") || field.equals("health") || field.equals("absorption")
				|| field.equals("damage_cooldown") || field.equals("hurt_time") || field.equals("attack_strength_ticks") ? "combat" : "state";
		difference.addProperty("category", category);
		differences.add(difference);
		if (firstDifference == null) firstDifference = difference.deepCopy();
		if (!firstByCategory.has(category)) firstByCategory.add(category, difference.deepCopy());
	}

	int firstDivergentTick() {
		return firstDifference == null ? -1 : firstDifference.get("tick").getAsInt();
	}

	void finish(String status, String reason) throws IOException {
		if (finished) return;
		finished = true;
		JsonObject summary = new JsonObject();
		summary.addProperty("format", "minecraft_sim_analysis_summary");
		summary.addProperty("version", 1);
		summary.addProperty("status", status);
		summary.addProperty("reason", reason);
		summary.add("metadata", metadata);
		summary.addProperty("compared_ticks", ticks);
		summary.addProperty("actor_ticks", actorTicks);
		summary.addProperty("divergent_actor_ticks", divergentActorTicks);
		summary.addProperty("within_tolerance_actor_ticks", actorTicks - divergentActorTicks);
		summary.add("first_difference", firstDifference == null ? com.google.gson.JsonNull.INSTANCE : firstDifference);
		summary.add("first_by_category", firstByCategory);
		JsonObject fields = new JsonObject();
		statistics.forEach((field, values) -> fields.add(field, values.json()));
		summary.add("fields", fields);
		summary.addProperty("details_file", detailsPath.toAbsolutePath().toString());
		JsonObject footer = new JsonObject();
		footer.addProperty("type", "summary");
		footer.add("summary", summary);
		try (writer) {
			write(footer);
		}
		Files.writeString(summaryPath, new GsonBuilder().serializeNulls().setPrettyPrinting().create().toJson(summary) + System.lineSeparator(),
				StandardCharsets.UTF_8, StandardOpenOption.CREATE_NEW);
	}

	private void write(JsonObject value) throws IOException {
		writer.write(JSON.toJson(value));
		writer.newLine();
	}

	private static final class Statistics {
		int count;
		int mismatches;
		int first = -1;
		int worst;
		double sum;
		double maximum;

		void add(int tick, double error, boolean mismatch) {
			count++;
			sum += error;
			if (count == 1 || error > maximum) {
				maximum = error;
				worst = tick;
			}
			if (mismatch) {
				mismatches++;
				if (first < 0) first = tick;
			}
		}

		JsonObject json() {
			JsonObject result = new JsonObject();
			result.addProperty("samples", count);
			result.addProperty("mismatches", mismatches);
			result.addProperty("mean_absolute_error", sum / count);
			result.addProperty("max_absolute_error", maximum);
			result.addProperty("worst_tick", worst);
			result.addProperty("first_divergent_tick", first);
			return result;
		}
	}
}