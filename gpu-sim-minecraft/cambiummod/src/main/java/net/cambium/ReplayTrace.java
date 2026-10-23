package net.cambium;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;
import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

final class ReplayTrace {

	record Controls(boolean forward, boolean backward, boolean left, boolean right,
			boolean jump, boolean sprint, boolean sneak, boolean attack, float yaw_delta, float pitch_delta,
			boolean select_weapon, String weapon, boolean use_item, Integer select_slot) {
		Controls {
			if (weapon == null) weapon = "minecraft:diamond_sword";
			if (select_slot == null) select_slot = -1;
			if (select_slot < -1 || select_slot >= PolicyCatalogue.MAX_WEAPON_SLOTS || !select_weapon && select_slot != -1) {
				throw new IllegalArgumentException("Invalid semantic selected slot");
			}
		}
		Controls(boolean forward, boolean backward, boolean left, boolean right, boolean jump, boolean sprint,
				boolean sneak, boolean attack, float yaw, float pitch, boolean select, String weapon, boolean use) {
			this(forward, backward, left, right, jump, sprint, sneak, attack, yaw, pitch, select, weapon, use, -1);
		}
		Controls(boolean forward, boolean backward, boolean left, boolean right, boolean jump, boolean sprint,
				boolean sneak, boolean attack, float yaw, float pitch) {
			this(forward, backward, left, right, jump, sprint, sneak, attack, yaw, pitch, false, "minecraft:diamond_sword", false);
		}
	}
	record Outcome(int target, boolean attempted, boolean accepted, boolean critical,
			boolean sprint_bonus, boolean sweep, float damage, float health_lost, float blocked_damage, boolean shield_disabled) {
		Outcome(int target, boolean attempted, boolean accepted, boolean critical, boolean sprint, boolean sweep, float damage, float lost) {
			this(target, attempted, accepted, critical, sprint, sweep, damage, lost, 0, false);
		}
	}
	record Actor(int id, double[] position, double[] velocity, float yaw, float pitch,
			boolean on_ground, boolean sprinting, int pose, float health, float absorption,
			int damage_cooldown, int hurt_time, int attack_strength_ticks, Controls controls, Outcome outcome,
			String weapon, boolean shield_equipped, boolean using_shield, int shield_use_ticks, int shield_cooldown,
			int item_use_delay, boolean attack_suppressed, JsonObject armor, Integer weapon_slot,
			int arrow_count, boolean bow_using, int bow_draw_ticks, float bow_power) {
		Actor {
			if (weapon == null) weapon = "minecraft:diamond_sword";
			if (armor == null) armor = new JsonObject();
			if (weapon_slot == null) weapon_slot = -1;
			if (weapon_slot < -1 || weapon_slot >= PolicyCatalogue.MAX_WEAPON_SLOTS) throw new IllegalArgumentException("Invalid actor weapon slot");
		}
		Actor(int id, double[] position, double[] velocity, float yaw, float pitch, boolean ground, boolean sprint,
				int pose, float health, float absorption, int cooldown, int hurt, int charge, Controls controls, Outcome outcome) {
			this(id, position, velocity, yaw, pitch, ground, sprint, pose, health, absorption, cooldown, hurt, charge,
					controls, outcome, "minecraft:diamond_sword", false, false, 0, 0, 0, false, new JsonObject(), -1,
					0, false, 0, 0);
		}
		Actor withLoadout(JsonObject loadout) {
			return new Actor(id, position, velocity, yaw, pitch, on_ground, sprinting, pose, health, absorption,
					damage_cooldown, hurt_time, attack_strength_ticks, controls, outcome,
					loadout.getAsJsonArray("weapons").get(loadout.get("selected").getAsInt()).getAsString(),
					loadout.get("offhand").getAsString().equals("minecraft:shield"), false, 0, 0, 0, false,
					loadout.getAsJsonObject("armor").deepCopy(), loadout.get("selected").getAsInt(),
					loadout.has("arrows") ? loadout.get("arrows").getAsInt() : 0, false, 0, 0);
		}
		Actor at(double[] position, float yaw, float pitch) {
			return new Actor(id, position, new double[]{0, 0, 0}, yaw, pitch, false, false, 0, health, absorption,
					0, 0, attack_strength_ticks, controls, outcome, weapon, shield_equipped, false, 0, 0, 0, false, armor.deepCopy(), weapon_slot,
					arrow_count, bow_using, bow_draw_ticks, bow_power);
		}
	}
	record Projectile(int owner, double[] position, double[] velocity, int age, boolean left_owner,
			boolean in_ground, int in_ground_ticks, boolean critical) {}
	record Frame(int tick, List<Actor> actors, List<Projectile> projectiles) {
		Frame(int tick, List<Actor> actors) { this(tick, actors, List.of()); }
	}

	static final class Environment {
		final JsonObject definition;
		final int[] min;
		final int[] size;
		final byte[] cells;
		final List<JsonObject> loadouts = new ArrayList<>();

		Environment(JsonObject definition) throws IOException {
			this.definition = definition;
			if (!definition.get("format").getAsString().equals("minecraft_sim_environment")
					|| definition.get("version").getAsInt() != 1
					|| !definition.get("minecraft").getAsString().equals("26.3")
					|| !List.of("configured", "unarmored_diamond_sword", "unarmored_sword_axe_shield").contains(
							definition.has("profile") ? definition.get("profile").getAsString() : "configured")) {
				throw new IOException("Unsupported embedded environment format/version/build/profile");
			}
			SimulationContract.capabilities(definition);
			if (definition.has("rules")) {
				JsonObject rules = definition.getAsJsonObject("rules");
				for (String rule : rules.keySet()) {
					if ((!rule.equals("simulate_food") && !rule.equals("natural_regeneration"))
							|| !rules.get(rule).equals(ReplayAnalysis.JSON.toJsonTree(false))) {
						throw new IOException("Scenario bridge requires frozen food and no natural regeneration");
					}
				}
			}
			min = integerVector(definition.getAsJsonObject("bounds").get("min"));
			size = integerVector(definition.getAsJsonObject("bounds").get("size"));
			int volume = 1;
			for (int extent : size) {
				if (extent < 1 || extent > 128) throw new IOException("Environment dimensions must be in 1..128");
				volume *= extent;
			}
			if (volume > 32768) throw new IOException("Environment exceeds 32768 cells");
			cells = new byte[volume];
			var fills = definition.getAsJsonArray("blocks");
			if (fills.size() > 256) throw new IOException("Environment exceeds 256 fill regions");
			for (JsonElement element : fills) {
				JsonObject fill = element.getAsJsonObject();
				int[] from = integerVector(fill.get("from"));
				int[] to = integerVector(fill.get("to"));
				String block = fill.get("block").getAsString();
				if (!block.equals("minecraft:air") && !block.equals("minecraft:stone")) {
					throw new IOException("Only minecraft:air and minecraft:stone are supported");
				}
				for (int axis = 0; axis < 3; axis++) {
					if (from[axis] < min[axis] || from[axis] > to[axis] || to[axis] >= min[axis] + size[axis]) {
						throw new IOException("Environment fill is outside bounds");
					}
				}
				for (int blockZ = from[2]; blockZ <= to[2]; blockZ++) {
					for (int blockY = from[1]; blockY <= to[1]; blockY++) {
						for (int blockX = from[0]; blockX <= to[0]; blockX++) {
							cells[((blockZ - min[2]) * size[1] + blockY - min[1]) * size[0] + blockX - min[0]]
									= (byte) (block.equals("minecraft:stone") ? 1 : 0);
						}
					}
				}
			}
			if (definition.getAsJsonArray("players").size() != 2) {
				throw new IOException("Environment needs two initial players");
			}
			for (JsonElement player : definition.getAsJsonArray("players")) {
				loadouts.add(SimulationContract.loadout(player.getAsJsonObject(), definition.has("profile") ? definition.get("profile").getAsString() : "configured"));
			}
			boolean bowLoadout = loadouts.stream().anyMatch(loadout -> loadout.getAsJsonArray("weapons")
					.contains(ReplayAnalysis.JSON.toJsonTree("minecraft:bow")));
			boolean bowRequired = definition.has("requires")
					&& definition.getAsJsonArray("requires").contains(ReplayAnalysis.JSON.toJsonTree("bow"));
			if (bowLoadout && !bowRequired) throw new IOException("Bow loadouts require the bow capability");
		}

		void validateInitial(Frame frame, Gson gson) throws IOException {
			for (int index = 0; index < 2; index++) {
				JsonObject spawn = definition.getAsJsonArray("players").get(index).getAsJsonObject();
				Actor actor = frame.actors().get(index);
				JsonObject loadout = loadouts.get(index);
				if (!actor.weapon().equals(loadout.getAsJsonArray("weapons").get(loadout.get("selected").getAsInt()).getAsString())
						|| actor.arrow_count() != (loadout.has("arrows") ? loadout.get("arrows").getAsInt() : 0)
						|| actor.bow_using() || actor.bow_draw_ticks() != 0 || actor.bow_power() != 0
						|| actor.weapon_slot() >= 0 && actor.weapon_slot() != loadout.get("selected").getAsInt()
						|| actor.shield_equipped() != loadout.get("offhand").getAsString().equals("minecraft:shield")
						|| actor.using_shield() || actor.shield_use_ticks() != 0 || actor.shield_cooldown() != 0
						|| actor.item_use_delay() != 0 || actor.attack_suppressed()) {
					throw new IOException("Frame zero equipment does not match embedded initial player " + index);
				}
				if (!Arrays.equals(actor.position(), gson.fromJson(spawn.get("position"), double[].class))
						|| !Arrays.equals(actor.velocity(), gson.fromJson(spawn.get("velocity"), double[].class))
						|| actor.yaw() != spawn.get("yaw").getAsFloat() || actor.pitch() != spawn.get("pitch").getAsFloat()
						|| actor.on_ground() != spawn.get("on_ground").getAsBoolean() || actor.health() <= 0
						|| actor.health() != (spawn.has("health") ? spawn.get("health").getAsFloat() : 20)
						|| actor.absorption() != (spawn.has("absorption") ? spawn.get("absorption").getAsFloat() : 0)
						|| actor.attack_strength_ticks() != (spawn.has("attack_strength_ticks") ? spawn.get("attack_strength_ticks").getAsInt() : 0)) {
					throw new IOException("Frame zero does not match embedded initial player " + index);
				}
				for (int axis = 0; axis < 3; axis++) {
					if (actor.position()[axis] < min[axis] || actor.position()[axis] >= min[axis] + size[axis]) {
						throw new IOException("Initial player is outside environment bounds");
					}
				}
			}
		}

		private static int[] integerVector(JsonElement element) throws IOException {
			var values = element.getAsJsonArray();
			if (values.size() != 3) throw new IOException("Environment coordinates need three integers");
			int[] result = new int[3];
			for (int axis = 0; axis < 3; axis++) {
				result[axis] = values.get(axis).getAsBigDecimal().intValueExact();
				if (Math.abs((long) result[axis]) > 4096) throw new IOException("Environment coordinates exceed -4096..4096");
			}
			return result;
		}
	}

	final List<Frame> frames;
	final Environment environment;
	final String backend;
	final Path path;
	final JsonObject header;

	private ReplayTrace(Path path, List<Frame> frames, String backend, Environment environment, JsonObject header) {
		this.path = path;
		this.frames = List.copyOf(frames);
		this.backend = backend;
		this.environment = environment;
		this.header = header.deepCopy();
	}

	static ReplayTrace read(Path path) throws IOException {
		if (Files.size(path) > 64L * 1024 * 1024) {
			throw new IOException("Replay exceeds the 64 MiB viewer limit");
		}
		Gson gson = new Gson();
		try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
			JsonObject header = JsonParser.parseString(reader.readLine()).getAsJsonObject();
			if (!header.get("format").getAsString().equals("minecraft_sim_replay")
					|| (header.get("version").getAsInt() != 2 && header.get("version").getAsInt() != 3)
					|| !header.get("minecraft").getAsString().equals("26.3")
					|| header.get("players").getAsInt() != 2
					|| header.get("ticks_per_second").getAsInt() != 20) {
				throw new IOException("Replay needs a supported envelope with an embedded environment");
			}
			Environment environment = new Environment(header.getAsJsonObject("environment"));
			List<Frame> frames = new ArrayList<>();
			String line;
			while ((line = reader.readLine()) != null) {
				JsonObject raw = JsonParser.parseString(line).getAsJsonObject();
				Frame decoded = gson.fromJson(raw, Frame.class);
				List<Projectile> projectiles = new ArrayList<>();
				if (raw.has("projectiles")) {
					if (!raw.get("projectiles").isJsonArray()) throw new IOException("Frame projectiles must be an array");
					for (JsonElement element : raw.getAsJsonArray("projectiles")) {
						if (!element.isJsonObject()) throw new IOException("Projectile state must be an object");
						JsonObject projectileJson = element.getAsJsonObject();
						for (String field : List.of("owner", "position", "velocity", "age", "left_owner", "in_ground", "in_ground_ticks", "critical")) {
							if (!projectileJson.has(field)) throw new IOException("Missing projectile field: " + field);
						}
						for (String field : List.of("left_owner", "in_ground", "critical")) {
							JsonElement value = projectileJson.get(field);
							if (!value.isJsonPrimitive() || !value.getAsJsonPrimitive().isBoolean()) throw new IOException("Invalid projectile boolean: " + field);
						}
						for (String field : List.of("position", "velocity")) {
							if (!projectileJson.get(field).isJsonArray() || projectileJson.getAsJsonArray(field).size() != 3) {
								throw new IOException("Projectile " + field + " needs three numeric coordinates");
							}
							for (JsonElement coordinate : projectileJson.getAsJsonArray(field)) {
								if (!coordinate.isJsonPrimitive() || !coordinate.getAsJsonPrimitive().isNumber()) {
									throw new IOException("Projectile coordinates must be numeric");
								}
							}
						}
						int owner = SimulationContract.integer(projectileJson.get("owner"));
						int age = SimulationContract.integer(projectileJson.get("age"));
						int inGroundTicks = SimulationContract.integer(projectileJson.get("in_ground_ticks"));
						Projectile projectile = gson.fromJson(projectileJson, Projectile.class);
						if (projectile == null || owner < 0 || owner > 1 || age < 0 || age > 1000000
								|| inGroundTicks < 0 || inGroundTicks > 1000000
								|| !vector(projectile.position(), 29999984) || !vector(projectile.velocity(), 1000)) {
							throw new IOException("Invalid projectile state at tick " + frames.size());
						}
						projectiles.add(projectile);
					}
				}
			Frame frame = decoded == null ? null : new Frame(decoded.tick(), decoded.actors(), projectiles);
				if (frame == null || frame.tick() != frames.size() || frame.actors() == null
						|| frame.actors().size() != 2 || frames.size() > 100000) {
					throw new IOException("Replay needs consecutive ticks from zero and exactly two actors per tick");
				}
				for (int index = 0; index < 2; index++) {
					Actor actor = frame.actors().get(index);
					if (actor == null) throw new IOException("Invalid actor state at tick " + frame.tick());
					JsonObject actorJson = raw.getAsJsonArray("actors").get(index).getAsJsonObject();
					for (String field : List.of("arrow_count", "bow_using", "bow_draw_ticks", "bow_power")) {
						if (actorJson.has(field) && actorJson.get(field).isJsonNull()) throw new IOException("Invalid bow state field: " + field);
					}
					if (actorJson.has("arrow_count")) SimulationContract.integer(actorJson.get("arrow_count"));
					if (actorJson.has("bow_draw_ticks")) SimulationContract.integer(actorJson.get("bow_draw_ticks"));
					if (actorJson.has("bow_using") && (!actorJson.get("bow_using").isJsonPrimitive()
							|| !actorJson.get("bow_using").getAsJsonPrimitive().isBoolean())) throw new IOException("Invalid bow_using flag");
					if (actorJson.has("bow_power") && (!actorJson.get("bow_power").isJsonPrimitive()
							|| !actorJson.get("bow_power").getAsJsonPrimitive().isNumber())) throw new IOException("Invalid bow_power value");
					if (actor == null || actor.id() != index || !vector(actor.position(), 29999984)
							|| !vector(actor.velocity(), 1000) || !Float.isFinite(actor.yaw())
							|| !Float.isFinite(actor.pitch()) || Math.abs(actor.pitch()) > 90
							|| !Float.isFinite(actor.health()) || actor.health() < 0 || actor.health() > 20
							|| !Float.isFinite(actor.absorption()) || actor.absorption() < 0
							|| actor.pose() < 0 || actor.pose() > 2 || actor.damage_cooldown() < 0
							|| actor.hurt_time() < 0 || actor.attack_strength_ticks() < 0
							|| actor.controls() == null || actor.outcome() == null
							|| !SimulationContract.weaponItem(actor.weapon()) || actor.shield_use_ticks() < 0 || actor.shield_use_ticks() > 72000
							|| actor.shield_cooldown() < 0 || actor.shield_cooldown() > 100 || actor.item_use_delay() < 0 || actor.item_use_delay() > 4
							|| actor.using_shield() && (!actor.shield_equipped() || actor.shield_cooldown() > 0)
							|| actor.arrow_count() < 0 || actor.arrow_count() > 64 || actor.bow_draw_ticks() < 0 || actor.bow_draw_ticks() > 72000
							|| !Float.isFinite(actor.bow_power()) || actor.bow_power() < 0 || actor.bow_power() > 1
							|| actor.bow_using() && !actor.weapon().equals("minecraft:bow")
							|| !actor.bow_using() && (actor.bow_draw_ticks() != 0 || actor.bow_power() != 0)
							|| !SimulationContract.weaponItem(actor.controls().weapon())
							|| !Float.isFinite(actor.controls().yaw_delta())
							|| !Float.isFinite(actor.controls().pitch_delta())) {
						throw new IOException("Invalid actor state at tick " + frame.tick());
					}
					if (header.get("version").getAsInt() == 3) {
						if (actorJson.has("weapon_slot")) SimulationContract.integer(actorJson.get("weapon_slot"));
						if (actorJson.getAsJsonObject("controls").has("select_slot")) {
							SimulationContract.integer(actorJson.getAsJsonObject("controls").get("select_slot"));
						}
						for (String field : List.of("weapon", "shield_equipped", "using_shield", "shield_use_ticks", "shield_cooldown",
								"item_use_delay", "attack_suppressed", "armor")) {
							if (!actorJson.has(field)) throw new IOException("Missing explicit actor equipment field: " + field);
						}
						for (String field : List.of("select_weapon", "weapon", "use_item")) {
							if (!actorJson.getAsJsonObject("controls").has(field)) throw new IOException("Missing semantic control: " + field);
						}
						JsonObject loadout = environment.loadouts.get(index);
						if (actor.weapon_slot() >= 0) SimulationContract.selectedSlot(loadout, actor.weapon(), actor.weapon_slot(), -1);
						if (actor.controls().select_slot() >= 0) {
							SimulationContract.selectedSlot(loadout, actor.controls().weapon(), actor.controls().select_slot(), -1);
							if (actor.weapon_slot() >= 0 && !actor.weapon_slot().equals(actor.controls().select_slot())) {
								throw new IOException("Actor slot disagrees with selected control at tick " + frame.tick());
							}
						}
						if (!loadout.getAsJsonArray("weapons").contains(ReplayAnalysis.JSON.toJsonTree(actor.weapon()))
								|| actor.shield_equipped() != loadout.get("offhand").getAsString().equals("minecraft:shield")
								|| !actor.armor().equals(loadout.getAsJsonObject("armor"))
								|| actor.controls().select_weapon() && !loadout.getAsJsonArray("weapons").contains(ReplayAnalysis.JSON.toJsonTree(actor.controls().weapon()))) {
							throw new IOException("Actor equipment disagrees with scenario loadout at tick " + frame.tick());
						}
					}
				}
				frames.add(frame);
			}
			if (frames.size() < 2) {
				throw new IOException("Replay has no transitions");
			}
			environment.validateInitial(frames.getFirst(), gson);
			return new ReplayTrace(path, frames, header.get("backend").getAsString(), environment, header);
		} catch (RuntimeException error) {
			throw new IOException("Malformed replay JSON: " + error.getMessage(), error);
		}
	}

	static ReplayTrace forLive(JsonObject definition) throws IOException {
		Environment environment = new Environment(definition.deepCopy());
		List<Actor> actors = new ArrayList<>();
		for (int index = 0; index < 2; index++) {
			JsonObject spawn = definition.getAsJsonArray("players").get(index).getAsJsonObject();
			Actor actor = new Actor(index, new Gson().fromJson(spawn.get("position"), double[].class),
					new Gson().fromJson(spawn.get("velocity"), double[].class), spawn.get("yaw").getAsFloat(),
					spawn.get("pitch").getAsFloat(), spawn.get("on_ground").getAsBoolean(), false, 0,
					spawn.has("health") ? spawn.get("health").getAsFloat() : 20,
					spawn.has("absorption") ? spawn.get("absorption").getAsFloat() : 0, 0, 0,
					spawn.has("attack_strength_ticks") ? spawn.get("attack_strength_ticks").getAsInt() : 0,
					new Controls(false, false, false, false, false, false, false, false, 0, 0),
					new Outcome(-1, false, false, false, false, false, 0, 0));
			if (!vector(actor.position(), 29999984) || !vector(actor.velocity(), 1000)
					|| !Float.isFinite(actor.yaw()) || !Float.isFinite(actor.pitch()) || Math.abs(actor.pitch()) > 90
					|| !Float.isFinite(actor.health()) || actor.health() <= 0 || actor.health() > 20
					|| !Float.isFinite(actor.absorption()) || actor.absorption() < 0 || actor.attack_strength_ticks() < 0) {
				throw new IOException("Invalid checkpoint spawn");
			}
			actors.add(actor.withLoadout(environment.loadouts.get(index)));
		}
		Frame initial = new Frame(0, actors);
		environment.validateInitial(initial, new Gson());
		JsonObject header = new JsonObject();
		header.addProperty("format", "minecraft_sim_live");
		header.add("environment", definition);
		return new ReplayTrace(Path.of("live-duel"), List.of(initial), "LIVE", environment, header);
	}

	private static boolean vector(double[] values, double limit) {
		return values != null && values.length == 3
				&& Double.isFinite(values[0]) && Math.abs(values[0]) <= limit
				&& Double.isFinite(values[1]) && Math.abs(values[1]) <= limit
				&& Double.isFinite(values[2]) && Math.abs(values[2]) <= limit;
	}
}