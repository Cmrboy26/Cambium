package net.cambium;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.CompletableFuture;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.phys.Vec3;

final class BridgePolicyClient implements AutoCloseable {
	private static final Set<String> LIVE_FEATURES = Set.of(
			"self.health_fraction", "self.yaw_fraction", "self.pitch_fraction", "self.velocity.x", "self.velocity.y", "self.velocity.z",
			"self.on_ground", "self.weapon_slot", "self.arrow_count_fraction", "self.bow_using", "self.bow_power_fraction",
			"opponent.alive", "opponent.health_fraction", "opponent.on_ground", "relative.x", "relative.y", "relative.z",
			"relative.velocity.x", "relative.velocity.y", "relative.velocity.z", "target.distance_fraction", "target.aim_dot",
			"target.yaw_error", "target.pitch_error", "arena.ray_world_0", "arena.ray_world_45", "arena.ray_world_90",
			"arena.ray_world_135", "arena.ray_world_180", "arena.ray_world_225", "arena.ray_world_270", "arena.ray_world_315");
	private static final Set<String> LIVE_ACTIONS = Set.of("movement", "jump", "sprint", "sneak", "attack", "yaw", "pitch", "weapon", "use_item");
	private static final List<String> DESCRIPTORS = List.of(
			"d708928f0009e2176270447a1bd327edb265ed66fb5ae4ba964c66ab21196380",
			"6c073518d3c85671184a0120166926af19cae9335bf7c43eee0df18c7d495201");
	private final HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(2)).build();
	private final URI base;
	private SimulationContract contract;
	private ReplayTrace.Environment environment;
	private List<JsonObject> loadouts;
	private String contractHash;
	private String descriptorHash;
	private int matchFeatures;

	BridgePolicyClient(int port) {
		if (port < 1 || port > 65535) throw new IllegalArgumentException("Invalid inference port");
		base = URI.create("http://127.0.0.1:" + port);
	}

	CompletableFuture<JsonObject> info() {
		return request("/info", null).thenApply(info -> {
			negotiate(info);
			return info;
		});
	}

	void negotiate(JsonObject info) {
		if (SimulationContract.integer(info.get("protocol")) != 2
				|| !PolicyCatalogue.HASH.equals(info.get("catalogue_hash").getAsString())) {
			throw new IllegalArgumentException("Unsupported Bridge inference protocol or catalogue");
		}
		JsonObject descriptor = info.getAsJsonObject("bridge_observation");
		String negotiatedDescriptorHash = digest(canonical(descriptor));
		if (!DESCRIPTORS.contains(negotiatedDescriptorHash) || !negotiatedDescriptorHash.equals(info.get("bridge_observation_hash").getAsString())) {
			throw new IllegalArgumentException("Unsupported Bridge observation descriptor");
		}
		int negotiatedMatchFeatures = descriptor.getAsJsonArray("match_features").size();
		String negotiatedContractHash = info.get("contract_hash").getAsString();
		if (!negotiatedContractHash.matches("[0-9a-f]{64}")) throw new IllegalArgumentException("Invalid policy contract fingerprint");
		SimulationContract negotiatedContract = new SimulationContract(info.getAsJsonObject("contract"));
		ReplayTrace.Environment parsed;
		try {
			parsed = new ReplayTrace.Environment(info.getAsJsonObject("environment").deepCopy());
			BridgeMatch.fromEnvironment(parsed, Vec3.ZERO);
			JsonObject bridge = parsed.definition.getAsJsonObject("experimental_bridge");
			if (bridge.has("kits")) {
				JsonArray kits = bridge.getAsJsonArray("kits");
				if (kits.size() != 2) throw new IllegalArgumentException("Bridge requires two kits");
				for (int actor = 0; actor < 2; actor++) new BridgeKit(kits.get(actor).getAsJsonObject(), parsed.loadouts.get(actor));
			}
			Map<Integer, Integer> initial = bridge.has("terrain_generation")
					? BridgeTerrainGenerator.generate(parsed, bridge.getAsJsonObject("terrain_generation"), 0) : Map.of();
			if (bridge.has("building")) new BridgeArena(parsed, bridge.getAsJsonObject("building"), initial);
		} catch (java.io.IOException error) {
			throw new IllegalArgumentException("Invalid Bridge scenario", error);
		}
		negotiatedContract.validateLoadouts(parsed.loadouts);
		descriptorHash = negotiatedDescriptorHash;
		matchFeatures = negotiatedMatchFeatures;
		contractHash = negotiatedContractHash;
		loadouts = parsed.loadouts;
		environment = parsed;
		contract = negotiatedContract;
	}

	int observationCount() { return contract.observationCount(); }
	int matchFeatureCount() { return matchFeatures; }
	void requireLiveSupport() {
		if (contract == null) throw new IllegalStateException("Bridge policy is not negotiated");
		contract.requireSupported(LIVE_FEATURES, LIVE_ACTIONS);
	}
	float[] encode(ServerPlayer source, ServerPlayer target, float[] rays) {
		if (contract == null) throw new IllegalStateException("Bridge policy is not negotiated");
		return LivePolicy.encode(source, target, rays, contract);
	}
	BridgeMatch newMatch(Vec3 origin) {
		if (environment == null) throw new IllegalStateException("Bridge policy is not negotiated");
		return BridgeMatch.fromEnvironment(environment, origin);
	}
	Map<Integer, Integer> initialTerrain(long playerRandomState) {
		if (environment == null) throw new IllegalStateException("Bridge policy is not negotiated");
		JsonObject bridge = environment.definition.getAsJsonObject("experimental_bridge");
		return bridge.has("terrain_generation")
				? BridgeTerrainGenerator.generate(environment, bridge.getAsJsonObject("terrain_generation"),
						BridgeTerrainGenerator.seed(playerRandomState)) : Map.of();
	}

	CompletableFuture<JsonObject> act(String session, int tick, int[] seats, float[][] scalar, float[][] match, byte[][][][] voxels) {
		if (contract == null || session == null || session.isEmpty() || session.length() > 64 || tick < 0
				|| seats == null || seats.length < 1 || seats.length > 2 || scalar == null || match == null || voxels == null
				|| scalar.length != seats.length || match.length != seats.length || voxels.length != seats.length) {
			throw new IllegalArgumentException("Invalid Bridge inference request");
		}
		for (int actor = 0; actor < seats.length; actor++) {
			if (seats[actor] < 0 || seats[actor] > 1 || actor > 0 && seats[actor] == seats[0]
					|| scalar[actor] == null || scalar[actor].length != observationCount()
					|| match[actor] == null || match[actor].length != matchFeatures
					|| voxels[actor] == null || voxels[actor].length != 5) throw new IllegalArgumentException("Invalid Bridge input shape or seat");
			for (float value : scalar[actor]) if (!Float.isFinite(value)) throw new IllegalArgumentException("Nonfinite Bridge scalar");
			for (float value : match[actor]) if (!Float.isFinite(value)) throw new IllegalArgumentException("Nonfinite Bridge feature");
			for (byte[][] layer : voxels[actor]) {
				if (layer == null || layer.length != 9) throw new IllegalArgumentException("Invalid Bridge voxel shape");
				for (byte[] row : layer) {
					if (row == null || row.length != 9) throw new IllegalArgumentException("Invalid Bridge voxel shape");
					for (byte value : row) if (value < 0 || value > 4) throw new IllegalArgumentException("Invalid Bridge voxel category");
				}
			}
		}
		JsonObject body = new JsonObject();
		body.addProperty("protocol", 2);
		body.addProperty("session", session);
		body.addProperty("tick", tick);
		body.addProperty("contract_hash", contractHash);
		body.addProperty("bridge_observation_hash", descriptorHash);
		body.add("players", ReplayAnalysis.JSON.toJsonTree(seats));
		JsonObject observations = new JsonObject();
		observations.add("scalar", ReplayAnalysis.JSON.toJsonTree(scalar));
		observations.add("match", ReplayAnalysis.JSON.toJsonTree(match));
		observations.add("voxels", ReplayAnalysis.JSON.toJsonTree(voxels));
		body.add("observations", observations);
		return request("/act", body);
	}

	ReplayTrace.Controls[] actions(JsonObject response, String session, int tick, int[] seats) {
		if (SimulationContract.integer(response.get("protocol")) != 2 || !session.equals(response.get("session").getAsString())
				|| SimulationContract.integer(response.get("tick")) != tick || !contractHash.equals(response.get("contract_hash").getAsString())
				|| !descriptorHash.equals(response.get("bridge_observation_hash").getAsString())
				|| !response.get("players").equals(ReplayAnalysis.JSON.toJsonTree(seats))) {
			throw new IllegalArgumentException("Stale or mismatched Bridge response");
		}
		JsonArray results = response.getAsJsonArray("actions");
		if (results.size() != seats.length) throw new IllegalArgumentException("Unexpected Bridge action count");
		ReplayTrace.Controls[] controls = new ReplayTrace.Controls[seats.length];
		for (int index = 0; index < seats.length; index++) controls[index] = contract.decode(results.get(index).getAsJsonArray(), loadouts.get(seats[index]));
		return controls;
	}

	private CompletableFuture<JsonObject> request(String path, JsonObject body) {
		var builder = HttpRequest.newBuilder(base.resolve(path)).timeout(Duration.ofSeconds(2));
		HttpRequest request = body == null ? builder.GET().build() : builder.header("Content-Type", "application/json")
				.POST(HttpRequest.BodyPublishers.ofString(ReplayAnalysis.JSON.toJson(body))).build();
		return client.sendAsync(request, HttpResponse.BodyHandlers.ofString()).thenApply(response -> {
			if (response.statusCode() != 200 || response.body().length() > 1024 * 1024) {
				throw new IllegalStateException("Bridge inference HTTP " + response.statusCode());
			}
			return JsonParser.parseString(response.body()).getAsJsonObject();
		});
	}

	private static JsonElement canonical(JsonElement element) {
		if (element.isJsonArray()) {
			JsonArray result = new JsonArray();
			for (JsonElement value : element.getAsJsonArray()) result.add(canonical(value));
			return result;
		}
		if (element.isJsonObject()) {
			JsonObject result = new JsonObject();
			var sorted = new TreeMap<String, JsonElement>();
			for (var entry : element.getAsJsonObject().entrySet()) sorted.put(entry.getKey(), entry.getValue());
			for (var entry : sorted.entrySet()) result.add(entry.getKey(), canonical(entry.getValue()));
			return result;
		}
		return element;
	}

	private static String digest(JsonElement element) {
		try {
			return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(element.toString().getBytes(StandardCharsets.UTF_8)));
		} catch (NoSuchAlgorithmException error) {
			throw new IllegalStateException(error);
		}
	}

	@Override public void close() { client.shutdownNow(); }
}