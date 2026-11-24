package net.cambium;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

final class BridgeTerrainGenerator {
	private record Pattern(boolean top, String type, int minimum, int maximum, int probability) {}

	private BridgeTerrainGenerator() {}

	static long seed(long playerRandomState) {
		return playerRandomState ^ 0x7465727261696eL;
	}

	static long initialPlayerState(long episodeSeed) {
		return (episodeSeed ^ 0x5deece66dL) & ((1L << 48) - 1);
	}

	static Map<Integer, Integer> generate(ReplayTrace.Environment environment, JsonObject generation, long seed) {
		if (environment == null || generation == null) {
			throw new IllegalArgumentException("Invalid Bridge terrain generation configuration");
		}
		if (generation.has("bridge")) return generatePatterns(environment, generation, seed);
		if (!generation.keySet().equals(
				Set.of("center_x", "floor_y", "start_z", "end_z", "segments"))) {
			throw new IllegalArgumentException("Invalid Bridge terrain generation configuration");
		}
		int centerX = SimulationContract.integer(generation.get("center_x"));
		int floorY = SimulationContract.integer(generation.get("floor_y"));
		int startZ = SimulationContract.integer(generation.get("start_z"));
		int endZ = SimulationContract.integer(generation.get("end_z"));
		int segments = SimulationContract.integer(generation.get("segments"));
		long span = (long) endZ - startZ;
		if (segments < 1 || segments > 16 || span < 1 || span > 10000
				|| centerX < Integer.MIN_VALUE + 2 || centerX > Integer.MAX_VALUE - 2 || floorY == Integer.MAX_VALUE) {
			throw new IllegalArgumentException("Invalid Bridge terrain generation bounds");
		}
		for (int blockZ = startZ; blockZ < endZ; blockZ++) {
			if (environment.cells[cell(environment, centerX, floorY, blockZ)] != 1) {
				throw new IllegalArgumentException("Bridge terrain requires a solid center lane");
			}
			for (int offset = -2; offset <= 2; offset++) {
				if (offset == 0) continue;
				for (int height = 0; height <= 1; height++) {
					if (environment.cells[cell(environment, centerX + offset, floorY + height, blockZ)] != 0) {
						throw new IllegalArgumentException("Bridge generated terrain requires clear base cells");
					}
				}
			}
		}
		Random random = new Random(seed);
		Map<Integer, Integer> overlay = new HashMap<>();
		for (int segment = 0; segment < segments; segment++) {
			int side = (random.nextInt() >>> 31) == 0 ? -1 : 1;
			int length = Math.min(2 + (random.nextInt() >>> 30), (int) span);
			int start = startZ + (random.nextInt() >>> 1) % ((int) span - length + 1);
			int width = 1 + (random.nextInt() >>> 31);
			boolean rail = (random.nextInt() >>> 31) != 0;
			int team = (random.nextInt() >>> 31) == 0 ? 1 : 2;
			for (int blockZ = start; blockZ < start + length; blockZ++) {
				for (int offset = 1; offset <= width; offset++) {
					overlay.put(cell(environment, centerX + side * offset, floorY, blockZ), team);
				}
				if (rail) overlay.put(cell(environment, centerX + side * width, floorY + 1, blockZ), team);
			}
		}
		return Map.copyOf(overlay);
	}

	private static Map<Integer, Integer> generatePatterns(ReplayTrace.Environment environment, JsonObject generation, long seed) {
		if (!generation.keySet().equals(Set.of("bridge", "patterns"))
				|| !generation.get("bridge").isJsonObject() || !generation.get("patterns").isJsonArray()) {
			throw new IllegalArgumentException("Invalid Bridge pattern generation configuration");
		}
		JsonObject bridge = generation.getAsJsonObject("bridge");
		if (!bridge.keySet().equals(Set.of("min", "max")) || !bridge.get("min").isJsonArray()
				|| !bridge.get("max").isJsonArray()) throw new IllegalArgumentException("Invalid Bridge pattern bounds");
		JsonArray lower = bridge.getAsJsonArray("min"), upper = bridge.getAsJsonArray("max");
		if (lower.size() != 3 || upper.size() != 3) throw new IllegalArgumentException("Invalid Bridge pattern bounds");
		int[] minimum = new int[3], maximum = new int[3];
		for (int axis = 0; axis < 3; axis++) {
			minimum[axis] = SimulationContract.integer(lower.get(axis));
			maximum[axis] = SimulationContract.integer(upper.get(axis));
			if (minimum[axis] < -1_000_000_000 || minimum[axis] > 1_000_000_000
					|| maximum[axis] < -1_000_000_000 || maximum[axis] > 1_000_000_000) {
				throw new IllegalArgumentException("Invalid Bridge pattern bounds");
			}
		}
		long width = (long) maximum[0] - minimum[0], span = (long) maximum[2] - minimum[2];
		if (width < 1 || width > 64 || maximum[1] - minimum[1] != 1 || span < 1 || span > 10000) {
			throw new IllegalArgumentException("Invalid Bridge pattern bounds");
		}
		JsonArray definitions = generation.getAsJsonArray("patterns");
		if (definitions.isEmpty() || definitions.size() > 32) throw new IllegalArgumentException("Invalid Bridge terrain patterns");
		List<Pattern> patterns = new ArrayList<>();
		for (JsonElement element : definitions) {
			if (!element.isJsonObject()) throw new IllegalArgumentException("Invalid Bridge terrain pattern");
			JsonObject pattern = element.getAsJsonObject();
			if (!pattern.has("location") || !pattern.get("location").isJsonPrimitive()
					|| !pattern.getAsJsonPrimitive("location").isString() || !pattern.has("type")
					|| !pattern.get("type").isJsonPrimitive() || !pattern.getAsJsonPrimitive("type").isString()
					|| !pattern.has("percent") || !pattern.get("percent").isJsonPrimitive()
					|| !pattern.getAsJsonPrimitive("percent").isNumber()) {
				throw new IllegalArgumentException("Invalid Bridge terrain pattern");
			}
			String location = pattern.get("location").getAsString(), type = pattern.get("type").getAsString();
			if (!location.equals("top") && !location.equals("side")) throw new IllegalArgumentException("Invalid Bridge pattern location");
			int low, high;
			if (type.equals("constant") && pattern.keySet().equals(Set.of("location", "type", "value", "percent"))) {
				low = high = SimulationContract.integer(pattern.get("value"));
			} else if (type.equals("random") && pattern.keySet().equals(Set.of("location", "type", "min", "max", "percent"))
					|| type.equals("function") && pattern.keySet().equals(Set.of("location", "type", "min", "max", "percent", "function"))
							&& pattern.get("function").isJsonPrimitive() && pattern.getAsJsonPrimitive("function").isString()
							&& pattern.get("function").getAsString().equals("half-sine")) {
				low = SimulationContract.integer(pattern.get("min"));
				high = SimulationContract.integer(pattern.get("max"));
			} else throw new IllegalArgumentException("Invalid Bridge terrain pattern fields");
			double probability = pattern.get("percent").getAsDouble();
			if (low < 0 || high < low || high > 8 || !Double.isFinite(probability) || probability < 0 || probability > 1) {
				throw new IllegalArgumentException("Invalid Bridge terrain pattern limits");
			}
			patterns.add(new Pattern(location.equals("top"), type, low, high, (int) Math.round(probability * (1 << 24))));
		}
		for (int blockZ = minimum[2]; blockZ < maximum[2]; blockZ++) {
			for (int blockX = minimum[0]; blockX < maximum[0]; blockX++) {
				if (environment.cells[cell(environment, blockX, minimum[1], blockZ)] != 1) {
					throw new IllegalArgumentException("Bridge terrain requires a solid bridge floor");
				}
			}
			for (Pattern pattern : patterns) {
				for (int offset = 1; offset <= pattern.maximum(); offset++) {
					int startX = pattern.top() ? minimum[0] : minimum[0] - offset;
					int endX = pattern.top() ? maximum[0] : maximum[0] + offset;
					for (int blockX = startX; blockX < endX; blockX++) {
						if (!pattern.top() && blockX != startX && blockX != endX - 1) continue;
						int blockY = minimum[1] + (pattern.top() ? offset : 0);
						if (environment.cells[cell(environment, blockX, blockY, blockZ)] != 0) {
							throw new IllegalArgumentException("Bridge generated terrain requires clear base cells");
						}
					}
				}
			}
		}
		Random random = new Random(seed);
		Map<Integer, Integer> overlay = new HashMap<>();
		for (Pattern pattern : patterns) {
			for (int blockZ = minimum[2]; blockZ < maximum[2]; blockZ++) {
				int columns = pattern.top() ? (int) width : 2;
				for (int column = 0; column < columns; column++) {
					if ((random.nextInt() >>> 8) >= pattern.probability()) continue;
					int extent = patternExtent(pattern, blockZ - minimum[2], (int) span, random);
					int team = (random.nextInt() >>> 31) + 1;
					for (int offset = 1; offset <= extent; offset++) {
						int blockX = pattern.top() ? minimum[0] + column
								: column == 0 ? minimum[0] - offset : maximum[0] + offset - 1;
						int blockY = minimum[1] + (pattern.top() ? offset : 0);
						overlay.put(cell(environment, blockX, blockY, blockZ), team);
					}
				}
			}
		}
		return Map.copyOf(overlay);
	}

	private static int patternExtent(Pattern pattern, int position, int span, Random random) {
		if (pattern.type().equals("random")) return pattern.minimum() + (random.nextInt() >>> 1) % (pattern.maximum() - pattern.minimum() + 1);
		if (pattern.type().equals("constant") || span == 1) return pattern.minimum();
		long denominator = span - 1L, product = position * (denominator - position);
		long divisor = 5 * denominator * denominator - 4 * product;
		long numerator = (long) (pattern.maximum() - pattern.minimum()) * 16 * product;
		return pattern.minimum() + (int) ((numerator + divisor / 2) / divisor);
	}

	private static int cell(ReplayTrace.Environment environment, int x, int y, int z) {
		long offsetX = (long) x - environment.min[0];
		long offsetY = (long) y - environment.min[1];
		long offsetZ = (long) z - environment.min[2];
		if (offsetX < 0 || offsetX >= environment.size[0] || offsetY < 0 || offsetY >= environment.size[1]
				|| offsetZ < 0 || offsetZ >= environment.size[2]) {
			throw new IllegalArgumentException("Bridge terrain generation is outside arena bounds");
		}
		return (int) ((offsetZ * environment.size[1] + offsetY) * environment.size[0] + offsetX);
	}
}
