package net.cambium;

import com.google.gson.JsonObject;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Set;

final class BridgeTerrainGenerator {
	private BridgeTerrainGenerator() {}

	static long seed(long playerRandomState) {
		return playerRandomState ^ 0x7465727261696eL;
	}

	static long initialPlayerState(long episodeSeed) {
		return (episodeSeed ^ 0x5deece66dL) & ((1L << 48) - 1);
	}

	static Map<Integer, Integer> generate(ReplayTrace.Environment environment, JsonObject generation, long seed) {
		if (environment == null || generation == null || !generation.keySet().equals(
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
