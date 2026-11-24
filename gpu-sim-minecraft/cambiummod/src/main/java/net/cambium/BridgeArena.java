package net.cambium;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import net.minecraft.world.phys.AABB;

final class BridgeArena {
	private static final class Mining {
		int cell = -1;
		int delay;
		float progress;
	}

	private final ReplayTrace.Environment environment;
	private final byte[] originalCells;
	private final List<AABB> zones = new ArrayList<>();
	private final Map<Integer, Integer> overlay = new HashMap<>();
	private final Mining[] mining = {new Mining(), new Mining()};
	private final float hardness;
	private final float miningSpeed;
	private final boolean correctTool;

	BridgeArena(ReplayTrace.Environment environment, JsonObject building, Map<Integer, Integer> initial) {
		if (environment == null || building == null || initial == null || !building.keySet().equals(
				Set.of("zones", "block_range", "entity_range", "hardness", "mining_speed", "correct_tool"))) {
			throw new IllegalArgumentException("Invalid Bridge building rules");
		}
		this.environment = environment;
		originalCells = environment.cells.clone();
		double blockRange = number(building.get("block_range"));
		double entityRange = number(building.get("entity_range"));
		hardness = (float) number(building.get("hardness"));
		miningSpeed = (float) number(building.get("mining_speed"));
		if (blockRange <= 0 || blockRange > 64 || entityRange < 0 || entityRange > 64
				|| !Float.isFinite(hardness) || hardness <= 0 || !Float.isFinite(miningSpeed) || miningSpeed <= 0
				|| !building.get("correct_tool").isJsonPrimitive() || !building.get("correct_tool").getAsJsonPrimitive().isBoolean()) {
			throw new IllegalArgumentException("Invalid Bridge building limits");
		}
		correctTool = building.get("correct_tool").getAsBoolean();
		JsonArray regions = building.getAsJsonArray("zones");
		if (regions == null || regions.isEmpty() || regions.size() > 8) throw new IllegalArgumentException("Invalid Bridge build zones");
		for (JsonElement element : regions) {
			JsonObject region = element.getAsJsonObject();
			if (!region.keySet().equals(Set.of("min", "max"))) throw new IllegalArgumentException("Invalid Bridge build zone");
			JsonArray minimum = region.getAsJsonArray("min");
			JsonArray maximum = region.getAsJsonArray("max");
			if (minimum == null || maximum == null || minimum.size() != 3 || maximum.size() != 3) throw new IllegalArgumentException("Invalid Bridge build zone");
			double[] lower = new double[3], upper = new double[3];
			for (int axis = 0; axis < 3; axis++) {
				lower[axis] = number(minimum.get(axis));
				upper[axis] = number(maximum.get(axis));
				if (lower[axis] >= upper[axis] || lower[axis] < environment.min[axis]
						|| upper[axis] > (long) environment.min[axis] + environment.size[axis]) throw new IllegalArgumentException("Bridge build zone outside arena");
			}
			zones.add(new AABB(lower[0], lower[1], lower[2], upper[0], upper[1], upper[2]));
		}
		for (var entry : initial.entrySet()) {
			int index = entry.getKey();
			if (index < 0 || index >= environment.cells.length || environment.cells[index] != 0
					|| entry.getValue() < 1 || entry.getValue() > 2 || !inZone(x(index), y(index), z(index))) {
				throw new IllegalArgumentException("Invalid initial Bridge overlay");
			}
			overlay.put(index, entry.getValue());
		}
	}

	private static double number(JsonElement element) {
		if (element == null || !element.isJsonPrimitive() || !element.getAsJsonPrimitive().isNumber()
				|| !Double.isFinite(element.getAsDouble())) throw new IllegalArgumentException("Invalid Bridge building number");
		return element.getAsDouble();
	}

	Map<Integer, Integer> overlay() { return Map.copyOf(overlay); }
	void editBase(int index, boolean solid) {
		if (index < 0 || index >= environment.cells.length) throw new IllegalArgumentException("Bridge edit outside arena");
		overlay.remove(index);
		environment.cells[index] = (byte) (solid ? 1 : 0);
	}
	void restoreBase() { System.arraycopy(originalCells, 0, environment.cells, 0, originalCells.length); }
	void clear() {
		overlay.clear();
		for (int actor = 0; actor < mining.length; actor++) {
			cancelMining(actor);
			mining[actor].delay = 0;
		}
	}
	float miningProgress(int actor) { return mining[actor].progress; }
	int miningDelay(int actor) { return mining[actor].delay; }

	int cell(int blockX, int blockY, int blockZ) {
		long offsetX = (long) blockX - environment.min[0], offsetY = (long) blockY - environment.min[1], offsetZ = (long) blockZ - environment.min[2];
		if (offsetX < 0 || offsetX >= environment.size[0] || offsetY < 0 || offsetY >= environment.size[1]
				|| offsetZ < 0 || offsetZ >= environment.size[2]) return -1;
		return (int) ((offsetZ * environment.size[1] + offsetY) * environment.size[0] + offsetX);
	}

	private int x(int index) { return environment.min[0] + index % environment.size[0]; }
	private int y(int index) { return environment.min[1] + index / environment.size[0] % environment.size[1]; }
	private int z(int index) { return environment.min[2] + index / (environment.size[0] * environment.size[1]); }

	private boolean inZone(int blockX, int blockY, int blockZ) {
		for (AABB zone : zones) {
			if (blockX >= zone.minX && blockY >= zone.minY && blockZ >= zone.minZ
					&& (double) blockX + 1 <= zone.maxX && (double) blockY + 1 <= zone.maxY && (double) blockZ + 1 <= zone.maxZ) return true;
		}
		return false;
	}

	boolean place(int actor, int targetX, int targetY, int targetZ, int blockX, int blockY, int blockZ, int remaining, AABB[] players) {
		if (actor < 0 || actor > 1 || remaining <= 0 || players == null || players.length != 2
				|| Math.abs((long) blockX - targetX) + Math.abs((long) blockY - targetY) + Math.abs((long) blockZ - targetZ) != 1) return false;
		int target = cell(targetX, targetY, targetZ), destination = cell(blockX, blockY, blockZ);
		if (target < 0 || destination < 0 || environment.cells[target] == 0 && !overlay.containsKey(target)
				|| environment.cells[destination] != 0 || overlay.containsKey(destination) || !inZone(blockX, blockY, blockZ)) return false;
		AABB cube = new AABB(blockX, blockY, blockZ, (double) blockX + 1, (double) blockY + 1, (double) blockZ + 1);
		for (AABB player : players) if (player == null || cube.intersects(player)) return false;
		overlay.put(destination, actor + 1);
		return true;
	}

	void cancelMining(int actor) {
		mining[actor].cell = -1;
		mining[actor].progress = 0;
	}

	boolean mine(int actor, int blockX, int blockY, int blockZ, boolean onGround) {
		Mining state = mining[actor];
		if (state.delay > 0) { state.delay--; return false; }
		int index = cell(blockX, blockY, blockZ);
		if (index < 0 || environment.cells[index] != 0 || !overlay.containsKey(index) || !inZone(blockX, blockY, blockZ)) {
			cancelMining(actor);
			return false;
		}
		float speed = onGround ? miningSpeed : miningSpeed / 5.0F;
		float increment = speed / hardness / (correctTool ? 30.0F : 100.0F);
		if (state.cell != index) {
			state.cell = index;
			state.progress = 0;
			if (increment < 1) return false;
		} else {
			state.progress += increment;
			if (state.progress < 1) return false;
		}
		overlay.remove(index);
		cancelMining(actor);
		state.delay = 5;
		return true;
	}
}