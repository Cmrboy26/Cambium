package net.cambium;

import java.util.Map;

final class BridgeObservationEncoder {
	private BridgeObservationEncoder() {}

	record MatchRules(int goalsToWin, int maxTicks, int cageTicks, int arrowRefillTicks) {}
	record MatchState(int[] scores, int ticks, int cageRemaining) {}
	record ActorState(int blocks, int goldenApples, int arrows, int refillElapsed, int selectedSlot,
			int eatingRemaining, double miningProgress, int useCooldown, int miningCooldown,
			double[] position, float yaw, float pitch) {}

	static float[] features(int version, int actor, MatchRules rules, MatchState match, ActorState state, double[] target) {
		if (version < 1 || version > 2 || actor < 0 || actor > 1 || rules == null || match == null || state == null
				|| rules.goalsToWin() < 1 || rules.maxTicks() < 1 || rules.cageTicks() < 0 || rules.arrowRefillTicks() < 0
				|| match.scores() == null || match.scores().length != 2 || match.scores()[0] < 0 || match.scores()[1] < 0
				|| match.scores()[0] > rules.goalsToWin() || match.scores()[1] > rules.goalsToWin()
				|| match.ticks() < 0 || match.ticks() > rules.maxTicks()
				|| match.cageRemaining() < 0 || match.cageRemaining() > rules.cageTicks()
				|| state.blocks() < 0 || state.blocks() > 64 || state.goldenApples() < 0 || state.goldenApples() > 64
				|| state.arrows() < 0 || state.arrows() > 64 || state.refillElapsed() < 0
				|| state.refillElapsed() > rules.arrowRefillTicks() || state.selectedSlot() < 0 || state.selectedSlot() > 8) {
			throw new IllegalArgumentException("Invalid Bridge match or kit state");
		}
		float[] result = new float[version == 1 ? 10 : 17];
		result[0] = (float) match.scores()[actor] / rules.goalsToWin();
		result[1] = (float) match.scores()[1 - actor] / rules.goalsToWin();
		result[2] = (float) (rules.maxTicks() - match.ticks()) / rules.maxTicks();
		result[3] = (float) match.cageRemaining() / Math.max(1, rules.cageTicks());
		result[4] = state.blocks() / 64.0F;
		result[5] = state.goldenApples() / 64.0F;
		result[6] = state.arrows() / 64.0F;
		result[7] = (float) state.refillElapsed() / Math.max(1, rules.arrowRefillTicks());
		result[8] = rules.arrowRefillTicks() > 0 ? 1 : 0;
		result[9] = state.selectedSlot() / 8.0F;
		if (version == 1) return result;
		if (state.eatingRemaining() < 0 || state.eatingRemaining() > 32
				|| !Double.isFinite(state.miningProgress()) || state.miningProgress() < 0 || state.miningProgress() > 1
				|| state.useCooldown() < 0 || state.useCooldown() > 4 || state.miningCooldown() < 0 || state.miningCooldown() > 5
				|| state.position() == null || state.position().length != 3 || target == null || target.length != 3
				|| !Float.isFinite(state.yaw()) || !Float.isFinite(state.pitch())) {
			throw new IllegalArgumentException("Invalid Bridge actor task state");
		}
		for (int axis = 0; axis < 3; axis++) {
			if (!Double.isFinite(state.position()[axis]) || !Double.isFinite(target[axis])) {
				throw new IllegalArgumentException("Invalid Bridge objective position");
			}
		}
		double deltaX = target[0] - state.position()[0];
		double deltaY = target[1] - state.position()[1];
		double deltaZ = target[2] - state.position()[2];
		double horizontal = Math.hypot(deltaX, deltaZ);
		double distance = Math.hypot(horizontal, deltaY);
		if (!Double.isFinite(distance)) throw new IllegalArgumentException("Invalid Bridge objective distance");
		double desiredYaw = Math.toDegrees(Math.atan2(-deltaX, deltaZ));
		double yawError = horizontal > 0 ? ((desiredYaw - state.yaw() + 180) % 360 + 360) % 360 - 180 : 0;
		double pitchError = distance > 0 ? Math.toDegrees(Math.atan2(-deltaY, horizontal)) - state.pitch() : 0;
		result[10] = state.eatingRemaining() > 0 ? 1.0F - state.eatingRemaining() / 32.0F : 0;
		result[11] = (float) state.miningProgress();
		result[12] = (float) (yawError / 180);
		result[13] = (float) (pitchError / 90);
		result[14] = (float) Math.min(distance / 50, 1);
		result[15] = state.useCooldown() / 4.0F;
		result[16] = state.miningCooldown() / 5.0F;
		return result;
	}

	static byte[][][] voxels(ReplayTrace.Environment environment, Map<Integer, Integer> overlay,
			double[] feet, float yaw, int actor) {
		if (environment == null || overlay == null || feet == null || feet.length != 3 || !Float.isFinite(yaw)
				|| actor < 0 || actor > 1) throw new IllegalArgumentException("Invalid Bridge voxel inputs");
		long[] origin = new long[3];
		for (int axis = 0; axis < 3; axis++) {
			if (!Double.isFinite(feet[axis])) throw new IllegalArgumentException("Invalid Bridge voxel position");
			origin[axis] = (long) Math.floor(Math.clamp(feet[axis], -4294967300.0, 4294967300.0));
		}
		double reduced = (double) yaw % 360.0;
		int quarter = reduced >= 0.0
				? reduced < 45.0 ? 0 : reduced < 135.0 ? 1 : reduced < 225.0 ? 2 : reduced < 315.0 ? 3 : 0
				: reduced >= -45.0 ? 0 : reduced >= -135.0 ? 3 : reduced >= -225.0 ? 2 : reduced >= -315.0 ? 1 : 0;
		int[] forwardX = {0, -1, 0, 1};
		int[] forwardZ = {1, 0, -1, 0};
		int[] rightX = {1, 0, -1, 0};
		int[] rightZ = {0, 1, 0, -1};
		byte[][][] result = new byte[5][9][9];
		for (int up = 0; up < 5; up++) {
			for (int right = 0; right < 9; right++) {
				for (int forward = 0; forward < 9; forward++) {
					long blockX = origin[0] + (right - 4) * rightX[quarter] + (forward - 4) * forwardX[quarter];
					long blockY = origin[1] + up - 2;
					long blockZ = origin[2] + (right - 4) * rightZ[quarter] + (forward - 4) * forwardZ[quarter];
					if (blockX < environment.min[0] || blockX >= (long) environment.min[0] + environment.size[0]
							|| blockY < environment.min[1] || blockY >= (long) environment.min[1] + environment.size[1]
							|| blockZ < environment.min[2] || blockZ >= (long) environment.min[2] + environment.size[2]) {
						result[up][right][forward] = 4;
						continue;
					}
					int cell = (int) (((blockZ - environment.min[2]) * environment.size[1] + blockY - environment.min[1])
							* environment.size[0] + blockX - environment.min[0]);
					int base = environment.cells[cell];
					int owner = overlay.getOrDefault(cell, 0);
					if (base < 0 || base > 1 || owner < 0 || owner > 2 || base == 1 && owner != 0) {
						throw new IllegalArgumentException("Invalid Bridge terrain cell");
					}
					result[up][right][forward] = (byte) (base == 1 ? 1 : owner == 0 ? 0 : owner == actor + 1 ? 2 : 3);
				}
			}
		}
		return result;
	}
}