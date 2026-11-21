package net.cambium;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.Vec3;

final class BridgeMatch {
	enum Outcome { ONGOING, TEAM0, TEAM1, DRAW }
	record Player(boolean alive, double feetY, AABB bounds) {}
	record Events(boolean advanced, boolean[] died, boolean[] scored, boolean[] respawn,
			boolean roundReset, Outcome outcome) {}
	record Refill(int arrows, int elapsed) {}

	private final AABB[] goals;
	private final double voidY;
	private final int goalsToWin;
	private final int maxTicks;
	private final int cageTicks;
	private final int[] scores = new int[2];
	private int ticks;
	private int cageRemaining;
	private Outcome outcome = Outcome.ONGOING;

	BridgeMatch(AABB[] goals, double voidY, int goalsToWin, int maxTicks, int cageTicks) {
		if (goals == null || goals.length != 2 || goals[0] == null || goals[1] == null
				|| !Double.isFinite(voidY) || goalsToWin < 1 || maxTicks < 1 || cageTicks < 0) {
			throw new IllegalArgumentException("Invalid Bridge match rules");
		}
		this.goals = goals.clone();
		this.voidY = voidY;
		this.goalsToWin = goalsToWin;
		this.maxTicks = maxTicks;
		this.cageTicks = cageTicks;
		this.cageRemaining = cageTicks;
	}

	static BridgeMatch fromEnvironment(ReplayTrace.Environment environment, Vec3 origin) {
		if (environment == null || origin == null || !Double.isFinite(origin.x)
				|| !Double.isFinite(origin.y) || !Double.isFinite(origin.z)) {
			throw new IllegalArgumentException("Invalid Bridge arena origin");
		}
		try {
			JsonObject bridge = environment.definition.getAsJsonObject("experimental_bridge");
			JsonObject mode = bridge.getAsJsonObject("mode");
			if (!"bridge".equals(mode.get("name").getAsString()) || SimulationContract.integer(mode.get("version")) != 1) {
				throw new IllegalArgumentException("Unsupported Bridge mode");
			}
			double voidY = number(bridge.get("void_y"));
			JsonArray regions = bridge.getAsJsonArray("goals");
			if (regions.size() != 2) throw new IllegalArgumentException("Bridge requires two goals");
			AABB[] goals = new AABB[2];
			for (int actor = 0; actor < goals.length; actor++) {
				JsonObject region = regions.get(actor).getAsJsonObject();
				JsonArray minimum = region.getAsJsonArray("min");
				JsonArray maximum = region.getAsJsonArray("max");
				if (minimum.size() != 3 || maximum.size() != 3) throw new IllegalArgumentException("Invalid Bridge goal");
				double[] lower = new double[3];
				double[] upper = new double[3];
				for (int axis = 0; axis < 3; axis++) {
					lower[axis] = number(minimum.get(axis));
					upper[axis] = number(maximum.get(axis));
					if (lower[axis] >= upper[axis] || lower[axis] < environment.min[axis]
							|| upper[axis] > (long) environment.min[axis] + environment.size[axis]) {
						throw new IllegalArgumentException("Bridge goal must be inside arena bounds");
					}
				}
				if (lower[1] < voidY) throw new IllegalArgumentException("Bridge goal must be above the void");
				goals[actor] = new AABB(lower[0], lower[1], lower[2], upper[0], upper[1], upper[2]).move(origin);
			}
			if (goals[0].intersects(goals[1])) throw new IllegalArgumentException("Bridge goals must not overlap");
			JsonObject rules = bridge.getAsJsonObject("match");
			int goalsToWin = SimulationContract.integer(rules.get("goals_to_win"));
			int maxTicks = SimulationContract.integer(rules.get("max_ticks"));
			int cageTicks = SimulationContract.integer(rules.get("cage_ticks"));
			if (goalsToWin > 1000000 || maxTicks > 1000000 || cageTicks > 1000000
					|| !Double.isFinite(voidY + origin.y)) throw new IllegalArgumentException("Invalid Bridge match rules");
			return new BridgeMatch(goals, voidY + origin.y, goalsToWin, maxTicks, cageTicks);
		} catch (NullPointerException | ClassCastException | IllegalStateException error) {
			throw new IllegalArgumentException("Invalid Bridge match configuration", error);
		}
	}

	private static double number(JsonElement value) {
		if (value == null || !value.isJsonPrimitive() || !value.getAsJsonPrimitive().isNumber()
				|| !Double.isFinite(value.getAsDouble())) throw new IllegalArgumentException("Expected a finite Bridge coordinate");
		return value.getAsDouble();
	}

	int score(int actor) { return scores[actor]; }
	int ticks() { return ticks; }
	int cageRemaining() { return cageRemaining; }
	Outcome outcome() { return outcome; }

	ReplayTrace.Controls controls(ReplayTrace.Controls input) {
		if (cageRemaining == 0) return input;
		return new ReplayTrace.Controls(false, false, false, false, false, false, false, false,
				input.yaw_delta(), input.pitch_delta());
	}

	Events advance(Player[] players) {
		if (players == null || players.length != 2) throw new IllegalArgumentException("Bridge requires two players");
		for (Player player : players) {
			if (player == null || !Double.isFinite(player.feetY()) || player.bounds() == null) {
				throw new IllegalArgumentException("Invalid Bridge player state");
			}
		}
		boolean[] died = new boolean[2];
		boolean[] scored = new boolean[2];
		boolean[] respawn = new boolean[2];
		if (outcome != Outcome.ONGOING) return new Events(false, died, scored, respawn, false, outcome);
		for (int actor = 0; actor < 2; actor++) {
			died[actor] = !players[actor].alive() || players[actor].feetY() < voidY;
			scored[actor] = !died[actor] && players[actor].bounds().intersects(goals[1 - actor]);
			if (scored[actor]) scores[actor]++;
		}
		ticks++;
		if (cageRemaining > 0) cageRemaining--;
		boolean firstWon = scores[0] >= goalsToWin;
		boolean secondWon = scores[1] >= goalsToWin;
		if (firstWon && secondWon) outcome = Outcome.DRAW;
		else if (firstWon) outcome = Outcome.TEAM0;
		else if (secondWon) outcome = Outcome.TEAM1;
		else if (ticks >= maxTicks) outcome = scores[0] == scores[1] ? Outcome.DRAW : scores[0] > scores[1] ? Outcome.TEAM0 : Outcome.TEAM1;
		if (outcome != Outcome.ONGOING) return new Events(true, died, scored, respawn, false, outcome);
		boolean reset = scored[0] || scored[1];
		if (reset) cageRemaining = cageTicks;
		else System.arraycopy(died, 0, respawn, 0, 2);
		return new Events(true, died, scored, respawn, reset, outcome);
	}

	static Refill refill(int arrowsBefore, int arrowsAfter, int elapsed, int delay) {
		if (arrowsBefore < 0 || arrowsBefore > 64 || arrowsAfter < 0 || arrowsAfter > 64
				|| elapsed < 0 || delay < 0 || elapsed > delay) throw new IllegalArgumentException("Invalid Bridge arrow state");
		if (arrowsAfter > 0 || delay == 0 || arrowsBefore > 0) return new Refill(arrowsAfter, 0);
		return elapsed >= delay - 1 ? new Refill(1, 0) : new Refill(0, elapsed + 1);
	}
}