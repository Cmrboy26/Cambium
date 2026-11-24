package net.cambium;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.util.ArrayList;
import java.util.Locale;
import java.util.Map;
import net.minecraft.core.BlockPos;
import net.minecraft.network.protocol.game.ServerboundPlayerActionPacket;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.damagesource.DamageSource;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.level.ClipContext;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.EntityHitResult;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec3;

final class LiveBridge implements LiveSession {
	private static final ReplayTrace.Controls NEUTRAL = new ReplayTrace.Controls(false, false, false, false, false, false, false, false, 0, 0);
	private final SimulationReplay.Session session;
	private final BridgePolicyClient policy;
	private final BridgeInference<JsonObject> inference = new BridgeInference<>();
	private final ServerPlayer[] fighters = new ServerPlayer[2];
	private final BridgeKit[] kits = new BridgeKit[2];
	private final int[] refill = new int[2];
	private final int[] arrowsBefore = new int[2];
	private final int[] useDelay = new int[2];
	private final boolean[] defeated = new boolean[2];
	private final JsonObject rules;
	private final int maxTicks;
	private final int cageTicks;
	private final int goalsToWin;
	private final int refillTicks;
	private final int externalLimit;
	private final boolean persistBlocks;
	private final String deploymentSummary;
	private BridgeMatch match;
	private BridgeWorld world;
	private int[] requestedSeats;
	private int humanSlot = -1;
	private ParticipantState savedHuman;
	private long episodes;
	private long requestNanos;
	private double latency;
	private Vec3[] lockedPositions;
	private BlockPos humanMining;

	LiveBridge(SimulationReplay.Session session, BridgePolicyClient policy, JsonObject info) {
		this.session = session;
		this.policy = policy;
		policy.requireLiveSupport();
		JsonObject bridge = session.trace.environment.definition.getAsJsonObject("experimental_bridge");
		if (bridge == null || !bridge.has("building") || !bridge.has("kits")) throw new IllegalArgumentException("Live Bridge requires building rules and explicit kits");
		rules = bridge.getAsJsonObject("match");
		maxTicks = SimulationContract.integer(rules.get("max_ticks"));
		cageTicks = SimulationContract.integer(rules.get("cage_ticks"));
		goalsToWin = SimulationContract.integer(rules.get("goals_to_win"));
		refillTicks = SimulationContract.integer(rules.get("arrow_refill_ticks"));
		externalLimit = SimulationContract.integer(info.get("max_ticks"));
		if (refillTicks < 0 || refillTicks > 1000000 || externalLimit < 1 || externalLimit > 1000000
				|| !rules.get("persist_blocks_on_goal").getAsJsonPrimitive().isBoolean()) throw new IllegalArgumentException("Invalid Bridge timers");
		persistBlocks = rules.get("persist_blocks_on_goal").getAsBoolean();
		JsonArray definitions = bridge.getAsJsonArray("kits");
		for (int actor = 0; actor < 2; actor++) kits[actor] = new BridgeKit(definitions.get(actor).getAsJsonObject(), session.trace.environment.loadouts.get(actor));
		match = policy.newMatch(session.origin);
		deploymentSummary = DeploymentSummary.describe(info);
	}

	@Override public void reset() {
		cancelForLifecycle();
		match = policy.newMatch(session.origin);
		world = null;
		lockedPositions = null;
		humanMining = null;
		for (int actor = 0; actor < 2; actor++) {
			refill[actor] = 0;
			useDelay[actor] = 0;
			defeated[actor] = false;
		}
	}

	@Override public void afterEnvironmentBuilt() {
		JsonObject bridge = session.trace.environment.definition.getAsJsonObject("experimental_bridge");
		Map<Integer, Integer> initial = policy.initialTerrain(BridgeTerrainGenerator.initialPlayerState(1 + episodes++ * 1000033L));
		world = new BridgeWorld(session, new BridgeArena(session.trace.environment, bridge.getAsJsonObject("building"), initial));
	}

	@Override public void afterActorSpawned(int index) {
		ServerPlayer player = fighters[index];
		player.removeAllEffects();
		kits[index].equip(player, index);
		refill[index] = 0;
		useDelay[index] = 0;
		defeated[index] = false;
		arrowsBefore[index] = arrows(player);
		if (index == humanSlot) humanMining = null;
	}

	@Override public void tick(ServerPlayer viewer) {
		try {
			if (!session.paused && match.outcome() == BridgeMatch.Outcome.ONGOING) {
				world.verify();
				advance();
			} else freezeHuman();
		} catch (RuntimeException error) {
			session.paused = true;
			cancelForLifecycle();
			freezeHuman();
			session.tell("Bridge paused: " + message(error) + ". Fix the cause, then /simduel resume.");
			CambiumFabric.LOGGER.warn("Bridge paused", error);
		}
		publish();
	}

	private void advance() {
		long now = System.nanoTime();
		BridgeInference.Result<JsonObject> result = inference.poll(match.ticks(), now);
		ReplayTrace.Controls[] controls = {NEUTRAL, NEUTRAL};
		if (result != null) {
			latency = (now - requestNanos) / 1_000_000.0;
			ReplayTrace.Controls[] actions = policy.actions(result.value(), result.epoch(), result.tick(), requestedSeats);
			for (int index = 0; index < requestedSeats.length; index++) controls[requestedSeats[index]] = actions[index];
		} else {
			for (ReplayPlayer bot : session.actors) if (bot.isUsingItem()) bot.stopUsingItem();
		}
		boolean caged = match.cageRemaining() > 0;
		if (caged && lockedPositions == null) lockedPositions = new Vec3[]{fighters[0].position(), fighters[1].position()};
		for (int actor = 0; actor < 2; actor++) {
			if (useDelay[actor] > 0) useDelay[actor]--;
			if (actor == humanSlot) {
				tickHumanMining(caged);
				continue;
			}
			ReplayPlayer bot = (ReplayPlayer) fighters[actor];
			ReplayTrace.Controls action = caged ? match.controls(controls[actor]) : controls[actor];
			String selected = kits[actor].slot(action.select_weapon() ? action.select_slot() : bot.getInventory().getSelectedSlot());
			boolean special = selected.equals("pickaxe") || selected.equals("blocks") || selected.equals("golden_apple");
			boolean use = action.use_item();
			if (!special && use && !bot.isUsingItem()) {
				if (useDelay[actor] > 0) use = false; else useDelay[actor] = 4;
			}
			ReplayTrace.Controls prepared = special ? new ReplayTrace.Controls(action.forward(), action.backward(), action.left(), action.right(),
					action.jump(), action.sprint(), action.sneak(), false, action.yaw_delta(), action.pitch_delta(),
					action.select_weapon(), action.weapon(), selected.equals("golden_apple") && action.use_item(), action.select_slot())
					: new ReplayTrace.Controls(action.forward(), action.backward(), action.left(), action.right(), action.jump(), action.sprint(),
							action.sneak(), action.attack(), action.yaw_delta(), action.pitch_delta(), action.select_weapon(), action.weapon(), use, action.select_slot());
			bot.prepare(prepared);
			if (!caged) itemAction(actor, action, selected);
		}
		for (ReplayPlayer bot : session.actors) {
			HitResult hit = bot.pickAttack();
			if (hit == null || !bot.isAlive()) continue;
			bot.swingReplay();
			if (hit instanceof EntityHitResult entity) bot.attack(entity.getEntity()); else bot.miss(hit);
			session.actors.forEach(ReplayPlayer::acceptMotion);
		}
		for (ReplayPlayer bot : session.actors) bot.advanceInput();
		if (caged) {
			for (int actor = 0; actor < 2; actor++) lock(fighters[actor], lockedPositions[actor]);
		} else lockedPositions = null;
		for (int actor = 0; actor < 2; actor++) {
			int current = arrows(fighters[actor]);
			BridgeMatch.Refill next = BridgeMatch.refill(arrowsBefore[actor], current, refill[actor], refillTicks);
			refill[actor] = next.elapsed();
			if (next.arrows() != current) fighters[actor].getInventory().setItem(9, new ItemStack(Items.ARROW, next.arrows()));
			arrowsBefore[actor] = next.arrows();
		}
		BridgeMatch.Player[] players = new BridgeMatch.Player[2];
		for (int actor = 0; actor < 2; actor++) players[actor] = new BridgeMatch.Player(!defeated[actor] && fighters[actor].isAlive(), fighters[actor].getY(), fighters[actor].getBoundingBox());
		BridgeMatch.Events events = match.advance(players);
		if (events.outcome() != BridgeMatch.Outcome.ONGOING || match.ticks() >= externalLimit) {
			session.paused = true;
			cancelForLifecycle();
			session.tell("Bridge finished: " + (events.outcome() == BridgeMatch.Outcome.ONGOING ? "external time limit" : events.outcome()));
			return;
		}
		if (events.roundReset()) {
			cancelForLifecycle();
			if (!persistBlocks) world.clearOverlay();
			session.respawnActors();
			lockedPositions = null;
		} else for (int actor = 0; actor < 2; actor++) if (events.respawn()[actor]) {
			cancelRequest();
			session.clearProjectiles(fighters[actor]);
			session.respawnActor(actor);
		}
		if (!inference.pending()) request();
	}

	private void itemAction(int actor, ReplayTrace.Controls action, String selected) {
		ServerPlayer player = fighters[actor];
		if (!selected.equals("pickaxe") || !action.attack()) world.cancelMining(actor);
		if (selected.equals("golden_apple") && action.use_item() && !player.isUsingItem() && useDelay[actor] == 0) {
			useDelay[actor] = 4;
			player.getMainHandItem().use(session.level, player, InteractionHand.MAIN_HAND);
		}
		if (!selected.equals("blocks") && !selected.equals("pickaxe")) return;
		BlockHitResult target = targetBlock(player, fighters[1 - actor]);
		if (selected.equals("blocks") && action.use_item() && useDelay[actor] == 0) {
			useDelay[actor] = 4;
			if (target != null && world.place(actor, target.getBlockPos(), target.getBlockPos().relative(target.getDirection()),
					player.getMainHandItem().getCount(), new AABB[]{fighters[0].getBoundingBox(), fighters[1].getBoundingBox()})) {
				player.getMainHandItem().shrink(1);
			}
		} else if (selected.equals("pickaxe") && action.attack() && target != null) {
			world.mine(actor, target.getBlockPos(), player.onGround());
		} else if (selected.equals("pickaxe")) world.cancelMining(actor);
	}

	private BlockHitResult targetBlock(ServerPlayer player, ServerPlayer opponent) {
		JsonObject building = session.trace.environment.definition.getAsJsonObject("experimental_bridge").getAsJsonObject("building");
		double blockRange = building.get("block_range").getAsDouble(), entityRange = building.get("entity_range").getAsDouble();
		Vec3 eye = player.getEyePosition();
		Vec3 end = eye.add(player.getLookAngle().scale(Math.max(blockRange, entityRange)));
		BlockHitResult hit = session.level.clip(new ClipContext(eye, end, ClipContext.Block.OUTLINE, ClipContext.Fluid.NONE, player));
		if (hit.getType() != HitResult.Type.BLOCK || eye.distanceToSqr(hit.getLocation()) >= blockRange * blockRange) return null;
		var entity = opponent.getBoundingBox().clip(eye, end);
		if (entity.isPresent() && eye.distanceToSqr(entity.get()) < eye.distanceToSqr(hit.getLocation())) return null;
		return hit;
	}

	boolean participant(ServerPlayer player) { return humanSlot >= 0 && fighters[humanSlot] == player; }
	boolean locked() { return session.paused || match.cageRemaining() > 0 || match.outcome() != BridgeMatch.Outcome.ONGOING; }
	boolean contains(BlockPos position) {
		BlockPos local = position.subtract(BlockPos.containing(session.origin));
		var environment = session.trace.environment;
		return local.getX() >= environment.min[0] && (long) local.getX() < (long) environment.min[0] + environment.size[0]
				&& local.getY() >= environment.min[1] && (long) local.getY() < (long) environment.min[1] + environment.size[1]
				&& local.getZ() >= environment.min[2] && (long) local.getZ() < (long) environment.min[2] + environment.size[2];
	}
	boolean attackAllowed(ServerPlayer player, int target) {
		if (!participant(player)) return fighters[0].getId() != target && fighters[1].getId() != target;
		String slot = kits[humanSlot].slot(player.getInventory().getSelectedSlot());
		return !locked() && fighters[1 - humanSlot].getId() == target && !slot.equals("pickaxe")
				&& !slot.equals("blocks") && !slot.equals("golden_apple");
	}
	void humanBreak(ServerboundPlayerActionPacket.Action action, BlockPos position) {
		if (action == ServerboundPlayerActionPacket.Action.ABORT_DESTROY_BLOCK) {
			humanMining = null;
			world.cancelMining(humanSlot);
		} else if (action == ServerboundPlayerActionPacket.Action.START_DESTROY_BLOCK && !locked()
				&& kits[humanSlot].slot(fighters[humanSlot].getInventory().getSelectedSlot()).equals("pickaxe")) {
			humanMining = position.immutable();
		}
	}
	private void tickHumanMining(boolean caged) {
		if (humanMining == null) return;
		ServerPlayer player = fighters[humanSlot];
		BlockHitResult target = targetBlock(player, fighters[1 - humanSlot]);
		if (caged || !kits[humanSlot].slot(player.getInventory().getSelectedSlot()).equals("pickaxe")
				|| target == null || !humanMining.equals(target.getBlockPos())) {
			humanMining = null;
			world.cancelMining(humanSlot);
			return;
		}
		if (world.mine(humanSlot, humanMining, player.onGround())) humanMining = null;
	}
	boolean humanPlace(ServerPlayer player, InteractionHand hand, BlockHitResult requested) {
		if (locked() || hand != InteractionHand.MAIN_HAND || useDelay[humanSlot] > 0
				|| !kits[humanSlot].slot(player.getInventory().getSelectedSlot()).equals("blocks")) return false;
		useDelay[humanSlot] = 4;
		BlockHitResult target = targetBlock(player, fighters[1 - humanSlot]);
		if (target == null || !target.getBlockPos().equals(requested.getBlockPos()) || target.getDirection() != requested.getDirection()) return false;
		if (!world.place(humanSlot, target.getBlockPos(), target.getBlockPos().relative(target.getDirection()),
				player.getMainHandItem().getCount(), new AABB[]{fighters[0].getBoundingBox(), fighters[1].getBoundingBox()})) return false;
		player.getMainHandItem().shrink(1);
		player.inventoryMenu.broadcastChanges();
		return true;
	}
	boolean humanUseAllowed(ServerPlayer player, InteractionHand hand) {
		if (locked() || hand != InteractionHand.MAIN_HAND) return false;
		String selected = kits[humanSlot].slot(player.getInventory().getSelectedSlot());
		if (!selected.equals("bow") && !selected.equals("golden_apple")) return false;
		if (!player.isUsingItem()) {
			if (useDelay[humanSlot] > 0) return false;
			useDelay[humanSlot] = 4;
		}
		return true;
	}

	private void request() {
		requestedSeats = humanSlot < 0 ? new int[]{0, 1} : new int[]{1 - humanSlot};
		float[][] scalar = new float[requestedSeats.length][];
		float[][] features = new float[requestedSeats.length][];
		byte[][][][] voxels = new byte[requestedSeats.length][][][];
		JsonArray goals = session.trace.environment.definition.getAsJsonObject("experimental_bridge").getAsJsonArray("goals");
		var featureRules = new BridgeObservationEncoder.MatchRules(goalsToWin, maxTicks, cageTicks, refillTicks);
		var matchState = new BridgeObservationEncoder.MatchState(new int[]{match.score(0), match.score(1)}, match.ticks(), match.cageRemaining());
		for (int index = 0; index < requestedSeats.length; index++) {
			int actor = requestedSeats[index];
			ServerPlayer player = fighters[actor];
			Vec3 eye = player.getEyePosition();
			BlockPos eyeBlock = BlockPos.containing(eye);
			if (!session.level.hasChunksAt(eyeBlock.offset(-50, 0, -50), eyeBlock.offset(50, 0, 50))) throw new IllegalStateException("Bridge observation radius is not loaded");
			float[] rays = new float[8];
			for (int ray = 0; ray < 8; ray++) {
				double angle = ray * Math.PI / 4;
				Vec3 end = eye.add(-Math.sin(angle) * 50, 0, Math.cos(angle) * 50);
				HitResult hit = session.level.clip(new ClipContext(eye, end, ClipContext.Block.OUTLINE, ClipContext.Fluid.NONE, player));
				rays[ray] = hit.getType() == HitResult.Type.MISS ? 1 : (float) (eye.distanceTo(hit.getLocation()) / 50);
			}
			scalar[index] = policy.encode(player, fighters[1 - actor], rays);
			Vec3 position = player.position().subtract(session.origin);
			double[] feet = {position.x, position.y, position.z};
			JsonObject goal = goals.get(1 - actor).getAsJsonObject();
			double[] target = new double[3];
			for (int axis = 0; axis < 3; axis++) target[axis] = (goal.getAsJsonArray("min").get(axis).getAsDouble() + goal.getAsJsonArray("max").get(axis).getAsDouble()) / 2;
			int eating = player.isUsingItem() && player.getUseItem().is(Items.GOLDEN_APPLE) ? Math.max(0, 32 - player.getTicksUsingItem()) : 0;
			var state = new BridgeObservationEncoder.ActorState(count(player, kits[actor].index("blocks")), count(player, kits[actor].index("golden_apple")),
					arrows(player), refill[actor], player.getInventory().getSelectedSlot(), eating, world.miningProgress(actor),
					useDelay[actor], world.miningDelay(actor), feet, player.getYRot(), player.getXRot());
			features[index] = BridgeObservationEncoder.features(policy.matchEncoderVersion(), actor, featureRules, matchState, state, target);
			voxels[index] = BridgeObservationEncoder.voxels(session.trace.environment, world.overlay(), feet, player.getYRot(), actor);
		}
		requestNanos = System.nanoTime();
		inference.submit(policy.act(inference.epoch(), match.ticks(), requestedSeats, scalar, features, voxels), match.ticks(), requestNanos);
	}

	private static int count(ServerPlayer player, int slot) { return slot < 0 ? 0 : player.getInventory().getItem(slot).getCount(); }
	private static int arrows(ServerPlayer player) { return player.getInventory().getItem(9).is(Items.ARROW) ? player.getInventory().getItem(9).getCount() : 0; }
	private void lock(ServerPlayer player, Vec3 position) {
		player.stopUsingItem();
		player.setDeltaMovement(Vec3.ZERO);
		if (player instanceof ReplayPlayer) player.setPos(position);
		else if (player.position().distanceToSqr(position) > 1.0E-8) player.teleportTo(session.level, position.x, position.y, position.z, java.util.Set.of(), player.getYRot(), player.getXRot(), true);
	}
	private void freezeHuman() {
		if (humanSlot < 0 || fighters[humanSlot] == null) return;
		if (lockedPositions == null) lockedPositions = new Vec3[]{fighters[0].position(), fighters[1].position()};
		lock(fighters[humanSlot], lockedPositions[humanSlot]);
	}
	private static String message(Throwable error) {
		while (error.getCause() != null) error = error.getCause();
		return error.getMessage() == null ? error.getClass().getSimpleName() : error.getMessage();
	}
	private void publish() {
		String state = session.paused ? "Paused" : match.cageRemaining() > 0 ? "Cage " + DuelHud.time(match.cageRemaining()) : "Running";
		var lines = new ArrayList<String>();
		lines.add(status());
		DuelHud.publishMatch(new DuelHud.Match("LIVE BRIDGE", state, "Arena", match.ticks(), maxTicks, 0, match.score(0), match.score(1), 0, latency),
				DuelHud.fighter(fighters[0], humanSlot == 0 ? "A (you)" : "A", ""),
				DuelHud.fighter(fighters[1], humanSlot == 1 ? "B (you)" : "B", ""), lines);
	}
	@Override public void spawnHuman() { if (humanSlot >= 0) fighters[humanSlot] = LiveDuel.spawnHuman(session, humanSlot); }
	@Override public boolean isHumanSlot(int index) { return index == humanSlot; }
	@Override public void setBot(ReplayPlayer bot) { fighters[bot.replayIndex] = bot; }
	@Override public void updateTargets() { for (ReplayPlayer bot : session.actors) bot.setLiveTarget(fighters[1 - bot.replayIndex]); }
	@Override public boolean ownsProjectile(Entity owner) { return owner != null && (owner == fighters[0] || owner == fighters[1]); }
	@Override public boolean allowDeath(LivingEntity entity) {
		for (int actor = 0; actor < 2; actor++) if (entity == fighters[actor]) { defeated[actor] = true; entity.setHealth(1); return false; }
		return true;
	}
	@Override public boolean allowDamage(LivingEntity entity, DamageSource source) {
		boolean victim = entity == fighters[0] || entity == fighters[1];
		Entity attacker = source.getEntity();
		boolean fighterAttack = attacker != null && (attacker == fighters[0] || attacker == fighters[1]);
		if (!victim) return !fighterAttack;
		if (locked()) return false;
		return fighterAttack || attacker == null && source.getDirectEntity() == null;
	}
	@Override public void selectHuman(ServerPlayer viewer, int slot) {
		if (savedHuman == null) { viewer.closeContainer(); savedHuman = new ParticipantState(viewer); }
		humanSlot = slot;
	}
	@Override public void restoreHuman() {
		ServerPlayer player = session.server.getPlayerList().getPlayer(session.owner);
		if (player != null) restoreHuman(player);
	}
	@Override public void restoreHuman(ServerPlayer player) {
		if (savedHuman == null) return;
		savedHuman.restore(player);
		savedHuman = null;
		humanSlot = -1;
	}
	@Override public void cancelRequest() { inference.cancel(); requestedSeats = null; }
	@Override public void cancelForLifecycle() {
		cancelRequest();
		for (ServerPlayer fighter : fighters) if (fighter != null && fighter.isUsingItem()) fighter.stopUsingItem();
		session.clearProjectiles();
	}
	@Override public String status() {
		return String.format(Locale.ROOT, "BRIDGE %s tick %d/%d | score %d:%d | %s | inference %.0fms",
				session.paused ? "PAUSED" : "RUNNING", match.ticks(), maxTicks, match.score(0), match.score(1),
				humanSlot < 0 ? "bots" : "you=" + (humanSlot + 1), latency);
	}
	@Override public String deploymentSummary() { return deploymentSummary; }
	@Override public void close() { cancelForLifecycle(); policy.close(); restoreHuman(); }
}