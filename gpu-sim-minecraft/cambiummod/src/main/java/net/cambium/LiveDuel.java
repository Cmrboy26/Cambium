package net.cambium;

import com.google.gson.JsonObject;
import com.mojang.brigadier.arguments.DoubleArgumentType;
import com.mojang.brigadier.arguments.IntegerArgumentType;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.function.BiConsumer;
import net.cambium.mixin.PlayerEquipmentAccess;
import net.cambium.mixin.LivingEntityRecordingAccessor;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.fabric.api.client.command.v2.ClientCommands;
import net.fabricmc.fabric.api.entity.event.v1.ServerLivingEntityEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.fabric.api.networking.v1.ServerPlayConnectionEvents;
import net.minecraft.client.Minecraft;
import net.minecraft.core.BlockPos;
import net.minecraft.network.chat.Component;
import net.minecraft.network.protocol.game.ClientboundSetHeldSlotPacket;
import net.minecraft.network.protocol.game.ClientboundSetEntityMotionPacket;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.Identifier;
import net.minecraft.world.entity.EquipmentSlot;
import net.minecraft.world.entity.ai.attributes.Attributes;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.food.FoodData;
import net.minecraft.world.level.GameType;
import net.minecraft.world.level.ClipContext;
import net.minecraft.world.phys.EntityHitResult;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec3;

final class LiveDuel implements AutoCloseable {

	private static final Placement[] placements = new Placement[2];
	private static long connectionGeneration;
	private final SimulationReplay.Session session;
	private final LivePolicy policy;
	private final int maxTicks;
	private CompletableFuture<JsonObject> pending;
	private String epoch;
	private int tick;
	private int round;
	private int restartDelay;
	private int winsA;
	private int winsB;
	private int draws;
	private long sentAt;
	private double inferenceMillis;
	private final ServerPlayer[] fighters = new ServerPlayer[2];
	private int humanSlot = -1;
	private HumanState savedHuman;
	private boolean humanDefeated;

	private record Placement(UUID owner, ServerLevel level, Vec3 position, float yaw, float pitch) {}

	LiveDuel(SimulationReplay.Session session, LivePolicy policy, JsonObject info) {
		this.session = session;
		this.policy = policy;
		this.maxTicks = info.get("max_ticks").getAsInt();
	}

	static void cancelConnection() {
		connectionGeneration++;
	}

	static void initialize() {
		ClientCommandRegistrationCallback.EVENT.register((dispatcher, registryAccess) -> {
			var root = ClientCommands.literal("simduel").executes(context -> dispatch(context.getSource().getClient(),
					(server, viewer) -> tell(viewer, "/simduel arena|world [port], pos1|pos2 [x y z], join [1|2], leave, pause, resume, reset, status, stop")));
			for (boolean arena : new boolean[]{true, false}) {
				root.then(ClientCommands.literal(arena ? "arena" : "world")
						.executes(context -> connect(context.getSource().getClient(), arena, LivePolicy.DEFAULT_PORT))
						.then(ClientCommands.argument("port", IntegerArgumentType.integer(1, 65535))
								.executes(context -> connect(context.getSource().getClient(), arena, IntegerArgumentType.getInteger(context, "port")))));
			}
			for (int index = 0; index < 2; index++) {
				final int player = index;
				root.then(ClientCommands.literal(index == 0 ? "pos1" : "pos2")
						.executes(context -> mark(context.getSource().getClient(), player, null))
						.then(ClientCommands.argument("x", DoubleArgumentType.doubleArg(-29999984, 29999984))
								.then(ClientCommands.argument("y", DoubleArgumentType.doubleArg(-20000000, 20000000))
										.then(ClientCommands.argument("z", DoubleArgumentType.doubleArg(-29999984, 29999984))
												.executes(context -> mark(context.getSource().getClient(), player, new Vec3(
														DoubleArgumentType.getDouble(context, "x"), DoubleArgumentType.getDouble(context, "y"), DoubleArgumentType.getDouble(context, "z"))))))));
			}
			root.then(ClientCommands.literal("join")
					.executes(context -> dispatch(context.getSource().getClient(), (server, viewer) -> join(viewer, 0)))
					.then(ClientCommands.argument("player", IntegerArgumentType.integer(1, 2))
							.executes(context -> dispatch(context.getSource().getClient(),
									(server, viewer) -> join(viewer, IntegerArgumentType.getInteger(context, "player") - 1)))));
			for (String command : List.of("leave", "pause", "resume", "reset", "status", "stop")) {
				root.then(ClientCommands.literal(command).executes(context -> dispatch(context.getSource().getClient(),
						(server, viewer) -> control(viewer, command))));
			}
			dispatcher.register(root);
		});
		ServerLifecycleEvents.SERVER_STOPPING.register(server -> {
			connectionGeneration++;
			for (int index = 0; index < 2; index++) if (placements[index] != null && placements[index].level.getServer() == server) placements[index] = null;
		});
		ServerLivingEntityEvents.ALLOW_DEATH.register((entity, source, amount) -> {
			SimulationReplay.Session active = SimulationReplay.active;
			if (active == null || active.live == null) return true;
			LiveDuel live = active.live;
			if (live.humanSlot < 0 || entity != live.fighters[live.humanSlot]) return true;
			live.humanDefeated = true;
			entity.setHealth(1);
			return false;
		});
		ServerLivingEntityEvents.ALLOW_DAMAGE.register((entity, source, amount) -> {
			SimulationReplay.Session active = SimulationReplay.active;
			if (active == null || active.live == null) return true;
			LiveDuel live = active.live;
			return live.humanSlot < 0 || entity != live.fighters[live.humanSlot]
					|| !active.paused && live.restartDelay == 0 && !live.humanDefeated;
		});
		ServerPlayConnectionEvents.DISCONNECT.register((handler, server) -> {
			SimulationReplay.Session active = SimulationReplay.active;
			if (active != null && active.server == server && active.live != null && active.owner.equals(handler.player.getUUID())) {
				active.live.restoreHuman(handler.player);
				active.close();
				SimulationReplay.active = null;
			}
		});
	}

	private static int dispatch(Minecraft client, BiConsumer<MinecraftServer, ServerPlayer> action) {
		MinecraftServer server = client.getSingleplayerServer();
		if (server == null || client.player == null || server.isPublished()) {
			client.gui.hud.getChat().addClientSystemMessage(Component.literal("Live duels require a private singleplayer world."));
			return 0;
		}
		UUID owner = client.player.getUUID();
		server.execute(() -> {
			ServerPlayer viewer = server.getPlayerList().getPlayer(owner);
			if (viewer != null && !server.isPublished()) action.accept(server, viewer);
		});
		return 1;
	}

	private static int mark(Minecraft client, int index, Vec3 position) {
		return dispatch(client, (server, viewer) -> {
			Vec3 selected = position == null ? viewer.position() : position;
			placements[index] = new Placement(viewer.getUUID(), viewer.level(), selected, viewer.getYRot(), viewer.getXRot());
			tell(viewer, "Player " + (index + 1) + " spawn: " + selected + " (current yaw/pitch saved; feet coordinates).");
		});
	}

	static int stop(Minecraft client) {
		return dispatch(client, (server, viewer) -> control(viewer, "stop"));
	}

	static int connect(Minecraft client, boolean arena, int port) {
		return dispatch(client, (server, viewer) -> {
			Placement[] chosen = placements.clone();
			if (!arena) {
				for (Placement point : chosen) {
					if (point == null || point.level != viewer.level() || !point.owner.equals(viewer.getUUID())) {
						tell(viewer, "Set /simduel pos1 and /simduel pos2 in this dimension first.");
						return;
					}
				}
			}
			long generation = ++connectionGeneration;
			UUID owner = viewer.getUUID();
			ServerLevel level = viewer.level();
			LivePolicy policy = new LivePolicy(port);
			tell(viewer, "Connecting to loopback policy on port " + port + "...");
			policy.info().whenComplete((info, failure) -> server.execute(() -> {
				ServerPlayer current = server.getPlayerList().getPlayer(owner);
				if (generation != connectionGeneration || current == null || current.level() != level || server.isPublished()) {
					policy.close();
					return;
				}
				if (failure != null) {
					policy.close();
					tell(current, "Policy connection failed: " + errorMessage(failure) + ". Start python -B -m simulator.training.serve <checkpoint>.");
					return;
				}
				SimulationReplay.Session candidate = null;
				try {
					ReplayTrace trace = ReplayTrace.forLive(info.getAsJsonObject("environment"));
					if (SimulationReplay.active != null) SimulationReplay.active.close();
					SimulationReplay.active = null;
					Vec3 origin = arena ? new Vec3(Math.floor(current.getX()), Math.floor(current.getY()) - 1, Math.floor(current.getZ())) : Vec3.ZERO;
					candidate = new SimulationReplay.Session(server, level, owner, trace, origin, Path.of("."));
					candidate.absolute = false;
					candidate.buildArena = arena;
					if (!arena) candidate.initialActors = worldSpawns(trace.frames.getFirst().actors(), chosen);
					candidate.live = new LiveDuel(candidate, policy, info);
					candidate.reset();
					candidate.paused = false;
					SimulationReplay.active = candidate;
					tell(current, "Live duel started: " + (arena ? "checkpoint arena" : "existing world; no arena edits")
							+ ". Policy: " + info.get("checkpoint").getAsString());
					if (!arena) tell(current, "World mode is experimental: terrain, fluids and hazards may be outside this policy's training distribution.");
					tell(current, "Use /simduel join [1|2] to fight, or hit either bot while they duel. Spectator mode gives an undisturbed match.");
					if (!level.isPvpAllowed()) {
						tell(current, "PvP is disabled in this world; bot attacks may be rejected. World rules were not changed.");
					}
				} catch (IOException | RuntimeException error) {
					if (candidate != null) candidate.close(); else policy.close();
					tell(current, "Could not start live duel: " + errorMessage(error));
					CambiumFabric.LOGGER.error("Live duel initialization failed", error);
				}
			}));
		});
	}

	private static List<ReplayTrace.Actor> worldSpawns(List<ReplayTrace.Actor> templates, Placement[] chosen) {
		List<ReplayTrace.Actor> result = new ArrayList<>();
		for (int index = 0; index < 2; index++) {
			ReplayTrace.Actor initial = templates.get(index);
			Placement point = chosen[index];
			result.add(initial.at(new double[]{point.position.x, point.position.y, point.position.z}, point.yaw, point.pitch));
		}
		return result;
	}

	private static void control(ServerPlayer viewer, String command) {
		if (command.equals("stop")) connectionGeneration++;
		SimulationReplay.Session active = SimulationReplay.active;
		if (active == null || active.live == null || !active.owner.equals(viewer.getUUID())) {
			tell(viewer, "No live duel. Use /simduel arena or /simduel world (stop also cancels a pending connection).");
			return;
		}
		switch (command) {
			case "leave" -> {
				active.live.restoreHuman();
				control(viewer, "reset");
				tell(viewer, "Returned to bot versus bot.");
			}
			case "pause" -> { active.paused = true; active.live.cancelRequest(); }
			case "resume" -> { active.live.cancelRequest(); active.paused = false; }
			case "reset" -> {
				try { active.reset(); active.paused = false; }
				catch (RuntimeException error) {
					active.close();
					SimulationReplay.active = null;
					tell(viewer, "Live reset failed: " + errorMessage(error));
				}
			}
			case "stop" -> { active.close(); SimulationReplay.active = null; tell(viewer, "Live duel stopped."); }
			case "status" -> {
				tell(viewer, active.live.status());
				tell(viewer, "Mod " + CambiumFabric.BUILD_ID + "; catalogue=" + PolicyCatalogue.HASH);
			}
			default -> throw new IllegalArgumentException("Unknown live command");
		}
	}

	void reset() {
		cancelRequest();
		tick = 0;
		round++;
		restartDelay = 0;
		humanDefeated = false;
	}

	private static void join(ServerPlayer viewer, int slot) {
		SimulationReplay.Session active = SimulationReplay.active;
		if (active == null || active.live == null || !active.owner.equals(viewer.getUUID())) {
			tell(viewer, "Start /simduel arena or /simduel world first, then /simduel join [1|2].");
			return;
		}
		if (!viewer.isAlive()) {
			tell(viewer, "Respawn before joining a duel.");
			return;
		}
		LiveDuel live = active.live;
		if (live.savedHuman == null) {
			viewer.closeContainer();
			live.savedHuman = new HumanState(viewer);
		}
		live.humanSlot = slot;
		control(viewer, "reset");
		if (SimulationReplay.active == active) tell(viewer, "You are player " + (slot + 1) + ". /simduel leave restores your player and resumes bot versus bot.");
	}

	boolean isHumanSlot(int index) { return index == humanSlot; }

	void setBot(ReplayPlayer bot) { fighters[bot.replayIndex] = bot; }

	void updateTargets() {
		for (ReplayPlayer bot : session.actors) bot.setLiveTarget(fighters[1 - bot.replayIndex]);
	}

	void spawnHuman() {
		if (humanSlot < 0) return;
		ServerPlayer player = session.server.getPlayerList().getPlayer(session.owner);
		if (player == null || player.level() != session.level) throw new IllegalStateException("Duel player is no longer in this dimension");
		ReplayTrace.Actor initial = session.initialActors.get(humanSlot);
		Vec3 spawn = session.origin.add(initial.position()[0], initial.position()[1], initial.position()[2]);
		BlockPos feet = BlockPos.containing(spawn);
		BlockPos head = BlockPos.containing(spawn.add(0, 1.8, 0));
		if (!session.level.hasChunksAt(feet.offset(-1, 0, -1), head.offset(1, 0, 1))
				|| session.level.isOutsideBuildHeight(feet) || session.level.isOutsideBuildHeight(head)
				|| !session.level.getWorldBorder().isWithinBounds(feet) || !session.level.getWorldBorder().isWithinBounds(head)) {
			throw new IllegalStateException("Live spawn must fit inside loaded chunks, world height and border.");
		}
		player.closeContainer();
		player.stopRiding();
		player.stopUsingItem();
		player.setGameMode(GameType.SURVIVAL);
		player.getInventory().clearContent();
		JsonObject loadout = session.trace.environment.loadouts.get(humanSlot);
		var weapons = loadout.getAsJsonArray("weapons");
		for (int index = 0; index < weapons.size(); index++) player.getInventory().setItem(index, item(weapons.get(index).getAsString()));
		player.getInventory().setSelectedSlot(loadout.get("selected").getAsInt());
		player.setItemSlot(EquipmentSlot.OFFHAND, initial.shield_equipped() ? new ItemStack(Items.SHIELD) : ItemStack.EMPTY);
		for (String slot : SimulationContract.ARMOR_SLOTS) {
			EquipmentSlot equipment = switch (slot) {
				case "boots" -> EquipmentSlot.FEET;
				case "leggings" -> EquipmentSlot.LEGS;
				case "chestplate" -> EquipmentSlot.CHEST;
				case "helmet" -> EquipmentSlot.HEAD;
				default -> throw new IllegalArgumentException("Unsupported armor slot");
			};
			player.setItemSlot(equipment, item(initial.armor().has(slot) ? initial.armor().get(slot).getAsString() : "minecraft:air"));
		}
		player.teleportTo(session.level, session.origin.x + initial.position()[0], session.origin.y + initial.position()[1],
				session.origin.z + initial.position()[2], java.util.Set.of(), initial.yaw(), initial.pitch(), true);
		if (!session.level.noCollision(player, player.getBoundingBox())) throw new IllegalStateException("Your duel spawn must be clear.");
		((LivingEntityRecordingAccessor) (Object) player).cambium$refreshEquipment();
		((PlayerEquipmentAccess) (Object) player).cambium$setFoodData(new ReplayPlayer.ReplayFoodData());
		player.setDeltaMovement(new Vec3(initial.velocity()[0], initial.velocity()[1], initial.velocity()[2]));
		player.setHealth(initial.health());
		player.getAttribute(Attributes.MAX_ABSORPTION).setBaseValue(initial.absorption());
		player.setAbsorptionAmount(initial.absorption());
		player.damageCooldownTime = 0;
		player.hurtTime = 0;
		player.fallDistance = 0;
		player.clearFire();
		player.setSprinting(false);
		player.setShiftKeyDown(false);
		player.getFoodData().setFoodLevel(20);
		player.getFoodData().setSaturation(0);
		ItemStack shield = new ItemStack(Items.SHIELD);
		player.getCooldowns().removeCooldown(player.getCooldowns().getCooldownGroup(shield));
		player.resetAttackStrengthTicker();
		player.inventoryMenu.broadcastChanges();
		player.connection.send(new ClientboundSetHeldSlotPacket(player.getInventory().getSelectedSlot()));
		player.connection.send(new ClientboundSetEntityMotionPacket(player));
		fighters[humanSlot] = player;
	}

	private static ItemStack item(String name) {
		ItemStack stack = name.equals("minecraft:air") ? ItemStack.EMPTY : new ItemStack(BuiltInRegistries.ITEM.getValue(Identifier.parse(name)));
		if (!stack.isEmpty()) stack.set(net.minecraft.core.component.DataComponents.UNBREAKABLE, net.minecraft.util.Unit.INSTANCE);
		return stack;
	}

	private void restoreHuman() {
		if (savedHuman == null) return;
		ServerPlayer player = session.server.getPlayerList().getPlayer(session.owner);
		if (player != null) restoreHuman(player);
	}

	private void restoreHuman(ServerPlayer player) {
		if (savedHuman == null) return;
		savedHuman.restore(player);
		savedHuman = null;
		humanSlot = -1;
	}

	private void cancelRequest() {
		if (pending != null) pending.cancel(true);
		pending = null;
		epoch = UUID.randomUUID().toString();
	}

	void tick(ServerPlayer viewer) {
		if (humanSlot >= 0) {
			viewer.getFoodData().setFoodLevel(20);
			viewer.getFoodData().setSaturation(0);
		}
		if (!session.paused && restartDelay == 0) checkRound();
		if (!session.paused) {
			if (restartDelay > 0) {
				if (--restartDelay == 0) {
					session.reset();
					session.paused = false;
				}
			} else {
				try {
					if (pending != null && pending.isDone()) {
						JsonObject response = pending.join();
						inferenceMillis = (System.nanoTime() - sentAt) / 1000000.0;
						ReplayTrace.Controls[] controls = policy.mapActions(response, epoch, tick);
						pending = null;
						step(controls);
						tick++;
						checkRound();
					}
					if (pending == null && restartDelay == 0) {
						float[][] observations = observations();
						sentAt = System.nanoTime();
						pending = policy.act(epoch, tick, observations);
					}
				} catch (RuntimeException error) {
					session.paused = true;
					cancelRequest();
					session.tell("Live duel paused: " + errorMessage(error) + ". Fix the cause, then /simduel resume.");
					CambiumFabric.LOGGER.warn("Live duel paused", error);
				}
			}
		}
		viewer.sendOverlayMessage(Component.literal(status()));
	}

	private float[][] observations() {
		float[][] result = new float[2][];
		for (int index = 0; index < 2; index++) {
			ServerPlayer source = fighters[index];
			ServerPlayer target = fighters[1 - index];
			Vec3 eye = source.getEyePosition();
			BlockPos eyeBlock = BlockPos.containing(eye);
			if (!session.level.hasChunksAt(eyeBlock.offset(-50, 0, -50), eyeBlock.offset(50, 0, 50))) {
				throw new IllegalStateException("The 50-block observation radius must stay loaded; move nearer to the bots or increase view distance");
			}
			float[] rays = new float[8];
			for (int ray = 0; ray < 8; ray++) {
				double angle = ray * 45.0 * 0.017453292519943295;
				Vec3 end = eye.add(-Math.sin(angle) * 50.0, 0, Math.cos(angle) * 50.0);
				HitResult hit = session.level.clip(new ClipContext(eye, end, ClipContext.Block.OUTLINE, ClipContext.Fluid.NONE, source));
				rays[ray] = hit.getType() == HitResult.Type.MISS ? 1 : (float) (eye.distanceTo(hit.getLocation()) / 50.0);
			}
			result[index] = policy.encode(source, target, rays, index);
		}
		return result;
	}

	private void step(ReplayTrace.Controls[] controls) {
		for (ReplayPlayer bot : session.actors) bot.prepare(controls[bot.replayIndex]);
		HitResult[] picks = session.actors.stream().map(ReplayPlayer::pickAttack).toArray(HitResult[]::new);
		for (int index = 0; index < session.actors.size(); index++) {
			ReplayPlayer player = session.actors.get(index);
			if (!player.isAlive() || picks[index] == null) continue;
			player.swingReplay();
			if (picks[index] instanceof EntityHitResult hit) player.attack(hit.getEntity());
			else player.miss(picks[index]);
			session.actors.forEach(ReplayPlayer::acceptMotion);
		}
		session.actors.forEach(ReplayPlayer::advanceInput);
	}

	private void checkRound() {
		boolean lostA = lost(fighters[0]);
		boolean lostB = lost(fighters[1]);
		if (!lostA && !lostB && tick < maxTicks) return;
		String result;
		if (lostA != lostB) {
			if (lostA) winsB++; else winsA++;
			result = lostA ? "B wins" : "A wins";
		} else {
			draws++;
			result = lostA ? "both lost" : "timeout";
		}
		session.tell("Live round " + round + ": " + result + " after " + tick + " ticks. Resetting in 1 second.");
		cancelRequest();
		restartDelay = 20;
	}

	private boolean lost(ServerPlayer player) {
		if (humanSlot >= 0 && player == fighters[humanSlot] && humanDefeated) return true;
		if (!player.isAlive() || player.getY() < session.level.getMinY()) return true;
		if (!session.buildArena) return false;
		var environment = session.trace.environment;
		Vec3 position = player.position().subtract(session.origin);
		return position.x < environment.min[0] || position.x >= environment.min[0] + environment.size[0]
				|| position.z < environment.min[2] || position.z >= environment.min[2] + environment.size[2]
				|| position.y < environment.min[1] - 4;
	}

	private String status() {
		return String.format(Locale.ROOT, "LIVE %s%s%s round %d tick %d/%d | A %.1f B %.1f | wins %d:%d draws %d | response %.0fms",
				session.buildArena ? "ARENA " : "WORLD ", session.paused ? "PAUSED" : restartDelay > 0 ? "RESETTING" : pending != null && !pending.isDone() ? "WAITING" : "",
				humanSlot < 0 ? "" : " YOU=" + (humanSlot == 0 ? "A" : "B"),
				round, tick, maxTicks, fighters[0].getHealth(), fighters[1].getHealth(), winsA, winsB, draws, inferenceMillis);
	}

	private static void tell(ServerPlayer viewer, String message) {
		viewer.sendSystemMessage(Component.literal(message));
	}

	private static String errorMessage(Throwable error) {
		while (error.getCause() != null) error = error.getCause();
		return error.getMessage() == null ? error.getClass().getSimpleName() : error.getMessage();
	}

	@Override
	public void close() {
		cancelRequest();
		policy.close();
		restoreHuman();
	}

	private static final class HumanState {
		private final ServerLevel level;
		private final Vec3 position;
		private final Vec3 velocity;
		private final float yaw;
		private final float pitch;
		private final GameType gameMode;
		private final List<ItemStack> inventory = new ArrayList<>();
		private final int selected;
		private final float health;
		private final float absorption;
		private final double maxAbsorption;
		private final FoodData food;

		HumanState(ServerPlayer player) {
			level = player.level();
			position = player.position();
			velocity = player.getDeltaMovement();
			yaw = player.getYRot();
			pitch = player.getXRot();
			gameMode = player.gameMode.getGameModeForPlayer();
			for (int index = 0; index < player.getInventory().getContainerSize(); index++) inventory.add(player.getInventory().getItem(index).copy());
			selected = player.getInventory().getSelectedSlot();
			health = player.getHealth();
			absorption = player.getAbsorptionAmount();
			maxAbsorption = player.getAttribute(Attributes.MAX_ABSORPTION).getBaseValue();
			food = player.getFoodData();
		}

		void restore(ServerPlayer player) {
			player.closeContainer();
			player.stopUsingItem();
			player.getInventory().clearContent();
			for (int index = 0; index < inventory.size(); index++) player.getInventory().setItem(index, inventory.get(index).copy());
			player.getInventory().setSelectedSlot(selected);
			((LivingEntityRecordingAccessor) (Object) player).cambium$refreshEquipment();
			player.setGameMode(gameMode);
			player.teleportTo(level, position.x, position.y, position.z, java.util.Set.of(), yaw, pitch, true);
			player.setDeltaMovement(velocity);
			player.setHealth(health);
			player.getAttribute(Attributes.MAX_ABSORPTION).setBaseValue(maxAbsorption);
			player.setAbsorptionAmount(absorption);
			((PlayerEquipmentAccess) (Object) player).cambium$setFoodData(food);
			player.damageCooldownTime = 0;
			player.hurtTime = 0;
			player.fallDistance = 0;
			player.clearFire();
			player.inventoryMenu.broadcastChanges();
			player.connection.send(new ClientboundSetHeldSlotPacket(selected));
			player.connection.send(new ClientboundSetEntityMotionPacket(player));
		}
	}
}