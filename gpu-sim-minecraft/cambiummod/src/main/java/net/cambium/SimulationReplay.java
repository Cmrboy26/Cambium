package net.cambium;

import com.mojang.brigadier.arguments.DoubleArgumentType;
import com.mojang.brigadier.arguments.StringArgumentType;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.util.HashSet;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.UUID;
import java.util.function.Consumer;
import net.fabricmc.fabric.api.client.command.v2.ClientCommands;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.minecraft.client.Minecraft;
import net.minecraft.core.BlockPos;
import net.minecraft.network.chat.Component;
import net.minecraft.network.chat.ClickEvent;
import net.minecraft.network.chat.HoverEvent;
import net.minecraft.network.protocol.game.ClientboundPlayerInfoRemovePacket;
import net.minecraft.network.protocol.game.ClientboundPlayerInfoUpdatePacket;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.util.Mth;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EntityTypes;
import net.minecraft.world.entity.Display.ItemDisplay;
import net.minecraft.world.entity.projectile.arrow.Arrow;
import net.minecraft.world.item.ItemDisplayContext;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.gamerules.GameRules;
import net.minecraft.world.phys.EntityHitResult;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec3;
import net.cambium.mixin.DisplayAccess;
import net.cambium.mixin.ItemDisplayAccess;

final class SimulationReplay {

	static Session active;

	// Vanilla reference: BowItem#releaseUsing creates an owner-bound Arrow through Arrow(Level, LivingEntity, ...);
	// the Session adapter records only arrows whose owner is one of its two actors.
	static void trackArrow(Arrow arrow) {
		Session current = active;
		if (current == null || current.level != arrow.level()) return;
		Entity owner = arrow.getOwner();
		if (current.actors.stream().anyMatch(actor -> actor == owner)
				|| current.live != null && current.live.ownsProjectile(owner)) current.ownedArrows.add(arrow);
	}

	static void trackEgg(net.minecraft.world.entity.projectile.throwableitemprojectile.ThrownEgg egg) {
		Session current = active;
		if (current == null || current.level != egg.level()) return;
		Entity owner = egg.getOwner();
		if (current.actors.stream().anyMatch(actor -> actor == owner)
				|| current.live != null && current.live.ownsProjectile(owner)) current.ownedEggs.add(egg);
	}

	static boolean ownsEgg(net.minecraft.world.entity.projectile.throwableitemprojectile.ThrownEgg egg) {
		return active != null && active.ownedEggs.contains(egg);
	}

	static void trackSnowball(net.minecraft.world.entity.projectile.throwableitemprojectile.Snowball snowball) {
		Session current = active;
		if (current == null || current.level != snowball.level()) return;
		Entity owner = snowball.getOwner();
		if (current.actors.stream().anyMatch(actor -> actor == owner)
				|| current.live != null && current.live.ownsProjectile(owner)) current.ownedSnowballs.add(snowball);
	}

	static void trackPearl(net.minecraft.world.entity.projectile.throwableitemprojectile.ThrownEnderpearl pearl) {
		Session current = active;
		if (current == null || current.level != pearl.level()) return;
		Entity owner = pearl.getOwner();
		if (current.actors.contains(owner) || current.live != null && current.live.ownsProjectile(owner)) current.ownedPearls.add(pearl);
	}

	static boolean ownsPearl(net.minecraft.world.entity.projectile.throwableitemprojectile.ThrownEnderpearl pearl) {
		return active != null && active.ownedPearls.contains(pearl);
	}

	static void trackWind(net.minecraft.world.entity.projectile.hurtingprojectile.windcharge.WindCharge wind) {
		Session current = active;
		if (current == null || current.level != wind.level()) return;
		Entity owner = wind.getOwner();
		if (current.actors.contains(owner) || current.live != null && current.live.ownsProjectile(owner)) current.ownedWind.add(wind);
	}

	static void initialize() {
		LiveDuel.initialize();
		ClientCommandRegistrationCallback.EVENT.register((dispatcher, registryAccess) -> {
			var root = ClientCommands.literal("simreplay")
					.executes(context -> message(context.getSource().getClient(),
							"/simreplay load [latest|file], absolute|input, analyze [stop], pause, resume, step, restart, reset, speed <0.05..8>, stop"));
			root.then(ClientCommands.literal("load")
					.executes(context -> load(context.getSource().getClient(), null))
					.then(ClientCommands.literal("latest")
							.executes(context -> load(context.getSource().getClient(), null)))
					.then(ClientCommands.argument("file", StringArgumentType.string())
							.executes(context -> load(context.getSource().getClient(), StringArgumentType.getString(context, "file")))));
			for (String mode : List.of("absolute", "input")) {
				root.then(ClientCommands.literal(mode)
						.executes(context -> control(context.getSource().getClient(), session -> {
							session.absolute = mode.equals("absolute");
							session.reset();
						})));
			}
			root.then(ClientCommands.literal("pause").executes(context -> control(
					context.getSource().getClient(), session -> session.paused = true)));
			root.then(ClientCommands.literal("resume").executes(context -> control(
					context.getSource().getClient(), session -> session.paused = false)));
			root.then(ClientCommands.literal("step").executes(context -> control(
					context.getSource().getClient(), session -> { session.paused = true; session.advance(); })));
			root.then(ClientCommands.literal("restart").executes(context -> control(
					context.getSource().getClient(), Session::restart)));
			root.then(ClientCommands.literal("reset").executes(context -> control(
					context.getSource().getClient(), Session::reset)));
			root.then(ClientCommands.literal("analyze")
					.executes(context -> control(context.getSource().getClient(), Session::startAnalysis))
					.then(ClientCommands.literal("stop").executes(context -> control(
							context.getSource().getClient(), session -> session.finishAnalysis("partial", "analysis_stopped")))));
			root.then(ClientCommands.literal("speed")
					.then(ClientCommands.argument("rate", DoubleArgumentType.doubleArg(0.05, 8))
							.executes(context -> control(context.getSource().getClient(), session -> {
								session.speed = DoubleArgumentType.getDouble(context, "rate");
								session.budget = 0;
							}))));
			root.then(ClientCommands.literal("stop").executes(context -> control(
					context.getSource().getClient(), session -> { session.close(); active = null; })));
			dispatcher.register(root);
		});
		ServerTickEvents.END_SERVER_TICK.register(server -> {
			if (active != null && active.server == server) {
				try {
					active.tick();
				} catch (RuntimeException error) {
					CambiumFabric.LOGGER.error("Replay stopped", error);
					active.tell("Replay stopped: " + error.getMessage());
					active.close();
					active = null;
				}
			}
		});
		ServerLifecycleEvents.SERVER_STOPPING.register(server -> {
			if (active != null && active.server == server) {
				active.close();
				active = null;
			}
		});
	}

	private static int load(Minecraft client, String filename) {
		return load(client, filename, false);
	}

	static int load(Minecraft client, String filename, boolean play) {
		var server = client.getSingleplayerServer();
		if (server == null || client.player == null || server.isPublished()) {
			return message(client, "Replay is restricted to a private singleplayer world.");
		}
		try {
			Path folder = client.gameDirectory.toPath().resolve("sim-replays");
			Files.createDirectories(folder);
			Path path;
			if (filename == null) {
				try (var files = Files.list(folder)) {
					path = files.filter(Files::isRegularFile).filter(file -> file.toString().endsWith(".jsonl"))
							.max(Comparator.comparingLong(file -> file.toFile().lastModified()))
							.orElseThrow(() -> new IOException("No .jsonl replay in " + folder));
				}
			} else {
				path = folder.resolve(filename).normalize();
			}
			ReplayTrace trace = ReplayTrace.read(path);
			UUID owner = client.player.getUUID();
			server.execute(() -> {
				ServerPlayer viewer = server.getPlayerList().getPlayer(owner);
				if (viewer == null) {
					return;
				}
				LiveDuel.cancelConnection();
				Vec3 origin = active != null && active.live == null && active.level == viewer.level() ? active.origin
						: new Vec3(Mth.floor(viewer.getX()), Mth.floor(viewer.getY()) - 1, Mth.floor(viewer.getZ()));
				if (active != null) {
					active.close();
				}
				active = new Session(server, viewer.level(), owner, trace, origin, folder.resolve("analysis"));
				try {
					active.reset();
					active.paused = !play;
					active.tell("Loaded " + trace.path.getFileName() + " (" + trace.backend
							+ ", absolute), arena rebuilt; " + (play ? "playing" : "paused at tick 0") + ", speed 1x.");
					active.warnFallDamageMismatch();
				} catch (RuntimeException error) {
					active.tell("Could not spawn replay: " + error.getMessage());
					active.close();
					active = null;
					CambiumFabric.LOGGER.error("Replay initialization failed", error);
				}
			});
			return 1;
		} catch (IOException error) {
			return message(client, error.getMessage());
		}
	}

	private static int control(Minecraft client, Consumer<Session> action) {
		var server = client.getSingleplayerServer();
		if (server == null || client.player == null) {
			return message(client, "No singleplayer replay is running.");
		}
		UUID owner = client.player.getUUID();
		server.execute(() -> {
			if (active != null && active.server == server && active.live == null) {
				try {
					action.accept(active);
				} catch (RuntimeException error) {
					active.tell("Replay command failed: " + error.getMessage());
					active.close();
					active = null;
				}
			} else {
				ServerPlayer viewer = server.getPlayerList().getPlayer(owner);
				if (viewer != null) viewer.sendSystemMessage(Component.literal("Use /simreplay load for replays or /simduel for live controls."));
			}
		});
		return 1;
	}

	private static int message(Minecraft client, String text) {
		client.gui.hud.getChat().addClientSystemMessage(Component.literal(text));
		return 1;
	}

	static final class Session {
		final MinecraftServer server;
		final ServerLevel level;
		final UUID owner;
		final ReplayTrace trace;
		final Vec3 origin;
		final Path analysisDirectory;
		boolean absolute = true;
		final List<ReplayPlayer> actors = new ArrayList<>();
		final Set<Arrow> ownedArrows = new HashSet<>();
		final Set<net.minecraft.world.entity.projectile.throwableitemprojectile.ThrownEgg> ownedEggs = new HashSet<>();
		final Set<net.minecraft.world.entity.projectile.throwableitemprojectile.Snowball> ownedSnowballs = new HashSet<>();
		final Set<net.minecraft.world.entity.projectile.throwableitemprojectile.ThrownEnderpearl> ownedPearls = new HashSet<>();
		final Set<net.minecraft.world.entity.projectile.hurtingprojectile.windcharge.WindCharge> ownedWind = new HashSet<>();
		final List<RenderedProjectile> renderedProjectiles = new ArrayList<>();
		BlockState[] originalBlocks;
		int frame;
		boolean paused = true;
		double speed = 1.0;
		double budget;
		int firstStateDivergence = -1;
		boolean targetingDiverged;
		ReplayAnalysis analysis;
		boolean buildArena = true;
		List<ReplayTrace.Actor> initialActors;
		LiveDuel live;
		private record RenderedProjectile(ReplayTrace.Projectile sample, ItemDisplay entity) {}

		void warnFallDamageMismatch() {
			if (trace.environment.simulateFallDamage != level.getGameRules().get(GameRules.FALL_DAMAGE)) {
				tell("World fall-damage rule differs from the simulator scenario; replay and duel behavior may diverge. World rules were not changed.");
			}
		}

		Session(MinecraftServer server, ServerLevel level, UUID owner, ReplayTrace trace, Vec3 origin, Path analysisDirectory) {
			this.server = server;
			this.level = level;
			this.owner = owner;
			this.trace = trace;
			this.origin = origin;
			this.analysisDirectory = analysisDirectory;
			this.initialActors = trace.frames.getFirst().actors();
		}

		void restart() {
			reset();
			paused = false;
		}

		void reset() {
			finishAnalysis("partial", "reset_or_mode_change");
			if (live != null) live.reset();
			clearProjectiles();
			removeActors();
			frame = 0;
			budget = 0;
			paused = true;
			firstStateDivergence = -1;
			targetingDiverged = false;
			if (buildArena) rebuildEnvironment();
			if (live != null) live.spawnHuman();
			for (ReplayTrace.Actor initial : initialActors) {
				if (live != null && live.isHumanSlot(initial.id())) continue;
				if (live != null) {
					Vec3 position = origin.add(initial.position()[0], initial.position()[1], initial.position()[2]);
					BlockPos feet = BlockPos.containing(position);
					BlockPos head = BlockPos.containing(position.add(0, 1.8, 0));
					if (!level.hasChunksAt(feet.offset(-1, 0, -1), head.offset(1, 0, 1))
							|| level.isOutsideBuildHeight(feet) || level.isOutsideBuildHeight(head)
							|| !level.getWorldBorder().isWithinBounds(feet) || !level.getWorldBorder().isWithinBounds(head)) {
						throw new IllegalStateException("Live spawn must fit inside loaded chunks, world height and border.");
					}
				}
				ReplayPlayer player = new ReplayPlayer(server, level, initial, origin, absolute);
				player.setLoadout(trace.environment.loadouts.get(initial.id()));
				if (live != null) {
					player.setLiveControlled();
					BlockPos feet = BlockPos.containing(player.position());
					BlockPos head = BlockPos.containing(player.getX(), player.getY() + player.getBbHeight(), player.getZ());
					if (!level.hasChunkAt(feet) || level.isOutsideBuildHeight(feet) || level.isOutsideBuildHeight(head)
							|| !level.getWorldBorder().isWithinBounds(feet) || !level.getWorldBorder().isWithinBounds(head)
							|| !level.noCollision(player, player.getBoundingBox())) {
						throw new IllegalStateException("Live spawn must be clear, loaded and inside world height/border. Move away or use spectator mode.");
					}
				}
				actors.add(player);
				if (live != null) live.setBot(player);
				server.getPlayerList().broadcastAll(ClientboundPlayerInfoUpdatePacket.createPlayerInitializing(List.of(player)));
				level.addNewPlayer(player);
			}
			for (int index = 0; index < actors.size(); index++) {
				ReplayPlayer player = actors.get(index);
				player.applyState(initialActors.get(player.replayIndex), origin);
				if (live != null && !buildArena) player.setOnGround(!level.noCollision(player, player.getBoundingBox().move(0, -0.001, 0)));
			}
			if (live != null) live.updateTargets();
			else if (absolute) renderRecordedProjectiles(trace.frames.getFirst().projectiles());
			tell("Replay origin: " + origin.x + ", " + origin.y + ", " + origin.z);
		}

		// Replay adapter invariant: discard only vanilla arrows captured from these actors and ItemDisplays created below.
		void clearProjectiles() {
			for (Arrow arrow : ownedArrows) if (!arrow.isRemoved()) arrow.discard();
			ownedArrows.clear();
			for (var egg : ownedEggs) if (!egg.isRemoved()) egg.discard();
			ownedEggs.clear();
			for (var snowball : ownedSnowballs) if (!snowball.isRemoved()) snowball.discard();
			ownedSnowballs.clear();
			for (var pearl : ownedPearls) if (!pearl.isRemoved()) pearl.discard();
			ownedPearls.clear();
			for (var wind : ownedWind) if (!wind.isRemoved()) wind.discard();
			ownedWind.clear();
			for (RenderedProjectile rendered : renderedProjectiles) if (!rendered.entity().isRemoved()) rendered.entity().discard();
			renderedProjectiles.clear();
		}

		// Visual-only replay adapter: Display's constructor sets noPhysics (minecraft-26.3-src/net/minecraft/world/entity/Display.java:129-132),
		// and ItemDisplay renders the recorded sample (Display.java:562-599); these are not vanilla projectile simulation or damage validation.
		private void renderRecordedProjectiles(List<ReplayTrace.Projectile> projectiles) {
			for (RenderedProjectile rendered : renderedProjectiles) if (!rendered.entity().isRemoved()) rendered.entity().discard();
			renderedProjectiles.clear();
			for (ReplayTrace.Projectile sample : projectiles) {
				ItemDisplay display = new ItemDisplay(EntityTypes.ITEM_DISPLAY, level);
				ItemDisplayAccess itemAccess = (ItemDisplayAccess) (Object) display;
				itemAccess.cambium$setItemStack(new ItemStack(switch (sample.type()) {
					case "snowball" -> Items.SNOWBALL;
					case "egg" -> Items.EGG;
					case "ender_pearl" -> Items.ENDER_PEARL;
					case "wind_charge" -> Items.WIND_CHARGE;
					default -> Items.ARROW;
				}));
				itemAccess.cambium$setItemTransform(sample.in_ground() ? ItemDisplayContext.GROUND : ItemDisplayContext.FIXED);
				display.setPos(origin.x + sample.position()[0], origin.y + sample.position()[1], origin.z + sample.position()[2]);
				Vec3 velocity = new Vec3(sample.velocity()[0], sample.velocity()[1], sample.velocity()[2]);
				if (velocity.lengthSqr() > 0) {
					display.setYRot((float) (Math.atan2(velocity.x, velocity.z) * 180.0 / Math.PI));
					display.setXRot((float) (Math.atan2(velocity.y, velocity.horizontalDistance()) * 180.0 / Math.PI));
				}
				display.setGlowingTag(true);
				int ownerColor = sample.owner() == 0 ? sample.critical() ? 0x99F0D4 : 0x38A98A : sample.critical() ? 0xFFD28A : 0xC98221;
				((DisplayAccess) (Object) display).cambium$setGlowColorOverride(ownerColor);
				display.setCustomName(Component.literal("Replay " + sample.type() + " owner " + sample.owner() + ", age " + sample.age()
						+ (sample.left_owner() ? ", left owner" : "") + (sample.in_ground() ? ", in ground " + sample.in_ground_ticks() : "")));
				display.setCustomNameVisible(false);
				if (level.addFreshEntity(display)) renderedProjectiles.add(new RenderedProjectile(sample, display));
			}
		}

		BlockPos environmentPosition(int index) {
			var environment = trace.environment;
			return BlockPos.containing(origin).offset(environment.min[0] + index % environment.size[0],
					environment.min[1] + index / environment.size[0] % environment.size[1],
					environment.min[2] + index / (environment.size[0] * environment.size[1]));
		}

		void rebuildEnvironment() {
			if (originalBlocks == null) {
				BlockState[] snapshot = new BlockState[trace.environment.cells.length];
				for (int index = 0; index < snapshot.length; index++) {
					BlockPos position = environmentPosition(index);
					if (level.isOutsideBuildHeight(position) || !level.getWorldBorder().isWithinBounds(position)
							|| !level.hasChunkAt(position)) {
						throw new IllegalStateException("Arena must fit inside loaded chunks, build height and world border.");
					}
					if (level.getBlockEntity(position) != null) {
						throw new IllegalStateException("Arena contains a block entity/container; choose an empty test area.");
					}
					snapshot[index] = level.getBlockState(position);
				}
				originalBlocks = snapshot;
			}
			for (int index = 0; index < trace.environment.cells.length; index++) {
				BlockState state = trace.environment.cells[index] == 0 ? Blocks.AIR.defaultBlockState() : Blocks.STONE.defaultBlockState();
				level.setBlock(environmentPosition(index), state, Block.UPDATE_CLIENTS | Block.UPDATE_KNOWN_SHAPE);
			}
		}

		void tick() {
			ServerPlayer viewer = server.getPlayerList().getPlayer(owner);
			if (viewer == null || viewer.level() != level || server.isPublished()
					|| actors.stream().anyMatch(Entity::isRemoved)) {
				close();
				active = null;
				return;
			}
			if (live != null) {
				live.tick(viewer);
				return;
			}
			if (!paused) {
				budget += speed;
				while (budget >= 1 && !paused) {
					budget--;
					advance();
				}
			}
			ReplayTrace.Frame sample = trace.frames.get(frame);
			var lines = new ArrayList<String>();
			lines.add((absolute ? "Absolute" : "Input") + (paused ? " / Paused" : " / Playing"));
			lines.add("Tick " + frame + " / " + (trace.frames.size() - 1) + " | " + speed + "x");
			for (int index = 0; index < 2; index++) {
				ReplayTrace.Actor reference = sample.actors().get(index);
				ReplayPlayer player = actors.get(index);
				lines.add((index == 0 ? "A: " : "B: ") + buttons(reference.controls()));
				lines.add(String.format(Locale.ROOT, "HP %.1f", player.getHealth()));
				if (!absolute) {
					Vec3 expected = origin.add(reference.position()[0], reference.position()[1], reference.position()[2]);
					lines.add(String.format(Locale.ROOT, "Position error %.4f", player.position().distanceTo(expected)));
				} else if (reference.outcome().attempted()) {
					lines.add(reference.outcome().accepted() ? "Hit accepted" : "Hit rejected");
				}
			}
			DuelHud.publishMatch(new DuelHud.Match("REPLAY", paused ? "Paused" : "Playing", (absolute ? "Absolute" : "Input") + " " + speed + "x",
					frame, trace.frames.size() - 1, 0, 0, 0, 0, 0),
					DuelHud.fighter(actors.get(0), "Replay A", buttons(sample.actors().get(0).controls())),
					DuelHud.fighter(actors.get(1), "Replay B", buttons(sample.actors().get(1).controls())), lines);
		}

		void advance() {
			if (frame + 1 >= trace.frames.size()) {
				paused = true;
				return;
			}
			ReplayTrace.Frame sample = trace.frames.get(++frame);
			if (absolute) {
				for (int index = 0; index < 2; index++) {
					actors.get(index).applyState(sample.actors().get(index), origin);
					if (sample.actors().get(index).outcome().attempted() && actors.get(index).isAlive()) {
						actors.get(index).swingReplay();
					}
				}
				renderRecordedProjectiles(sample.projectiles());
			} else {
				JsonObject phases = analysis == null ? null : new JsonObject();
				if (phases != null) phases.add("before_prepare", analysisActors());
				for (int index = 0; index < 2; index++) {
					actors.get(index).prepare(sample.actors().get(index).controls());
				}
				HitResult[] picks = actors.stream().map(ReplayPlayer::pickAttack).toArray(HitResult[]::new);
				JsonArray outcomes = new JsonArray();
				JsonArray attackPhases = new JsonArray();
				if (phases != null) {
					phases.add("before_attacks", analysisActors());
					JsonArray targeting = new JsonArray();
					for (int index = 0; index < 2; index++) {
						targeting.add(actors.get(index).analysisPick(picks[index], actors.get(1 - index), origin));
					}
					phases.add("targeting", targeting);
				}
				for (int index = 0; index < 2; index++) {
					ReplayPlayer player = actors.get(index);
					checkTarget(sample, index, picks[index]);
					boolean executed = picks[index] != null && player.isAlive();
					if (executed) {
						player.swingReplay();
						if (picks[index] instanceof EntityHitResult hit) {
							player.attack(hit.getEntity());
						} else {
							player.miss(picks[index]);
						}
						actors.forEach(ReplayPlayer::acceptMotion);
					}
					if (phases != null) {
						outcomes.add(player.analysisOutcome(picks[index], executed));
						if (executed) {
							JsonObject attack = new JsonObject();
							attack.addProperty("actor", index);
							attack.add("after_attack_and_motion", analysisActors());
							attackPhases.add(attack);
						}
					}
				}
				if (phases != null) {
					phases.add("ordered_attacks", attackPhases);
					phases.add("after_combat", analysisActors());
				}
				actors.forEach(ReplayPlayer::advanceInput);
				checkState(sample);
				if (phases != null) {
					JsonArray actual = analysisActors();
					for (int index = 0; index < 2; index++) actual.get(index).getAsJsonObject().add("outcome", outcomes.get(index));
					try {
						analysis.record(sample.tick(), ReplayAnalysis.JSON.toJsonTree(sample.actors()).getAsJsonArray(), actual, phases);
					} catch (IOException error) {
						paused = true;
						finishAnalysis("failed", "report_write_error");
						tell("Analysis write failed: " + error.getMessage());
						CambiumFabric.LOGGER.error("Analysis write failed", error);
					}
				}
			}
			if (frame + 1 >= trace.frames.size()) {
				paused = true;
				finishAnalysis("complete", "end_of_replay");
			}
		}

		JsonArray analysisActors() {
			JsonArray result = new JsonArray();
			for (ReplayPlayer player : actors) result.add(player.analysisState(origin));
			return result;
		}

		void startAnalysis() {
			absolute = false;
			reset();
			JsonObject metadata = new JsonObject();
			metadata.addProperty("minecraft", "26.3");
			metadata.addProperty("source_replay", trace.path.toAbsolutePath().toString());
			metadata.add("source_header", trace.header);
			metadata.addProperty("reference_backend", trace.backend);
			metadata.addProperty("reference_ticks", trace.frames.size() - 1);
			metadata.addProperty("comparison", "once-seeded vanilla ServerPlayer input adapter versus recorded simulator states");
			metadata.addProperty("dimension", level.dimension().toString());
			metadata.addProperty("difficulty", level.getDifficulty().name());
			metadata.addProperty("pvp_allowed", level.isPvpAllowed());
			metadata.addProperty("natural_regeneration", level.getGameRules().get(GameRules.NATURAL_HEALTH_REGENERATION));
			metadata.addProperty("actor_natural_regeneration", false);
			metadata.addProperty("actor_food_simulated", false);
			metadata.addProperty("speed_at_start", speed);
			metadata.add("origin", ReplayPlayer.vector(origin));
			ServerPlayer viewer = server.getPlayerList().getPlayer(owner);
			metadata.addProperty("observer_spectator", viewer != null && viewer.isSpectator());
			if (viewer != null) metadata.add("observer_position", ReplayPlayer.vector(viewer.position().subtract(origin)));
			metadata.add("expected_initial", ReplayAnalysis.JSON.toJsonTree(trace.frames.getFirst().actors()));
			metadata.add("actual_initial", analysisActors());
			metadata.addProperty("limitations", "Not a full LocalPlayer/network or RNG reconstruction; world entities/rules can affect results. "
					+ "Critical/sprint/sweep flags are reference-only; actual outcome comparisons use target, attempted, accepted, damage and health_lost. "
					+ "Positions are replay-local; phase snapshots are vanilla observations, not isolated simulator phase comparisons. Frame zero is unscored.");
			try {
				analysis = new ReplayAnalysis(analysisDirectory, metadata);
				actors.forEach(player -> player.setAnalysisEnabled(true));
				paused = false;
				tell("Analysis started from tick 0 in input mode (reference: " + trace.backend + "). Finishes automatically at replay end.");
				tellFile("Detailed report: ", analysis.detailsPath);
				if (viewer != null && !viewer.isSpectator()) tell("Use spectator mode or stay outside the arena to avoid affecting player contact.");
				tell("Replay actors use fixed food and no natural regeneration. World rules have not been changed.");
				if (!level.isPvpAllowed()) {
					tell("Comparison warning: PvP must be enabled for player damage.");
				}
			} catch (IOException error) {
				tell("Could not create analysis: " + error.getMessage());
				CambiumFabric.LOGGER.error("Could not create replay analysis", error);
			}
		}

		void finishAnalysis(String status, String reason) {
			if (analysis == null) return;
			ReplayAnalysis report = analysis;
			analysis = null;
			actors.forEach(player -> player.setAnalysisEnabled(false));
			try {
				report.finish(status, reason);
				tell("Analysis " + status + ". First difference: "
						+ (report.firstDivergentTick() < 0 ? "none in compared ticks" : "tick " + report.firstDivergentTick()) + ".");
				tellFile("Summary: ", report.summaryPath);
				tellFile("Tick details: ", report.detailsPath);
			} catch (IOException error) {
				tell("Could not finalize analysis: " + error.getMessage() + ". Partial details: " + report.detailsPath);
				CambiumFabric.LOGGER.error("Could not finalize replay analysis {}", report.detailsPath, error);
			}
		}

		void checkTarget(ReplayTrace.Frame sample, int index, HitResult pick) {
			ReplayPlayer player = actors.get(index);
			int actual = player.isAlive() && pick instanceof EntityHitResult hit
					&& hit.getEntity() instanceof ReplayPlayer target ? target.replayIndex : -1;
			int expected = sample.actors().get(index).outcome().target();
			if (targetingDiverged || actual == expected) return;
			targetingDiverged = true;
			ReplayPlayer opponent = actors.get(1 - index);
			Vec3 eye = player.getEyePosition();
			Vec3 look = player.getHeadLookAngle();
			double reach = player.getAttributeValue(net.minecraft.world.entity.ai.attributes.Attributes.ENTITY_INTERACTION_RANGE)
					+ Math.max(0.0, player.getKnownMovement().dot(look));
			CambiumFabric.LOGGER.warn(
					"Replay targeting divergence: file={}, tick={}, actor={}, expectedTarget={}, actualTarget={}, pick={}, "
							+ "firstStateDivergence={}, eye={}, look={}, velocity={}, reach={}, targetBox={}, eyeInsideTarget={}, eyeToBox={}",
					trace.path, sample.tick(), index, expected, actual, pick == null ? "NO_ATTACK" : pick.getType(),
					firstStateDivergence, eye.subtract(origin), look, player.getKnownMovement(), reach,
					opponent.getBoundingBox().move(origin.scale(-1)), opponent.getBoundingBox().contains(eye),
					Math.sqrt(opponent.getBoundingBox().distanceToSqr(eye)));
			tell("First targeting difference: tick " + sample.tick() + ", actor " + index
					+ ", expected " + expected + ", actual " + actual + ". Details in latest.log.");
		}

		void checkState(ReplayTrace.Frame sample) {
			if (firstStateDivergence >= 0) return;
			for (int index = 0; index < actors.size(); index++) {
				ReplayPlayer player = actors.get(index);
				ReplayTrace.Actor expected = sample.actors().get(index);
				Vec3 position = origin.add(expected.position()[0], expected.position()[1], expected.position()[2]);
				Vec3 velocity = new Vec3(expected.velocity()[0], expected.velocity()[1], expected.velocity()[2]);
				double positionError = player.position().distanceTo(position);
				double velocityError = player.getDeltaMovement().distanceTo(velocity);
				if (positionError <= 1.0E-6 && velocityError <= 1.0E-6
						&& Math.abs(player.getHealth() - expected.health()) <= 1.0E-5
						&& Math.abs(player.getAbsorptionAmount() - expected.absorption()) <= 1.0E-5) continue;
				firstStateDivergence = sample.tick();
				CambiumFabric.LOGGER.warn(
						"Replay state divergence: file={}, tick={}, actor={}, positionError={}, velocityError={}, "
								+ "actualPosition={}, expectedPosition={}, actualVelocity={}, expectedVelocity={}, "
								+ "actualHealth={}, expectedHealth={}, actualAbsorption={}, expectedAbsorption={}",
						trace.path, sample.tick(), index, positionError, velocityError,
						player.position().subtract(origin), position.subtract(origin), player.getDeltaMovement(), velocity,
						player.getHealth(), expected.health(), player.getAbsorptionAmount(), expected.absorption());
				tell("First state difference: tick " + sample.tick() + ", actor " + index + ". Details in latest.log.");
				return;
			}
		}

		void removeActors() {
			if (!actors.isEmpty()) {
				var ids = actors.stream().map(Entity::getUUID).toList();
				for (ReplayPlayer player : actors) {
					level.removePlayerImmediately(player, Entity.RemovalReason.DISCARDED);
					player.getTextFilter().leave();
				}
				server.getPlayerList().broadcastAll(new ClientboundPlayerInfoRemovePacket(ids));
				actors.clear();
			}
		}

		void close() {
			DuelHud.clear();
			finishAnalysis("partial", "session_closed");
			clearProjectiles();
			if (live != null) live.close();
			removeActors();
			if (originalBlocks != null) {
				for (int index = 0; index < originalBlocks.length; index++) {
					level.setBlock(environmentPosition(index), originalBlocks[index], Block.UPDATE_CLIENTS | Block.UPDATE_KNOWN_SHAPE);
				}
				originalBlocks = null;
			}
		}

		void tell(String text) {
			ServerPlayer viewer = server.getPlayerList().getPlayer(owner);
			if (viewer != null) {
				viewer.sendSystemMessage(Component.literal(text));
			}
		}

		void tellFile(String label, Path path) {
			ServerPlayer viewer = server.getPlayerList().getPlayer(owner);
			if (viewer == null) return;
			Component file = Component.literal(path.toAbsolutePath().toString()).withStyle(style -> style
					.withUnderlined(true)
					.withClickEvent(new ClickEvent.CopyToClipboard("\"" + path.toAbsolutePath() + "\""))
					.withHoverEvent(new HoverEvent.ShowText(Component.literal("Copy quoted report path"))));
			viewer.sendSystemMessage(Component.literal(label).append(file));
		}
	}

	static String buttons(ReplayTrace.Controls input) {
		StringBuilder text = new StringBuilder();
		if (input.forward()) text.append("W ");
		if (input.backward()) text.append("S ");
		if (input.left()) text.append("A ");
		if (input.right()) text.append("D ");
		if (input.jump()) text.append("JUMP ");
		if (input.sprint()) text.append("SPRINT ");
		if (input.sneak()) text.append("SNEAK ");
		if (input.attack()) text.append("ATTACK ");
		return text.isEmpty() ? "-" : text.toString().stripTrailing();
	}
}