package net.cambium;

import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.UUID;
import net.cambium.mixin.LivingEntityRecordingAccessor;
import net.cambium.mixin.LocalPlayerRecordingAccessor;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.ChatFormatting;
import net.minecraft.client.Minecraft;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.network.chat.ClickEvent;
import net.minecraft.network.chat.Component;
import net.minecraft.network.chat.HoverEvent;
import net.minecraft.world.level.Level;
import net.minecraft.world.phys.Vec3;

public final class MovementRecorder {

	private static final DateTimeFormatter FILE_TIME = DateTimeFormatter
			.ofPattern("uuuu-MM-dd_HH-mm-ss-SSS").withZone(ZoneOffset.UTC);
	private static MovementCsv recording;
	private static Path path;
	private static LocalPlayer recordedPlayer;
	private static Level recordedLevel;
	private static MovementCsv.State before;
	private static MovementCsv.State previous;
	private static int input;
	private static boolean inputCaptured;

	private MovementRecorder() {
	}

	public static void toggle(Minecraft client) {
		if (recording != null) {
			stop("Recording stopped");
			return;
		}
		LocalPlayer player = client.player;
		if (player == null || client.level == null) {
			message("Join a world before recording", ChatFormatting.YELLOW);
			return;
		}
		if (player.input.keyPresses.jump() || snapshot(player).jumpDelay() != 0) {
			message("Release jump and wait for the jump cooldown before starting", ChatFormatting.YELLOW);
			return;
		}
		MovementCsv opened = null;
		try {
			Path directory = FabricLoader.getInstance().getGameDir().resolve("movement-recordings");
			Files.createDirectories(directory);
			Instant now = Instant.now();
			Path destination = directory.resolve("movement-" + FILE_TIME.format(now) + "-" + UUID.randomUUID() + ".csv");
			opened = new MovementCsv(destination);
			JsonObject metadata = new JsonObject();
			metadata.addProperty("schema_version", 2);
			metadata.addProperty("minecraft_version", FabricLoader.getInstance().getModContainer("minecraft")
					.orElseThrow().getMetadata().getVersion().getFriendlyString());
			metadata.addProperty("created_utc", now.toString());
			metadata.addProperty("dimension", player.level().dimension().toString());
			metadata.addProperty("phase", "before LocalPlayer.aiStep through end of LocalPlayer.tick (including pose update)");
			metadata.addProperty("inputs", "current effective keyPresses and absolute orientation before movement");
			metadata.addProperty("initial_row", "tick 0 is the before-state; subsequent rows contain resulting state");
			metadata.addProperty("discontinuity_policy", "stop before a step whose position/velocity/ground/jump-delay differs from prior post-state");
			metadata.addProperty("world", "not captured; comparison requires matching terrain and default movement attributes");
			metadata.addProperty("pose_encoding", "0=standing, 1=crouching, 2=crawling (dry swimming pose)");
			metadata.addProperty("movement_state", "first row initializes bookkeeping; later rows are reference only, not corrections");
			Files.writeString(Path.of(destination + ".json"), new GsonBuilder().setPrettyPrinting().create().toJson(metadata),
					StandardCharsets.UTF_8, StandardOpenOption.CREATE_NEW);
			recording = opened;
			path = destination;
			recordedPlayer = player;
			recordedLevel = player.level();
			before = null;
			previous = null;
			inputCaptured = false;
			message("Movement recording started", ChatFormatting.GREEN);
		} catch (IOException | RuntimeException error) {
			if (opened != null) {
				try {
					opened.close();
				} catch (IOException closeError) {
					error.addSuppressed(closeError);
				}
			}
			CambiumFabric.LOGGER.error("Could not start movement recording", error);
			message("Could not start recording; see the game log", ChatFormatting.RED);
		}
	}

	public static void checkSession(Minecraft client) {
		if (recording != null && (client.player != recordedPlayer || client.level != recordedLevel)) {
			stop("Player or world changed");
		}
	}

	public static void beforeMovement(LocalPlayer player) {
		if (recording == null) {
			return;
		}
		if (player != recordedPlayer || player.level() != recordedLevel) {
			stop("Player or world changed");
			return;
		}
		MovementCsv.State state = snapshot(player);
		if (state.flying() || state.pose() < 0
				|| player.isInWater() || player.isInLava() || player.isPassenger() || player.isUsingItem()
				|| player.onClimbable() || !player.getActiveEffects().isEmpty()) {
			stop("Movement mode is outside the current simulator's supported land-movement slice");
			return;
		}
		if (previous != null && !state.continues(previous)) {
			stop("State changed between movement steps (possible correction, teleport or external force)");
			return;
		}
		before = state;
		inputCaptured = false;
	}

	public static void captureInput(LocalPlayer player) {
		if (recording == null || before == null || player != recordedPlayer) {
			return;
		}
		var keys = player.input.keyPresses;
		input = (keys.forward() ? 1 : 0) | (keys.backward() ? 2 : 0)
				| (keys.left() ? 4 : 0) | (keys.right() ? 8 : 0) | (keys.jump() ? 16 : 0)
				| (keys.shift() ? 32 : 0) | (keys.sprint() ? 64 : 0);
		if (previous == null && (before.jumpDelay() != 0 || keys.jump())) {
			stop("Start with jump released and jump cooldown zero");
			return;
		}
		inputCaptured = true;
		if (previous == null) {
			try {
				recording.append(input, before, before);
			} catch (IOException error) {
				writeFailure(error);
			}
		}
	}

	public static void afterMovement(LocalPlayer player) {
		if (recording == null || before == null || !inputCaptured || player != recordedPlayer) {
			return;
		}
		try {
			MovementCsv.State after = snapshot(player);
			recording.append(input, before, after);
			previous = after;
			before = null;
			inputCaptured = false;
		} catch (IOException error) {
			writeFailure(error);
		}
	}

	private static MovementCsv.State snapshot(LocalPlayer player) {
		Vec3 velocity = player.getDeltaMovement();
		int pose = switch (player.getPose()) {
			case STANDING -> 0;
			case CROUCHING -> 1;
			case SWIMMING -> 2;
			default -> -1;
		};
		return new MovementCsv.State(player.getX(), player.getY(), player.getZ(), velocity.x, velocity.y, velocity.z,
				player.getYRot(), player.getXRot(), player.onGround(), player.horizontalCollision, player.verticalCollision,
				((LivingEntityRecordingAccessor) player).cambium$getJumpDelay(), player.isSprinting(),
				player.isCrouching(), player.getAbilities().flying, pose, player.input.hasForwardImpulse(),
				player.input.keyPresses.shift(), ((LocalPlayerRecordingAccessor) player).cambium$getSprintTriggerTime(),
				Minecraft.getInstance().options.sprintWindow().get(), player.getFoodData().getFoodLevel(),
				player.getAbilities().mayfly, player.fallDistance, player.minorHorizontalCollision);
	}

	public static void stop(String reason) {
		if (recording == null) {
			return;
		}
		MovementCsv completed = recording;
		Path savedPath = path;
		recording = null;
		before = null;
		previous = null;
		inputCaptured = false;
		recordedPlayer = null;
		recordedLevel = null;
		try {
			completed.close();
			long steps = Math.max(0, completed.rows() - 1);
			CambiumFabric.LOGGER.info("{}; saved {} movement steps to {}", reason, steps, savedPath.toAbsolutePath());
			Component file = Component.literal(savedPath.toAbsolutePath().toString()).withStyle(style -> style
					.withColor(ChatFormatting.AQUA).withUnderlined(true)
					.withClickEvent(new ClickEvent.CopyToClipboard("\"" + savedPath.toAbsolutePath() + "\""))
					.withHoverEvent(new HoverEvent.ShowText(Component.literal("Copy quoted CSV path"))));
			Minecraft.getInstance().gui.hud.getChat().addClientSystemMessage(Component.literal("[Cambium] " + reason
					+ ". Saved " + steps + " movement steps: ").append(file));
			if (steps == 0) {
				message("Capture too short to compare; record at least one movement step", ChatFormatting.YELLOW);
			}
		} catch (IOException error) {
			CambiumFabric.LOGGER.error("Could not finish recording {}", savedPath, error);
			message("Could not finish CSV; see the game log", ChatFormatting.RED);
		}
	}

	private static void writeFailure(IOException error) {
		CambiumFabric.LOGGER.error("Movement CSV write failed", error);
		message("CSV write failed; recording may be incomplete", ChatFormatting.RED);
		stop("Recording stopped after an I/O error");
	}

	private static void message(String text, ChatFormatting color) {
		Minecraft.getInstance().gui.hud.getChat().addClientSystemMessage(Component.literal("[Cambium] " + text).withStyle(color));
	}
}