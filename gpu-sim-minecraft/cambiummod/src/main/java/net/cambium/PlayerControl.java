package net.cambium;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.mojang.blaze3d.platform.InputConstants;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import net.cambium.mixin.PlayerControlGameModeAccess;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientLifecycleEvents;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keymapping.v1.KeyMappingHelper;
import net.fabricmc.fabric.api.client.networking.v1.ClientPlayConnectionEvents;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.Minecraft;
import net.minecraft.client.player.ClientInput;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.network.chat.Component;
import net.minecraft.world.entity.player.Input;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.Items;
import net.minecraft.world.phys.Vec2;
import net.minecraft.world.phys.Vec3;

final class PlayerControl {

	private static volatile long generation;
	private static Process process;
	private static LivePolicy policy;
	private static CompletableFuture<JsonObject> ready;
	private static CompletableFuture<JsonObject> pending;
	private static Object world;
	private static UUID owner;
	private static UUID targetId;
	private static String session;
	private static ClientInput previousInput;
	private static PolicyInput input;
	private static Vec3 previousTarget;
	private static int tick;
	private static int requestTick;
	private static long requestedAt;
	private static boolean useAction;
	private static String status = "Stopped";

	static void initialize() {
		KeyMapping stop = KeyMappingHelper.registerKeyMapping(new KeyMapping("key.cambiumfabric.stop_control", InputConstants.Type.KEYBOARD,
				InputConstants.KEY_F10, KeyMapping.Category.register(CambiumFabric.id("interface"))));
		ClientTickEvents.START_CLIENT_TICK.register(client -> {
			while (stop.consumeClick()) if (active()) stop(client, "Stopped by F10");
			if (active()) tick(client);
		});
		ClientPlayConnectionEvents.DISCONNECT.register((handler, client) -> stop(client, "Disconnected"));
		ClientLifecycleEvents.CLIENT_STOPPING.register(client -> stop(client, "Client stopping"));
	}

	static boolean active() { return world != null; }
	static String status() { return status; }

	static void start(Minecraft client, Path checkpoint, UUID target) {
		if (active()) throw new IllegalStateException("Stop the current player controller first");
		if (client.level == null || client.player == null || client.player.getUUID().equals(target)) throw new IllegalArgumentException("Choose another player as target");
		long selected = ++generation;
		world = client.level;
		owner = client.player.getUUID();
		targetId = target;
		session = UUID.randomUUID().toString();
		status = "Starting local inference";
		Path game = client.gameDirectory.toPath().toAbsolutePath();
		Thread.ofVirtual().name("cambium-player-policy").start(() -> {
			Process started = null;
			try {
				Files.createDirectories(game.resolve("logs"));
				ProcessBuilder builder;
				if (BundledInference.available()) builder = BundledInference.helper(BundledInference.prepare(game), game, "local-policy", checkpoint.toString());
				else {
					Path configPath = game.resolve("config/cambium-rosie.json");
					if (!Files.isRegularFile(configPath)) throw new IOException("No bundled inference runtime for this platform");
					JsonObject config = JsonParser.parseString(Files.readString(configPath)).getAsJsonObject();
					builder = RosieLauncher.helper(config, game, "local-policy", "latest", checkpoint.toString());
				}
				started = builder.start();
				synchronized (PlayerControl.class) {
					if (generation != selected) { terminate(started); return; }
					process = started;
				}
				try (var output = new BufferedReader(new InputStreamReader(started.getInputStream(), StandardCharsets.UTF_8))) {
					String line;
					while ((line = output.readLine()) != null) {
						JsonObject event = JsonParser.parseString(line).getAsJsonObject();
						client.execute(() -> {
							if (generation != selected) return;
							String name = event.get("event").getAsString();
							if (name.equals("policy")) {
								policy = new LivePolicy(event.get("port").getAsInt());
								ready = policy.info();
								status = "Checking policy contract";
							} else if (name.equals("error")) stop(client, event.get("message").getAsString());
						});
					}
				}
				int exit = started.waitFor();
				client.execute(() -> { if (generation == selected) stop(client, "Inference process stopped (" + exit + ")"); });
			} catch (IOException | RuntimeException | InterruptedException error) {
				if (started != null) terminate(started);
				client.execute(() -> { if (generation == selected) stop(client, "Inference failed: " + error.getMessage()); });
			}
		});
	}

	private static void tick(Minecraft client) {
		try {
			LocalPlayer player = client.player;
			if (client.level != world || player == null || !owner.equals(player.getUUID()) || !player.isAlive()) throw new IllegalStateException("Player or world unavailable");
			if (client.gui.screen() != null || client.gui.overlay() != null || !client.isWindowActive()) throw new IllegalStateException("Screen opened or game lost focus");
			Player target = client.level.players().stream().filter(candidate -> candidate.getUUID().equals(targetId)).findFirst().orElse(null);
			if (target == null || !target.isAlive()) throw new IllegalStateException("Target unavailable");
			Vec3 position = target.position();
			Vec3 velocity = previousTarget == null ? null : position.subtract(previousTarget);
			previousTarget = position;
			if (ready != null && ready.isDone()) {
				ready.join();
				ready = null;
				policy.requirePlayerSupport(ClientObservation.FEATURES, ClientObservation.ACTIONS);
				useAction = policy.acts("use_item");
				if (useAction && (!player.getMainHandItem().is(Items.BOW) || player.getOffhandItem().is(Items.BOW))) {
					throw new IllegalStateException("Equip a main-hand bow and no offhand bow before starting");
				}
				if (useAction && player.isUsingItem()) throw new IllegalStateException("Stop using your item before starting");
				previousInput = player.input;
				input = new PolicyInput();
				player.input = input;
				if (useAction) client.options.keyUse.setDown(false);
				status = "Running (remote movement and arrows estimated); F10 stops";
				message(client, status);
			}
			if (input == null || velocity == null) return;
			if (useAction && (!player.getMainHandItem().is(Items.BOW) || player.getOffhandItem().is(Items.BOW))) throw new IllegalStateException("Bow equipment changed");
			if (pending != null) {
				if (expired(tick, requestTick, System.nanoTime() - requestedAt)) throw new IllegalStateException("Inference timed out");
				if (pending.isDone()) {
					ReplayTrace.Controls controls = policy.mapOneAction(pending.join(), session, requestTick);
					pending = null;
					input.controls = controls;
					player.setYRot(player.getYRot() + controls.yaw_delta());
					player.setXRot(Math.clamp(player.getXRot() + controls.pitch_delta(), -90.0F, 90.0F));
					if (useAction) client.options.keyUse.setDown(controls.use_item());
				}
			}
			if (pending == null) {
				requestTick = tick;
				requestedAt = System.nanoTime();
				pending = policy.actOne(session, requestTick, ClientObservation.sample(policy, player, target, velocity));
			}
			tick++;
		} catch (RuntimeException error) {
			Throwable cause = error.getCause() == null ? error : error.getCause();
			stop(client, cause.getMessage() == null ? cause.toString() : cause.getMessage());
		}
	}

	static boolean expired(int currentTick, int requestedTick, long elapsedNanos) {
		return currentTick - requestedTick > 5 || elapsedNanos > 250_000_000L;
	}

	static void stop(Minecraft client, String reason) {
		if (!active()) return;
		generation++;
		if (pending != null) pending.cancel(true);
		if (ready != null) ready.cancel(true);
		if (client.player != null && client.level == world && owner.equals(client.player.getUUID()) && input != null) {
			LocalPlayer player = client.player;
			if (useAction && player.isUsingItem() && player.getUseItem().is(Items.BOW)
					&& player.getUsedItemHand() == net.minecraft.world.InteractionHand.MAIN_HAND && client.gameMode != null) {
				var gameMode = (PlayerControlGameModeAccess) client.gameMode;
				player.getInventory().setSelectedSlot(Math.floorMod(gameMode.cambium$carriedIndex() + 1, 9));
				gameMode.cambium$sendCarriedItem();
				player.stopUsingItem();
			}
			if (useAction) client.options.keyUse.setDown(false);
			if (player.input == input) player.input = previousInput;
		}
		if (policy != null) policy.close();
		synchronized (PlayerControl.class) { if (process != null) terminate(process); process = null; }
		world = null; owner = null; targetId = null; session = null; policy = null; pending = null; ready = null;
		previousInput = null; input = null; previousTarget = null; tick = 0; useAction = false;
		status = "Stopped: " + reason;
		if (client.player != null) message(client, status);
	}

	private static void terminate(Process worker) {
		worker.descendants().forEach(ProcessHandle::destroyForcibly);
		worker.destroyForcibly();
	}

	private static void message(Minecraft client, String text) { client.gui.hud.getChat().addClientSystemMessage(Component.literal("Player control: " + text)); }

	private static final class PolicyInput extends ClientInput {
		private ReplayTrace.Controls controls = new ReplayTrace.Controls(false, false, false, false, false, false, false, false, 0, 0);
		@Override public void tick() {
			keyPresses = new Input(controls.forward(), controls.backward(), controls.left(), controls.right(), controls.jump(), controls.sneak(), controls.sprint());
			float forward = (controls.forward() ? 1 : 0) - (controls.backward() ? 1 : 0);
			float left = (controls.left() ? 1 : 0) - (controls.right() ? 1 : 0);
			moveVector = new Vec2(left, forward).normalized();
		}
	}

	private PlayerControl() {}
}