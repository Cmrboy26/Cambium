package net.cambium;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.mojang.brigadier.arguments.StringArgumentType;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.ArrayList;
import java.util.UUID;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.fabric.api.client.command.v2.ClientCommands;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientLifecycleEvents;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.networking.v1.ClientPlayConnectionEvents;
import net.minecraft.client.Minecraft;
import net.minecraft.network.chat.ClickEvent;
import net.minecraft.network.chat.Component;

final class RosieLauncher {

	private static Process worker;
	private static volatile long generation;
	private static boolean busy;
	private static boolean serving;
	private static Object world;
	private static String operation = "idle";
	private static String lastMessage = "Local library ready";
	private static List<JsonObject> remoteRuns = List.of();
	private static boolean arenaMode = true;

	static void initialize() {
		ClientCommandRegistrationCallback.EVENT.register((dispatcher, registryAccess) -> {
			var root = ClientCommands.literal("simrosie").executes(context -> message(context.getSource().getClient(),
					"/simrosie runs, replays [run], replay [run] [file], duel [run], sync [run], status, stop"));
			root.then(ClientCommands.literal("runs").executes(context -> start(context.getSource().getClient(), "runs", "latest", "latest")));
			for (String action : List.of("replays", "replay", "duel", "sync")) {
				var command = ClientCommands.literal(action)
						.executes(context -> start(context.getSource().getClient(), action, "latest", "latest"));
				var run = ClientCommands.argument("run", StringArgumentType.word())
						.executes(context -> start(context.getSource().getClient(), action,
								StringArgumentType.getString(context, "run"), "latest"));
				if (action.equals("replay")) {
					run.then(ClientCommands.argument("file", StringArgumentType.word())
							.executes(context -> start(context.getSource().getClient(), action,
									StringArgumentType.getString(context, "run"), StringArgumentType.getString(context, "file"))));
				}
				root.then(command.then(run));
			}
			root.then(ClientCommands.literal("status").executes(context -> message(context.getSource().getClient(), "Rosie: " + operation)));
			root.then(ClientCommands.literal("local").executes(context -> start(context.getSource().getClient(), "local", "latest", "latest")));
			root.then(ClientCommands.literal("stop").executes(context -> {
				Minecraft client = context.getSource().getClient();
				boolean stopDuel = serving;
				cancel();
				if (stopDuel) LiveDuel.stop(client);
				return message(client, "Rosie launcher stopped; cached files were kept.");
			}));
			dispatcher.register(root);
		});
		ClientPlayConnectionEvents.DISCONNECT.register((handler, client) -> cancel());
		ClientLifecycleEvents.CLIENT_STOPPING.register(client -> cancel());
		ClientTickEvents.END_CLIENT_TICK.register(client -> {
			if (busy && (client.level != world || client.getSingleplayerServer() == null || client.getSingleplayerServer().isPublished())) cancel();
		});
	}

	static boolean isBusy() { return busy; }
	static boolean isServing() { return serving; }
	static String status() { return busy ? operation : lastMessage; }
	static List<JsonObject> remoteRuns() { return remoteRuns; }

	static void stopOwned(Minecraft client) {
		cancel();
		LiveDuel.stop(client);
		lastMessage = "Stopped; local files kept";
	}

	static int start(Minecraft client, String action, String run, String replay) {
		var server = client.getSingleplayerServer();
		if (server == null || client.player == null || server.isPublished()) {
			return message(client, "Rosie commands require a private singleplayer world.");
		}
		if (busy) return message(client, "Rosie is busy: " + operation + ". Use /simrosie stop before switching.");
		Path game = client.gameDirectory.toPath().toAbsolutePath();
		Path configuration = game.resolve("config/cambium-rosie.json");
		if (!Files.isRegularFile(configuration)) {
			return message(client, "Configure Rosie once locally: scripts/rosie-play.ps1 -Setup -MinecraftDirectory <this game folder>.");
		}
		long selected = ++generation;
		UUID owner = client.player.getUUID();
		Object level = client.level;
		world = level;
		busy = true;
		arenaMode = !action.equals("local-world");
		operation = action + " " + run;
		message(client, "Rosie: " + operation + "...");
		Thread.ofVirtual().name("cambium-rosie").start(() -> {
			Process process = null;
			boolean reported = false;
			try {
				JsonObject config = JsonParser.parseString(Files.readString(configuration)).getAsJsonObject();
				Path logs = game.resolve("logs");
				Files.createDirectories(logs);
				process = helper(config, game, action, run, replay).start();
				synchronized (RosieLauncher.class) {
					if (generation != selected) {
						terminate(process);
						return;
					}
					worker = process;
				}
				try (var output = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
					String line;
					while ((line = output.readLine()) != null) {
						JsonObject event = JsonParser.parseString(line).getAsJsonObject();
						if (event.get("event").getAsString().equals("error")) reported = true;
						client.execute(() -> {
							if (generation == selected && client.getSingleplayerServer() == server && !server.isPublished()
									&& client.level == level && client.player != null && client.player.getUUID().equals(owner)) handle(client, event);
						});
					}
				}
				int status = process.waitFor();
				if (status != 0 && !reported) throw new IOException("Helper exited " + status + "; see logs/rosie-launcher.log (check Python dependencies and SSH login).");
			} catch (IOException | RuntimeException | InterruptedException error) {
				if (process != null) terminate(process);
				client.execute(() -> {
					if (generation == selected) message(client, "Rosie failed: " + error.getMessage());
				});
			} finally {
				client.execute(() -> {
					if (generation == selected) {
						if (serving) {
							LiveDuel.stop(client);
							message(client, "Rosie policy process stopped.");
						}
						cancel();
					}
				});
			}
		});
		return 1;
	}

	static ProcessBuilder helper(JsonObject config, Path game, String action, String run, String replay) {
		boolean local = action.equals("local") || action.equals("local-world");
		var arguments = new ArrayList<>(List.of(config.get("python").getAsString(), "-u", "-B", "-m",
				"simulator.training.rosie", local ? "local" : action, "--minecraft-directory", game.toString(),
				"--run", run, "--replay", replay, "--json"));
		if (local) arguments.addAll(List.of("--duel", game.resolve("sim-rosie-cache/duel.json").toString(), "--offline"));
		return new ProcessBuilder(arguments)
				.directory(Path.of(config.get("repository").getAsString()).toFile())
				.redirectError(ProcessBuilder.Redirect.appendTo(game.resolve("logs/rosie-launcher.log").toFile()));
	}

	private static void handle(Minecraft client, JsonObject event) {
		switch (event.get("event").getAsString()) {
			case "error" -> message(client, "Rosie: " + event.get("message").getAsString());
			case "runs", "replays" -> {
				boolean runs = event.get("event").getAsString().equals("runs");
				var items = event.getAsJsonArray("items");
				if (runs) {
					var rows = new ArrayList<JsonObject>();
					for (var item : items) rows.add(item.getAsJsonObject().deepCopy());
					remoteRuns = List.copyOf(rows);
					lastMessage = "Loaded " + rows.size() + " remote runs";
					if (client.gui.screen() instanceof ControlPanel panel) { panel.remoteChanged(); return; }
				}
				message(client, "Rosie: " + items.size() + (runs ? " saved runs (newest first)." : " replays (newest first)."));
				for (var item : items) {
					JsonObject row = item.getAsJsonObject();
					String name = row.get("name").getAsString();
					String command = runs ? "/simrosie replay " + name : "/simrosie replay " + event.get("run").getAsString() + " " + name;
					client.gui.hud.getChat().addClientSystemMessage(Component.literal(name
							+ (runs && row.get("checkpoint").getAsBoolean() ? " [checkpoint]" : ""))
							.withStyle(style -> style.withClickEvent(new ClickEvent.SuggestCommand(command))));
				}
			}
			case "replay" -> SimulationReplay.load(client, event.get("path").getAsString(), true);
			case "policy" -> {
				serving = true;
				operation = "serving " + event.get("run").getAsString() + " locally";
				LiveDuel.connect(client, arenaMode, event.get("port").getAsInt());
			}
			case "synced" -> {
				message(client, "Cached " + event.get("run").getAsString() + " and " + event.get("replays").getAsInt() + " replays.");
				if (client.gui.screen() instanceof ControlPanel panel) panel.refreshLibrary();
			}
			default -> throw new IllegalArgumentException("Unknown Rosie helper event");
		}
	}

	private static int message(Minecraft client, String text) {
		lastMessage = text;
		client.gui.hud.getChat().addClientSystemMessage(Component.literal(text));
		return 1;
	}

	private static void terminate(Process process) {
		process.descendants().forEach(ProcessHandle::destroyForcibly);
		process.destroyForcibly();
	}

	private static synchronized void cancel() {
		generation++;
		if (worker != null) terminate(worker);
		worker = null;
		busy = false;
		serving = false;
		world = null;
		operation = "idle";
	}
}