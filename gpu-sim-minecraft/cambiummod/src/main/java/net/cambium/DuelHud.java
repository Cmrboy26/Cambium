package net.cambium;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import net.fabricmc.fabric.api.client.networking.v1.ClientPlayConnectionEvents;
import net.fabricmc.fabric.api.client.rendering.v1.hud.HudElementRegistry;
import net.minecraft.client.Minecraft;
import net.minecraft.network.chat.Component;

final class DuelHud {

	private record Snapshot(String title, List<String> lines) {}
	private static volatile Snapshot snapshot;
	private static boolean enabled = true;

	static void initialize() {
		Path path = settings(Minecraft.getInstance());
		try {
			if (Files.isRegularFile(path)) enabled = JsonParser.parseString(Files.readString(path)).getAsJsonObject().get("enabled").getAsBoolean();
		} catch (IOException | RuntimeException error) {
			CambiumFabric.LOGGER.warn("Could not read Cambium HUD settings", error);
		}
		ClientPlayConnectionEvents.DISCONNECT.register((handler, client) -> clear());
		HudElementRegistry.addLast(CambiumFabric.id("duel_sidebar"), (graphics, delta) -> {
			Minecraft client = Minecraft.getInstance();
			Snapshot current = snapshot;
			if (!enabled || current == null || client.level == null || client.gui.screen() != null || client.gui.hud.isHidden()) return;
			int panelWidth = Math.min(216, graphics.guiWidth() / 3 + 12);
			int left = graphics.guiWidth() - panelWidth - 8;
			var font = client.font;
			var lines = new java.util.ArrayList<net.minecraft.util.FormattedCharSequence>();
			for (String line : current.lines()) lines.addAll(font.split(Component.literal(line), panelWidth - 16));
			int available = Math.max(1, (graphics.guiHeight() - 54) / 11);
			int shown = Math.min(available, lines.size());
			graphics.fill(left, 12, left + panelWidth, 39 + shown * 11, 0xDD151A1C);
			graphics.fill(left, 12, left + panelWidth, 14, 0xFF75CDA0);
			graphics.text(font, font.plainSubstrByWidth(current.title(), panelWidth - 16), left + 8, 21, 0xFF92DDB3);
			for (int index = 0; index < shown; index++) graphics.text(font, lines.get(index), left + 8, 36 + index * 11, 0xFFE2E8E5);
		});
	}

	static void publish(String title, List<String> lines) {
		snapshot = new Snapshot(title, List.copyOf(lines));
	}

	static List<String> lines() {
		Snapshot current = snapshot;
		return current == null ? List.of("No active duel or replay") : current.lines();
	}

	static boolean enabled() { return enabled; }

	static void toggle(Minecraft client) {
		enabled = !enabled;
		JsonObject value = new JsonObject();
		value.addProperty("enabled", enabled);
		try { DuelLibrary.write(settings(client), value); }
		catch (IOException error) { CambiumFabric.LOGGER.warn("Could not save Cambium HUD settings", error); }
	}

	static void clear() { snapshot = null; }

	private static Path settings(Minecraft client) {
		return client.gameDirectory.toPath().resolve("config/cambium-hud.json");
	}
}