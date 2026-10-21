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
import net.minecraft.client.gui.Font;
import net.minecraft.client.gui.GuiGraphicsExtractor;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.EquipmentSlot;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;

final class DuelHud {

	record Fighter(String name, float health, float maximum, float absorption, float charge, String shield, String actions, List<ItemStack> equipment) {
		Fighter { equipment = equipment.stream().map(ItemStack::copy).toList(); }
	}
	record Match(String title, String state, String mode, int tick, int maxTicks, int round, int winsA, int winsB, int draws, double latency) {}
	record Layout(int left, int center, int right, int fighterWidth, int centerWidth) {}
	private record Snapshot(String title, List<String> lines, Match match, Fighter first, Fighter second) {}
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
			if (current.match() != null) {
				renderMatch(graphics, client.font, current);
				return;
			}
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
		snapshot = new Snapshot(title, List.copyOf(lines), null, null, null);
	}

	static void publishMatch(Match match, Fighter first, Fighter second, List<String> lines) {
		snapshot = new Snapshot(match.title(), List.copyOf(lines), match, first, second);
	}

	static Fighter fighter(ServerPlayer player, String name, String actions) {
		int cooldown = LivePolicy.shieldCooldown(player);
		String shield = cooldown > 0 ? "Disabled " + time(cooldown) : !player.getOffhandItem().is(Items.SHIELD) ? "No shield" : player.isUsingItem() ? "Shield raised" : "Shield ready";
		return new Fighter(name, player.getHealth(), player.getMaxHealth(), player.getAbsorptionAmount(), player.getAttackStrengthScale(0.5F), shield, actions,
				List.of(player.getMainHandItem(), player.getOffhandItem(), player.getItemBySlot(EquipmentSlot.HEAD), player.getItemBySlot(EquipmentSlot.CHEST),
						player.getItemBySlot(EquipmentSlot.LEGS), player.getItemBySlot(EquipmentSlot.FEET)));
	}

	static Layout layout(int width) {
		int centerWidth = Math.min(152, Math.max(88, width / 4));
		int fighterWidth = Math.min(218, Math.max(1, (width - 32 - centerWidth) / 2));
		return new Layout(8, (width - centerWidth) / 2, width - fighterWidth - 8, fighterWidth, centerWidth);
	}

	static double fraction(double value, double maximum) { return maximum <= 0 ? 0 : Math.clamp(value / maximum, 0, 1); }
	static String time(int ticks) {
		int seconds = Math.max(0, (ticks + 19) / 20);
		return String.format(java.util.Locale.ROOT, "%d:%02d", seconds / 60, seconds % 60);
	}

	private static void renderMatch(GuiGraphicsExtractor graphics, Font font, Snapshot current) {
		Layout layout = layout(graphics.guiWidth());
		drawFighter(graphics, font, current.first(), "P1", layout.left(), layout.fighterWidth(), 0xFF80C9F7);
		drawFighter(graphics, font, current.second(), "P2", layout.right(), layout.fighterWidth(), 0xFFF19C94);
		Match match = current.match();
		int left = layout.center();
		int width = layout.centerWidth();
		panel(graphics, left, width, 114, 0xFF92DDB3);
		text(graphics, font, match.title(), left, 20, width, 0xFF92DDB3);
		text(graphics, font, match.round() > 0 ? "Round " + match.round() : match.mode(), left, 35, width, 0xFFE7ECE9);
		text(graphics, font, match.state(), left, 48, width, 0xFFB9C8C1);
		text(graphics, font, time(match.maxTicks() - match.tick()) + " / " + time(match.maxTicks()), left, 63, width, 0xFFFFFFFF);
		bar(graphics, left + 8, 77, width - 16, 5, fraction(match.maxTicks() - match.tick(), match.maxTicks()), 0xFF92DDB3);
		text(graphics, font, match.round() > 0 ? match.winsA() + " : " + match.winsB() + " | D " + match.draws() : "Tick " + match.tick(), left, 88, width, 0xFFE7ECE9);
		text(graphics, font, match.round() > 0 ? String.format(java.util.Locale.ROOT, "%.0f ms | %s", match.latency(), match.mode()) : match.title(), left, 103, width, 0xFF94A9A0);
	}

	private static void drawFighter(GuiGraphicsExtractor graphics, Font font, Fighter fighter, String side, int left, int width, int accent) {
		panel(graphics, left, width, 148, accent);
		text(graphics, font, side + " / " + fighter.name(), left, 20, width, accent);
		String health = String.format(java.util.Locale.ROOT, "%.0f / %.0f HP", fighter.health(), fighter.maximum());
		if (fighter.absorption() > 0) health += String.format(java.util.Locale.ROOT, " +%.0f", fighter.absorption());
		text(graphics, font, health, left, 35, width, 0xFFE7ECE9);
		bar(graphics, left + 8, 48, width - 16, 7, fraction(fighter.health(), fighter.maximum()), 0xFF6FD195);
		if (fighter.absorption() > 0) bar(graphics, left + 8, 56, width - 16, 2, fraction(fighter.absorption(), fighter.maximum()), 0xFFF2D17A);
		text(graphics, font, String.format(java.util.Locale.ROOT, "Charge %.0f%%", Math.clamp(fighter.charge(), 0, 1) * 100), left, 65, width, 0xFFB9C8C1);
		bar(graphics, left + 8, 77, width - 16, 4, fraction(fighter.charge(), 1), accent);
		text(graphics, font, fighter.shield(), left, 87, width, 0xFFB9C8C1);
		int columns = Math.max(1, (width - 16) / 18);
		for (int index = 0; index < fighter.equipment().size(); index++) {
			int position = left + 8 + index % columns * 18;
			int top = 101 + index / columns * 18;
			graphics.fill(position, top, position + 16, top + 16, 0x663D4D4A);
			graphics.item(fighter.equipment().get(index), position, top);
		}
		text(graphics, font, fighter.actions(), left, 143, width, 0xFF94A9A0);
	}

	private static void panel(GuiGraphicsExtractor graphics, int left, int width, int height, int accent) {
		graphics.fill(left, 12, left + width, 12 + height, 0xDE141C1F);
		graphics.fill(left, 12, left + width, 14, accent);
	}

	private static void text(GuiGraphicsExtractor graphics, Font font, String value, int left, int top, int width, int color) {
		String shown = font.width(value) <= width - 16 ? value : font.plainSubstrByWidth(value, Math.max(0, width - 16 - font.width("..."))) + "...";
		graphics.text(font, shown, left + 8, top, color);
	}

	private static void bar(GuiGraphicsExtractor graphics, int left, int top, int width, int height, double fraction, int color) {
		graphics.fill(left, top, left + width, top + height, 0xFF33423F);
		graphics.fill(left, top, left + (int) (width * fraction), top + height, color);
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