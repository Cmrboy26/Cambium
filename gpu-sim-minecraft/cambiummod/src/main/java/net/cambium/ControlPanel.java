package net.cambium;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.mojang.blaze3d.platform.InputConstants;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.function.Consumer;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.fabric.api.client.command.v2.ClientCommands;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keymapping.v1.KeyMappingHelper;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.GuiGraphicsExtractor;
import net.minecraft.client.gui.components.Button;
import net.minecraft.client.gui.components.CycleButton;
import net.minecraft.client.gui.components.EditBox;
import net.minecraft.client.gui.components.Tooltip;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.network.chat.Component;

final class ControlPanel extends Screen {

	private record Label(String text, int left, int top, int width, int color) {}
	private static final List<String> MODES = List.of("Policy", "Basic melee", "Stationary target");
	private static Object requestedWorld;
	private final Screen previous;
	private final List<Label> labels = new ArrayList<>();
	private DuelLibrary library;
	private String environmentId = "";
	private final String[] policyIds = {"", ""};
	private final String[] modes = {"Policy", "Basic melee"};
	private final JsonObject[] scriptOptions = {new JsonObject(), new JsonObject()};
	private boolean arena = true;
	private int tab;
	private int page;
	private String query = "";
	private String selectedPolicy = "";
	private String libraryRun = "";
	private int maxTicks;
	private String selectedRun = "";
	private String syncScope = "policy";
	private String status = "";
	private boolean helperBusy;
	private boolean helperServing;
	private int left;
	private int contentWidth;

	ControlPanel(Screen previous) {
		super(Component.literal("Cambium"));
		this.previous = previous;
		refreshLibrary();
		try {
			if (library != null && Files.isRegularFile(library.root.resolve("selections.json"))) {
				JsonObject saved = JsonParser.parseString(Files.readString(library.root.resolve("selections.json"))).getAsJsonObject();
				environmentId = saved.get("environment").getAsString();
				arena = saved.get("arena").getAsBoolean();
				maxTicks = saved.has("max_ticks") ? saved.get("max_ticks").getAsInt() : 0;
				if (saved.has("sync_scope") && List.of("policy", "latest", "all").contains(saved.get("sync_scope").getAsString())) syncScope = saved.get("sync_scope").getAsString();
				for (int index = 0; index < 2; index++) {
					JsonObject fighter = saved.getAsJsonArray("fighters").get(index).getAsJsonObject();
					policyIds[index] = fighter.get("policy").getAsString();
					String mode = fighter.get("mode").getAsString();
					modes[index] = MODES.contains(mode) ? mode : "Policy";
					scriptOptions[index] = fighter.getAsJsonObject("options").deepCopy();
				}
			}
		} catch (IOException | RuntimeException error) { status = "Could not restore selections: " + error.getMessage(); }
	}

	static void initialize() {
		KeyMapping menu = KeyMappingHelper.registerKeyMapping(new KeyMapping("key.cambiumfabric.panel", InputConstants.Type.KEYBOARD,
				InputConstants.KEY_F8, KeyMapping.Category.register(CambiumFabric.id("interface"))));
		ClientTickEvents.END_CLIENT_TICK.register(client -> {
			while (menu.consumeClick()) open(client);
			Object world = requestedWorld;
			requestedWorld = null;
			if (world != null && client.level == world && client.player != null && client.getSingleplayerServer() != null
					&& !client.getSingleplayerServer().isPublished() && !(client.gui.screen() instanceof ControlPanel)) {
				client.gui.setScreen(new ControlPanel(client.gui.screen()));
			}
		});
		ClientCommandRegistrationCallback.EVENT.register((dispatcher, registry) -> dispatcher.register(
				ClientCommands.literal("simmenu").executes(context -> open(context.getSource().getClient()))));
	}

	static int open(Minecraft client) {
		if (client.getSingleplayerServer() == null || client.getSingleplayerServer().isPublished() || client.player == null) {
			client.gui.hud.getChat().addClientSystemMessage(Component.literal("Cambium requires a private singleplayer world."));
			return 0;
		}
		requestedWorld = client.level;
		return 1;
	}

	void refreshLibrary() {
		try {
			library = new DuelLibrary(Minecraft.getInstance().gameDirectory.toPath());
			if (library.find(library.runs, libraryRun) == null && !library.runs.isEmpty()) libraryRun = identity(library.runs.getFirst());
			selectVersion();
			if (library.find(library.environments, environmentId) == null && !library.environments.isEmpty()) environmentId = identity(library.environments.getFirst());
			for (int index = 0; index < 2; index++) {
				if (library.find(library.policies, policyIds[index]) == null && !library.policies.isEmpty()) policyIds[index] = identity(library.policies.getFirst());
			}
		} catch (IOException | RuntimeException error) { library = null; status = "Library: " + error.getMessage(); }
		if (width > 0) rebuildWidgets();
	}

	void remoteChanged() { if (width > 0) rebuildWidgets(); }

	@Override
	public void tick() {
		if (helperBusy != RosieLauncher.isBusy() || helperServing != RosieLauncher.isServing()) {
			helperBusy = RosieLauncher.isBusy();
			helperServing = RosieLauncher.isServing();
			rebuildWidgets();
		}
	}

	@Override
	protected void init() {
		labels.clear();
		contentWidth = Math.min(740, width - 24);
		left = (width - contentWidth) / 2;
		button("Close", left + contentWidth - 60, 10, 60, this::onClose);
		String[] tabs = {"Duel", "Library", "Rosie", "Live"};
		int tabWidth = (contentWidth - 12) / 4;
		for (int index = 0; index < tabs.length; index++) {
			final int selected = index;
			Button item = button(tabs[index], left + index * (tabWidth + 4), 34, tabWidth, () -> { tab = selected; page = 0; status = ""; rebuildWidgets(); });
			item.active = tab != index;
		}
		switch (tab) {
			case 0 -> duel();
			case 1 -> shelf();
			case 2 -> rosie();
			case 3 -> live();
			default -> throw new IllegalStateException("Unknown panel tab");
		}
	}

	private void duel() {
		JsonObject environment = environment();
		label("Environment", left, 63, contentWidth, 0xFFB5C2BE);
		button(environment == null ? "No local environments" : name(environment), left, 76, contentWidth - 100,
				() -> choose("Environment", library == null ? List.of() : library.environmentChoices(environmentId), entry -> { environmentId = identity(entry); persist(); }));
		addRenderableWidget(CycleButton.builder(value -> Component.literal(value ? "Arena" : "World"), arena)
				.withValues(true, false).displayOnlyValue().create(left + contentWidth - 96, 76, 96, 20, Component.literal("Location"),
						(widget, value) -> { arena = value; persist(); rebuildWidgets(); }));
		int columnWidth = (contentWidth - 12) / 2;
		for (int index = 0; index < 2; index++) {
			final int fighter = index;
			int column = left + index * (columnWidth + 12);
			label(index == 0 ? "Fighter A" : "Fighter B", column, 107, columnWidth, index == 0 ? 0xFF85C5F5 : 0xFFF6A397);
			addRenderableWidget(CycleButton.builder(Component::literal, modes[index]).withValues(MODES).displayOnlyValue()
					.create(column, 120, columnWidth, 20, Component.literal("Controller"), (widget, value) -> { modes[fighter] = value; persist(); rebuildWidgets(); }));
			JsonObject policy = library == null ? null : library.find(library.policies, policyIds[index]);
			String title = modes[index].equals("Policy") ? policy == null ? "Select local policy" : name(policy)
					: modes[index].equals("Basic melee") ? "Script settings" : "Idle target";
			Button selection = button(title, column, 146, columnWidth, () -> {
				if (modes[fighter].equals("Policy")) {
					List<JsonObject> compatible = library == null ? List.of() : library.policies.stream()
							.filter(entry -> environment == null || entry.get("contract_hash").equals(environment.get("contract_hash"))).toList();
					choose("Fighter " + (fighter == 0 ? "A" : "B"), compatible, entry -> { policyIds[fighter] = identity(entry); persist(); });
				} else minecraft.gui.setScreen(new ScriptScreen(fighter));
			});
			selection.active = !modes[index].equals("Stationary target");
			if (!arena) button("Mark spawn " + (index + 1), column, 172, columnWidth, () -> LiveDuel.markSpawn(minecraft, fighter));
			else if (modes[index].equals("Policy") && policy != null) label("Update " + policy.get("update").getAsLong() + " / " + policy.get("stage").getAsString(), column, 176, columnWidth, 0xFFACBDB5);
		}
		button(RosieLauncher.isServing() ? "Replace live duel" : "Start live duel", left, height - 30, contentWidth - 124, this::launch)
				.active = environment != null && RosieLauncher.canStartLocalMatch();
		button(maxTicks == 0 ? "Duration: Default" : String.format(Locale.ROOT, "Duration: %.1f s", maxTicks / 20.0), left + contentWidth - 120, height - 30, 120,
				() -> minecraft.gui.setScreen(new DurationScreen()));
	}

	private void shelf() {
		EditBox search = new EditBox(font, left, 64, contentWidth - 84, 20, Component.literal("Search local policies"));
		search.setHint(Component.literal("Search local policies"));
		search.setValue(query);
		addRenderableWidget(search);
		search.setResponder(value -> { query = value; page = 0; rebuildWidgets(); });
		setInitialFocus(search);
		button("Refresh", left + contentWidth - 80, 64, 80, this::refreshLibrary);
		String filter = query.toLowerCase(Locale.ROOT);
		List<JsonObject> entries = library == null ? List.of() : library.runs.stream().filter(entry -> name(entry).toLowerCase(Locale.ROOT).contains(filter)
				|| identity(entry).toLowerCase(Locale.ROOT).contains(filter) || library.versions(identity(entry)).stream().anyMatch(policy -> name(policy).toLowerCase(Locale.ROOT).contains(filter))).toList();
		int rows = Math.max(1, (height - 212) / 24);
		page = Math.min(page, Math.max(0, (entries.size() - 1) / rows));
		for (int index = page * rows; index < Math.min(entries.size(), (page + 1) * rows); index++) {
			JsonObject entry = entries.get(index);
			button((identity(entry).equals(libraryRun) ? "> " : "") + name(entry) + " / " + library.versions(identity(entry)).size() + " versions",
					left, 92 + (index % rows) * 24, contentWidth, () -> { libraryRun = identity(entry); selectVersion(); rebuildWidgets(); });
		}
		if (entries.isEmpty()) label("No matching local runs", left, 100, contentWidth, 0xFFACBDB5);
		pager(entries.size(), rows, height - 108);
		List<JsonObject> versions = library == null ? List.of() : library.versions(libraryRun);
		JsonObject version = library == null ? null : library.find(versions, selectedPolicy);
		if (version != null) addRenderableWidget(CycleButton.builder(entry -> Component.literal(fit(DuelLibrary.versionName(entry), contentWidth - 24)), version)
				.withValues(versions).displayOnlyValue().create(left, height - 80, contentWidth, 20, Component.literal("Policy version"),
						(widget, value) -> { selectedPolicy = identity(value); rebuildWidgets(); })).setTooltip(Tooltip.create(Component.literal(DuelLibrary.versionName(version))));
		int actionWidth = (contentWidth - 12) / 4;
		for (int index = 0; index < 2; index++) {
			final int fighter = index;
			button(index == 0 ? "Assign A" : "Assign B", left + index * (actionWidth + 4), height - 30, actionWidth,
					() -> { policyIds[fighter] = selectedPolicy; modes[fighter] = "Policy"; tab = 0; persist(); rebuildWidgets(); })
					.active = library != null && library.find(library.policies, selectedPolicy) != null;
		}
		button("Name policy", left + 2 * (actionWidth + 4), height - 30, actionWidth, () -> {
			JsonObject entry = library.find(library.policies, selectedPolicy);
			minecraft.gui.setScreen(new RenameScreen(entry));
		}).active = !RosieLauncher.isBusy() && library != null && library.find(library.policies, selectedPolicy) != null;
		button("Name run", left + 3 * (actionWidth + 4), height - 30, actionWidth,
				() -> minecraft.gui.setScreen(new RenameScreen(library.find(library.runs, libraryRun))))
				.active = !RosieLauncher.isBusy() && library != null && library.find(library.runs, libraryRun) != null;
	}

	private void selectVersion() {
		List<JsonObject> versions = library.versions(libraryRun);
		if (library.find(versions, selectedPolicy) == null) selectedPolicy = versions.isEmpty() ? "" : identity(versions.getFirst());
	}

	private void rosie() {
		button("Refresh remote runs", left, 64, (contentWidth - 4) / 2, () -> RosieLauncher.start(minecraft, "runs", "latest", "latest")).active = !RosieLauncher.isBusy();
		button("Sync selected run", left + (contentWidth + 4) / 2, 64, (contentWidth - 4) / 2,
				() -> RosieLauncher.start(minecraft, "sync", selectedRun, "latest", syncScope)).active = !selectedRun.isEmpty() && !RosieLauncher.isBusy();
		addRenderableWidget(CycleButton.builder(value -> Component.literal(switch (value) {
			case "policy" -> "Policy only";
			case "latest" -> "Policy + latest replay";
			default -> "All policies and replays";
		}), syncScope).withValues("policy", "latest", "all").displayOnlyValue().create(left, 90, contentWidth, 20,
				Component.literal("Sync scope"), (widget, value) -> { syncScope = value; persist(); })).active = !RosieLauncher.isBusy();
		List<JsonObject> runs = RosieLauncher.remoteRuns();
		int rows = Math.max(1, (height - 218) / 24);
		page = Math.min(page, Math.max(0, (runs.size() - 1) / rows));
		for (int index = page * rows; index < Math.min(runs.size(), (page + 1) * rows); index++) {
			JsonObject entry = runs.get(index);
			String run = name(entry);
			button((run.equals(selectedRun) ? "> " : "") + run, left, 116 + index % rows * 24, contentWidth,
					() -> { selectedRun = run; rebuildWidgets(); }).active = entry.get("checkpoint").getAsBoolean();
		}
		if (runs.isEmpty()) label("No remote run list loaded", left, 121, contentWidth, 0xFFACBDB5);
		pager(runs.size(), rows, height - 96);
		button(RosieLauncher.hasInference() ? "Shutdown inference" : "Cancel helper", left, height - 30, contentWidth,
				() -> { RosieLauncher.stopOwned(minecraft); rebuildWidgets(); }).active = RosieLauncher.isBusy();
	}

	private void live() {
		String[] commands = {"pause", "resume", "reset", "stop"};
		int actionWidth = (contentWidth - 12) / 4;
		for (int index = 0; index < commands.length; index++) {
			String command = commands[index];
			button(command.equals("stop") ? "Stop match" : Character.toUpperCase(command.charAt(0)) + command.substring(1), left + index * (actionWidth + 4), 66, actionWidth,
					() -> { if (command.equals("stop")) RosieLauncher.stopMatch(minecraft); else LiveDuel.command(minecraft, command); });
		}
		button("Join A", left, 92, actionWidth, () -> LiveDuel.joinSide(minecraft, 0));
		button("Join B", left + actionWidth + 4, 92, actionWidth, () -> LiveDuel.joinSide(minecraft, 1));
		button("Leave", left + 2 * (actionWidth + 4), 92, actionWidth, () -> LiveDuel.command(minecraft, "leave"));
		button("Sidebar: " + (DuelHud.enabled() ? "On" : "Off"), left + 3 * (actionWidth + 4), 92, actionWidth,
				() -> { DuelHud.toggle(minecraft); rebuildWidgets(); });
		button("Return to game", left, height - 30, (contentWidth - 4) / 2, () -> minecraft.gui.setScreen(null));
		button("Shutdown inference", left + (contentWidth + 4) / 2, height - 30, (contentWidth - 4) / 2,
				() -> { RosieLauncher.stopOwned(minecraft); rebuildWidgets(); }).active = RosieLauncher.hasInference();
	}

	private void pager(int count, int rows) {
		pager(count, rows, height - 80);
	}

	private void pager(int count, int rows, int top) {
		button("Previous", left, top, 76, () -> { page--; rebuildWidgets(); }).active = page > 0;
		label((page + 1) + " / " + Math.max(1, (count + rows - 1) / rows), left + 84, top + 7, contentWidth - 168, 0xFFACBDB5);
		button("Next", left + contentWidth - 76, top, 76, () -> { page++; rebuildWidgets(); }).active = (page + 1) * rows < count;
	}

	private void launch() {
		try {
			JsonObject environment = environment();
			JsonObject[] fighters = new JsonObject[2];
			for (int index = 0; index < 2; index++) fighters[index] = modes[index].equals("Policy")
					? library.policyFighter(library.find(library.policies, policyIds[index]), environment)
					: DuelLibrary.scriptedFighter(modes[index].equals("Basic melee") ? "basic_melee" : "stationary_target", scriptOptions[index]);
			library.saveDuel(environment, fighters[0], fighters[1], maxTicks);
			persist();
			RosieLauncher.start(minecraft, arena ? "local" : "local-world", "latest", "latest");
			tab = 3;
			rebuildWidgets();
		} catch (IOException | RuntimeException error) { status = error.getMessage(); }
	}

	private void persist() {
		if (library == null) return;
		JsonObject saved = new JsonObject();
		saved.addProperty("environment", environmentId);
		saved.addProperty("arena", arena);
		saved.addProperty("max_ticks", maxTicks);
		saved.addProperty("sync_scope", syncScope);
		var fighters = new com.google.gson.JsonArray();
		for (int index = 0; index < 2; index++) {
			JsonObject fighter = new JsonObject();
			fighter.addProperty("policy", policyIds[index]);
			fighter.addProperty("mode", modes[index]);
			fighter.add("options", scriptOptions[index].deepCopy());
			fighters.add(fighter);
		}
		saved.add("fighters", fighters);
		try { DuelLibrary.write(library.root.resolve("selections.json"), saved); }
		catch (IOException error) { status = "Could not save selections: " + error.getMessage(); }
	}

	private void choose(String title, List<JsonObject> entries, Consumer<JsonObject> selected) {
		minecraft.gui.setScreen(new ChoiceScreen(title, entries, entry -> { selected.accept(entry); status = ""; }));
	}

	private JsonObject environment() { return library == null ? null : library.find(library.environmentChoices(environmentId), environmentId); }
	private static String identity(JsonObject entry) { return entry.get("id").getAsString(); }
	private static String name(JsonObject entry) { return entry.get("name").getAsString(); }
	private void label(String text, int position, int top, int limit, int color) { labels.add(new Label(text, position, top, limit, color)); }

	private Button button(String text, int position, int top, int size, Runnable action) {
		return addRenderableWidget(Button.builder(Component.literal(fit(text, size - 12)), widget -> action.run()).bounds(position, top, size, 20)
				.tooltip(Tooltip.create(Component.literal(text))).build());
	}

	private String fit(String text, int limit) {
		return font.width(text) <= limit ? text : font.plainSubstrByWidth(text, Math.max(0, limit - font.width("..."))) + "...";
	}

	@Override
	public void extractRenderState(GuiGraphicsExtractor graphics, int mouseX, int mouseY, float delta) {
		graphics.fill(0, 0, width, height, 0xE515191C);
		graphics.fill(0, 0, width, 58, 0xF0202829);
		graphics.text(font, "CAMBIUM", left, 15, 0xFF92DDB3);
		for (Label label : labels) graphics.text(font, fit(label.text(), label.width()), label.left(), label.top(), label.color());
		String message = status.isEmpty() ? RosieLauncher.status() : status;
		graphics.text(font, fit(message, contentWidth), left, height - 45, status.isEmpty() ? 0xFFACBDB5 : 0xFFF3B298);
		JsonObject progress = RosieLauncher.progress();
		if (tab == 2 && progress != null) {
			long total = progress.get("bytes_total").getAsLong();
			long done = progress.get("bytes_done").getAsLong();
			String detail = progress.get("phase").getAsString() + " | " + progress.get("completed").getAsInt() + "/" + progress.get("total").getAsInt()
					+ " files | " + String.format(Locale.ROOT, "%.1f/%.1f MiB", done / 1048576.0, total / 1048576.0)
					+ " | " + progress.get("cached").getAsInt() + " cached";
			graphics.text(font, fit(detail, contentWidth), left, height - 68, 0xFFACBDB5);
			graphics.fill(left, height - 56, left + contentWidth, height - 52, 0xFF354244);
			if (total > 0) graphics.fill(left, height - 56, left + (int) (contentWidth * Math.clamp(done / (double) total, 0, 1)), height - 52, 0xFF92DDB3);
		}
		if (tab == 3) {
			List<String> lines = DuelHud.lines();
			for (int index = 0; index < Math.min(lines.size(), Math.max(0, (height - 178) / 12)); index++) graphics.text(font, fit(lines.get(index), contentWidth), left, 130 + index * 12, 0xFFE2E8E5);
		}
		super.extractRenderState(graphics, mouseX, mouseY, delta);
	}

	@Override public boolean isPauseScreen() { return false; }
	@Override public void onClose() { persist(); minecraft.gui.setScreen(previous); }

	private final class ChoiceScreen extends Screen {
		private final List<JsonObject> entries;
		private final Consumer<JsonObject> selected;
		private String filter = "";
		private int selectedPage;
		private int position;
		private int size;
		private boolean empty;

		ChoiceScreen(String title, List<JsonObject> entries, Consumer<JsonObject> selected) {
			super(Component.literal(title));
			this.entries = List.copyOf(entries);
			this.selected = selected;
		}

		@Override protected void init() {
			size = Math.min(650, width - 32);
			position = (width - size) / 2;
			EditBox search = addRenderableWidget(new EditBox(font, position, 42, size, 20, Component.literal("Search")));
			search.setHint(Component.literal("Search"));
			search.setValue(filter);
			search.setResponder(value -> { filter = value; selectedPage = 0; rebuildWidgets(); });
			setInitialFocus(search);
			List<JsonObject> filtered = entries.stream().filter(entry -> name(entry).toLowerCase(Locale.ROOT).contains(filter.toLowerCase(Locale.ROOT))).toList();
			empty = filtered.isEmpty();
			int rows = Math.max(1, (height - 122) / 24);
			selectedPage = Math.min(selectedPage, Math.max(0, (filtered.size() - 1) / rows));
			for (int index = selectedPage * rows; index < Math.min(filtered.size(), (selectedPage + 1) * rows); index++) {
				JsonObject entry = filtered.get(index);
				String title = name(entry) + (entry.has("update") ? " / update " + entry.get("update").getAsLong() : "");
				addRenderableWidget(Button.builder(Component.literal(fit(title, size - 12)), widget -> { selected.accept(entry); onClose(); })
						.bounds(position, 74 + index % rows * 24, size, 20).tooltip(Tooltip.create(Component.literal(title))).build());
			}
			addRenderableWidget(Button.builder(Component.literal("Previous"), widget -> { selectedPage--; rebuildWidgets(); }).bounds(position, height - 32, 90, 20).build()).active = selectedPage > 0;
			addRenderableWidget(Button.builder(Component.literal("Back"), widget -> onClose()).bounds(position + size / 2 - 40, height - 32, 80, 20).build());
			addRenderableWidget(Button.builder(Component.literal("Next"), widget -> { selectedPage++; rebuildWidgets(); }).bounds(position + size - 90, height - 32, 90, 20).build()).active = (selectedPage + 1) * rows < filtered.size();
		}

		@Override public void extractRenderState(GuiGraphicsExtractor graphics, int mouseX, int mouseY, float delta) {
			graphics.fill(0, 0, width, height, 0xF0181E20);
			graphics.text(font, title, position, 18, 0xFF92DDB3);
			if (empty) graphics.text(font, "No matching local entries", position, 82, 0xFFACBDB5);
			super.extractRenderState(graphics, mouseX, mouseY, delta);
		}
		@Override public boolean isPauseScreen() { return false; }
		@Override public void onClose() { minecraft.gui.setScreen(ControlPanel.this); }
	}

	private final class RenameScreen extends Screen {
		private final JsonObject entry;
		RenameScreen(JsonObject entry) { super(Component.literal("Local display name")); this.entry = entry; }
		@Override protected void init() {
			int size = Math.min(400, width - 40);
			int position = (width - size) / 2;
			EditBox name = addRenderableWidget(new EditBox(font, position, 74, size, 20, title));
			name.setMaxLength(80);
			name.setValue(ControlPanel.name(entry));
			setInitialFocus(name);
			addRenderableWidget(Button.builder(Component.literal("Save"), widget -> {
				try { library.rename(entry, name.getValue()); status = ""; onClose(); }
				catch (IOException error) { status = error.getMessage(); }
			}).bounds(position, 108, (size - 8) / 2, 20).build());
			addRenderableWidget(Button.builder(Component.literal("Cancel"), widget -> onClose()).bounds(position + (size + 8) / 2, 108, (size - 8) / 2, 20).build());
		}
		@Override public void extractRenderState(GuiGraphicsExtractor graphics, int mouseX, int mouseY, float delta) {
			graphics.fill(0, 0, width, height, 0xF0181E20);
			graphics.text(font, title, (width - Math.min(400, width - 40)) / 2, 48, 0xFF92DDB3);
			graphics.text(font, fit(status, width - 40), 20, height - 35, 0xFFF3B298);
			super.extractRenderState(graphics, mouseX, mouseY, delta);
		}
		@Override public boolean isPauseScreen() { return false; }
		@Override public void onClose() { minecraft.gui.setScreen(ControlPanel.this); }
	}

	private final class DurationScreen extends Screen {
		private String error = "";
		DurationScreen() { super(Component.literal("Maximum round duration (seconds)")); }
		@Override protected void init() {
			int size = Math.min(400, width - 40);
			int position = (width - size) / 2;
			JsonObject environment = environment();
			int initial = maxTicks > 0 ? maxTicks : environment != null && environment.getAsJsonObject("environment").has("max_ticks")
					? environment.getAsJsonObject("environment").get("max_ticks").getAsInt() : 300;
			EditBox duration = addRenderableWidget(new EditBox(font, position, 74, size, 20, title));
			duration.setMaxLength(12);
			duration.setValue(String.format(Locale.ROOT, "%.2f", initial / 20.0));
			setInitialFocus(duration);
			addRenderableWidget(Button.builder(Component.literal("Apply"), widget -> {
				try { maxTicks = DuelLibrary.durationTicks(duration.getValue()); persist(); onClose(); }
				catch (IllegalArgumentException invalid) { error = invalid.getMessage(); }
			}).bounds(position, 108, (size - 8) / 2, 20).build());
			addRenderableWidget(Button.builder(Component.literal("Cancel"), widget -> onClose()).bounds(position + (size + 8) / 2, 108, (size - 8) / 2, 20).build());
			addRenderableWidget(Button.builder(Component.literal("Environment default"), widget -> { maxTicks = 0; persist(); onClose(); }).bounds(position, 136, size, 20).build());
		}
		@Override public void extractRenderState(GuiGraphicsExtractor graphics, int mouseX, int mouseY, float delta) {
			graphics.fill(0, 0, width, height, 0xF0181E20);
			graphics.text(font, fit(title.getString(), width - 40), (width - Math.min(400, width - 40)) / 2, 48, 0xFF92DDB3);
			graphics.text(font, fit(error, width - 40), 20, height - 35, 0xFFF3B298);
			super.extractRenderState(graphics, mouseX, mouseY, delta);
		}
		@Override public boolean isPauseScreen() { return false; }
		@Override public void onClose() { minecraft.gui.setScreen(ControlPanel.this); }
	}

	private final class ScriptScreen extends Screen {
		private final int fighter;
		private EditBox interval;
		private EditBox distance;
		private EditBox strafeInterval;
		private EditBox axe;
		private EditBox sword;
		private boolean strafe;
		private boolean sprint;
		private boolean shield;
		private int position;
		private int column;
		private String error = "";

		ScriptScreen(int fighter) {
			super(Component.literal("Basic melee / Fighter " + (fighter == 0 ? "A" : "B")));
			this.fighter = fighter;
			JsonObject options = scriptOptions[fighter];
			strafe = options.has("strafe") && options.get("strafe").getAsBoolean();
			sprint = options.has("sprint") && options.get("sprint").getAsBoolean();
			shield = options.has("shield") && options.get("shield").getAsBoolean();
		}

		@Override protected void init() {
			int size = Math.min(480, width - 32);
			position = (width - size) / 2;
			column = (size - 12) / 2;
			JsonObject options = scriptOptions[fighter];
			interval = field(position, 76, column, "Attack interval", options, "attack_interval", "13");
			distance = field(position + column + 12, 76, column, "Melee distance", options, "melee_distance", "3");
			strafeInterval = field(position, 116, column, "Strafe interval", options, "strafe_interval", "20");
			axe = field(position + column + 12, 116, (column - 4) / 2, "Axe slot", options, "axe_slot", "");
			sword = field(position + column + 16 + (column - 4) / 2, 116, (column - 4) / 2, "Sword slot", options, "sword_slot", "");
			addRenderableWidget(CycleButton.onOffBuilder(strafe).create(position, 148, column, 20, Component.literal("Strafe"), (widget, value) -> strafe = value));
			addRenderableWidget(CycleButton.onOffBuilder(sprint).create(position + column + 12, 148, column, 20, Component.literal("Sprint"), (widget, value) -> sprint = value));
			addRenderableWidget(CycleButton.onOffBuilder(shield).create(position, 174, column, 20, Component.literal("Shield"), (widget, value) -> shield = value));
			addRenderableWidget(Button.builder(Component.literal("Apply"), widget -> apply()).bounds(position, height - 30, column, 20).build());
			addRenderableWidget(Button.builder(Component.literal("Cancel"), widget -> onClose()).bounds(position + column + 12, height - 30, column, 20).build());
		}

		private EditBox field(int position, int top, int size, String name, JsonObject options, String key, String fallback) {
			EditBox field = addRenderableWidget(new EditBox(font, position, top, size, 20, Component.literal(name)));
			field.setValue(options.has(key) ? options.get(key).getAsString() : fallback);
			field.setHint(Component.literal(name));
			field.setMaxLength(12);
			field.setTooltip(Tooltip.create(Component.literal(name)));
			return field;
		}

		private void apply() {
			try {
				JsonObject options = new JsonObject();
				options.addProperty("attack_interval", positive(interval.getValue()));
				options.addProperty("strafe_interval", positive(strafeInterval.getValue()));
				double reach = Double.parseDouble(distance.getValue());
				if (!Double.isFinite(reach) || reach <= 0 || reach >= 10) throw new IllegalArgumentException("Melee distance must be between 0 and 10");
				options.addProperty("melee_distance", reach);
				options.addProperty("strafe", strafe);
				options.addProperty("sprint", sprint);
				options.addProperty("shield", shield);
				if (!axe.getValue().isBlank() || !sword.getValue().isBlank()) {
					int axeSlot = Integer.parseInt(axe.getValue());
					int swordSlot = Integer.parseInt(sword.getValue());
					if (axeSlot < 0 || axeSlot > 8 || swordSlot < 0 || swordSlot > 8 || axeSlot == swordSlot) throw new IllegalArgumentException("Choose distinct axe/sword slots from 0 to 8");
					options.addProperty("axe_slot", axeSlot);
					options.addProperty("sword_slot", swordSlot);
				}
				scriptOptions[fighter] = options;
				persist();
				onClose();
			} catch (IllegalArgumentException invalid) { error = invalid.getMessage(); }
		}

		private int positive(String text) {
			int value = Integer.parseInt(text);
			if (value < 1 || value > 1000000) throw new IllegalArgumentException("Tick intervals must be from 1 to 1000000");
			return value;
		}

		@Override public void extractRenderState(GuiGraphicsExtractor graphics, int mouseX, int mouseY, float delta) {
			graphics.fill(0, 0, width, height, 0xF0181E20);
			graphics.text(font, title, position, 22, 0xFF92DDB3);
			graphics.text(font, "Attack interval (ticks)", position, 62, 0xFFB5C2BE);
			graphics.text(font, "Melee distance (blocks)", position + column + 12, 62, 0xFFB5C2BE);
			graphics.text(font, "Strafe interval (ticks)", position, 103, 0xFFB5C2BE);
			graphics.text(font, "Axe / sword slots", position + column + 12, 103, 0xFFB5C2BE);
			graphics.text(font, fit(error, column * 2 + 12), position, height - 45, 0xFFF3B298);
			super.extractRenderState(graphics, mouseX, mouseY, delta);
		}
		@Override public boolean isPauseScreen() { return false; }
		@Override public void onClose() { minecraft.gui.setScreen(ControlPanel.this); }
	}
}