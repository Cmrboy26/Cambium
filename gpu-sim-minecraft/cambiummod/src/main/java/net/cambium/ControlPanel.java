package net.cambium;

import com.google.gson.JsonObject;
import com.google.gson.JsonArray;
import com.google.gson.JsonParser;
import com.mojang.blaze3d.platform.InputConstants;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.function.BiConsumer;
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
	private static final List<String> MODES = List.of("Policy", "Basic melee", "Stationary melee", "Tactical melee", "Random walk", "Stationary target");
	private static Object requestedWorld;
	private final Screen previous;
	private final List<Label> labels = new ArrayList<>();
	private DuelLibrary library;
	private String environmentId = "";
	private final String[] policyIds = {"", ""};
	private final String[] modes = {"Policy", "Basic melee"};
	private final JsonObject[] scriptOptions = {new JsonObject(), new JsonObject()};
	private final JsonObject[] loadouts = {DuelLibrary.defaultWorldLoadout(), DuelLibrary.defaultWorldLoadout()};
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
	private final boolean[] markedSpawns = {false, false};
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
					if (saved.has("loadouts")) loadouts[index] = saved.getAsJsonArray("loadouts").get(index).getAsJsonObject().deepCopy();
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
			if (environment() == null && !library.environments.isEmpty()) environmentId = identity(library.environments.getFirst());
			for (int index = 0; index < 2; index++) {
				if (library.find(library.policies, policyIds[index]) == null && !library.policies.isEmpty()) policyIds[index] = identity(library.policies.getFirst());
			}
		} catch (IOException | RuntimeException error) { library = null; status = "Library: " + error.getMessage(); }
		if (width > 0) rebuildWidgets();
	}

	void remoteChanged() { if (width > 0) rebuildWidgets(); }
	void imported(String run) { libraryRun = run; selectedPolicy = ""; refreshLibrary(); }

	@Override
	public void tick() {
		boolean spawnsChanged = false;
		for (int index = 0; index < markedSpawns.length; index++) {
			boolean ready = !arena && LiveDuel.worldSpawnReady(minecraft, index);
			if (ready != markedSpawns[index]) { markedSpawns[index] = ready; spawnsChanged = true; }
		}
		if (helperBusy != RosieLauncher.isBusy() || helperServing != RosieLauncher.isServing() || spawnsChanged) {
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
		JsonObject environment = arena ? compatibleEnvironment() : null;
		label("Environment", left, 63, contentWidth, 0xFFB5C2BE);
		List<JsonObject> environmentChoices = compatibleEnvironmentChoices();
		String environmentLabel = !arena ? "Not used in World mode" : environment != null ? name(environment)
				: library == null || library.environments.isEmpty() ? "No local environments" : "No compatible environment";
		button(environmentLabel, left, 76, contentWidth - 100, this::chooseEnvironment).active = arena && !environmentChoices.isEmpty();
		addRenderableWidget(CycleButton.builder(value -> Component.literal(value ? "Arena" : "World"), arena)
				.withValues(true, false).displayOnlyValue().create(left + contentWidth - 96, 76, 96, 20, Component.literal("Location"),
						(widget, value) -> { arena = value; persist(); rebuildWidgets(); }));
		int columnWidth = (contentWidth - 12) / 2;
		for (int index = 0; index < 2; index++) {
			final int fighter = index;
			int column = left + index * (columnWidth + 12);
			label(index == 0 ? "Fighter A" : "Fighter B", column, 107, columnWidth, index == 0 ? 0xFF85C5F5 : 0xFFF6A397);
			addRenderableWidget(CycleButton.builder(Component::literal, modes[index]).withValues(MODES).displayOnlyValue()
					.create(column, 120, columnWidth, 20, Component.literal("Controller"), (widget, value) -> {
						if (!modes[fighter].equals(value)) scriptOptions[fighter] = new JsonObject();
						modes[fighter] = value;
						persist();
						rebuildWidgets();
					}));
			JsonObject policy = library == null ? null : library.find(library.policies, policyIds[index]);
			String title = modes[index].equals("Policy") ? policy == null ? "Select local policy" : name(policy)
					: modes[index].equals("Stationary target") ? "Idle target" : "Script settings";
			Button selection = button(title, column, 146, columnWidth, () -> {
				if (modes[fighter].equals("Policy")) {
					List<JsonObject> compatible = library == null ? List.of() : library.policies;
					choose("Fighter " + (fighter == 0 ? "A" : "B"), compatible, entry -> { policyIds[fighter] = identity(entry); persist(); });
				} else minecraft.gui.setScreen(new ScriptScreen(fighter));
			});
			selection.active = !modes[index].equals("Stationary target");
			if (!arena) button((LiveDuel.worldSpawnReady(minecraft, fighter) ? "Spawn " + (index + 1) + " marked" : "Mark spawn " + (index + 1)),
					column, 172, columnWidth, () -> LiveDuel.markSpawn(minecraft, fighter));
			else if (modes[index].equals("Policy") && policy != null) label("Update " + policy.get("update").getAsLong() + " / " + policy.get("stage").getAsString(), column, 176, columnWidth, 0xFFACBDB5);
			if (!arena) button("Loadout: " + loadoutSummary(loadouts[index]), column, 198, columnWidth, () -> minecraft.gui.setScreen(new LoadoutScreen(fighter)));
		}
		button(RosieLauncher.isServing() ? "Replace live duel" : "Start live duel", left, height - 30, contentWidth - 124, this::launch)
				.active = RosieLauncher.canStartLocalMatch() && (arena ? environment != null && canStartArena(environment) : canStartWorld());
		button(maxTicks == 0 ? arena ? "Duration: Default" : "Set duration" : String.format(Locale.ROOT, "Duration: %.1f s", maxTicks / 20.0), left + contentWidth - 120, height - 30, 120,
				() -> minecraft.gui.setScreen(new DurationScreen()));
	}

	private void shelf() {
		EditBox search = new EditBox(font, left, 64, contentWidth - 168, 20, Component.literal("Search local policies"));
		search.setHint(Component.literal("Search local policies"));
		search.setValue(query);
		addRenderableWidget(search);
		search.setResponder(value -> { query = value; page = 0; rebuildWidgets(); });
		setInitialFocus(search);
		button("Import", left + contentWidth - 164, 64, 80, this::importPolicy).active = !RosieLauncher.isBusy();
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

	private void importPolicy() {
		Object level = minecraft.level;
		java.awt.EventQueue.invokeLater(() -> {
			try {
				java.awt.FileDialog dialog = new java.awt.FileDialog((java.awt.Frame) null, "Import policy", java.awt.FileDialog.LOAD);
				dialog.setFilenameFilter((directory, filename) -> filename.toLowerCase(Locale.ROOT).endsWith(".pt"));
				dialog.setVisible(true);
				String file = dialog.getFile();
				String directory = dialog.getDirectory();
				dialog.dispose();
				if (file != null && directory != null) minecraft.execute(() -> {
					if (minecraft.gui.screen() == this && minecraft.level == level) {
						RosieLauncher.start(minecraft, "import", "latest", java.nio.file.Path.of(directory, file).toString());
					}
				});
			} catch (RuntimeException error) {
				minecraft.execute(() -> { status = "File picker: " + error.getMessage(); });
			}
		});
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
			JsonObject environment = arena ? compatibleEnvironment() : null;
			JsonObject[] fighters = new JsonObject[2];
			JsonObject[] policyEntries = new JsonObject[2];
			for (int index = 0; index < 2; index++) {
				policyEntries[index] = modes[index].equals("Policy") ? library.find(library.policies, policyIds[index]) : null;
				fighters[index] = modes[index].equals("Policy")
						? arena ? library.policyFighter(policyEntries[index], environment)
								: library.policyFighter(policyEntries[index], policyEntries[index].get("contract_hash").getAsString())
						: DuelLibrary.scriptedFighter(modes[index].equals("Basic melee") ? "basic_melee" : "stationary_target", scriptOptions[index]);
			}
			if (arena) library.saveDuel(environment, fighters[0], fighters[1], maxTicks);
			else library.saveWorldDuel(library.worldPolicy(policyEntries[0], policyEntries[1]), fighters[0], fighters[1], loadouts[0], loadouts[1], maxTicks);
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
		var savedLoadouts = new com.google.gson.JsonArray();
		for (JsonObject loadout : loadouts) savedLoadouts.add(loadout.deepCopy());
		saved.add("loadouts", savedLoadouts);
		try { DuelLibrary.write(library.root.resolve("selections.json"), saved); }
		catch (IOException error) { status = "Could not save selections: " + error.getMessage(); }
	}

	private void choose(String title, List<JsonObject> entries, Consumer<JsonObject> selected) {
		minecraft.gui.setScreen(new ChoiceScreen(title, entries, entry -> { selected.accept(entry); status = ""; }));
	}

	private void chooseEnvironment() {
		List<JsonObject> choices = compatibleEnvironmentChoices();
		var families = new java.util.LinkedHashMap<String, List<JsonObject>>();
		for (JsonObject choice : choices) {
			JsonObject environment = choice.getAsJsonObject("environment");
			String family = environment.has("name") && !environment.get("name").getAsString().isBlank()
					? environment.get("name").getAsString() : name(choice);
			families.computeIfAbsent(family, ignored -> new ArrayList<>()).add(choice);
		}
		List<JsonObject> familyEntries = new ArrayList<>();
		families.forEach((family, variants) -> {
			JsonObject entry = new JsonObject();
			entry.addProperty("id", family);
			entry.addProperty("name", family + " (" + variants.size() + ")");
			entry.addProperty("family", family);
			familyEntries.add(entry);
		});
		minecraft.gui.setScreen(new ChoiceScreen("Environment families", familyEntries, (familyEntry, familyScreen) -> {
			String family = familyEntry.get("family").getAsString();
			List<JsonObject> variants = families.get(family).stream().map(choice -> {
				JsonObject entry = choice.deepCopy();
				String label = name(entry);
				String prefix = family + " / ";
				if (label.startsWith(prefix)) label = label.substring(prefix.length());
				else if (label.equals(family)) label = "Default";
				entry.addProperty("name", label);
				return entry;
			}).toList();
			minecraft.gui.setScreen(new ChoiceScreen(family, variants, (entry, screen) -> {
				environmentId = identity(entry);
				persist();
				status = "";
			}, familyScreen, ControlPanel.this));
		}, ControlPanel.this, null));
	}

	private JsonObject environment() { return library == null ? null : library.find(library.environmentChoices(environmentId), environmentId); }

	private List<String> selectedPolicyHashes() {
		if (library == null) return List.of();
		List<String> hashes = new ArrayList<>();
		for (int index = 0; index < 2; index++) {
			JsonObject policy = modes[index].equals("Policy") ? library.find(library.policies, policyIds[index]) : null;
			if (policy != null) hashes.add(policy.get("contract_hash").getAsString());
		}
		return hashes;
	}

	private List<JsonObject> compatibleEnvironmentChoices() {
		return library == null ? List.of() : library.environmentChoices(environmentId, selectedPolicyHashes());
	}

	private JsonObject compatibleEnvironment() {
		return library == null ? null : library.find(compatibleEnvironmentChoices(), environmentId);
	}

	private boolean canStartWorld() {
		if (library == null || maxTicks < 1 || !LiveDuel.worldSpawnsReady(minecraft)) return false;
		JsonObject[] entries = new JsonObject[2];
		try {
			for (int index = 0; index < 2; index++) {
				entries[index] = modes[index].equals("Policy") ? library.find(library.policies, policyIds[index]) : null;
				if (modes[index].equals("Policy") && entries[index] == null) return false;
			}
			JsonObject policy = library.worldPolicy(entries[0], entries[1]);
			String contractHash = library.worldPolicyHash(entries[0], entries[1]);
			var contract = new SimulationContract(policy);
			List<JsonObject> layouts = new ArrayList<>();
			for (int index = 0; index < 2; index++) {
				layouts.add(DuelLibrary.normalizeLoadout(loadouts[index]));
				if (entries[index] != null) library.policyFighter(entries[index], contractHash);
			}
			contract.validateLoadouts(layouts);
			return true;
		} catch (IOException | RuntimeException invalid) { return false; }
	}

	private boolean canStartArena(JsonObject environment) {
		if (library == null) return false;
		try {
			for (int index = 0; index < 2; index++) {
				if (modes[index].equals("Policy")) library.policyFighter(library.find(library.policies, policyIds[index]), environment);
			}
			return true;
		} catch (IOException | RuntimeException invalid) { return false; }
	}

	private static String loadoutSummary(JsonObject loadout) {
		List<String> items = new ArrayList<>();
		for (var weapon : loadout.getAsJsonArray("weapons")) if (!weapon.getAsString().equals("minecraft:air")) items.add(shortItem(weapon.getAsString()));
		if (!loadout.get("offhand").getAsString().equals("minecraft:air")) items.add(shortItem(loadout.get("offhand").getAsString()));
		for (String slot : SimulationContract.ARMOR_SLOTS) {
			String item = loadout.getAsJsonObject("armor").get(slot).getAsString();
			if (!item.equals("minecraft:air")) items.add(shortItem(item));
		}
		return items.isEmpty() ? "empty" : String.join(" + ", items);
	}
	private static String identity(JsonObject entry) { return entry.get("id").getAsString(); }
	private static String name(JsonObject entry) { return entry.get("name").getAsString(); }

	private static String environmentVariant(JsonObject entry) {
		JsonObject environment = entry.getAsJsonObject("environment");
		List<String> details = new ArrayList<>();
		JsonObject task = environment.has("task") ? environment.getAsJsonObject("task") : environment;
		if (task.has("max_ticks")) details.add(task.get("max_ticks").getAsString() + " ticks");
		if (environment.has("bounds") && environment.getAsJsonObject("bounds").has("size")) {
			var size = environment.getAsJsonObject("bounds").getAsJsonArray("size");
			details.add(size.get(0).getAsString() + "x" + size.get(1).getAsString() + "x" + size.get(2).getAsString());
		} else if (environment.has("arena") && environment.get("arena").isJsonObject()
				&& environment.getAsJsonObject("arena").has("width")) {
			details.add(environment.getAsJsonObject("arena").get("width").getAsString() + " wide");
		}
		JsonObject players = environment.has("players") && environment.get("players").isJsonObject()
				? environment.getAsJsonObject("players") : null;
		JsonObject player = players == null ? null : players.has("shared") ? players.getAsJsonObject("shared") : null;
		if (player == null && environment.has("players") && environment.get("players").isJsonArray()
				&& !environment.getAsJsonArray("players").isEmpty()) {
			player = environment.getAsJsonArray("players").get(0).getAsJsonObject();
		}
		JsonObject loadout = player != null && player.has("loadout") ? player.getAsJsonObject("loadout") :
				environment.has("loadout") ? environment.getAsJsonObject("loadout") : null;
		if (loadout != null) {
			List<String> gear = new ArrayList<>();
			if (loadout.has("weapon")) {
				String weapon = loadout.get("weapon").getAsString();
				if (!weapon.equals("minecraft:air")) gear.add(shortItem(weapon));
			} else if (loadout.has("weapons")) {
				for (var weapon : loadout.getAsJsonArray("weapons")) {
					if (weapon.isJsonPrimitive() && !weapon.getAsString().equals("minecraft:air")) gear.add(shortItem(weapon.getAsString()));
				}
			}
			if (loadout.has("offhand") && !loadout.get("offhand").getAsString().equals("minecraft:air")) {
				gear.add(shortItem(loadout.get("offhand").getAsString()));
			}
			if (loadout.has("armor") && loadout.get("armor").isJsonObject()) {
				for (var armor : loadout.getAsJsonObject("armor").entrySet()) {
					if (armor.getValue().isJsonPrimitive() && !armor.getValue().getAsString().equals("minecraft:air")) {
						gear.add(shortItem(armor.getValue().getAsString()));
					}
				}
			}
			if (!gear.isEmpty()) details.add(String.join("+", gear));
		}
		return String.join(" / ", details);
	}

	private static String shortItem(String item) {
		return item.substring(item.indexOf(':') + 1);
	}
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
		private final BiConsumer<JsonObject, Screen> selected;
		private final Screen returnScreen;
		private final Screen selectionScreen;
		private String filter = "";
		private int selectedPage;
		private int position;
		private int size;
		private boolean empty;

		ChoiceScreen(String title, List<JsonObject> entries, Consumer<JsonObject> selected) {
			this(title, entries, (entry, screen) -> selected.accept(entry), ControlPanel.this, ControlPanel.this);
		}

		ChoiceScreen(String title, List<JsonObject> entries, BiConsumer<JsonObject, Screen> selected, Screen returnScreen, Screen selectionScreen) {
			super(Component.literal(title));
			this.entries = List.copyOf(entries);
			this.selected = selected;
			this.returnScreen = returnScreen;
			this.selectionScreen = selectionScreen;
		}

		@Override protected void init() {
			size = Math.min(650, width - 32);
			position = (width - size) / 2;
			EditBox search = addRenderableWidget(new EditBox(font, position, 42, size, 20, Component.literal("Search")));
			search.setHint(Component.literal("Search"));
			search.setValue(filter);
			search.setResponder(value -> { filter = value; selectedPage = 0; rebuildWidgets(); });
			setInitialFocus(search);
			List<JsonObject> filtered = entries.stream().filter(entry -> {
				String searchText = name(entry) + (entry.has("environment") ? " " + environmentVariant(entry) : "");
				return searchText.toLowerCase(Locale.ROOT).contains(filter.toLowerCase(Locale.ROOT));
			}).toList();
			empty = filtered.isEmpty();
			int rows = Math.max(1, (height - 122) / 24);
			selectedPage = Math.min(selectedPage, Math.max(0, (filtered.size() - 1) / rows));
			for (int index = selectedPage * rows; index < Math.min(filtered.size(), (selectedPage + 1) * rows); index++) {
				JsonObject entry = filtered.get(index);
				String title = name(entry) + (entry.has("environment") && !environmentVariant(entry).isEmpty()
						? " / " + environmentVariant(entry) : entry.has("update") ? " / update " + entry.get("update").getAsLong() : "");
				addRenderableWidget(Button.builder(Component.literal(fit(title, size - 12)), widget -> {
					selected.accept(entry, this);
					if (selectionScreen != null) minecraft.gui.setScreen(selectionScreen);
				})
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
		@Override public void onClose() { minecraft.gui.setScreen(returnScreen); }
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
			int initial = maxTicks > 0 ? maxTicks : arena && environment != null && environment.getAsJsonObject("environment").has("max_ticks")
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
			if (arena) addRenderableWidget(Button.builder(Component.literal("Environment default"), widget -> { maxTicks = 0; persist(); onClose(); }).bounds(position, 136, size, 20).build());
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

	private final class LoadoutScreen extends Screen {
		private final int fighter;
		private final JsonObject edited;
		private int position;
		private int size;
		private int editorWidth;
		private int columnWidth;
		private int labelWidth;
		private int pickerWidth;
		private int pickerLeft;
		private int pageRows;
		private int fieldPage;
		private int openDropdown = -1;
		private int optionPage;
		private String error = "";

		LoadoutScreen(int fighter) {
			super(Component.literal("Fighter " + (fighter == 0 ? "A" : "B") + " loadout"));
			this.fighter = fighter;
			this.edited = loadouts[fighter].deepCopy();
		}

		@Override protected void init() {
			size = Math.min(900, width - 24);
			position = (width - size) / 2;
			pickerWidth = openDropdown >= 0 ? Math.min(300, Math.max(160, size / 3)) : 0;
			editorWidth = size - (pickerWidth == 0 ? 0 : pickerWidth + 12);
			pickerLeft = position + editorWidth + 12;
			columnWidth = (editorWidth - 12) / 2;
			labelWidth = Math.min(90, Math.max(32, columnWidth / 3));
			pageRows = Math.max(1, Math.min(8, (height - 150) / 24));
			int fieldPages = (8 + pageRows - 1) / pageRows;
			fieldPage = Math.max(0, Math.min(fieldPage, fieldPages - 1));
			int buttonWidth = columnWidth - labelWidth - 6;
			int visibleRows = Math.min(pageRows, 8 - fieldPage * pageRows);
			for (int row = 0; row < visibleRows; row++) {
				int fieldRow = fieldPage * pageRows + row;
				for (int column = 0; column < 2; column++) {
					int field = column * 8 + fieldRow;
					if (field >= 15) continue;
					int fieldX = position + column * (columnWidth + 12);
					int fieldY = 52 + row * 24;
					int dropdown = field;
					String label = fieldName(field);
					String value = fieldDisplay(field);
					addRenderableWidget(Button.builder(Component.literal(fit(value, buttonWidth - 12)), widget -> {
						openDropdown = openDropdown == dropdown ? -1 : dropdown;
						optionPage = 0;
						rebuildWidgets();
					})
						.bounds(fieldX + labelWidth, fieldY, buttonWidth, 20)
						.tooltip(Tooltip.create(Component.literal(label + ": " + value)))
						.build());
				}
			}
			if (fieldPages > 1) {
				int navigationY = height - 58;
				addRenderableWidget(Button.builder(Component.literal("Previous"), widget -> {
					fieldPage--;
					openDropdown = -1;
					rebuildWidgets();
				}).bounds(position, navigationY, 90, 20).build()).active = fieldPage > 0;
				addRenderableWidget(Button.builder(Component.literal("Next"), widget -> {
					fieldPage++;
					openDropdown = -1;
					rebuildWidgets();
				}).bounds(position + editorWidth - 90, navigationY, 90, 20).build()).active = fieldPage + 1 < fieldPages;
			}
			int half = (size - 8) / 2;
			addRenderableWidget(Button.builder(Component.literal("Apply"), widget -> apply()).bounds(position, height - 30, half, 20).build());
			addRenderableWidget(Button.builder(Component.literal("Cancel"), widget -> onClose()).bounds(position + half + 8, height - 30, half, 20).build());
			if (openDropdown >= 0) addDropdownWidgets();
		}

		private String fieldName(int field) {
			if (field < 9) return "Hotbar " + (field + 1);
			if (field == 9) return "Selected slot";
			if (field == 10) return "Offhand";
			String slot = SimulationContract.ARMOR_SLOTS.get(field - 11);
			return slot.substring(0, 1).toUpperCase(Locale.ROOT) + slot.substring(1);
		}

		private String fieldValue(int field) {
			JsonArray weapons = edited.getAsJsonArray("weapons");
			if (field < 9) return weapons.get(field).getAsString();
			if (field == 9) return Integer.toString(edited.get("selected").getAsInt());
			if (field == 10) return edited.get("offhand").getAsString();
			return edited.getAsJsonObject("armor").get(SimulationContract.ARMOR_SLOTS.get(field - 11)).getAsString();
		}

		private String fieldDisplay(int field) {
			return field == 9 ? "Slot " + (Integer.parseInt(fieldValue(field)) + 1) : shortItem(fieldValue(field));
		}

		private List<String> fieldChoices(int field) {
			if (field < 9) return SimulationContract.weaponChoices();
			if (field == 9) return java.util.stream.IntStream.range(0, 9).mapToObj(Integer::toString).toList();
			if (field == 10) return List.of("minecraft:air", "minecraft:shield");
			return SimulationContract.armorChoices(SimulationContract.ARMOR_SLOTS.get(field - 11));
		}

		private void chooseFieldValue(int field, String value) {
			if (field < 9) edited.getAsJsonArray("weapons").set(field, ReplayAnalysis.JSON.toJsonTree(value));
			else if (field == 9) edited.addProperty("selected", Integer.parseInt(value));
			else if (field == 10) edited.addProperty("offhand", value);
			else edited.getAsJsonObject("armor").addProperty(SimulationContract.ARMOR_SLOTS.get(field - 11), value);
		}

		private void addDropdownWidgets() {
			List<String> choices = fieldChoices(openDropdown);
			int visibleRows = Math.max(1, Math.min(8, (height - 150) / 22));
			int pageCount = (choices.size() + visibleRows - 1) / visibleRows;
			optionPage = Math.max(0, Math.min(optionPage, pageCount - 1));
			int first = optionPage * visibleRows;
			int rows = Math.min(visibleRows, choices.size() - first);
			for (int row = 0; row < rows; row++) {
				String value = choices.get(first + row);
				int optionIndex = first + row;
				int y = 78 + row * 22;
				String display = fieldDisplayForChoice(openDropdown, value);
				addRenderableWidget(Button.builder(Component.literal(fit(display, pickerWidth - 12)), widget -> {
					chooseFieldValue(openDropdown, choices.get(optionIndex));
					openDropdown = -1;
					rebuildWidgets();
				}).bounds(pickerLeft, y, pickerWidth, 20)
					.tooltip(Tooltip.create(Component.literal(value)))
					.build());
			}
			if (pageCount > 1) {
				int navigationY = 78 + rows * 22 + 4;
				int half = (pickerWidth - 4) / 2;
				addRenderableWidget(Button.builder(Component.literal("Previous"), widget -> {
					optionPage--;
					rebuildWidgets();
				}).bounds(pickerLeft, navigationY, half, 20).build()).active = optionPage > 0;
				addRenderableWidget(Button.builder(Component.literal("Next"), widget -> {
					optionPage++;
					rebuildWidgets();
				}).bounds(pickerLeft + half + 4, navigationY, pickerWidth - half - 4, 20).build()).active = optionPage + 1 < pageCount;
			}
		}

		private void apply() {
			try {
				JsonObject player = new JsonObject();
				player.add("loadout", edited.deepCopy());
				loadouts[fighter] = SimulationContract.loadout(player, "configured");
				persist();
				status = "";
				onClose();
			} catch (IllegalArgumentException invalid) { error = invalid.getMessage(); }
		}

		@Override public void extractRenderState(GuiGraphicsExtractor graphics, int mouseX, int mouseY, float delta) {
			graphics.fill(0, 0, width, height, 0xF0181E20);
			graphics.text(font, title, position, 22, 0xFF92DDB3);
			int visibleRows = Math.min(pageRows, 8 - fieldPage * pageRows);
			for (int row = 0; row < visibleRows; row++) {
				int fieldRow = fieldPage * pageRows + row;
				for (int column = 0; column < 2; column++) {
					int field = column * 8 + fieldRow;
					if (field >= 15) continue;
					int fieldX = position + column * (columnWidth + 12);
					graphics.text(font, fit(fieldName(field), labelWidth), fieldX, 58 + row * 24, 0xFFB5C2BE);
				}
			}
			if (openDropdown >= 0) drawDropdownBackground(graphics);
			super.extractRenderState(graphics, mouseX, mouseY, delta);
			int fieldPages = (8 + pageRows - 1) / pageRows;
			if (fieldPages > 1) graphics.text(font, "Page " + (fieldPage + 1) + " / " + fieldPages,
					position + (size - font.width("Page " + (fieldPage + 1) + " / " + fieldPages)) / 2, height - 52, 0xFFB5C2BE);
			graphics.text(font, fit(error, size), position, height - 78, 0xFFF3B298);
		}

		private void drawDropdownBackground(GuiGraphicsExtractor graphics) {
			graphics.fill(pickerLeft - 6, 42, pickerLeft + pickerWidth + 6, height - 64, 0xFF07100F);
			graphics.text(font, fit("Choose " + fieldName(openDropdown), pickerWidth), pickerLeft, 54, 0xFF92DDB3);
		}

		private String fieldDisplayForChoice(int field, String value) {
			return field == 9 ? "Slot " + (Integer.parseInt(value) + 1) : shortItem(value);
		}
		@Override public boolean isPauseScreen() { return false; }
		@Override public void onClose() { minecraft.gui.setScreen(ControlPanel.this); }
	}

	private final class ScriptScreen extends Screen {
		private final int fighter;
		private final String mode;
		private EditBox interval;
		private EditBox distance;
		private EditBox strafeInterval;
		private EditBox axe;
		private EditBox sword;
		private EditBox changeInterval;
		private EditBox wallMargin;
		private boolean strafe;
		private boolean sprint;
		private boolean shield;
		private String tacticalStyle;
		private int position;
		private int column;
		private String error = "";

		ScriptScreen(int fighter) {
			super(Component.literal(modes[fighter] + " / Fighter " + (fighter == 0 ? "A" : "B")));
			this.fighter = fighter;
			this.mode = modes[fighter];
			JsonObject options = scriptOptions[fighter];
			strafe = options.has("strafe") && options.get("strafe").getAsBoolean();
			sprint = options.has("sprint") && options.get("sprint").getAsBoolean();
			shield = options.has("shield") && options.get("shield").getAsBoolean();
			tacticalStyle = options.has("style") ? options.get("style").getAsString() : "evasive";
		}

		@Override protected void init() {
			int size = Math.min(480, width - 32);
			position = (width - size) / 2;
			column = (size - 12) / 2;
			JsonObject options = scriptOptions[fighter];
			if (mode.equals("Random walk")) {
				changeInterval = field(position, 76, column, "Change interval", options, "change_interval", "20");
				wallMargin = field(position + column + 12, 76, column, "Wall margin", options, "wall_margin", "1.25");
			} else {
				interval = field(position, 76, column, "Attack interval", options, "attack_interval", "13");
				distance = field(position + column + 12, 76, column, "Melee distance", options, "melee_distance", "3");
				if (mode.equals("Basic melee") || mode.equals("Tactical melee")) {
					strafeInterval = field(position, 116, column, "Strafe interval", options, "strafe_interval", "20");
					addWeaponSlotFields(position + column + 12, options);
					addRenderableWidget(CycleButton.onOffBuilder(strafe).create(position, 148, column, 20, Component.literal("Strafe"), (widget, value) -> strafe = value));
					addRenderableWidget(CycleButton.onOffBuilder(sprint).create(position + column + 12, 148, column, 20, Component.literal("Sprint"), (widget, value) -> sprint = value));
					if (mode.equals("Tactical melee")) {
						addRenderableWidget(CycleButton.onOffBuilder(shield).create(position, 174, column, 20, Component.literal("Shield"), (widget, value) -> shield = value));
						addRenderableWidget(CycleButton.builder(Component::literal, tacticalStyle).withValues("evasive", "aggressive", "patient").displayOnlyValue()
								.create(position + column + 12, 174, column, 20, Component.literal("Style"), (widget, value) -> tacticalStyle = value));
					} else {
						addRenderableWidget(CycleButton.onOffBuilder(shield).create(position, 174, column, 20, Component.literal("Shield"), (widget, value) -> shield = value));
					}
				} else {
					addWeaponSlotFields(position, options);
					addRenderableWidget(CycleButton.onOffBuilder(shield).create(position, 148, column, 20, Component.literal("Shield"), (widget, value) -> shield = value));
				}
			}
			addRenderableWidget(Button.builder(Component.literal("Apply"), widget -> apply()).bounds(position, height - 30, column, 20).build());
			addRenderableWidget(Button.builder(Component.literal("Cancel"), widget -> onClose()).bounds(position + column + 12, height - 30, column, 20).build());
		}

		private void addWeaponSlotFields(int left, JsonObject options) {
			int slotWidth = (column - 4) / 2;
			axe = field(left, 116, slotWidth, "Axe slot", options, "axe_slot", "");
			sword = field(left + slotWidth + 4, 116, slotWidth, "Sword slot", options, "sword_slot", "");
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
				if (mode.equals("Random walk")) {
					options.addProperty("change_interval", positive(changeInterval.getValue()));
					double margin = Double.parseDouble(wallMargin.getValue());
					if (!Double.isFinite(margin) || margin <= 0 || margin >= 10) throw new IllegalArgumentException("Wall margin must be between 0 and 10");
					options.addProperty("wall_margin", margin);
				} else {
					options.addProperty("attack_interval", positive(interval.getValue()));
					double reach = Double.parseDouble(distance.getValue());
					if (!Double.isFinite(reach) || reach <= 0 || reach >= 10) throw new IllegalArgumentException("Melee distance must be between 0 and 10");
					options.addProperty("melee_distance", reach);
					if (mode.equals("Basic melee") || mode.equals("Tactical melee")) {
						options.addProperty("strafe_interval", positive(strafeInterval.getValue()));
						options.addProperty("strafe", strafe);
						options.addProperty("sprint", sprint);
					}
					options.addProperty("shield", shield);
					if (mode.equals("Tactical melee")) options.addProperty("style", tacticalStyle);
					if (!axe.getValue().isBlank() || !sword.getValue().isBlank()) {
						if (axe.getValue().isBlank() || sword.getValue().isBlank()) throw new IllegalArgumentException("Set both axe and sword slots");
						int axeSlot = Integer.parseInt(axe.getValue());
						int swordSlot = Integer.parseInt(sword.getValue());
						if (axeSlot < 0 || axeSlot > 8 || swordSlot < 0 || swordSlot > 8 || axeSlot == swordSlot) throw new IllegalArgumentException("Choose distinct axe/sword slots from 0 to 8");
						options.addProperty("axe_slot", axeSlot);
						options.addProperty("sword_slot", swordSlot);
					}
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
			if (mode.equals("Random walk")) {
				graphics.text(font, "Change interval (ticks)", position, 62, 0xFFB5C2BE);
				graphics.text(font, "Wall margin (blocks)", position + column + 12, 62, 0xFFB5C2BE);
			} else {
				graphics.text(font, "Attack interval (ticks)", position, 62, 0xFFB5C2BE);
				graphics.text(font, "Melee distance (blocks)", position + column + 12, 62, 0xFFB5C2BE);
				if (mode.equals("Basic melee") || mode.equals("Tactical melee")) {
					graphics.text(font, "Strafe interval (ticks)", position, 103, 0xFFB5C2BE);
					graphics.text(font, "Axe / sword slots", position + column + 12, 103, 0xFFB5C2BE);
				} else graphics.text(font, "Axe / sword slots", position, 103, 0xFFB5C2BE);
			}
			graphics.text(font, fit(error, column * 2 + 12), position, 48, 0xFFF3B298);
			super.extractRenderState(graphics, mouseX, mouseY, delta);
		}
		@Override public boolean isPauseScreen() { return false; }
		@Override public void onClose() { minecraft.gui.setScreen(ControlPanel.this); }
	}
}