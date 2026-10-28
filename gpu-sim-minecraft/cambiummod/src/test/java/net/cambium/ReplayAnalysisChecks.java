package net.cambium;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import net.minecraft.world.phys.Vec3;

public final class ReplayAnalysisChecks {

	public static void main(String[] args) throws Exception {
		if (args.length == 2 && args[0].equals("--replay")) {
			checkReplay(Path.of(args[1]));
			return;
		}
		if (args.length == 1 || args.length == 3) {
			try (LivePolicy policy = new LivePolicy(Integer.parseInt(args[0]))) {
				JsonObject info = policy.info().get(5, java.util.concurrent.TimeUnit.SECONDS);
				if (info.has("contract") && info.getAsJsonObject("contract").has("requires")
						&& info.getAsJsonObject("contract").getAsJsonArray("requires").contains(ReplayAnalysis.JSON.toJsonTree("bow"))) {
					require(policy.fullBowBridgeReady(), "HTTP bridge must negotiate the complete bow observations, use_item choices, and finite bow loadouts");
				}
				ReplayTrace trace = ReplayTrace.forLive(info.getAsJsonObject("environment"));
				require(trace.frames.getFirst().actors().size() == 2, "Policy environment must produce two initial actors");
				JsonObject response = policy.act("java-python-check", 42, new float[2][policy.observationCount()]).get(5, java.util.concurrent.TimeUnit.SECONDS);
				require(policy.mapActions(response, "java-python-check", 42).length == 2, "Python must return two valid mapped controls");
				response.addProperty("contract_hash", "0".repeat(64));
				boolean rejected = false;
				try { policy.mapActions(response, "java-python-check", 42); } catch (IllegalArgumentException expected) { rejected = true; }
				require(rejected, "Negotiated contract fingerprint must be checked on responses");
				System.out.println("PASS: Java HTTP -> Python checkpoint inference -> validated Java actions and checkpoint arena");
			}
			if (args.length == 3) checkReplay(Path.of(args[2]));
			return;
		}
		checkSprintInput();
		checkMinorCollision();
		checkRosieLauncher();
		checkBundledInference();
		checkWarmInference();
		checkDuelLibrary();
		for (int width : new int[]{320, 426, 640, 960, 1280}) {
			DuelHud.Layout layout = DuelHud.layout(width);
			require(layout.left() + layout.fighterWidth() + 8 <= layout.center() && layout.center() + layout.centerWidth() + 8 <= layout.right(), "Player and shared HUD columns must never overlap");
			require(layout.right() + layout.fighterWidth() <= width - 8, "HUD must stay inside the viewport");
		}
		require(DuelHud.time(1201).equals("1:01") && DuelHud.time(-1).equals("0:00"), "Round time must clamp at zero and round up partial seconds");
		require(DuelHud.fraction(40, 20) == 1 && DuelHud.fraction(-1, 20) == 0, "HUD bars must stay within bounds");
		var replayFood = new ReplayPlayer.ReplayFoodData();
		replayFood.setFoodLevel(12);
		replayFood.setSaturation(2.0F);
		replayFood.addExhaustion(40.0F);
		replayFood.tick(null);
		require(!replayFood.live && replayFood.getFoodLevel() == 12 && replayFood.getSaturationLevel() == 2.0F,
				"Replay food must stay frozen without consulting the world's natural-regeneration rule");
		System.out.println("PASS: replay food profile is frozen independently of world rules");
		checkLivePolicy();
		checkDescriptiveContract();
		checkBowContract();
		checkBowReplay();
		BoundaryContractChecks.check();
		checkPolicyFixtures();
		EquipmentObservationChecks.check();
		require(CambiumFabric.BUILD_ID.matches(".*source=[0-9a-f]{64} built=.+"), "Mod build identity must be packaged");
		Path folder = Files.createTempDirectory("cambium-analysis-checks-");
		try {
			JsonObject metadata = new JsonObject();
			metadata.addProperty("backend", "cuda");
			ReplayAnalysis report = new ReplayAnalysis(folder, metadata);
			JsonArray expected = actors();
			report.record(1, expected, expected.deepCopy(), new JsonObject());
			require(report.firstDivergentTick() == -1, "Identical states must not diverge");
			JsonArray actual = expected.deepCopy();
			actual.get(0).getAsJsonObject().getAsJsonArray("position").set(0, ReplayAnalysis.JSON.toJsonTree(0.01));
			actual.get(1).getAsJsonObject().getAsJsonObject("outcome").addProperty("target", -1);
			report.record(2, expected, actual, new JsonObject());
			require(report.firstDivergentTick() == 2, "First divergence must use transition tick");
			report.finish("complete", "end_of_replay");
			JsonObject summary = JsonParser.parseString(Files.readString(report.summaryPath)).getAsJsonObject();
			require(summary.get("compared_ticks").getAsInt() == 2, "Frame zero must not count as a transition");
			require(summary.get("divergent_actor_ticks").getAsInt() == 2, "Count actors once, not once per differing field");
			require(summary.getAsJsonObject("first_by_category").has("movement")
					&& summary.getAsJsonObject("first_by_category").has("targeting"), "Separate movement from targeting");
			JsonObject position = summary.getAsJsonObject("fields").getAsJsonObject("0.position.x");
			require(position.get("mean_absolute_error").getAsDouble() == 0.005
					&& position.get("max_absolute_error").getAsDouble() == 0.01
					&& position.get("worst_tick").getAsInt() == 2, "Aggregate error statistics must include matching ticks");
			var rows = Files.readAllLines(report.detailsPath);
			require(rows.size() == 4 && JsonParser.parseString(rows.get(0)).getAsJsonObject().get("type").getAsString().equals("header")
					&& JsonParser.parseString(rows.get(3)).getAsJsonObject().get("type").getAsString().equals("summary"),
					"Detailed export must contain provenance, all compared ticks and final summary");
			ReplayAnalysis partial = new ReplayAnalysis(folder, metadata);
			partial.finish("partial", "reset_or_stop");
			require(!partial.detailsPath.equals(report.detailsPath), "Exports must not overwrite prior analysis");
			JsonObject interrupted = JsonParser.parseString(Files.readString(partial.summaryPath)).getAsJsonObject();
			require(interrupted.get("status").getAsString().equals("partial")
					&& interrupted.get("first_difference").isJsonNull(), "An interrupted empty run is partial, not a parity pass");
			ReplayAnalysis tolerance = new ReplayAnalysis(folder, metadata);
			actual = expected.deepCopy();
			actual.get(0).getAsJsonObject().getAsJsonArray("position").set(0, ReplayAnalysis.JSON.toJsonTree(5.0E-7));
			tolerance.record(1, expected, actual, new JsonObject());
			require(tolerance.firstDivergentTick() == -1, "Sub-tolerance floating point error must not flag a mismatch");
			tolerance.finish("complete", "end_of_replay");
			System.out.println("PASS: replay analysis matching, divergence, summaries, tolerances and partial exports");
		} finally {
			try (var files = Files.walk(folder)) {
				for (Path path : files.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
			}
		}
	}

	private static void checkRosieLauncher() {
		Path root = Path.of(System.getProperty("java.io.tmpdir")).resolve("Minecraft with spaces").toAbsolutePath();
		JsonObject config = new JsonObject();
		config.addProperty("python", root.resolve("venv with spaces/python.exe").toString());
		config.addProperty("repository", root.resolve("checkout with spaces").toString());
		ProcessBuilder builder = RosieLauncher.helper(config, root, "replay", "run;not-a-shell", "sample.jsonl");
		require(builder.command().getFirst().equals(config.get("python").getAsString()), "Python executable must remain one argument");
		require(builder.command().contains(root.toString()) && builder.command().contains("run;not-a-shell"), "Helper arguments must be literal, including spaces and shell metacharacters");
		require(builder.command().contains("--json") && !builder.redirectErrorStream(), "Helper stdout must remain a separate JSON event stream");
		require(builder.command().contains("--connect"), "Remote commands must automatically open or reuse the terminal connection");
		require(builder.directory().toPath().equals(root.resolve("checkout with spaces")), "Module must start from configured repository");
		for (String scope : java.util.List.of("policy", "latest", "all")) {
			var sync = RosieLauncher.helper(config, root, "sync", "run", "latest", scope).command();
			require(sync.get(sync.indexOf("--scope") + 1).equals(scope) && sync.get(sync.indexOf("--workers") + 1).equals("3"), "Sync scopes and bounded concurrency must reach the helper unchanged");
		}
		for (String action : java.util.List.of("local", "local-world")) {
			ProcessBuilder local = RosieLauncher.helper(config, root, action, "latest", "latest");
			require(local.command().get(5).equals("local") && local.command().contains("--offline"), "Local launch must select offline mode, including world mode");
			require(!local.command().contains("--connect"), "Offline launches must never open an SSH login terminal");
			int argument = local.command().indexOf("--duel");
			require(argument >= 0 && local.command().get(argument + 1).equals(root.resolve("sim-rosie-cache/duel.json").toString()), "Offline duel path must remain one literal argument");
		}
		String checkpoint = root.resolve("my policy;test.pt").toString();
		var imported = RosieLauncher.helper(config, root, "import", "latest", checkpoint).command();
		require(imported.contains("--offline") && !imported.contains("--connect"), "Import must not connect to Rosie");
		require(imported.get(imported.indexOf("--checkpoint") + 1).equals(checkpoint), "Import path must remain a literal argument");
		var bundled = BundledInference.helper(root.resolve("runtime.exe"), root, "import", checkpoint).command();
		require(bundled.getFirst().equals(root.resolve("runtime.exe").toString()) && bundled.contains("--offline"), "Bundled import must run offline without Python");
		require(bundled.get(bundled.indexOf("--checkpoint") + 1).equals(checkpoint), "Bundled import must preserve the checkpoint path");
		var bundledDuel = BundledInference.helper(root.resolve("runtime.exe"), root, "local-world", "").command();
		require(bundledDuel.get(1).equals("local") && bundledDuel.contains("--duel"), "Bundled world duels must use local inference");
		System.out.println("PASS: Rosie helper uses literal arguments, configured repository and separate JSON output");
	}

	private static void checkBundledInference() throws Exception {
		String checkpoint = System.getenv("CAMBIUM_BUNDLED_CHECKPOINT");
		if (checkpoint == null || checkpoint.isBlank()) return;
		require(BundledInference.available(), "Bundled runtime must be available on this platform");
		Path game = Files.createTempDirectory("cambium-bundled-checks-");
		try {
			Files.createDirectories(game.resolve("logs"));
			Path executable = BundledInference.prepare(game);
			Files.writeString(executable, "corrupted runtime");
			require(BundledInference.prepare(game).equals(executable), "Corrupted extracted runtime must be restored");
			Process process = BundledInference.helper(executable, game, "import", checkpoint).start();
			if (!process.waitFor(120, java.util.concurrent.TimeUnit.SECONDS)) {
				process.descendants().forEach(ProcessHandle::destroyForcibly);
				process.destroyForcibly();
				throw new AssertionError("Bundled import must finish");
			}
			String output = new String(process.getInputStream().readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
			require(process.exitValue() == 0 && output.contains("\"event\": \"imported\""), "Bundled import failed: " + output);
			JsonObject library = JsonParser.parseString(Files.readString(game.resolve("sim-rosie-cache/library.json"))).getAsJsonObject();
			require(library.getAsJsonArray("policies").size() == 1 && library.getAsJsonArray("environments").size() == 1,
					"Bundled import must index both policy and environment");
			System.out.println("PASS: bundled runtime extraction, integrity repair and checkpoint import");
		} finally {
			try (var files = Files.walk(game)) {
				for (Path path : files.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
			}
		}
	}

	private static void checkWarmInference() throws Exception {
		var fields = new java.util.HashMap<String, java.lang.reflect.Field>();
		var previous = new java.util.LinkedHashMap<java.lang.reflect.Field, Object>();
		for (String name : java.util.List.of("worker", "busy", "serving", "reloading", "matchStopped", "reloadGeneration",
				"generation", "reloadToken", "reloadPort", "world", "operation")) {
			var field = RosieLauncher.class.getDeclaredField(name);
			field.setAccessible(true);
			fields.put(name, field);
			previous.put(field, field.get(null));
		}
		try {
			LauncherProcess process = new LauncherProcess();
			fields.get("worker").set(null, process);
			fields.get("busy").set(null, true);
			fields.get("serving").set(null, true);
			fields.get("reloading").set(null, true);
			fields.get("matchStopped").set(null, false);
			fields.get("generation").set(null, 42L);
			fields.get("reloadGeneration").set(null, 7L);
			fields.get("reloadToken").set(null, "fixture-reload-token");
			fields.get("reloadPort").set(null, 12345);
			require(!RosieLauncher.canStartLocalMatch(), "A pending reload must disable starting another match");
			RosieLauncher.stopMatch();
			require(process.isAlive() && fields.get("worker").get(null) == process, "Stop match must retain the same ready inference process");
			require(RosieLauncher.hasInference() && !RosieLauncher.isServing() && RosieLauncher.canStartLocalMatch(), "Stopped inference must be ready for another local match without reporting active actors");
			require(fields.get("reloadGeneration").getLong(null) > 7 && fields.get("generation").getLong(null) == 42, "Stop match must invalidate pending reloads without invalidating the process monitor");
			require(fields.get("reloadToken").get(null).equals("fixture-reload-token") && fields.get("reloadPort").getInt(null) == 12345, "Stop match must retain authenticated reload access");
			RosieLauncher.stopMatch();
			require(process.isAlive() && RosieLauncher.status().contains("inference ready"), "Repeated match stops must leave inference warm");
			var shutdown = RosieLauncher.class.getDeclaredMethod("cancel");
			shutdown.setAccessible(true);
			shutdown.invoke(null);
			require(!process.isAlive() && !RosieLauncher.isBusy() && !RosieLauncher.hasInference(), "Explicit shutdown and lifecycle cleanup must terminate owned inference");
			require(fields.get("reloadToken").get(null) == null && fields.get("reloadPort").getInt(null) == 0, "Shutdown must clear inference credentials");
			LauncherProcess loading = new LauncherProcess();
			fields.get("worker").set(null, loading);
			fields.get("busy").set(null, true);
			RosieLauncher.stopMatch();
			require(!loading.isAlive() && !RosieLauncher.isBusy(), "Stopping before inference is ready must cancel startup rather than retain an unusable helper");
			System.out.println("PASS: warm inference retention, reload cancellation, restart readiness and explicit shutdown");
		} finally {
			for (var entry : previous.entrySet()) entry.getKey().set(null, entry.getValue());
		}
	}

	private static final class LauncherProcess extends Process {
		private boolean alive = true;
		@Override public java.io.OutputStream getOutputStream() { return java.io.OutputStream.nullOutputStream(); }
		@Override public java.io.InputStream getInputStream() { return java.io.InputStream.nullInputStream(); }
		@Override public java.io.InputStream getErrorStream() { return java.io.InputStream.nullInputStream(); }
		@Override public int waitFor() { alive = false; return 0; }
		@Override public int exitValue() { if (alive) throw new IllegalThreadStateException(); return 0; }
		@Override public void destroy() { alive = false; }
		@Override public Process destroyForcibly() { destroy(); return this; }
		@Override public boolean isAlive() { return alive; }
		@Override public java.util.stream.Stream<ProcessHandle> descendants() { return java.util.stream.Stream.empty(); }
	}

	private static void checkDuelLibrary() throws Exception {
		Path game = Files.createTempDirectory("cambium-library-checks-");
		try {
			DuelLibrary library = new DuelLibrary(game);
			require(library.policies.isEmpty() && library.environments.isEmpty(), "An unsynced library must be empty");
			JsonObject environment = new JsonObject();
			environment.addProperty("contract_hash", "a".repeat(64));
			environment.add("policy", new JsonObject());
			environment.add("environment", new JsonObject());
			Path checkpoint = library.root.resolve("run/latest.pt");
			Files.createDirectories(checkpoint.getParent());
			Files.writeString(checkpoint, "fixture");
			JsonObject entry = new JsonObject();
			entry.addProperty("name", "My fighter");
			entry.addProperty("path", checkpoint.toString());
			entry.addProperty("contract_hash", "a".repeat(64));
			entry.add("policy", JsonParser.parseString("""
				{"format":"minecraft_sim_policy","version":1,"minecraft":"26.3","ticks_per_action":1,
				 "observations":[{"feature":"self.health_fraction"}],
				 "actions":[{"name":"weapon","values":[0,2]}]}
				""").getAsJsonObject());
			JsonObject first = library.policyFighter(entry, environment);
			JsonObject meleeOptions = new JsonObject();
			meleeOptions.addProperty("attack_interval", 5);
			meleeOptions.addProperty("strafe", true);
			JsonObject second = DuelLibrary.scriptedFighter("basic_melee", meleeOptions);
			require(second.get("options").equals(meleeOptions), "Scripted fighter options must survive duel serialization");
			for (String[] controller : new String[][]{
					{"Basic melee", "basic_melee"}, {"Stationary melee", "stationary_melee"},
					{"Tactical melee", "tactical_melee"}, {"Random walk", "random_walk"},
					{"Stationary target", "stationary_target"}}) {
				JsonObject scripted = DuelLibrary.scriptedFighter(controller[1], new JsonObject());
				require(DuelLibrary.scriptedController(controller[0]).equals(controller[1])
						&& scripted.get("controller").getAsString().equals(controller[1])
						&& scripted.get("name").getAsString().equals(controller[0]), "Duel panel modes must map to registered scripted controllers");
			}
			boolean unknownController = false;
			try { DuelLibrary.scriptedFighter("unknown", new JsonObject()); }
			catch (IllegalArgumentException expected) { unknownController = true; }
			require(unknownController, "Unknown scripted controllers must be rejected");
			JsonObject duel = JsonParser.parseString(Files.readString(library.saveDuel(environment, first, second))).getAsJsonObject();
			require(duel.getAsJsonArray("fighters").get(0).getAsJsonObject().get("name").getAsString().equals("My fighter"), "Readable fighter names must survive launch");
			require(duel.getAsJsonArray("fighters").get(1).getAsJsonObject().get("type").getAsString().equals("scripted"), "Each side must retain its own controller type");
			JsonObject limited = JsonParser.parseString(Files.readString(library.saveDuel(environment, first, second, 1200))).getAsJsonObject();
			require(limited.get("max_ticks").getAsInt() == 1200 && !environment.has("max_ticks"), "Duration override must not mutate the library environment");
			require(DuelLibrary.durationTicks("60") == 1200 && DuelLibrary.durationTicks("0.05") == 1, "Duration uses twenty controlled ticks per second");
			JsonObject worldLoadout = DuelLibrary.defaultWorldLoadout();
			worldLoadout.getAsJsonArray("weapons").set(2, JsonParser.parseString("\"minecraft:iron_sword\""));
			worldLoadout.addProperty("selected", 2);
			worldLoadout.addProperty("offhand", "minecraft:shield");
			worldLoadout.getAsJsonObject("armor").addProperty("chestplate", "minecraft:diamond_chestplate");
			JsonObject worldDuel = JsonParser.parseString(Files.readString(library.saveWorldDuel(library.worldPolicy(entry, null), first, second,
					worldLoadout, DuelLibrary.defaultWorldLoadout(), 1200))).getAsJsonObject();
			JsonObject worldScenario = worldDuel.getAsJsonObject("environment");
			JsonObject worldPlayer = new JsonObject();
			worldPlayer.add("loadout", worldLoadout.deepCopy());
			require(worldDuel.get("max_ticks").getAsInt() == 1200 && worldDuel.getAsJsonObject("policy").equals(entry.getAsJsonObject("policy")),
					"World duel must carry the policy contract and explicit duration");
			require(worldScenario.getAsJsonArray("blocks").isEmpty() && !worldDuel.has("environment_id"),
					"World duel must use a runtime-only scenario rather than a cached environment preset");
			require(worldScenario.getAsJsonArray("players").get(0).getAsJsonObject().getAsJsonObject("loadout").equals(
					SimulationContract.loadout(worldPlayer, "configured")),
					"World duel must serialize the explicitly selected weapon, offhand and armor layout");
			require(DuelLibrary.defaultWorldLoadout().getAsJsonArray("weapons").size() == 9,
					"World loadouts must expose all nine physical hotbar slots");
			boolean worldDurationRequired = false;
			try { library.saveWorldDuel(library.worldPolicy(entry, null), first, second, worldLoadout, worldLoadout, 0); }
			catch (java.io.IOException expected) { worldDurationRequired = true; }
			require(worldDurationRequired, "World duels must reject an implicit duration");
			for (String invalid : java.util.List.of("0", "-2", "NaN", "Infinity", "50001")) {
				boolean invalidDuration = false;
				try { DuelLibrary.durationTicks(invalid); } catch (IllegalArgumentException expected) { invalidDuration = true; }
				require(invalidDuration, "Invalid round duration must be rejected");
			}
			entry.addProperty("contract_hash", "b".repeat(64));
			boolean rejected = false;
			try { library.policyFighter(entry, environment); } catch (java.io.IOException expected) { rejected = true; }
			require(rejected, "Incompatible policies must fail before launch");
			entry.addProperty("contract_hash", "a".repeat(64));
			entry.addProperty("path", game.resolve("outside.pt").toString());
			Files.writeString(game.resolve("outside.pt"), "outside");
			rejected = false;
			try { library.policyFighter(entry, environment); } catch (java.io.IOException expected) { rejected = true; }
			require(rejected, "Checkpoint selection must remain inside the local library cache");
			JsonObject index = new JsonObject();
			index.addProperty("version", 1);
			JsonArray entries = new JsonArray();
			entry.addProperty("id", "fixture:latest.pt");
			entry.addProperty("run", "fixture");
			entries.add(entry);
			JsonObject snapshot = entry.deepCopy();
			snapshot.addProperty("id", "fixture:snapshots/early.pt");
			snapshot.addProperty("update", 200);
			entries.add(snapshot);
			index.add("policies", entries);
			JsonArray environments = new JsonArray();
			environment.addProperty("id", "fixture:latest.pt");
			environment.addProperty("name", "Fixture / latest environment");
			environment.add("environment", JsonParser.parseString("""
				{"name":"Shared arena","max_ticks":200,"arena":{"width":5},"loadout":{"weapon":"sword"}}
				"""));
			environments.add(environment);
			JsonObject duplicate = environment.deepCopy();
			duplicate.addProperty("id", "another:snapshots/early.pt");
			duplicate.addProperty("name", "Another policy / early environment");
			duplicate.getAsJsonObject("environment").addProperty("name", "Another label");
			environments.add(duplicate);
			for (String field : java.util.List.of("max_ticks", "arena", "loadout", "contract_hash")) {
				JsonObject different = environment.deepCopy();
				different.addProperty("id", "fixture:" + field);
				different.addProperty("name", "Fixture / " + field);
				switch (field) {
					case "max_ticks" -> different.getAsJsonObject("environment").addProperty("max_ticks", 400);
					case "arena" -> different.getAsJsonObject("environment").getAsJsonObject("arena").addProperty("width", 9);
					case "loadout" -> different.getAsJsonObject("environment").getAsJsonObject("loadout").addProperty("weapon", "axe");
					case "contract_hash" -> different.addProperty("contract_hash", "b".repeat(64));
				}
				environments.add(different);
			}
			index.add("environments", environments);
			DuelLibrary.write(library.root.resolve("library.json"), index);
			DuelLibrary reopened = new DuelLibrary(game);
			var choices = reopened.environmentChoices("another:snapshots/early.pt");
			var compatibleChoices = reopened.environmentChoices("another:snapshots/early.pt", java.util.List.of("a".repeat(64)));
			require(choices.size() == 5 && reopened.environments.size() == 6, "Environment choices must group policy copies without removing cached entries or distinct duration, geometry, loadout and contract variants");
			require(compatibleChoices.size() == 4 && compatibleChoices.stream().allMatch(choice -> choice.get("contract_hash").getAsString().equals("a".repeat(64))),
					"Environment candidates must be filtered by selected policy layouts");
			require(choices.getFirst().get("id").getAsString().equals("another:snapshots/early.pt"), "Grouping must preserve an already selected duplicate's identity");
			require(choices.getFirst().get("name").getAsString().equals("Another label"), "Environment choices must use environment names rather than checkpoint labels");
			require(choices.stream().map(choice -> choice.get("name").getAsString()).distinct().count() == choices.size(), "Different setups sharing an environment name must retain distinguishable source labels");
			require(reopened.environments.getFirst().get("name").getAsString().equals("Fixture / latest environment"), "Selector grouping must not modify cached display names");
			require(reopened.runs.size() == 1 && reopened.versions("fixture").size() == 2, "Legacy libraries must group downloaded versions by run");
			require(reopened.versions("fixture").getFirst().get("id").getAsString().endsWith(":latest.pt"), "Latest must appear before named policy versions");
			JsonObject compact = environment.deepCopy();
			compact.add("sources", JsonParser.parseString("""
				[{"id":"fixture:latest.pt","name":"Fixture / latest environment","run":"fixture"},
				 {"id":"another:snapshots/early.pt","name":"Another policy / early environment","run":"another"}]
				""").getAsJsonArray());
			JsonArray compactEntries = new JsonArray();
			compactEntries.add(compact);
			JsonObject compactVariant = compact.deepCopy();
			compactVariant.addProperty("id", "fixture:variant");
			compactVariant.getAsJsonObject("environment").addProperty("max_ticks", 400);
			compactVariant.add("sources", JsonParser.parseString("""
				[{"id":"fixture:variant","name":"Fixture / duration variant","run":"fixture"}]
				""").getAsJsonArray());
			compactEntries.add(compactVariant);
			index.add("environments", compactEntries);
			DuelLibrary.write(library.root.resolve("library.json"), index);
			DuelLibrary compactLibrary = new DuelLibrary(game);
			var compactChoices = compactLibrary.environmentChoices("another:snapshots/early.pt");
			JsonObject selectedCompact = compactChoices.stream()
					.filter(choice -> choice.get("id").getAsString().equals("another:snapshots/early.pt")).findFirst().orElseThrow();
			require(compactChoices.size() == 2,
					"Compacted environments with distinct durations must remain separate");
			require(selectedCompact.get("id").getAsString().equals("another:snapshots/early.pt"),
					"Compacted environment aliases must resolve persisted selection IDs");
			require(selectedCompact.get("name").getAsString().equals("Shared arena / Another policy / early environment"),
					"Aliased source names must disambiguate matching environment names");
			require(compactLibrary.find(compactChoices, "another:snapshots/early.pt") == selectedCompact,
					"Persisted compacted selections must resolve to a duel environment");
			reopened.rename(reopened.runs.getFirst(), "My local run");
			reopened.rename(reopened.policies.getFirst(), " My saved alias ");
			require(new DuelLibrary(game).runs.getFirst().get("name").getAsString().equals("My local run"), "Run display aliases must persist without renaming run IDs");
			require(new DuelLibrary(game).policies.getFirst().get("name").getAsString().equals("My saved alias"), "Aliases must be trimmed and survive reopening Minecraft");
			require(Files.readString(library.root.resolve("library.json")).contains("\n  \"policies\""), "The local library must remain human-readable after renaming");
			System.out.println("PASS: local library and independent duel configuration");
		} finally {
			try (var paths = Files.walk(game)) {
				for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
			}
		}
	}

	private static void checkReplay(Path path) throws Exception {
		ReplayTrace trace = ReplayTrace.read(path);
		require(trace.frames.size() > 1 && trace.environment.loadouts.size() == 2, "Native replay must carry a resolved playable scenario");
		ReplayTrace.forLive(trace.header.getAsJsonObject("environment"));
		Path invalid = Files.createTempFile("cambium-invalid-replay-", ".jsonl");
		try {
			for (boolean missing : new boolean[]{false, true}) {
				var lines = Files.readAllLines(path);
				JsonObject frame = JsonParser.parseString(lines.get(1)).getAsJsonObject();
				JsonObject actor = frame.getAsJsonArray("actors").get(0).getAsJsonObject();
				if (missing) actor.remove("weapon"); else actor.addProperty("shield_use_ticks", 1);
				lines.set(1, ReplayAnalysis.JSON.toJson(frame));
				Files.write(invalid, lines);
				boolean rejected = false;
				try { ReplayTrace.read(invalid); } catch (java.io.IOException expected) { rejected = true; }
				require(rejected, "Missing equipment or inconsistent initial state must not silently default");
			}
		} finally { Files.deleteIfExists(invalid); }
		System.out.println("PASS: native replay/live scenario read and invalid equipment rejected (" + trace.frames.size() + " frames)");
	}

	private static JsonArray actors() {
		JsonArray actors = new JsonArray();
		for (int actor = 0; actor < 2; actor++) {
			JsonObject state = new JsonObject();
			state.add("position", ReplayAnalysis.JSON.toJsonTree(new double[]{0, 1, actor * 2}));
			state.add("velocity", ReplayAnalysis.JSON.toJsonTree(new double[]{0, 0, 0}));
			state.addProperty("health", 20);
			state.addProperty("on_ground", true);
			JsonObject outcome = new JsonObject();
			outcome.addProperty("target", 1 - actor);
			state.add("outcome", outcome);
			actors.add(state);
		}
		return actors;
	}

	private static void require(boolean condition, String message) {
		if (!condition) throw new AssertionError(message);
	}

	private static void checkSprintInput() {
		var input = new ReplayPlayer.SprintInput();
		var forward = new ReplayTrace.Controls(true, false, false, false, true, false, false, false, 0, 0);
		var released = new ReplayTrace.Controls(false, false, false, false, true, false, false, false, 0, 0);
		var sprint = new ReplayTrace.Controls(true, false, false, false, false, true, false, false, 0, 0);
		var backward = new ReplayTrace.Controls(false, true, false, false, false, false, false, false, 0, 0);
		var sneaking = new ReplayTrace.Controls(false, false, false, false, false, false, true, false, 0, 0);
		require(!input.update(forward, false, true, true, false), "A first forward tap must only arm the sprint window");
		require(!input.update(released, false, true, true, false), "Releasing forward must not start sprinting");
		require(input.update(forward, false, true, true, false), "Ticks 31/33: a second forward tap starts sprinting without the sprint key");

		input.reset();
		require(input.update(sprint, false, true, true, false), "The sprint key must still start sprinting immediately");
		input.update(released, false, false, true, false);
		require(input.update(forward, false, true, true, false), "Ticks 21/23: a sprint-key tap must also arm keyless double-tap sprinting");

		input.reset();
		require(!input.update(forward, false, true, true, false), "Reset must clear the sprint window");
		input.update(released, false, true, true, false);
		require(input.update(forward, false, true, true, false), "Reset must clear previous forward input so a fresh tap arms the window");

		input.reset();
		input.update(forward, false, true, true, false);
		require(!input.update(forward, false, true, true, false), "Holding forward must not count as another tap");
		for (int tick = 0; tick < 5; tick++) input.update(released, false, true, true, false);
		require(!input.update(forward, false, true, true, false), "The seven-tick window must expire before checking the next tap");

		input.reset();
		input.update(forward, false, true, true, false);
		for (int tick = 0; tick < 5; tick++) input.update(released, false, true, true, false);
		require(input.update(forward, false, true, true, false), "A second tap just inside the sprint window must still start sprinting");

		input.reset();
		input.update(forward, false, true, true, false);
		input.update(backward, false, true, true, false);
		require(!input.update(forward, false, true, true, false), "Backward input must cancel an armed sprint window");

		input.reset();
		input.update(forward, false, true, true, false);
		input.update(sneaking, false, false, true, false);
		require(!input.update(forward, false, true, true, false), "Previous sneak input must cancel an armed sprint window");

		input.reset();
		require(!input.update(sprint, false, false, true, false), "A blocked sprint start must ignore the sprint key and not arm a window");
		input.update(released, false, true, true, false);
		require(!input.update(forward, false, true, true, false), "Ineligible taps must not arm later double-tap sprinting");

		input.reset();
		input.update(forward, false, true, true, false);
		input.update(released, false, false, true, true);
		require(!input.update(forward, false, true, true, false), "Item-use slowdown must cancel an armed double-tap window");

		var sneakForward = new ReplayTrace.Controls(true, false, false, false, true, true, true, false, 0, 0);
		input.reset();
		require(input.update(sneakForward, true, false, true, false), "Replay tick 8: pressing sneak must not cancel an existing sprint");
		require(input.update(sneakForward, true, false, true, false), "Held crouch must not cancel an otherwise eligible existing sprint");
		require(input.update(forward, true, false, true, true), "Starting shield use must not cancel an existing sprint");
		require(!input.update(forward, true, false, false, false), "Food ineligibility or a non-minor wall collision must stop sprinting");
		require(!input.update(backward, true, false, true, false), "Losing forward impulse must stop sprinting");
		input.reset();
		require(!input.crouching(true, false), "New sneak input must not slow movement before the input refresh");
		require(input.update(sneakForward, false, !input.crouching(true, false), true, false),
				"New sneak input must not block a sprint start before the delayed crouch flag updates");
		require(input.crouching(true, false), "The next movement tick must use the previous sneak input");
		require(!input.update(sneakForward, false, !input.crouching(true, false), true, false), "An established crouch must block a new sprint start");
		require(input.crouching(true, false), "The first released-sneak tick must still use crouch slowdown");
		input.update(forward, true, false, true, false);
		require(!input.crouching(true, false), "Crouch slowdown must end on the tick after sneak is released");
		require(input.crouching(true, true) && !input.crouching(false, true), "Forced crouching requires room for the crouching pose");
		input.reset(true);
		require(input.crouching(true, false), "A replay seeded crouching must retain its initial sneak state");
		input.reset();
		require(!input.crouching(true, false), "Reset must clear previous sneak input");
		System.out.println("PASS: sprint persistence, delayed crouch eligibility, item-use reset, double taps and cancellation");
	}

	private static void checkMinorCollision() {
		var forward = new Vec3(0, 0, 1);
		require(ReplayPlayer.SprintInput.minorCollision(forward, new Vec3(0, 1, 0.1)),
				"Aligned accepted horizontal movement is a minor collision regardless of vertical movement");
		require(ReplayPlayer.SprintInput.minorCollision(forward, new Vec3(Math.sin(Math.toRadians(3.10)), 0, Math.cos(Math.toRadians(3.10)))),
				"Replay tick 90: a three-degree glancing wall contact must be minor");
		require(ReplayPlayer.SprintInput.minorCollision(forward, new Vec3(Math.sin(Math.toRadians(7.99)), 0, Math.cos(Math.toRadians(7.99)))),
				"The minor-collision threshold includes angles just below eight degrees");
		require(!ReplayPlayer.SprintInput.minorCollision(forward, new Vec3(Math.sin(Math.toRadians(8.01)), 0, Math.cos(Math.toRadians(8.01)))),
				"The minor-collision threshold excludes angles just above eight degrees");
		require(!ReplayPlayer.SprintInput.minorCollision(forward, new Vec3(0, 0, 0.003))
				&& !ReplayPlayer.SprintInput.minorCollision(new Vec3(0, 0, 0.003), forward),
				"Negligible desired or accepted horizontal movement must not be a minor collision");
		require(!ReplayPlayer.SprintInput.minorCollision(forward, new Vec3(0, 0, -1)),
				"Movement opposite the input is not a minor collision");
		System.out.println("PASS: local-player minor horizontal collision angle and displacement thresholds");
	}

	private static void checkLivePolicy() {
		var source = new ObservationEncoder.State(0.5, 1, -1.5, 20, 0, 0, true, 0.1, -0.2, 0.3);
		var target = new ObservationEncoder.State(0.5, 1, 2.5, 10, 180, 0, true, -0.2, 0.4, -0.1);
		float[] rays = {0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F, 0.7F, 0.8F};
		float[] observation = ObservationEncoder.observe(source, target, rays);
		require(observation.length == 22 && observation[0] == 1 && observation[4] == 0.5F
				&& observation[7] == 0.4F && observation[8] == -1 && observation[18] == 0.4F
				&& observation[19] == 1 && observation[20] == 0 && observation[21] == 0,
				"Live feature normalization and wrapped opponent yaw must match training");
		for (int ray = 0; ray < 8; ray++) require(observation[10 + ray] == rays[ray], "Wall-ray ordering must stay world-relative");
		float[] velocityCatalogue = new float[PolicyCatalogue.FEATURE_NAMES.size()];
		ObservationEncoder.velocityObservation(velocityCatalogue, source, target);
		require(velocityCatalogue[79] == 0.1F && velocityCatalogue[80] == -0.2F && velocityCatalogue[81] == 0.3F
				&& Math.abs(velocityCatalogue[82] + 0.3F) < 1e-6F
				&& Math.abs(velocityCatalogue[83] - 0.6F) < 1e-6F
				&& Math.abs(velocityCatalogue[84] + 0.4F) < 1e-6F,
				"Self and opponent-relative velocities must use world-axis blocks per tick");
		var bowSource = new ObservationEncoder.State(0.5, 1, -1.5, 20, 0, 0, true, 0.1, -0.2, 0.3,
				new ObservationEncoder.BowState(32, true, 10, 0.41666667F, true));
		var bowTarget = new ObservationEncoder.State(2, 1, 3, 20, 0, 0, true, -0.2, 0.4, -0.1,
				new ObservationEncoder.BowState(4, false, 0, 0, true));
		var arrow = new ObservationEncoder.ArrowState(1.5, 2, -1.5, 0.6, 0.2, 0.4);
		float[] bowCatalogue = new float[PolicyCatalogue.FEATURE_NAMES.size()];
		ObservationEncoder.bowObservation(bowCatalogue, bowSource, bowTarget, arrow);
		require(Math.abs(bowCatalogue[89] - 0.5F) < 1e-6F && bowCatalogue[90] == 1
				&& Math.abs(bowCatalogue[91] - 0.41666667F) < 1e-6F
				&& bowCatalogue[92] == 1 && bowCatalogue[93] == 1 && bowCatalogue[94] == 0,
				"Bow draw, ammo, and equipped observations must use actual actor state");
		require(bowCatalogue[95] == 1 && Math.abs(bowCatalogue[96] - 0.1F) < 1e-6F
				&& Math.abs(bowCatalogue[97] - 0.1F) < 1e-6F && bowCatalogue[98] == 0
				&& Math.abs(bowCatalogue[99] - 0.5F) < 1e-6F && Math.abs(bowCatalogue[100] - 0.4F) < 1e-6F
				&& Math.abs(bowCatalogue[101] - 0.1F) < 1e-6F,
				"Arrow features must be actor-relative position and velocity");
		float[] reverseBow = new float[PolicyCatalogue.FEATURE_NAMES.size()];
		ObservationEncoder.bowObservation(reverseBow, bowTarget, bowSource, arrow);
		require(reverseBow[89] == 0.0625F && reverseBow[90] == 0 && reverseBow[91] == 0
				&& reverseBow[92] == 1 && reverseBow[93] == 1 && reverseBow[94] == 1
				&& Math.abs(reverseBow[96] + 0.05F) < 1e-6F,
				"Bow and arrow observations must reverse with the actor perspective");
		float[] noArrow = new float[PolicyCatalogue.FEATURE_NAMES.size()];
		ObservationEncoder.bowObservation(noArrow, bowSource, bowTarget, null);
		for (int index = 95; index <= 101; index++) require(noArrow[index] == 0, "Absent arrows must zero every arrow feature");
		float[] coincident = ObservationEncoder.observe(source, source, rays);
		require(coincident[18] == 0 && coincident[19] == 0 && coincident[20] == 0 && coincident[21] == 0,
				"Coincident players must not produce undefined aiming observations");
		JsonObject response = JsonParser.parseString("{\"protocol\":1,\"session\":\"round\",\"tick\":7,"
				+ "\"actions\":[[2,1,1,1,8,0],[7,0,0,0,4,2]]}").getAsJsonObject();
		var actions = LivePolicy.actions(response, "round", 7);
		require(actions[0].forward() && actions[0].right() && !actions[0].left() && actions[0].jump()
				&& actions[0].sprint() && actions[0].attack() && actions[0].yaw_delta() == 20 && actions[0].pitch_delta() == -5
				&& actions[1].left() && !actions[1].forward() && actions[1].yaw_delta() == 0 && !actions[1].sneak(),
				"Live action head order, diagonal movement and neutral turn bins must match training");
		boolean rejected = false;
		try { LivePolicy.actions(response, "round", 8); } catch (IllegalArgumentException expected) { rejected = true; }
		require(rejected, "Stale tick responses must be rejected");
		rejected = false;
		try { LivePolicy.actions(response, "other-round", 7); } catch (IllegalArgumentException expected) { rejected = true; }
		require(rejected, "Responses from previous reset/session must be rejected");
		response.getAsJsonArray("actions").get(0).getAsJsonArray().set(0, ReplayAnalysis.JSON.toJsonTree(1.5));
		rejected = false;
		try { LivePolicy.actions(response, "round", 7); } catch (IllegalArgumentException expected) { rejected = true; }
		require(rejected, "Fractional action indices must not be silently truncated");
		System.out.println("PASS: live observation/action contract and stale-response checks");
	}

	private static ObservationEncoder.State fixtureState(JsonObject value) {
		JsonArray position = value.getAsJsonArray("position");
		JsonArray velocity = value.getAsJsonArray("velocity");
		return new ObservationEncoder.State(position.get(0).getAsDouble(), position.get(1).getAsDouble(), position.get(2).getAsDouble(),
				value.get("health").getAsFloat(), value.get("yaw").getAsFloat(), value.get("pitch").getAsFloat(), value.get("health").getAsFloat() > 0,
				velocity.get(0).getAsDouble(), velocity.get(1).getAsDouble(), velocity.get(2).getAsDouble());
	}

	private static void checkPolicyFixtures() throws Exception {
		JsonObject fixture = JsonParser.parseString(Files.readString(Path.of(System.getProperty("cambium.contractFixtures")))).getAsJsonObject();
		float tolerance = fixture.get("tolerance").getAsFloat();
		for (var sample : fixture.getAsJsonArray("observations")) {
			JsonObject value = sample.getAsJsonObject();
			var source = fixtureState(value.getAsJsonObject("source"));
			var target = fixtureState(value.getAsJsonObject("target"));
			float[] actual = ObservationEncoder.observe(source, target,
					ReplayAnalysis.JSON.fromJson(value.get("rays"), float[].class));
			float[] expected = ReplayAnalysis.JSON.fromJson(value.get("expected"), float[].class);
			require(actual.length == expected.length, "Shared observation size");
			for (int index = 0; index < actual.length; index++) require(Math.abs(actual[index] - expected[index]) <= tolerance,
					value.get("name").getAsString() + ": " + index + " actual=" + actual[index] + " expected=" + expected[index]);
			float[] velocityCatalogue = new float[PolicyCatalogue.FEATURE_NAMES.size()];
			ObservationEncoder.velocityObservation(velocityCatalogue, source, target);
			float[] expectedVelocity = ReplayAnalysis.JSON.fromJson(value.get("expected_velocity"), float[].class);
			for (int index = 0; index < expectedVelocity.length; index++) require(Math.abs(velocityCatalogue[79 + index] - expectedVelocity[index]) <= tolerance,
					value.get("name").getAsString() + ": velocity " + index);
			JsonObject sourceState = value.getAsJsonObject("source");
			JsonObject targetState = value.getAsJsonObject("target");
			ObservationEncoder.movementObservation(velocityCatalogue, sourceState.get("on_ground").getAsBoolean(),
					targetState.get("on_ground").getAsBoolean(), sourceState.get("sprinting").getAsBoolean(), targetState.get("sprinting").getAsBoolean());
			float[] expectedMovement = ReplayAnalysis.JSON.fromJson(value.get("expected_movement"), float[].class);
			for (int index = 0; index < expectedMovement.length; index++) require(velocityCatalogue[85 + index] == expectedMovement[index],
					value.get("name").getAsString() + ": movement " + index);
			ObservationEncoder.movementObservation(velocityCatalogue, targetState.get("on_ground").getAsBoolean(),
					sourceState.get("on_ground").getAsBoolean(), targetState.get("sprinting").getAsBoolean(), sourceState.get("sprinting").getAsBoolean());
			for (int index = 0; index < expectedMovement.length; index++) require(velocityCatalogue[85 + index] == expectedMovement[index ^ 1],
					value.get("name").getAsString() + ": movement perspective " + index);
		}
		SimulationContract contract = new SimulationContract(fixture.getAsJsonObject("policy"));
		float[] catalogue = ReplayAnalysis.JSON.fromJson(fixture.get("catalogue"), float[].class);
		if (catalogue.length == PolicyCatalogue.FEATURE_NAMES.size() - 6) {
			catalogue = java.util.Arrays.copyOf(catalogue, PolicyCatalogue.FEATURE_NAMES.size());
		}
		float[] actual = contract.project(catalogue);
		float[] expected = ReplayAnalysis.JSON.fromJson(fixture.get("projected"), float[].class);
		require(actual.length == expected.length, "Shared projection size");
		for (int index = 0; index < actual.length; index++) require(Math.abs(actual[index] - expected[index]) <= tolerance, "Shared projection values");
		for (var element : fixture.getAsJsonArray("projection_cases")) {
			JsonObject sample = element.getAsJsonObject();
			JsonObject descriptor = fixture.getAsJsonObject("policy").deepCopy();
			descriptor.add("observations", sample.get("observations"));
			float[] projected = new SimulationContract(descriptor).project(catalogue);
			float[] wanted = ReplayAnalysis.JSON.fromJson(sample.get("expected"), float[].class);
			require(projected.length == wanted.length, "Expanded observation projection size");
			for (int index = 0; index < projected.length; index++) require(Math.abs(projected[index] - wanted[index]) <= tolerance,
					sample.get("name").getAsString() + ": " + index);
		}
		JsonObject oversized = fixture.getAsJsonObject("policy").deepCopy();
		JsonArray expanded = new JsonArray();
		for (int index = 0; index < 15; index++) expanded.add(JsonParser.parseString("""
			{"feature":"self.weapon_slot","encoding":"one_hot","categories":[0,1,2,3,4,5,6,7,8]}
			"""));
		oversized.add("observations", expanded);
		boolean overflowRejected = false;
		try { new SimulationContract(oversized); } catch (IllegalArgumentException error) { overflowRejected = true; }
		require(overflowRejected, "Expanded observation limit must be enforced");
		JsonObject loadout = new JsonObject();
		loadout.add("weapons", fixture.get("weapons"));
		contract.validateLoadouts(java.util.List.of(loadout, loadout));
		var controls = contract.decode(fixture.getAsJsonArray("action"), loadout);
		JsonObject semantic = ReplayAnalysis.JSON.toJsonTree(controls).getAsJsonObject();
		semantic.remove("select_slot");
		require(semantic.equals(fixture.get("controls")), "Shared semantic action decoding retains historical item fields");
		for (var invalid : fixture.getAsJsonArray("invalid")) {
			JsonObject change = invalid.getAsJsonObject();
			JsonObject descriptor = fixture.getAsJsonObject("policy").deepCopy();
			String[] path = change.get("pointer").getAsString().substring(1).split("/");
			com.google.gson.JsonElement parent = descriptor;
			for (int index = 0; index < path.length - 1; index++) parent = parent.isJsonArray()
					? parent.getAsJsonArray().get(Integer.parseInt(path[index])) : parent.getAsJsonObject().get(path[index]);
			String key = path[path.length - 1];
			if (parent.isJsonArray()) parent.getAsJsonArray().set(Integer.parseInt(key), change.get("value"));
			else parent.getAsJsonObject().add(key, change.get("value"));
			boolean rejected = false;
			try { new SimulationContract(descriptor); } catch (IllegalArgumentException error) { rejected = true; }
			require(rejected, "Shared invalid descriptor must fail: " + change.get("pointer"));
		}
		System.out.println("PASS: shared native/Java observation, projection, semantic controls and rejection fixtures");
	}

	private static void checkDescriptiveContract() {
		JsonObject descriptor = JsonParser.parseString("""
			{"format":"minecraft_sim_policy","version":1,"minecraft":"26.3","ticks_per_action":1,
			 "observations":[{"feature":"self.attack_charge"},{"feature":"self.health_fraction","scale":2}],
			 "actions":[{"name":"weapon","values":[2,1,0]},{"name":"yaw","values":[-15,0,15]},{"name":"shield","values":[false,true]}]}
			""").getAsJsonObject();
		SimulationContract contract = new SimulationContract(descriptor);
		float[] catalogue = new float[SimulationContract.FEATURES.size()];
		catalogue[SimulationContract.FEATURES.indexOf("self.health_fraction")] = 1;
		catalogue[SimulationContract.FEATURES.indexOf("self.attack_charge")] = 0.75F;
		float[] projected = contract.project(catalogue);
		require(projected.length == 2 && projected[0] == 0.75F && projected[1] == 2, "Feature order and scales come from descriptors");
		JsonObject spawn = JsonParser.parseString("""
			{"loadout":{"weapons":["minecraft:iron_sword","minecraft:netherite_axe","minecraft:air"],
			 "offhand":"minecraft:shield","armor":{"helmet":"minecraft:diamond_helmet"}}}
			""").getAsJsonObject();
		JsonObject loadout = SimulationContract.loadout(spawn, "configured");
		contract.validateLoadouts(java.util.List.of(loadout, loadout));
		ReplayTrace.Controls controls = contract.decode(JsonParser.parseString("[1,2,1]").getAsJsonArray(), loadout);
		require(controls.select_weapon() && controls.weapon().equals("minecraft:netherite_axe") && controls.use_item()
				&& controls.yaw_delta() == 15 && !controls.attack(), "Actions decode by semantic primitive rather than head position");
		descriptor.getAsJsonArray("observations").get(0).getAsJsonObject().addProperty("feature", "server.secret");
		boolean rejected = false;
		try { new SimulationContract(descriptor); } catch (IllegalArgumentException expected) { rejected = true; }
		require(rejected, "Unknown mechanics/observations must fail explicitly");
		System.out.println("PASS: descriptive policy projection, arbitrary head ordering, loadouts and capability rejection");
	}

	private static void checkBowContract() {
		JsonObject descriptor = JsonParser.parseString("""
			{"format":"minecraft_sim_policy","version":1,"minecraft":"26.3","ticks_per_action":1,"requires":["bow"],
			 "observations":[
			  {"feature":"self.arrow_count_fraction"},
			  {"feature":"self.bow_using","encoding":"one_hot","categories":[0,1]},
			  {"feature":"self.bow_power_fraction"},
			  {"feature":"self.bow_equipped","encoding":"one_hot","categories":[0,1]},
			  {"feature":"opponent.bow_equipped","encoding":"one_hot","categories":[0,1]},
			  {"feature":"opponent.bow_using","encoding":"one_hot","categories":[0,1]},
			  {"feature":"arrow.present","encoding":"one_hot","categories":[0,1]},
			  {"feature":"arrow.relative.x"},{"feature":"arrow.relative.y"},{"feature":"arrow.relative.z"},
			  {"feature":"arrow.relative.velocity.x"},{"feature":"arrow.relative.velocity.y"},{"feature":"arrow.relative.velocity.z"}],
			 "actions":[{"name":"use_item","values":[false,true]}]}
			""").getAsJsonObject();
		SimulationContract contract = new SimulationContract(descriptor);
		require(contract.supportsFullBowContract() && contract.observationCount() == 18,
				"Complete bow feature and binary-use descriptors must be admitted");
		JsonObject spawn = JsonParser.parseString("""
			{"loadout":{"weapons":["minecraft:bow"],"arrows":64}}
			""").getAsJsonObject();
		JsonObject bowLoadout = SimulationContract.loadout(spawn, "configured");
		contract.validateLoadouts(java.util.List.of(bowLoadout, bowLoadout));
		ReplayTrace.Controls controls = contract.decode(JsonParser.parseString("[1]").getAsJsonArray(), bowLoadout);
		require(controls.use_item() && bowLoadout.get("arrows").getAsInt() == 64
				&& SimulationContract.weaponChoices().contains("minecraft:bow"),
				"Bow loadouts must carry finite ammo and decode use_item semantically");
		JsonObject defaultLoadout = SimulationContract.loadout(JsonParser.parseString(
				"{\"loadout\":{\"weapons\":[\"minecraft:diamond_sword\"]}}").getAsJsonObject(), "configured");
		require(defaultLoadout.get("arrows").getAsInt() == 0, "Historical loadouts must default to zero arrows");
		for (String invalid : new String[]{"-1", "65", "1.5"}) {
			JsonObject invalidSpawn = spawn.deepCopy();
			invalidSpawn.getAsJsonObject("loadout").add("arrows", JsonParser.parseString(invalid));
			boolean rejected = false;
			try { SimulationContract.loadout(invalidSpawn, "configured"); } catch (IllegalArgumentException expected) { rejected = true; }
			require(rejected, "Out-of-range/fractional arrow counts must fail");
		}
		JsonObject missingCapability = descriptor.deepCopy();
		missingCapability.remove("requires");
		boolean rejected = false;
		try { new SimulationContract(missingCapability); } catch (IllegalArgumentException expected) { rejected = true; }
		require(rejected, "Bow observation descriptors must require the bow capability");
		JsonObject alias = descriptor.deepCopy();
		alias.getAsJsonArray("actions").add(JsonParser.parseString("{\"name\":\"shield\",\"values\":[false,true]}"));
		rejected = false;
		try { new SimulationContract(alias); } catch (IllegalArgumentException expected) { rejected = true; }
		require(rejected, "A policy cannot declare both use_item and its historical shield alias");
		JsonObject shieldDescriptor = JsonParser.parseString("""
			{"format":"minecraft_sim_policy","version":1,"minecraft":"26.3","ticks_per_action":1,
			 "observations":[{"feature":"self.health_fraction"}],
			 "actions":[{"name":"shield","values":[false,true]}]}
			""").getAsJsonObject();
		SimulationContract shieldContract = new SimulationContract(shieldDescriptor);
		require(shieldContract.decode(JsonParser.parseString("[1]").getAsJsonArray(), defaultLoadout).use_item(),
				"Historical shield actions must remain an alias for held item use");
		System.out.println("PASS: complete bow descriptors, finite arrows, capability gating and legacy shield alias");
	}

	private static void checkBowReplay() throws Exception {
		Path folder = Files.createTempDirectory("cambium-bow-replay-");
		try {
			JsonObject environment = JsonParser.parseString("""
				{"format":"minecraft_sim_environment","version":1,"minecraft":"26.3","requires":["bow"],
				 "bounds":{"min":[0,0,0],"size":[16,8,16]},"blocks":[],
				 "players":[
				  {"position":[2,1,2],"velocity":[0,0,0],"yaw":0,"pitch":0,"on_ground":false,
				   "loadout":{"weapons":["minecraft:bow"],"arrows":64}},
				  {"position":[8,1,2],"velocity":[0,0,0],"yaw":0,"pitch":0,"on_ground":false,
				   "loadout":{"weapons":["minecraft:bow"],"arrows":64}}]}
				""").getAsJsonObject();
			JsonObject header = new JsonObject();
			JsonObject rules = new JsonObject();
			rules.addProperty("simulate_fall_damage", true);
			environment.add("rules", rules);
			require(ReplayTrace.forLive(environment).environment.simulateFallDamage, "Live scenario preserves opt-in fall damage");
			rules.addProperty("simulate_fall_damage", false);
			require(!ReplayTrace.forLive(environment).environment.simulateFallDamage, "Live scenario preserves disabled fall damage");
			rules.addProperty("simulate_fall_damage", "true");
			boolean invalidRule = false;
			try { ReplayTrace.forLive(environment); } catch (java.io.IOException expected) { invalidRule = true; }
			require(invalidRule, "Live scenario rejects nonboolean fall-damage rules");
			environment.remove("rules");
			header.addProperty("format", "minecraft_sim_replay");
			header.addProperty("version", 3);
			header.addProperty("minecraft", "26.3");
			header.addProperty("players", 2);
			header.addProperty("ticks_per_second", 20);
			header.addProperty("backend", "native");
			header.add("environment", environment);
			JsonArray emptyProjectiles = new JsonArray();
			JsonObject first = bowReplayFrame(0, false, false, emptyProjectiles);
			JsonArray projectileSamples = new JsonArray();
			projectileSamples.add(ReplayAnalysis.JSON.toJsonTree(new ReplayTrace.Projectile(0,
					new double[]{3, 2, 3}, new double[]{0.1, 0.2, 1.5}, 4, true, false, 0, true)));
			JsonObject second = bowReplayFrame(1, true, true, projectileSamples);
			Path replay = folder.resolve("bow.jsonl");
			Files.writeString(replay, ReplayAnalysis.JSON.toJson(header) + "\n" + ReplayAnalysis.JSON.toJson(first) + "\n"
					+ ReplayAnalysis.JSON.toJson(second) + "\n");
			ReplayTrace trace = ReplayTrace.read(replay);
			require(trace.frames.get(1).projectiles().size() == 1 && trace.frames.get(1).projectiles().getFirst().owner() == 0
					&& trace.frames.get(1).projectiles().getFirst().critical()
					&& trace.frames.get(1).actors().getFirst().bow_using()
					&& trace.frames.get(1).actors().getFirst().bow_draw_ticks() == 5,
				"Envelope-3 reader must preserve projectile ownership and actual bow draw state");
			JsonObject thrownHeader = header.deepCopy();
			thrownHeader.getAsJsonObject("environment").getAsJsonArray("players").get(0).getAsJsonObject().add("loadout",
					JsonParser.parseString("{\"weapons\":[\"minecraft:snowball\",\"minecraft:egg\"],\"snowballs\":1,\"eggs\":2}"));
			JsonObject thrownFirst = bowReplayFrame(0, false, false, new JsonArray());
			JsonObject thrownSecond = bowReplayFrame(1, false, true, new JsonArray());
			for (JsonObject sample : java.util.List.of(thrownFirst, thrownSecond)) {
				JsonObject actor = sample.getAsJsonArray("actors").get(0).getAsJsonObject();
				actor.addProperty("weapon", sample == thrownFirst ? "minecraft:snowball" : "minecraft:air");
				actor.getAsJsonObject("controls").addProperty("weapon", "minecraft:snowball");
				actor.addProperty("arrow_count", 0);
				actor.addProperty("snowball_count", sample == thrownFirst ? 1 : 0);
				actor.addProperty("egg_count", 2);
			}
			JsonObject snowball = ReplayAnalysis.JSON.toJsonTree(new ReplayTrace.Projectile(0,
					new double[]{3, 2, 3}, new double[]{0.1, 0.2, 1.5}, 0, false, false, 0, false, "snowball")).getAsJsonObject();
			thrownSecond.getAsJsonArray("projectiles").add(snowball);
			Path thrownReplay = folder.resolve("thrown.jsonl");
			Files.writeString(thrownReplay, ReplayAnalysis.JSON.toJson(thrownHeader) + "\n" + ReplayAnalysis.JSON.toJson(thrownFirst)
					+ "\n" + ReplayAnalysis.JSON.toJson(thrownSecond) + "\n");
			ReplayTrace thrownTrace = ReplayTrace.read(thrownReplay);
			require(thrownTrace.frames.get(1).actors().getFirst().snowball_count() == 0
					&& thrownTrace.frames.get(1).actors().getFirst().egg_count() == 2
					&& thrownTrace.frames.get(1).projectiles().getFirst().type().equals("snowball"),
					"Replay preserves depleted throwable slots and typed projectiles");
			JsonObject pearlHeader = thrownHeader.deepCopy();
			pearlHeader.getAsJsonObject("environment").getAsJsonArray("players").get(0).getAsJsonObject().add("loadout",
					JsonParser.parseString("{\"weapons\":[\"minecraft:ender_pearl\"],\"ender_pearls\":1}"));
			JsonObject pearlFirst = thrownFirst.deepCopy();
			JsonObject pearlSecond = thrownSecond.deepCopy();
			for (JsonObject sample : java.util.List.of(pearlFirst, pearlSecond)) {
				JsonObject actor = sample.getAsJsonArray("actors").get(0).getAsJsonObject();
				actor.addProperty("weapon", sample == pearlFirst ? "minecraft:ender_pearl" : "minecraft:air");
				actor.getAsJsonObject("controls").addProperty("weapon", "minecraft:ender_pearl");
				actor.addProperty("snowball_count", 0);
				actor.addProperty("egg_count", 0);
				actor.addProperty("ender_pearl_count", sample == pearlFirst ? 1 : 0);
				actor.addProperty("ender_pearl_cooldown", sample == pearlFirst ? 0 : 19);
			}
			pearlSecond.getAsJsonArray("projectiles").get(0).getAsJsonObject().addProperty("type", "ender_pearl");
			Path pearlReplay = folder.resolve("pearl.jsonl");
			Files.writeString(pearlReplay, ReplayAnalysis.JSON.toJson(pearlHeader) + "\n" + ReplayAnalysis.JSON.toJson(pearlFirst)
					+ "\n" + ReplayAnalysis.JSON.toJson(pearlSecond) + "\n");
			ReplayTrace pearlTrace = ReplayTrace.read(pearlReplay);
			require(pearlTrace.frames.get(1).actors().getFirst().ender_pearl_count() == 0
					&& pearlTrace.frames.get(1).actors().getFirst().ender_pearl_cooldown() == 19
					&& pearlTrace.frames.get(1).projectiles().getFirst().type().equals("ender_pearl"),
					"Replay preserves depleted pearls, cooldown, and typed projectiles");
			JsonObject badOwner = second.deepCopy();
			badOwner.getAsJsonArray("projectiles").get(0).getAsJsonObject().addProperty("owner", 2);
			Path invalidOwner = folder.resolve("invalid-owner.jsonl");
			Files.writeString(invalidOwner, ReplayAnalysis.JSON.toJson(header) + "\n" + ReplayAnalysis.JSON.toJson(first) + "\n"
					+ ReplayAnalysis.JSON.toJson(badOwner) + "\n");
			boolean rejected = false;
			try { ReplayTrace.read(invalidOwner); } catch (java.io.IOException expected) { rejected = true; }
			require(rejected, "Projectile owner indices outside the two actors must fail");
			JsonObject badFinite = second.deepCopy();
			badFinite.getAsJsonArray("projectiles").get(0).getAsJsonObject()
					.add("velocity", JsonParser.parseString("[0,1e309,0]").getAsJsonArray());
			Path invalidVelocity = folder.resolve("invalid-velocity.jsonl");
			Files.writeString(invalidVelocity, ReplayAnalysis.JSON.toJson(header) + "\n" + ReplayAnalysis.JSON.toJson(first) + "\n"
					+ ReplayAnalysis.JSON.toJson(badFinite) + "\n");
			rejected = false;
			try { ReplayTrace.read(invalidVelocity); } catch (java.io.IOException expected) { rejected = true; }
			require(rejected, "Non-finite projectile vectors must fail");
			JsonObject oldHeader = header.deepCopy();
			oldHeader.addProperty("version", 2);
			JsonObject oldEnvironment = JsonParser.parseString("""
				{"format":"minecraft_sim_environment","version":1,"minecraft":"26.3",
				 "bounds":{"min":[0,0,0],"size":[16,8,16]},"blocks":[],
				 "players":[{"position":[2,1,2],"velocity":[0,0,0],"yaw":0,"pitch":0,"on_ground":false},
				            {"position":[8,1,2],"velocity":[0,0,0],"yaw":0,"pitch":0,"on_ground":false}]}
				""").getAsJsonObject();
			oldHeader.add("environment", oldEnvironment);
			JsonObject oldFirst = legacyReplayFrame(0);
			JsonObject oldSecond = legacyReplayFrame(1);
			Path legacy = folder.resolve("legacy.jsonl");
			Files.writeString(legacy, ReplayAnalysis.JSON.toJson(oldHeader) + "\n" + ReplayAnalysis.JSON.toJson(oldFirst) + "\n"
					+ ReplayAnalysis.JSON.toJson(oldSecond) + "\n");
			ReplayTrace oldTrace = ReplayTrace.read(legacy);
			require(oldTrace.frames.getFirst().projectiles().isEmpty() && oldTrace.frames.getFirst().actors().getFirst().arrow_count() == 0,
				"Older envelopes must default absent bow/projectile fields without changing playback");
			System.out.println("PASS: envelope-3 bow/projectile validation, owner/finite rejection and legacy defaults");
		} finally {
			try (var files = Files.walk(folder)) {
				for (Path path : files.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
			}
		}
	}

	private static JsonObject bowReplayFrame(int tick, boolean bowUsing, boolean useItem, JsonArray projectiles) {
		JsonArray actors = new JsonArray();
		for (int index = 0; index < 2; index++) {
			var controls = new ReplayTrace.Controls(false, false, false, false, false, false, false, false, 0, 0,
					false, "minecraft:bow", useItem, -1);
			JsonObject actor = new JsonObject();
			actor.addProperty("id", index);
			actor.add("position", ReplayAnalysis.JSON.toJsonTree(index == 0 ? new double[]{2, 1, 2} : new double[]{8, 1, 2}));
			actor.add("velocity", ReplayAnalysis.JSON.toJsonTree(new double[]{0, 0, 0}));
			actor.addProperty("yaw", 0);
			actor.addProperty("pitch", 0);
			actor.addProperty("on_ground", false);
			actor.addProperty("sprinting", false);
			actor.addProperty("pose", 0);
			actor.addProperty("health", 20);
			actor.addProperty("absorption", 0);
			actor.addProperty("damage_cooldown", 0);
			actor.addProperty("hurt_time", 0);
			actor.addProperty("attack_strength_ticks", 0);
			actor.add("controls", ReplayAnalysis.JSON.toJsonTree(controls));
			actor.add("outcome", ReplayAnalysis.JSON.toJsonTree(new ReplayTrace.Outcome(-1, false, false, false, false, false, 0, 0)));
			actor.addProperty("weapon", "minecraft:bow");
			actor.addProperty("weapon_slot", 0);
			actor.addProperty("shield_equipped", false);
			actor.addProperty("using_shield", false);
			actor.addProperty("shield_use_ticks", 0);
			actor.addProperty("shield_cooldown", 0);
			actor.addProperty("item_use_delay", 0);
			actor.addProperty("attack_suppressed", false);
			actor.add("armor", JsonParser.parseString("""
				{"boots":"minecraft:air","leggings":"minecraft:air","chestplate":"minecraft:air","helmet":"minecraft:air"}
				"""));
			actor.addProperty("arrow_count", 64);
			actor.addProperty("bow_using", index == 0 && bowUsing);
			actor.addProperty("bow_draw_ticks", index == 0 && bowUsing ? 5 : 0);
			actor.addProperty("bow_power", index == 0 && bowUsing ? 0.1875F : 0);
			actors.add(actor);
		}
		JsonObject frame = new JsonObject();
		frame.addProperty("tick", tick);
		frame.add("actors", actors);
		frame.add("projectiles", projectiles);
		return frame;
	}

	private static JsonObject legacyReplayFrame(int tick) {
		JsonArray actors = new JsonArray();
		for (int index = 0; index < 2; index++) {
			JsonObject actor = new JsonObject();
			actor.addProperty("id", index);
			actor.add("position", ReplayAnalysis.JSON.toJsonTree(index == 0 ? new double[]{2, 1, 2} : new double[]{8, 1, 2}));
			actor.add("velocity", ReplayAnalysis.JSON.toJsonTree(new double[]{0, 0, 0}));
			actor.addProperty("yaw", 0);
			actor.addProperty("pitch", 0);
			actor.addProperty("on_ground", false);
			actor.addProperty("sprinting", false);
			actor.addProperty("pose", 0);
			actor.addProperty("health", 20);
			actor.addProperty("absorption", 0);
			actor.addProperty("damage_cooldown", 0);
			actor.addProperty("hurt_time", 0);
			actor.addProperty("attack_strength_ticks", 0);
			actor.add("controls", ReplayAnalysis.JSON.toJsonTree(new ReplayTrace.Controls(false, false, false, false,
					false, false, false, false, 0, 0)));
			actor.add("outcome", new JsonObject());
			actors.add(actor);
		}
		JsonObject frame = new JsonObject();
		frame.addProperty("tick", tick);
		frame.add("actors", actors);
		return frame;
	}
}