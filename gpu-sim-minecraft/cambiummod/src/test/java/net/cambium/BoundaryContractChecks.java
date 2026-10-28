package net.cambium;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

public final class BoundaryContractChecks {
	public static void main(String[] args) throws Exception {
		check();
		if (args.length == 1) ReplayTrace.read(java.nio.file.Path.of(args[0]));
	}

	static void check() {
		JsonObject spawn = JsonParser.parseString("""
			{"loadout":{"weapons":["minecraft:diamond_sword","minecraft:air","minecraft:air","minecraft:air",
			 "minecraft:air","minecraft:air","minecraft:air","minecraft:air","minecraft:air"],"selected":7}}
			""").getAsJsonObject();
		JsonObject loadout = SimulationContract.loadout(spawn, "configured");
		require(loadout.get("selected").getAsInt() == 7 && loadout.getAsJsonArray("weapons").size() == 9,
				"Repeated empty slots retain their explicit initial selection");
		JsonObject environment = JsonParser.parseString("""
			{"format":"minecraft_sim_environment","version":1,"minecraft":"26.3","profile":"configured",
			 "bounds":{"min":[0,0,0],"size":[5,4,5]},"blocks":[],
			 "players":[{"position":[1.5,1,1.5],"velocity":[0,0,0],"yaw":0,"pitch":0,"on_ground":false},
			            {"position":[3.5,1,3.5],"velocity":[0,0,0],"yaw":180,"pitch":0,"on_ground":false}]}
			""").getAsJsonObject();
		for (var player : environment.getAsJsonArray("players")) player.getAsJsonObject().add("loadout", loadout.deepCopy());
		try {
			ReplayTrace trace = ReplayTrace.forLive(environment);
			require(trace.frames.getFirst().actors().getFirst().weapon_slot() == 7,
					"Concrete nine-slot environment initializes live replay state with the requested empty slot");
			require(trace.environment.loadouts.getFirst().get("selected").getAsInt() == 7,
					"Live replay preserves resolved loadout selection");
		} catch (java.io.IOException error) {
			throw new AssertionError("Resolved concrete environment must remain the only runtime contract", error);
		}
		JsonObject descriptor = JsonParser.parseString("""
			{"format":"minecraft_sim_policy","version":1,"minecraft":"26.3","ticks_per_action":1,
			 "observations":[{"feature":"self.weapon_slot"},{"feature":"opponent.weapon_slot"}],
			 "actions":[{"name":"weapon","values":[8,0,7,6,5,4,3,2,1]}]}
			""").getAsJsonObject();
		SimulationContract contract = new SimulationContract(descriptor);
		contract.validateLoadouts(java.util.List.of(loadout, loadout));
		for (int choice = 0; choice < 9; choice++) {
			JsonArray action = new JsonArray();
			action.add(choice);
			ReplayTrace.Controls controls = contract.decode(action, loadout);
			int requested = descriptor.getAsJsonArray("actions").get(0).getAsJsonObject().getAsJsonArray("values").get(choice).getAsInt();
			require(controls.select_weapon() && controls.select_slot() == requested,
					"Weapon decoding preserves the semantic slot, not the choice index or first matching air");
			require(SimulationContract.selectedSlot(loadout, controls.weapon(), controls.select_slot(), 7) == requested,
					"Replay selection uses the requested physical hotbar slot");
			ReplayTrace.Controls restored = ReplayAnalysis.JSON.fromJson(ReplayAnalysis.JSON.toJson(controls), ReplayTrace.Controls.class);
			require(restored.equals(controls), "Semantic slot survives JSON round trip");
		}
		require(SimulationContract.selectedSlot(loadout, "minecraft:air", -1, 7) == 7,
				"Legacy item-only controls preserve a matching current empty slot");
		ReplayTrace.Controls legacy = ReplayAnalysis.JSON.fromJson("{\"select_weapon\":true,\"weapon\":\"minecraft:air\"}", ReplayTrace.Controls.class);
		require(legacy.select_slot() == -1, "Historical controls omit semantic slot without selecting slot zero");
		reject(() -> SimulationContract.selectedSlot(loadout, "minecraft:diamond_sword", 8, 7));
		reject(() -> SimulationContract.selectedSlot(loadout, "minecraft:air", 9, 7));
		reject(() -> ReplayAnalysis.JSON.fromJson("{\"select_weapon\":false,\"select_slot\":8}", ReplayTrace.Controls.class));
		JsonObject duplicate = spawn.deepCopy();
		duplicate.getAsJsonObject("loadout").getAsJsonArray("weapons").set(8, ReplayAnalysis.JSON.toJsonTree("minecraft:diamond_sword"));
		reject(() -> SimulationContract.loadout(duplicate, "configured"));
		JsonObject thrown = JsonParser.parseString("{\"loadout\":{\"weapons\":[\"minecraft:snowball\",\"minecraft:egg\",\"minecraft:ender_pearl\"],\"snowballs\":16,\"eggs\":2,\"ender_pearls\":8}}").getAsJsonObject();
		JsonObject thrownLoadout = SimulationContract.loadout(thrown, "configured");
		require(thrownLoadout.get("snowballs").getAsInt() == 16 && thrownLoadout.get("eggs").getAsInt() == 2 && thrownLoadout.get("ender_pearls").getAsInt() == 8,
				"Throwable counts survive loadout normalization");
		thrown.getAsJsonObject("loadout").addProperty("eggs", 0);
		reject(() -> SimulationContract.loadout(thrown, "configured"));
		require(DeploymentSummary.describe(new JsonObject()).contains("no curriculum runtime"),
				"Historical inference explicitly remains fixed-scenario shared learner");
		JsonObject info = JsonParser.parseString("""
			{"experiment":{"name":"split","stage":"shield","stage_index":2,"environment":"duel","arenas":["flat"],
			 "controllers":"shared_learner","sampling":"fixed_checkpoint_episode"}}
			""").getAsJsonObject();
		require(DeploymentSummary.describe(info).contains("Experiment split | stage 3 (shield) | environment duel"),
				"Status reports actual optional deployment metadata");
		System.out.println("PASS: nine-slot contract, repeated air, semantic replay selection, legacy controls and deployment metadata");
	}

	private static void reject(Runnable operation) {
		try { operation.run(); } catch (RuntimeException expected) { return; }
		throw new AssertionError("Invalid slot or duplicate nonair must be rejected");
	}

	private static void require(boolean condition, String message) {
		if (!condition) throw new AssertionError(message);
	}
}
