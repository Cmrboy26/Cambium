package net.cambium;

import com.google.gson.JsonObject;

final class DeploymentSummary {
	private DeploymentSummary() {}

	static String describe(JsonObject info) {
		String deployment = "";
		if (info.has("experiment") && info.get("experiment").isJsonObject()) {
			JsonObject experiment = info.getAsJsonObject("experiment");
			deployment = "Experiment " + experiment.get("name").getAsString()
					+ " | stage " + (SimulationContract.integer(experiment.get("stage_index")) + 1)
					+ " (" + experiment.get("stage").getAsString() + ")"
					+ " | environment " + experiment.get("environment").getAsString() + ". ";
		}
		return deployment + "Live rounds use a fixed embedded scenario and shared learner; no curriculum runtime.";
	}
}
