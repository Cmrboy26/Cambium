package net.cambium;

import com.mojang.blaze3d.platform.InputConstants;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientLifecycleEvents;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keymapping.v1.KeyMappingHelper;
import net.fabricmc.fabric.api.client.networking.v1.ClientPlayConnectionEvents;

import net.minecraft.client.KeyMapping;
import net.minecraft.resources.Identifier;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CambiumFabric implements ClientModInitializer {
	public static final String MOD_ID = "cambiumfabric";
	public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);
	static final String BUILD_ID = readBuildIdentity();
	static final KeyMapping.Category INTERFACE_CATEGORY = KeyMapping.Category.register(id("interface"));

	private static String readBuildIdentity() {
		try (var stream = CambiumFabric.class.getResourceAsStream("/cambium-build.properties")) {
			if (stream == null) return "unknown (rebuild mod)";
			var properties = new java.util.Properties();
			properties.load(stream);
			return properties.getProperty("version") + " source=" + properties.getProperty("source_hash")
					+ " built=" + properties.getProperty("built_utc");
		} catch (java.io.IOException error) {
			return "unreadable (rebuild mod)";
		}
	}

	@Override
	public void onInitializeClient() {
		SimulationReplay.initialize();
		RosieLauncher.initialize();
		PlayerControl.initialize();
		DuelHud.initialize();
		ControlPanel.initialize();
		KeyMapping toggle = KeyMappingHelper.registerKeyMapping(new KeyMapping(
				"key.cambiumfabric.toggle_recording", InputConstants.Type.KEYBOARD, InputConstants.KEY_F9,
				KeyMapping.Category.register(id("recording"))));
		ClientTickEvents.END_CLIENT_TICK.register(client -> {
			MovementRecorder.checkSession(client);
			while (toggle.consumeClick()) {
				if (client.gui.screen() == null) {
					MovementRecorder.toggle(client);
				}
			}
		});
		ClientPlayConnectionEvents.DISCONNECT.register((handler, client) -> MovementRecorder.stop("Disconnected"));
		ClientLifecycleEvents.CLIENT_STOPPING.register(client -> MovementRecorder.stop("Client stopping"));
		LOGGER.info("Cambium {} catalogue={} loaded_from={}", BUILD_ID, PolicyCatalogue.HASH,
				CambiumFabric.class.getProtectionDomain().getCodeSource().getLocation());
	}

	public static Identifier id(String path) {
		return Identifier.fromNamespaceAndPath(MOD_ID, path);
	}
}
