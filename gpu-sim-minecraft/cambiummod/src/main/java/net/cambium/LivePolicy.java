package net.cambium;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Comparator;
import java.util.concurrent.CompletableFuture;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.item.BowItem;
import net.minecraft.core.component.DataComponents;
import net.minecraft.world.entity.EquipmentSlot;
import net.minecraft.world.entity.ai.attributes.Attributes;
import net.minecraft.world.entity.projectile.arrow.Arrow;
import net.minecraft.world.item.component.ItemAttributeModifiers;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.Vec3;
import net.cambium.mixin.ArrowStateAccess;
import net.cambium.ObservationEncoder.Equipment;
import net.cambium.ObservationEncoder.Weapon;

final class LivePolicy implements AutoCloseable {

	static final int DEFAULT_PORT = 8766;
	private static final int[] SIZES = {9, 2, 2, 2, 9, 5};
	private static final int[] FORWARD = {0, 1, 1, 0, -1, -1, -1, 0, 1};
	private static final int[] STRAFE = {0, 0, -1, -1, -1, 0, 1, 1, 1};
	private static final float[] YAW = {-20, -10, -5, -2, 0, 2, 5, 10, 20};
	private static final float[] PITCH = {-5, -2, 0, 2, 5};
	private final HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(2))
			.version(HttpClient.Version.HTTP_1_1).build();
	private final URI base;
	private SimulationContract contract;
	private String contractHash;
	private java.util.List<JsonObject> loadouts;

	LivePolicy(int port) {
		if (port < 1 || port > 65535) throw new IllegalArgumentException("Invalid inference port");
		base = URI.create("http://127.0.0.1:" + port);
	}

	CompletableFuture<JsonObject> info() {
		return request("/info", null).thenApply(info -> {
			if (info.has("catalogue_hash") && !PolicyCatalogue.HASH.equals(info.get("catalogue_hash").getAsString())) {
				throw new IllegalArgumentException("Policy catalogue differs from mod; rebuild/reinstall mod and restart inference service");
			}
			if (info.has("contract")) {
				contract = new SimulationContract(info.getAsJsonObject("contract"));
				contractHash = info.get("contract_hash").getAsString();
				if (!contractHash.matches("[0-9a-f]{64}")) throw new IllegalArgumentException("Invalid policy contract fingerprint");
				try {
					loadouts = new ReplayTrace.Environment(info.getAsJsonObject("environment")).loadouts;
				} catch (java.io.IOException error) {
					throw new IllegalArgumentException("Invalid policy scenario", error);
				}
				contract.validateLoadouts(loadouts);
				int limit = integer(info.get("max_ticks"));
				if (limit < 1 || limit > 1000000) throw new IllegalArgumentException("Invalid policy round limit");
				return info;
			}
			JsonObject schema = info.getAsJsonObject("schema");
			if (!"minecraft_sim_duel_22_v1".equals(schema.get("id").getAsString())
					|| !"26.3".equals(schema.get("minecraft").getAsString())
					|| integer(schema.get("observation_dim")) != 22 || integer(schema.get("ticks_per_action")) != 1
					|| !schema.get("action_sizes").equals(ReplayAnalysis.JSON.toJsonTree(SIZES))
					|| !schema.get("action_names").equals(ReplayAnalysis.JSON.toJsonTree(new String[]{"movement", "jump", "sprint", "attack", "yaw", "pitch"}))
					|| !schema.get("yaw_bins").equals(ReplayAnalysis.JSON.toJsonTree(YAW))
					|| !schema.get("pitch_bins").equals(ReplayAnalysis.JSON.toJsonTree(PITCH))
					|| !schema.get("movement").equals(ReplayAnalysis.JSON.toJsonTree(new String[]{"neutral", "forward", "forward_right", "right",
							"backward_right", "backward", "backward_left", "left", "forward_left"}))) {
				throw new IllegalArgumentException("Inference observation/action schema does not match this mod");
			}
			int limit = integer(info.get("max_ticks"));
			if (limit < 1 || limit > 1000000) throw new IllegalArgumentException("Invalid policy round limit");
			return info;
		});
	}

	CompletableFuture<JsonObject> act(String session, int tick, float[][] observations) {
		JsonObject body = new JsonObject();
		body.addProperty("protocol", 1);
		body.addProperty("session", session);
		body.addProperty("tick", tick);
		if (contractHash != null) body.addProperty("contract_hash", contractHash);
		body.add("observations", ReplayAnalysis.JSON.toJsonTree(observations));
		return request("/act", body);
	}

	int observationCount() { return contract == null ? 22 : contract.observationCount(); }

	boolean fullBowBridgeReady() {
		return contract != null && contract.supportsFullBowContract() && loadouts != null && loadouts.size() == 2
				&& loadouts.stream().allMatch(loadout -> loadout.getAsJsonArray("weapons")
						.contains(ReplayAnalysis.JSON.toJsonTree("minecraft:bow")) && loadout.get("arrows").getAsInt() > 0);
	}

	ReplayTrace.Controls[] mapActions(JsonObject response, String session, int tick) {
		if (contract == null) return actions(response, session, tick);
		if (integer(response.get("protocol")) != 1 || !session.equals(response.get("session").getAsString())
				|| integer(response.get("tick")) != tick || !contractHash.equals(response.get("contract_hash").getAsString())) {
			throw new IllegalArgumentException("Stale response or policy contract mismatch");
		}
		JsonArray actors = response.getAsJsonArray("actions");
		if (actors.size() != 2) throw new IllegalArgumentException("Expected two policy actions");
		return new ReplayTrace.Controls[]{contract.decode(actors.get(0).getAsJsonArray(), loadouts.get(0)),
				contract.decode(actors.get(1).getAsJsonArray(), loadouts.get(1))};
	}

	float[] encode(ServerPlayer source, ServerPlayer target, float[] rays, int sourceIndex) {
		var ownVelocity = source.getDeltaMovement();
		var targetVelocity = target.getDeltaMovement();
		var own = new ObservationEncoder.State(source.getX(), source.getY(), source.getZ(), source.getHealth(), source.getYRot(), source.getXRot(),
				source.isAlive(), ownVelocity.x, ownVelocity.y, ownVelocity.z, bowState(source));
		var other = new ObservationEncoder.State(target.getX(), target.getY(), target.getZ(), target.getHealth(), target.getYRot(), target.getXRot(),
				target.isAlive(), targetVelocity.x, targetVelocity.y, targetVelocity.z, bowState(target));
		float[] base = ObservationEncoder.observe(own, other, rays);
		if (contract == null) return base;
		float[] catalogue = java.util.Arrays.copyOf(base, SimulationContract.FEATURES.size());
		catalogue[22] = weaponIsAxe(source) ? 1 : 0;
		catalogue[23] = weaponIsAxe(target) ? 1 : 0;
		catalogue[24] = usingShield(source) ? 1 : 0;
		catalogue[25] = usingShield(target) ? 1 : 0;
		int cooldown = shieldCooldown(source);
		catalogue[26] = cooldown > 0 ? 1 : 0;
		catalogue[27] = source instanceof ReplayPlayer bot && bot.observedOpponentDisabled() ? 1 : 0;
		catalogue[28] = Math.min(cooldown / 100.0F, 1);
		catalogue[29] = source.getAttackStrengthScale(0.5F);
		catalogue[30] = usingShield(source) ? Math.min(source.getTicksUsingItem() / 5.0F, 1) : 0;
		catalogue[31] = source.getInventory().getSelectedSlot();
		catalogue[32] = target.getInventory().getSelectedSlot();
		ObservationEncoder.velocityObservation(catalogue, own, other);
		ObservationEncoder.movementObservation(catalogue, source.onGround(), target.onGround(), source.isSprinting(), target.isSprinting());
		ObservationEncoder.equipment(catalogue, equipment(source), equipment(target), weaponSlots(source));
		ObservationEncoder.bowObservation(catalogue, own, other, nearestOpponentArrow(source, target));
		return contract.project(catalogue);
	}

	// Vanilla reference: minecraft-26.3-src/net/minecraft/world/entity/LivingEntity.java, getUseItem (3589-3590);
	// item-use identity, rather than the generic using flag, controls shield observations.
	static boolean usingShield(ServerPlayer player) {
		return player.isUsingItem() && player.getUseItem().is(Items.SHIELD);
	}

	// Vanilla reference: minecraft-26.3-src/net/minecraft/world/item/BowItem.java, use (95-103), and
	// minecraft-26.3-src/net/minecraft/world/entity/LivingEntity.java, getTicksUsingItem (3597-3598).
	private static ObservationEncoder.BowState bowState(ServerPlayer player) {
		boolean using = usingBow(player);
		int drawTicks = using ? player.getTicksUsingItem() : 0;
		return new ObservationEncoder.BowState(ordinaryArrowCount(player), using, drawTicks,
				using ? BowItem.getPowerForTime(drawTicks) : 0, player.getMainHandItem().is(Items.BOW));
	}

	// Vanilla reference: minecraft-26.3-src/net/minecraft/world/item/BowItem.java, use (95-103), which
	// checks Player#getProjectile; this feature intentionally counts only ordinary finite ARROW stacks.
	private static int ordinaryArrowCount(ServerPlayer player) {
		int count = 0;
		for (ItemStack stack : player.getInventory().getNonEquipmentItems()) {
			if (stack.is(Items.ARROW)) count += stack.getCount();
		}
		return Math.min(count, 64);
	}

	// Vanilla reference: minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java,
	// tick (173-269), isInGround (358), and getPickupItemStackOrigin (669); Projectile#getOwner (88-89).
	private static ObservationEncoder.ArrowState nearestOpponentArrow(ServerPlayer source, ServerPlayer target) {
		AABB search = source.getBoundingBox().inflate(50.0);
		Arrow nearest = source.level().getEntitiesOfClass(Arrow.class, search, arrow -> arrow.isAlive()
				&& arrow.getOwner() == target && arrow.getPickupItemStackOrigin().is(Items.ARROW)
				&& !((ArrowStateAccess) (Object) arrow).cambium$isInGround()
				&& arrow.getDeltaMovement().lengthSqr() > 0)
			.stream().filter(arrow -> arrow.position().distanceToSqr(source.position()) <= 2500.0)
			.min(Comparator.comparingDouble(arrow -> arrow.position().distanceToSqr(source.position()))).orElse(null);
		if (nearest == null) return null;
		Vec3 velocity = nearest.getDeltaMovement();
		return new ObservationEncoder.ArrowState(nearest.getX(), nearest.getY(), nearest.getZ(), velocity.x, velocity.y, velocity.z);
	}

	// Vanilla reference: minecraft-26.3-src/net/minecraft/world/entity/LivingEntity.java, getUseItem (3589-3590);
	// `isUsingItem` alone also covers shields, so only the actual main-hand bow counts.
	static boolean usingBow(ServerPlayer player) {
		return player.isUsingItem() && player.getUseItem().is(Items.BOW);
	}

	private static Weapon[] weaponSlots(ServerPlayer player) {
		Weapon[] slots = new Weapon[9];
		for (int slot = 0; slot < slots.length; slot++) slots[slot] = weapon(player.getInventory().getItem(slot));
		return slots;
	}

	private static Equipment equipment(ServerPlayer player) {
		return new Equipment(player.getOffhandItem().is(Items.SHIELD), player.getAttributeValue(Attributes.ATTACK_DAMAGE),
				player.getAttributeValue(Attributes.ATTACK_SPEED), player.getAttributeValue(Attributes.ARMOR),
				player.getAttributeValue(Attributes.ARMOR_TOUGHNESS));
	}

	static Weapon weapon(ItemStack stack) {
		String name = BuiltInRegistries.ITEM.getKey(stack.getItem()).toString();
		boolean axe = name.endsWith("_axe");
		boolean present = !stack.isEmpty() && (axe || name.endsWith("_sword"));
		if (!present) return new Weapon(false, false, 0, 0);
		var modifiers = stack.getOrDefault(DataComponents.ATTRIBUTE_MODIFIERS, ItemAttributeModifiers.EMPTY);
		return new Weapon(true, axe, modifiers.compute(Attributes.ATTACK_DAMAGE, 1.0, EquipmentSlot.MAINHAND),
				modifiers.compute(Attributes.ATTACK_SPEED, 4.0, EquipmentSlot.MAINHAND));
	}

	private static boolean weaponIsAxe(ServerPlayer player) {
		return BuiltInRegistries.ITEM.getKey(player.getMainHandItem().getItem()).toString().endsWith("_axe");
	}

	static int shieldCooldown(ServerPlayer player) {
		return Math.round(player.getCooldowns().getCooldownPercent(new ItemStack(Items.SHIELD), 0) * 100);
	}

	private CompletableFuture<JsonObject> request(String path, JsonObject body) {
		var builder = HttpRequest.newBuilder(base.resolve(path)).timeout(Duration.ofSeconds(2));
		HttpRequest request = body == null ? builder.GET().build() : builder.header("Content-Type", "application/json")
				.POST(HttpRequest.BodyPublishers.ofString(ReplayAnalysis.JSON.toJson(body))).build();
		return client.sendAsync(request, HttpResponse.BodyHandlers.ofString()).thenApply(response -> {
			if (response.statusCode() != 200 || response.body().length() > 1024 * 1024) {
				throw new IllegalStateException("Inference HTTP " + response.statusCode() + ": "
						+ response.body().substring(0, Math.min(200, response.body().length())));
			}
			JsonObject result = JsonParser.parseString(response.body()).getAsJsonObject();
			if (integer(result.get("protocol")) != 1) throw new IllegalArgumentException("Unsupported inference protocol");
			return result;
		});
	}

	static ReplayTrace.Controls[] actions(JsonObject response, String session, int tick) {
		if (integer(response.get("protocol")) != 1 || !session.equals(response.get("session").getAsString())
				|| integer(response.get("tick")) != tick) throw new IllegalArgumentException("Stale or mismatched inference response");
		JsonArray actors = response.getAsJsonArray("actions");
		if (actors.size() != 2) throw new IllegalArgumentException("Expected two policy actions");
		ReplayTrace.Controls[] result = new ReplayTrace.Controls[2];
		for (int actor = 0; actor < 2; actor++) {
			JsonArray heads = actors.get(actor).getAsJsonArray();
			if (heads.size() != 6) throw new IllegalArgumentException("Expected six action heads");
			int[] action = new int[6];
			for (int head = 0; head < 6; head++) {
				action[head] = integer(heads.get(head));
				if (action[head] < 0 || action[head] >= SIZES[head]) throw new IllegalArgumentException("Policy action outside head range");
			}
			int move = action[0];
			result[actor] = new ReplayTrace.Controls(FORWARD[move] > 0, FORWARD[move] < 0,
					STRAFE[move] > 0, STRAFE[move] < 0, action[1] != 0, action[2] != 0, false,
					action[3] != 0, YAW[action[4]], PITCH[action[5]]);
		}
		return result;
	}

	private static int integer(JsonElement value) {
		if (value == null || !value.isJsonPrimitive() || !value.getAsJsonPrimitive().isNumber()) {
			throw new IllegalArgumentException("Expected an integer protocol value");
		}
		double number = value.getAsDouble();
		if (!Double.isFinite(number) || number != Math.rint(number) || number < Integer.MIN_VALUE || number > Integer.MAX_VALUE) {
			throw new IllegalArgumentException("Invalid integer protocol value");
		}
		return (int) number;
	}

	@Override
	public void close() {
		client.shutdownNow();
	}
}