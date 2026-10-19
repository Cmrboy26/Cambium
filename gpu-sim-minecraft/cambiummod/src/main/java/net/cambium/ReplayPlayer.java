package net.cambium;

import com.mojang.authlib.GameProfile;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import io.netty.channel.ChannelFutureListener;
import java.util.UUID;
import net.cambium.mixin.ReplayInputAccess;
import net.cambium.mixin.PlayerEquipmentAccess;
import net.cambium.mixin.LivingEntityRecordingAccessor;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.Identifier;
import net.minecraft.core.component.DataComponents;
import net.minecraft.network.protocol.Packet;
import net.minecraft.network.protocol.PacketFlow;
import net.minecraft.network.protocol.game.ClientboundSetEntityMotionPacket;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ClientInformation;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.server.network.CommonListenerCookie;
import net.minecraft.server.network.ServerGamePacketListenerImpl;
import net.minecraft.util.Mth;
import net.minecraft.util.Unit;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.damagesource.DamageSource;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.EquipmentSlot;
import net.minecraft.world.entity.Pose;
import net.minecraft.world.entity.ai.attributes.Attributes;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.food.FoodData;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.item.component.AttackRange;
import net.minecraft.world.item.component.SwingAnimation;
import net.minecraft.world.level.GameType;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.EntityHitResult;
import net.minecraft.world.phys.Vec2;
import net.minecraft.world.phys.Vec3;

final class ReplayPlayer extends ServerPlayer {

	private ReplayTrace.Controls controls;
	private Vec3 pendingMotion;
	private int missTime;
	private final SprintInput sprintInput = new SprintInput();
	private final ReplayFoodData replayFood = new ReplayFoodData();
	private boolean crouching;
	private boolean absolute;
	private boolean analysisEnabled;
	private boolean liveControlled;
	private ServerPlayer liveTarget;
	private boolean attackSuppressed;
	private int itemUseDelay;
	private int observedOpponentDisable;
	private JsonObject loadout;
	private JsonArray damageEvents = new JsonArray();
	final int replayIndex;

	ReplayPlayer(MinecraftServer server, ServerLevel level, ReplayTrace.Actor initial, Vec3 origin, boolean absolute) {
		super(server, level, new GameProfile(UUID.randomUUID(), "Sim_" + (initial.id() == 0 ? "A" : "B")),
				ClientInformation.createDefault());
		this.replayIndex = initial.id();
		this.absolute = absolute;
		this.foodData = replayFood;
		this.connection = new ServerGamePacketListenerImpl(server, new net.minecraft.network.Connection(PacketFlow.SERVERBOUND), this,
				CommonListenerCookie.createInitial(getGameProfile(), false)) {
			@Override
			public boolean hasClientLoaded() {
				return true;
			}

			@Override
			public void send(Packet<?> packet, ChannelFutureListener listener) {
				if (packet instanceof ClientboundSetEntityMotionPacket motion && motion.id() == getId()) {
					pendingMotion = motion.movement();
				}
			}
		};
		gameMode.changeGameModeForPlayer(GameType.SURVIVAL);
		applyState(initial, origin);
		super.doTick();
		applyState(initial, origin);
	}

	@Override
	public void tick() {
	}

	@Override
	protected void tickRegeneration() {
		if (replayFood.live) super.tickRegeneration();
	}

	static final class ReplayFoodData extends FoodData {
		boolean live;

		@Override
		public void tick(ServerPlayer player) {
			if (live) super.tick(player);
		}

		@Override
		public void addExhaustion(float amount) {
			if (live) super.addExhaustion(amount);
		}
	}

	@Override
	public boolean isClientAuthoritative() {
		return false;
	}

	@Override
	public Vec3 getKnownMovement() {
		return getDeltaMovement();
	}

	@Override
	public boolean canHarmPlayer(Player other) {
		return (liveControlled || other instanceof ReplayPlayer) && super.canHarmPlayer(other);
	}

	@Override
	protected boolean considersEntityAsAlly(Entity other) {
		return (!liveControlled && !(other instanceof ReplayPlayer)) || super.considersEntityAsAlly(other);
	}

	void applyState(ReplayTrace.Actor actor, Vec3 origin) {
		sprintInput.reset(actor.pose() == 1);
		crouching = actor.pose() == 1;
		observedOpponentDisable = 0;
		stopUsingItem();
		equip(EquipmentSlot.MAINHAND, actor.weapon());
		equip(EquipmentSlot.OFFHAND, actor.shield_equipped() ? "minecraft:shield" : "minecraft:air");
		for (String slot : SimulationContract.ARMOR_SLOTS) {
			equip(armorSlot(slot), actor.armor().has(slot) ? actor.armor().get(slot).getAsString() : "minecraft:air");
		}
		ItemStack shield = new ItemStack(Items.SHIELD);
		getCooldowns().removeCooldown(getCooldowns().getCooldownGroup(shield));
		if (actor.shield_cooldown() > 0) getCooldowns().addCooldown(shield, actor.shield_cooldown());
		if (actor.using_shield()) {
			startUsingItem(InteractionHand.OFF_HAND);
			useItemRemaining = Math.max(0, getUseItem().getUseDuration(this) - actor.shield_use_ticks());
		}
		itemUseDelay = actor.item_use_delay();
		attackSuppressed = actor.attack_suppressed();
		setOldPosAndRot();
		setPos(origin.x + actor.position()[0], origin.y + actor.position()[1], origin.z + actor.position()[2]);
		setDeltaMovement(new Vec3(actor.velocity()[0], actor.velocity()[1], actor.velocity()[2]));
		setYRot(actor.yaw());
		setXRot(actor.pitch());
		setYHeadRot(actor.yaw());
		setYBodyRot(actor.yaw());
		setOnGround(actor.on_ground());
		setSprinting(actor.sprinting());
		setShiftKeyDown(actor.pose() == 1);
		setPose(actor.pose() == 0 ? Pose.STANDING : actor.pose() == 1 ? Pose.CROUCHING : Pose.SWIMMING);
		setHealth(actor.health());
		getAttribute(Attributes.MAX_ABSORPTION).setBaseValue(actor.absorption());
		setAbsorptionAmount(actor.absorption());
		damageCooldownTime = actor.damage_cooldown();
		hurtTime = actor.hurt_time();
		attackStrengthTicker = actor.attack_strength_ticks();
		getAbilities().invulnerable = absolute;
		noPhysics = absolute;
	}

	void prepare(ReplayTrace.Controls input) {
		controls = input;
		if (input.select_weapon()) {
			if (loadout != null && !loadout.getAsJsonArray("weapons").contains(ReplayAnalysis.JSON.toJsonTree(input.weapon()))) {
				throw new IllegalArgumentException("Replay selected an item outside the scenario loadout");
			}
			if (equip(EquipmentSlot.MAINHAND, input.weapon())) resetAttackStrengthTicker();
		}
		attackSuppressed = isUsingItem();
		if (itemUseDelay > 0) itemUseDelay--;
		if (!input.use_item() || !isAlive()) stopUsingItem();
		if (analysisEnabled) damageEvents = new JsonArray();
		if (liveControlled) acceptMotion(); else pendingMotion = null;
		setOldPosAndRot();
		setYRot(getYRot() + input.yaw_delta());
		setXRot(Mth.clamp(getXRot() + input.pitch_delta(), -90, 90));
		setYHeadRot(getYRot());
		setShiftKeyDown(input.sneak());
		if (damageCooldownTime > 0) {
			damageCooldownTime--;
		}
		if (missTime > 0) {
			missTime--;
		}
		if (!input.attack()) {
			missTime = 0;
		}
	}

	HitResult pickAttack() {
		if (!isAlive() || !controls.attack() || missTime > 0 || attackSuppressed) {
			return null;
		}
		return AttackRange.defaultFor(this).getClosesetHit(this, 1.0F,
				entity -> (liveControlled ? entity == liveTarget : entity instanceof ReplayPlayer) && entity != this && entity.isAlive());
	}

	void miss(HitResult hit) {
		if (hit.getType() == HitResult.Type.MISS) {
			resetAttackStrengthTicker();
			missTime = 10;
		}
	}

	void acceptMotion() {
		if (pendingMotion != null) {
			setDeltaMovement(pendingMotion);
			pendingMotion = null;
		}
	}

	void advanceInput() {
		if (isAlive()) {
			crouching = sprintInput.crouching(!getAbilities().flying && !isSwimming() && !isPassenger()
					&& canPlayerFitWithinBlocksAndEntitiesWhen(Pose.CROUCHING),
					!isSleeping() && !canPlayerFitWithinBlocksAndEntitiesWhen(Pose.STANDING));
			if (controls.use_item() && !isUsingItem() && itemUseDelay == 0) {
				itemUseDelay = 4;
				if (getOffhandItem().is(Items.SHIELD) && !getCooldowns().isOnCooldown(getOffhandItem())) startUsingItem(InteractionHand.OFF_HAND);
			}
			boolean possible = !isMobilityRestricted() && (getFoodData().getFoodLevel() > 6 || getAbilities().mayfly);
			setSprinting(sprintInput.update(controls, isSprinting(),
					possible && !isMovingSlowly() && !isUsingItem(),
					possible && (!horizontalCollision || minorHorizontalCollision), isUsingItem()));
			super.doTick();
			if (observedOpponentDisable > 0) observedOpponentDisable--;
		}
	}

	@Override
	public boolean isCrouching() {
		return crouching;
	}

	private boolean isMovingSlowly() {
		return isCrouching() || isVisuallyCrawling();
	}

	@Override
	protected boolean isHorizontalCollisionMinor(Vec3 movement) {
		float angle = getYRot() * 0.017453292F;
		double sine = Mth.sin(angle);
		double cosine = Mth.cos(angle);
		return SprintInput.minorCollision(new Vec3(xxa * cosine - zza * sine, 0, zza * cosine + xxa * sine), movement);
	}

	static final class SprintInput {
		private int triggerTime;
		private boolean previousForward;
		private boolean previousSneak;

		boolean crouching(boolean canCrouch, boolean forced) {
			return canCrouch && (previousSneak || forced);
		}

		static boolean minorCollision(Vec3 desired, Vec3 accepted) {
			double desiredSquared = desired.horizontalDistanceSqr();
			double acceptedSquared = accepted.horizontalDistanceSqr();
			if (desiredSquared < (double) 1.0E-5F || acceptedSquared < (double) 1.0E-5F) return false;
			double dot = desired.x * accepted.x + desired.z * accepted.z;
			return Math.acos(dot / Math.sqrt(desiredSquared * acceptedSquared)) < (double) 0.13962634F;
		}

		boolean update(ReplayTrace.Controls controls, boolean sprinting, boolean canStart, boolean canContinue, boolean slowedByItem) {
			if (triggerTime > 0) triggerTime--;
			boolean forward = controls.forward() && !controls.backward();
			if (previousSneak || controls.backward() || slowedByItem) triggerTime = 0;
			boolean start = false;
			if (!sprinting && canStart && forward) {
				if (!previousForward) {
					if (triggerTime > 0) {
						start = true;
					} else {
						triggerTime = 7;
					}
				}
				start |= controls.sprint();
			}
			previousForward = forward;
			previousSneak = controls.sneak();
			return (sprinting || start) && forward && canContinue;
		}

		void reset() {
			reset(false);
		}

		void reset(boolean sneaking) {
			triggerTime = 0;
			previousForward = false;
			previousSneak = sneaking;
		}
	}

	@Override
	public void applyInput() {
		if (controls == null) {
			return;
		}
		Vec2 input = new Vec2((controls.left() ? 1 : 0) - (controls.right() ? 1 : 0),
				(controls.forward() ? 1 : 0) - (controls.backward() ? 1 : 0)).normalized().scale(0.98F);
		if (isUsingItem()) input = input.scale(0.2F);
		if (isCrouching() || isVisuallyCrawling()) {
			input = input.scale((float) getAttributeValue(Attributes.SNEAKING_SPEED));
		}
		input = ReplayInputAccess.cambium$scaleMovement(input);
		xxa = input.x;
		zza = input.y;
		jumping = controls.jump();
	}

	void swingReplay() {
		swing(InteractionHand.MAIN_HAND, SwingAnimation.DEFAULT, true);
	}

	void setAnalysisEnabled(boolean enabled) {
		analysisEnabled = enabled;
		damageEvents = new JsonArray();
	}

	void setLiveControlled() {
		liveControlled = true;
		replayFood.live = false;
	}

	void setLoadout(JsonObject loadout) { this.loadout = loadout.deepCopy(); }

	void setLiveTarget(ServerPlayer target) { liveTarget = target; }

	@Override
	public void attack(Entity target) {
		boolean blocking = liveControlled && target instanceof ServerPlayer player && player.isBlocking();
		super.attack(target);
		if (blocking && ((ServerPlayer) target).getCooldowns().isOnCooldown(new ItemStack(Items.SHIELD))) observedOpponentDisable = 100;
	}

	private static EquipmentSlot armorSlot(String slot) {
		return switch (slot) {
			case "boots" -> EquipmentSlot.FEET;
			case "leggings" -> EquipmentSlot.LEGS;
			case "chestplate" -> EquipmentSlot.CHEST;
			case "helmet" -> EquipmentSlot.HEAD;
			default -> throw new IllegalArgumentException("Unsupported armor slot");
		};
	}

	private boolean equip(EquipmentSlot slot, String name) {
		String current = BuiltInRegistries.ITEM.getKey(getItemBySlot(slot).getItem()).toString();
		if (current.equals(name)) return false;
		ItemStack stack = name.equals("minecraft:air") ? ItemStack.EMPTY
				: new ItemStack(BuiltInRegistries.ITEM.getValue(Identifier.parse(name)));
		if (!stack.isEmpty()) stack.set(DataComponents.UNBREAKABLE, Unit.INSTANCE);
		setItemSlot(slot, stack);
		((LivingEntityRecordingAccessor) (Object) this).cambium$refreshEquipment();
		if (slot == EquipmentSlot.MAINHAND) ((PlayerEquipmentAccess) (Object) this).cambium$setLastMainHand(stack.copy());
		return true;
	}

	int weaponSlot() {
		if (loadout == null) return 0;
		String selected = BuiltInRegistries.ITEM.getKey(getMainHandItem().getItem()).toString();
		for (int index = 0; index < loadout.getAsJsonArray("weapons").size(); index++) {
			if (loadout.getAsJsonArray("weapons").get(index).getAsString().equals(selected)) return index;
		}
		throw new IllegalStateException("Selected weapon outside scenario loadout");
	}

	boolean weaponIsAxe() { return BuiltInRegistries.ITEM.getKey(getMainHandItem().getItem()).toString().endsWith("_axe"); }
	int shieldCooldown() { return Math.round(getCooldowns().getCooldownPercent(new ItemStack(Items.SHIELD), 0) * 100); }
	boolean observedOpponentDisabled() { return observedOpponentDisable > 0; }

	@Override
	protected void blockUsingItem(ServerLevel level, LivingEntity attacker, DamageSource source, float damage, boolean fullyBlocked) {
		boolean active = isBlocking();
		super.blockUsingItem(level, attacker, source, damage, fullyBlocked);
		if (active && shieldCooldown() > 0 && attacker instanceof ReplayPlayer player) player.observedOpponentDisable = 100;
	}

	@Override
	protected void dropEquipment(ServerLevel level) {
		if (!liveControlled) super.dropEquipment(level);
	}

	@Override
	public boolean hurtServer(ServerLevel level, DamageSource source, float damage) {
		if (!analysisEnabled) return super.hurtServer(level, source, damage);
		float healthBefore = getHealth();
		float absorptionBefore = getAbsorptionAmount();
		Vec3 velocityBefore = getDeltaMovement();
		boolean accepted = super.hurtServer(level, source, damage);
		JsonObject event = new JsonObject();
		event.addProperty("source_actor", source.getEntity() instanceof ReplayPlayer player ? player.replayIndex : -1);
		event.addProperty("source", source.toString());
		event.addProperty("damage", damage);
		event.addProperty("accepted", accepted);
		event.addProperty("health_lost", healthBefore - getHealth());
		event.addProperty("absorption_lost", absorptionBefore - getAbsorptionAmount());
		event.add("velocity_before", vector(velocityBefore));
		event.add("velocity_after_default_knockback", vector(getDeltaMovement()));
		damageEvents.add(event);
		return accepted;
	}

	JsonObject analysisState(Vec3 origin) {
		JsonObject state = new JsonObject();
		state.addProperty("id", replayIndex);
		state.add("position", vector(position().subtract(origin)));
		state.add("velocity", vector(getDeltaMovement()));
		state.addProperty("yaw", getYRot());
		state.addProperty("pitch", getXRot());
		state.addProperty("on_ground", onGround());
		state.addProperty("sprinting", isSprinting());
		state.addProperty("pose", getPose() == Pose.STANDING ? 0 : getPose() == Pose.CROUCHING ? 1 : getPose() == Pose.SWIMMING ? 2 : -1);
		state.addProperty("pose_name", getPose().name());
		state.addProperty("health", getHealth());
		state.addProperty("absorption", getAbsorptionAmount());
		state.addProperty("weapon", BuiltInRegistries.ITEM.getKey(getMainHandItem().getItem()).toString());
		state.addProperty("shield_equipped", getOffhandItem().is(Items.SHIELD));
		state.addProperty("using_shield", isUsingItem());
		state.addProperty("shield_use_ticks", getTicksUsingItem());
		state.addProperty("shield_cooldown", shieldCooldown());
		state.addProperty("item_use_delay", itemUseDelay);
		state.addProperty("attack_suppressed", attackSuppressed);
		JsonObject armor = new JsonObject();
		for (String slot : SimulationContract.ARMOR_SLOTS) armor.addProperty(slot, BuiltInRegistries.ITEM.getKey(getItemBySlot(armorSlot(slot)).getItem()).toString());
		state.add("armor", armor);
		state.addProperty("max_absorption", getMaxAbsorption());
		state.addProperty("damage_cooldown", damageCooldownTime);
		state.addProperty("hurt_time", hurtTime);
		state.addProperty("attack_strength_ticks", attackStrengthTicker);
		state.addProperty("attack_charge", getAttackStrengthScale(0.5F));
		state.addProperty("attack_miss_time", missTime);
		state.addProperty("fall_distance", fallDistance);
		state.addProperty("horizontal_collision", horizontalCollision);
		state.addProperty("minor_horizontal_collision", minorHorizontalCollision);
		state.addProperty("food_level", getFoodData().getFoodLevel());
		state.addProperty("food_saturation", getFoodData().getSaturationLevel());
		state.add("controls", ReplayAnalysis.JSON.toJsonTree(controls));
		state.add("damage_events", damageEvents.deepCopy());
		return state;
	}

	JsonObject analysisPick(HitResult pick, ReplayPlayer opponent, Vec3 origin) {
		JsonObject result = new JsonObject();
		Vec3 eye = getEyePosition();
		Vec3 look = getHeadLookAngle();
		double projection = getKnownMovement().dot(look);
		result.addProperty("actor", replayIndex);
		result.addProperty("hit_type", pick == null ? "NO_ATTACK" : pick.getType().name());
		result.addProperty("target", pick instanceof EntityHitResult hit && hit.getEntity() instanceof ReplayPlayer player ? player.replayIndex : -1);
		result.add("eye", vector(eye.subtract(origin)));
		result.add("look", vector(look));
		result.addProperty("movement_projection", projection);
		result.addProperty("reach", getAttributeValue(Attributes.ENTITY_INTERACTION_RANGE) + Math.max(0.0, projection));
		if (pick != null) result.add("hit_position", vector(pick.getLocation().subtract(origin)));
		var bounds = opponent.getBoundingBox().move(origin.scale(-1));
		result.add("target_bounds_min", vector(new Vec3(bounds.minX, bounds.minY, bounds.minZ)));
		result.add("target_bounds_max", vector(new Vec3(bounds.maxX, bounds.maxY, bounds.maxZ)));
		result.addProperty("eye_inside_target", opponent.getBoundingBox().contains(eye));
		result.addProperty("eye_to_target_box", Math.sqrt(opponent.getBoundingBox().distanceToSqr(eye)));
		result.addProperty("near_coincident_knockback", position().subtract(opponent.position()).horizontalDistanceSqr() < (double) 1.0E-5F);
		return result;
	}

	JsonObject analysisOutcome(HitResult pick, boolean executed) {
		JsonObject result = new JsonObject();
		result.addProperty("target", -1);
		result.addProperty("attempted", executed && pick.getType() != HitResult.Type.BLOCK);
		result.addProperty("accepted", false);
		result.addProperty("damage", 0);
		result.addProperty("health_lost", 0);
		if (executed && pick instanceof EntityHitResult hit && hit.getEntity() instanceof ReplayPlayer target) {
			result.addProperty("target", target.replayIndex);
			for (int index = target.damageEvents.size() - 1; index >= 0; index--) {
				JsonObject event = target.damageEvents.get(index).getAsJsonObject();
				if (event.get("source_actor").getAsInt() != replayIndex) continue;
				for (String field : new String[]{"accepted", "damage", "health_lost"}) result.add(field, event.get(field));
				break;
			}
		}
		return result;
	}

	static JsonArray vector(Vec3 value) {
		return ReplayAnalysis.JSON.toJsonTree(new double[]{value.x, value.y, value.z}).getAsJsonArray();
	}
}