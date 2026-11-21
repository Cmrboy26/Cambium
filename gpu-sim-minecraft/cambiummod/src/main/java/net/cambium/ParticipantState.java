package net.cambium;

import java.util.ArrayList;
import java.util.List;
import net.cambium.mixin.LivingEntityRecordingAccessor;
import net.cambium.mixin.PlayerEquipmentAccess;
import net.minecraft.network.protocol.game.ClientboundSetEntityMotionPacket;
import net.minecraft.network.protocol.game.ClientboundSetHeldSlotPacket;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.effect.MobEffectInstance;
import net.minecraft.world.entity.ai.attributes.Attributes;
import net.minecraft.world.food.FoodData;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.GameType;
import net.minecraft.world.phys.Vec3;

final class ParticipantState {
	private final ServerLevel level;
	private final Vec3 position;
	private final Vec3 velocity;
	private final float yaw;
	private final float pitch;
	private final GameType gameMode;
	private final List<ItemStack> inventory = new ArrayList<>();
	private final int selected;
	private final float health;
	private final float absorption;
	private final double maxAbsorption;
	private final FoodData food;
	private final List<MobEffectInstance> effects;
	private final int fireTicks;
	private final double fallDistance;
	private final int damageCooldown;
	private final int hurtTime;
	private final boolean sprinting;
	private final boolean sneaking;

	ParticipantState(ServerPlayer player) {
		level = player.level();
		position = player.position();
		velocity = player.getDeltaMovement();
		yaw = player.getYRot();
		pitch = player.getXRot();
		gameMode = player.gameMode.getGameModeForPlayer();
		for (int index = 0; index < player.getInventory().getContainerSize(); index++) inventory.add(player.getInventory().getItem(index).copy());
		selected = player.getInventory().getSelectedSlot();
		health = player.getHealth();
		absorption = player.getAbsorptionAmount();
		maxAbsorption = player.getAttribute(Attributes.MAX_ABSORPTION).getBaseValue();
		food = player.getFoodData();
		effects = player.getActiveEffects().stream().map(MobEffectInstance::new).toList();
		fireTicks = player.getRemainingFireTicks();
		fallDistance = player.fallDistance;
		damageCooldown = player.damageCooldownTime;
		hurtTime = player.hurtTime;
		sprinting = player.isSprinting();
		sneaking = player.isShiftKeyDown();
	}

	void restore(ServerPlayer player) {
		player.closeContainer();
		player.stopUsingItem();
		player.getInventory().clearContent();
		for (int index = 0; index < inventory.size(); index++) player.getInventory().setItem(index, inventory.get(index).copy());
		player.getInventory().setSelectedSlot(selected);
		((LivingEntityRecordingAccessor) (Object) player).cambium$refreshEquipment();
		player.setGameMode(gameMode);
		player.teleportTo(level, position.x, position.y, position.z, java.util.Set.of(), yaw, pitch, true);
		player.setDeltaMovement(velocity);
		player.getAttribute(Attributes.MAX_ABSORPTION).setBaseValue(maxAbsorption);
		player.removeAllEffects();
		for (MobEffectInstance effect : effects) player.addEffect(new MobEffectInstance(effect));
		player.setHealth(health);
		player.setAbsorptionAmount(absorption);
		((PlayerEquipmentAccess) (Object) player).cambium$setFoodData(food);
		player.damageCooldownTime = damageCooldown;
		player.hurtTime = hurtTime;
		player.fallDistance = fallDistance;
		player.setRemainingFireTicks(fireTicks);
		player.setSprinting(sprinting);
		player.setShiftKeyDown(sneaking);
		player.inventoryMenu.broadcastChanges();
		player.connection.send(new ClientboundSetHeldSlotPacket(selected));
		player.connection.send(new ClientboundSetEntityMotionPacket(player));
	}
}
