package net.cambium;

import java.util.HashSet;
import java.util.Set;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.core.component.DataComponents;
import net.minecraft.core.BlockPos;
import net.minecraft.world.entity.EquipmentSlot;
import net.minecraft.world.entity.ai.attributes.Attributes;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.entity.projectile.arrow.Arrow;
import net.minecraft.world.item.BowItem;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.item.component.ItemAttributeModifiers;
import net.minecraft.world.level.ClipContext;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec3;
import net.cambium.mixin.ArrowStateAccess;

final class ClientObservation {

	static final Set<String> FEATURES = supportedFeatures();
	static final Set<String> ACTIONS = Set.of("movement", "jump", "sprint", "yaw", "pitch", "use_item", "sneak");

	private static Set<String> supportedFeatures() {
		var result = new HashSet<String>(SimulationContract.FEATURES.subList(0, 22));
		result.addAll(SimulationContract.FEATURES.subList(22, 79));
		result.remove("opponent.shield_disabled_observed");
		result.remove("opponent.weapon_slot");
		result.addAll(SimulationContract.FEATURES.subList(79, 102));
		return Set.copyOf(result);
	}

	static float[] sample(LivePolicy policy, LocalPlayer source, Player target, Vec3 targetVelocity) {
		Vec3 eye = source.getEyePosition();
		BlockPos origin = BlockPos.containing(eye);
		boolean raysRequired = SimulationContract.FEATURES.subList(10, 18).stream().anyMatch(policy::observes);
		boolean arrowsRequired = SimulationContract.FEATURES.subList(95, 102).stream().anyMatch(policy::observes);
		if ((raysRequired || arrowsRequired) && !source.level().hasChunksAt(origin.offset(-50, 0, -50), origin.offset(50, 0, 50))) {
			throw new IllegalStateException("The 50-block observation area is not loaded");
		}
		float[] rays = new float[8];
		if (raysRequired) for (int ray = 0; ray < 8; ray++) {
			double angle = Math.toRadians(ray * 45.0);
			Vec3 end = eye.add(-Math.sin(angle) * 50.0, 0, Math.cos(angle) * 50.0);
			HitResult hit = source.level().clip(new ClipContext(eye, end, ClipContext.Block.OUTLINE, ClipContext.Fluid.NONE, source));
			rays[ray] = hit.getType() == HitResult.Type.MISS ? 1 : (float) (eye.distanceTo(hit.getLocation()) / 50.0);
		}
		var own = state(source, source.getDeltaMovement(), arrowCount(source));
		var other = state(target, targetVelocity, 0);
		ObservationEncoder.ArrowState arrow = null;
		if (arrowsRequired) {
			Arrow nearest = null;
			double distance = 2500.0;
			for (Arrow candidate : source.level().getEntitiesOfClass(Arrow.class, source.getBoundingBox().inflate(50.0))) {
				if (!candidate.isAlive() || ((ArrowStateAccess) (Object) candidate).cambium$isInGround()
						|| candidate.getDeltaMovement().lengthSqr() == 0) continue;
				double current = candidate.position().distanceToSqr(source.position());
				if (current > 2500.0) continue;
				if (candidate.getOwner() == null) throw new IllegalStateException("Nearby arrow owner is unavailable");
				if (candidate.getOwner() == target && current < distance) { nearest = candidate; distance = current; }
			}
			if (nearest != null) {
				Vec3 velocity = nearest.getDeltaMovement();
				arrow = new ObservationEncoder.ArrowState(nearest.getX(), nearest.getY(), nearest.getZ(), velocity.x, velocity.y, velocity.z);
			}
		}
		float[] catalogue = ObservationEncoder.catalogue(own, other, rays, source.onGround(), target.onGround(),
				source.isSprinting(), target.isSprinting(), arrow);
		equipment(catalogue, source, target);
		return policy.project(catalogue);
	}

	static void equipment(float[] catalogue, Player source, Player target) {
		boolean ownShield = LivePolicy.usingShield(source);
		combat(catalogue, LivePolicy.weaponIsAxe(source), LivePolicy.weaponIsAxe(target), ownShield,
				LivePolicy.usingShield(target), LivePolicy.shieldCooldown(source), source.getAttackStrengthScale(0.5F),
				source.getTicksUsingItem(), source.getInventory().getSelectedSlot());
		ObservationEncoder.equipment(catalogue, visibleEquipment(source), visibleEquipment(target), LivePolicy.weaponSlots(source));
	}

	static void combat(float[] catalogue, boolean ownAxe, boolean opponentAxe, boolean ownShield, boolean opponentShield,
			int cooldown, float charge, int shieldTicks, int slot) {
		catalogue[22] = ownAxe ? 1 : 0;
		catalogue[23] = opponentAxe ? 1 : 0;
		catalogue[24] = ownShield ? 1 : 0;
		catalogue[25] = opponentShield ? 1 : 0;
		catalogue[26] = cooldown > 0 ? 1 : 0;
		catalogue[28] = Math.min(cooldown / 100.0F, 1);
		catalogue[29] = charge;
		catalogue[30] = ownShield ? Math.min(shieldTicks / 5.0F, 1) : 0;
		catalogue[31] = slot;
	}

	private static ObservationEncoder.Equipment visibleEquipment(Player player) {
		return new ObservationEncoder.Equipment(player.getOffhandItem().is(Items.SHIELD), visibleDamage(player.getMainHandItem()),
				player.getAttributeValue(Attributes.ATTACK_SPEED), player.getAttributeValue(Attributes.ARMOR),
				player.getAttributeValue(Attributes.ARMOR_TOUGHNESS));
	}

	static double visibleDamage(ItemStack stack) {
		var modifiers = stack.getOrDefault(DataComponents.ATTRIBUTE_MODIFIERS, ItemAttributeModifiers.EMPTY);
		return modifiers.compute(Attributes.ATTACK_DAMAGE, 1.0, EquipmentSlot.MAINHAND);
	}

	private static ObservationEncoder.State state(Player player, Vec3 velocity, int arrows) {
		boolean using = player.isUsingItem() && player.getUseItem().is(Items.BOW);
		int ticks = using ? player.getTicksUsingItem() : 0;
		var bow = new ObservationEncoder.BowState(arrows, using, ticks, using ? BowItem.getPowerForTime(ticks) : 0,
				player.getMainHandItem().is(Items.BOW));
		return new ObservationEncoder.State(player.getX(), player.getY(), player.getZ(), player.getHealth(),
				player.getYRot(), player.getXRot(), player.isAlive(), velocity.x, velocity.y, velocity.z, bow);
	}

	private static int arrowCount(LocalPlayer player) {
		int count = 0;
		for (var stack : player.getInventory().getNonEquipmentItems()) if (stack.is(Items.ARROW)) count += stack.getCount();
		return Math.min(count, 64);
	}

	private ClientObservation() {}
}