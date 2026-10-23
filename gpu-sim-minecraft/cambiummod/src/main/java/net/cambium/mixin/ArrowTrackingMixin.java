package net.cambium.mixin;

import net.cambium.ProjectileTrackingBridge;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.projectile.arrow.Arrow;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.Level;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(Arrow.class)
abstract class ArrowTrackingMixin {

	// Vanilla reference: minecraft-26.3-src/net/minecraft/world/item/BowItem.java, releaseUsing (26-61), creates
	// minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/Arrow.java, owner constructor (29-34).
	@Inject(method = "<init>(Lnet/minecraft/world/level/Level;Lnet/minecraft/world/entity/LivingEntity;Lnet/minecraft/world/item/ItemStack;Lnet/minecraft/world/item/ItemStack;)V", at = @At("RETURN"))
	private void cambium$trackOwnedRoundArrow(Level level, LivingEntity owner, ItemStack pickupItem,
			ItemStack firedFromWeapon, CallbackInfo callback) {
		ProjectileTrackingBridge.track((Arrow) (Object) this);
	}
}
