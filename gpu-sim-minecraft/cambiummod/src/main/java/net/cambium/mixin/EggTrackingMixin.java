package net.cambium.mixin;

import net.cambium.ProjectileTrackingBridge;
import net.minecraft.util.RandomSource;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.projectile.throwableitemprojectile.ThrownEgg;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.Level;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.Redirect;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(ThrownEgg.class)
abstract class EggTrackingMixin {

	// Minecraft 26.3: ThrownEgg.java:32-33 and ThrownEgg.java:60-86.
	@Inject(method = "<init>(Lnet/minecraft/world/level/Level;Lnet/minecraft/world/entity/LivingEntity;Lnet/minecraft/world/item/ItemStack;)V", at = @At("RETURN"))
	private void cambium$trackRoundEgg(Level level, LivingEntity owner, ItemStack item, CallbackInfo callback) {
		ProjectileTrackingBridge.track((ThrownEgg) (Object) this);
	}

	@Redirect(method = "onHit", at = @At(value = "INVOKE", target = "Lnet/minecraft/util/RandomSource;nextInt(I)I", ordinal = 0))
	private int cambium$preventRoundHatching(RandomSource random, int bound) {
		int result = random.nextInt(bound);
		return ProjectileTrackingBridge.suppressHatch((ThrownEgg) (Object) this) ? 1 : result;
	}
}