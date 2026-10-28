package net.cambium.mixin;

import net.cambium.ProjectileTrackingBridge;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.entity.projectile.hurtingprojectile.windcharge.WindCharge;
import net.minecraft.world.level.Level;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(WindCharge.class)
abstract class WindTrackingMixin {
	// Minecraft 26.3: WindCharge.java:43-44.
	@Inject(method = "<init>(Lnet/minecraft/world/entity/player/Player;Lnet/minecraft/world/level/Level;DDD)V", at = @At("RETURN"))
	private void cambium$trackRoundWind(Player owner, Level level, double x, double y, double z, CallbackInfo callback) {
		ProjectileTrackingBridge.track((WindCharge) (Object) this);
	}
}