package net.cambium.mixin;

import net.cambium.ProjectileTrackingBridge;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.projectile.throwableitemprojectile.ThrownEnderpearl;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.Level;
import net.minecraft.server.network.ServerGamePacketListenerImpl;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.Redirect;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(ThrownEnderpearl.class)
abstract class PearlTrackingMixin {
	// Minecraft 26.3: ThrownEnderpearl.java:43-44.
	@Inject(method = "<init>(Lnet/minecraft/world/level/Level;Lnet/minecraft/world/entity/LivingEntity;Lnet/minecraft/world/item/ItemStack;)V", at = @At("RETURN"))
	private void cambium$trackRoundPearl(Level level, LivingEntity owner, ItemStack item, CallbackInfo callback) {
		ProjectileTrackingBridge.track((ThrownEnderpearl) (Object) this);
	}

	// Minecraft 26.3: ThrownEnderpearl.java:116-117; ServerGamePacketListenerImpl.java:414, 1410-1417.
	@Redirect(method = "onHit", at = @At(value = "INVOKE", target = "Lnet/minecraft/server/network/ServerGamePacketListenerImpl;isAcceptingMessages()Z"))
	private boolean cambium$allowSyntheticOwner(ServerGamePacketListenerImpl connection) {
		return connection.isAcceptingMessages() || ProjectileTrackingBridge.syntheticPearlOwner((ThrownEnderpearl) (Object) this);
	}
}