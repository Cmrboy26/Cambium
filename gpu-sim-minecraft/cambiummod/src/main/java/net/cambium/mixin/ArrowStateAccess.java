package net.cambium.mixin;

import net.minecraft.world.entity.projectile.arrow.AbstractArrow;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Invoker;

@Mixin(AbstractArrow.class)
public interface ArrowStateAccess {

	// Vanilla reference: minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/AbstractArrow.java, isInGround (358-359).
	@Invoker("isInGround")
	boolean cambium$isInGround();
}
