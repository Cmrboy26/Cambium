package net.cambium.mixin;

import net.minecraft.world.entity.Display;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Invoker;

@Mixin(Display.class)
public interface DisplayAccess {

	// Replay adapter access to vanilla Display#setGlowColorOverride (Display.java:368).
	@Invoker("setGlowColorOverride")
	void cambium$setGlowColorOverride(int color);
}
