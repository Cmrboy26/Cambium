package net.cambium.mixin;

import net.minecraft.client.player.LocalPlayer;
import net.minecraft.world.phys.Vec2;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Invoker;

@Mixin(LocalPlayer.class)
public interface ReplayInputAccess {

	@Invoker("modifyInputSpeedForSquareMovement")
	static Vec2 cambium$scaleMovement(Vec2 input) {
		throw new AssertionError("Mixin not applied");
	}
}