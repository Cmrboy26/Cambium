package net.cambium.mixin;

import net.cambium.MovementRecorder;
import net.minecraft.client.player.LocalPlayer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(LocalPlayer.class)
public abstract class LocalPlayerRecordingMixin {

	@Inject(method = "aiStep", at = @At("HEAD"))
	private void cambium$beginMovement(CallbackInfo callback) {
		MovementRecorder.beforeMovement((LocalPlayer) (Object) this);
	}

	@Inject(method = "aiStep", at = @At(value = "INVOKE",
			target = "Lnet/minecraft/client/player/AbstractClientPlayer;aiStep()V"))
	private void cambium$beforeMovement(CallbackInfo callback) {
		MovementRecorder.captureInput((LocalPlayer) (Object) this);
	}

	@Inject(method = "tick", at = @At("RETURN"))
	private void cambium$afterMovement(CallbackInfo callback) {
		MovementRecorder.afterMovement((LocalPlayer) (Object) this);
	}
}