package net.cambium.mixin;

import net.minecraft.client.player.LocalPlayer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

@Mixin(LocalPlayer.class)
public interface LocalPlayerRecordingAccessor {

	@Accessor("sprintTriggerTime")
	int cambium$getSprintTriggerTime();
}