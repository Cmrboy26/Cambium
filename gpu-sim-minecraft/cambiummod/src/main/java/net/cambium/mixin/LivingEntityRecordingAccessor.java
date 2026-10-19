package net.cambium.mixin;

import net.minecraft.world.entity.LivingEntity;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;
import org.spongepowered.asm.mixin.gen.Invoker;

@Mixin(LivingEntity.class)
public interface LivingEntityRecordingAccessor {

	@Accessor("noJumpDelay")
	int cambium$getJumpDelay();

	@Invoker("detectEquipmentUpdates")
	void cambium$refreshEquipment();
}