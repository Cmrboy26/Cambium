package net.cambium.mixin;

import net.minecraft.world.entity.Display;
import net.minecraft.world.item.ItemDisplayContext;
import net.minecraft.world.item.ItemStack;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Invoker;

@Mixin(Display.ItemDisplay.class)
public interface ItemDisplayAccess {

	// Replay adapter access to vanilla Display.ItemDisplay#setItemStack/setItemTransform (Display.java:595-599).
	@Invoker("setItemStack")
	void cambium$setItemStack(ItemStack item);

	@Invoker("setItemTransform")
	void cambium$setItemTransform(ItemDisplayContext transform);
}
