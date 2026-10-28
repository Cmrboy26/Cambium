package net.cambium.mixin;

import net.minecraft.client.multiplayer.MultiPlayerGameMode;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;
import org.spongepowered.asm.mixin.gen.Invoker;

@Mixin(MultiPlayerGameMode.class)
public interface PlayerControlGameModeAccess {
	@Accessor("carriedIndex")
	int cambium$carriedIndex();

	@Invoker("ensureHasSentCarriedItem")
	void cambium$sendCarriedItem();
}