package net.cambium.mixin;

import net.cambium.BridgeInteractionBridge;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.network.protocol.game.ServerboundPlayerActionPacket;
import net.minecraft.network.protocol.game.ClientboundBlockUpdatePacket;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.server.level.ServerPlayerGameMode;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.InteractionResult;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.Level;
import net.minecraft.world.phys.BlockHitResult;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

@Mixin(ServerPlayerGameMode.class)
abstract class BridgeGameModeMixin {
	@Shadow @Final protected ServerPlayer player;

	@Inject(method = "handleBlockBreakAction", at = @At("HEAD"), cancellable = true)
	private void cambium$break(BlockPos position, ServerboundPlayerActionPacket.Action action, Direction direction,
			int maxY, int sequence, CallbackInfo callback) {
		if (BridgeInteractionBridge.breakBlock(player, position, action)) {
			player.connection.send(new ClientboundBlockUpdatePacket(player.level(), position));
			callback.cancel();
		}
	}

	@Inject(method = "useItemOn", at = @At("HEAD"), cancellable = true)
	private void cambium$useOn(ServerPlayer player, Level level, ItemStack stack, InteractionHand hand,
			BlockHitResult hit, CallbackInfoReturnable<InteractionResult> callback) {
		InteractionResult result = BridgeInteractionBridge.useBlock(player, hand, hit);
		if (result != null) callback.setReturnValue(result);
	}
}