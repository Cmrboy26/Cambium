package net.cambium.mixin;

import net.cambium.BridgeInteractionBridge;
import net.minecraft.network.protocol.game.ClientboundSetHeldSlotPacket;
import net.minecraft.network.protocol.game.ServerboundAttackPacket;
import net.minecraft.network.protocol.game.ServerboundContainerClickPacket;
import net.minecraft.network.protocol.game.ServerboundInteractPacket;
import net.minecraft.network.protocol.game.ServerboundPlayerActionPacket;
import net.minecraft.network.protocol.game.ServerboundSetCarriedItemPacket;
import net.minecraft.network.protocol.game.ServerboundUseItemPacket;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.server.network.ServerGamePacketListenerImpl;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(ServerGamePacketListenerImpl.class)
abstract class BridgePacketMixin {
	@Shadow public ServerPlayer player;

	private boolean cambium$serverThread() { return player.level().getServer().isSameThread(); }

	@Inject(method = "handlePlayerAction", at = @At("HEAD"), cancellable = true)
	private void cambium$action(ServerboundPlayerActionPacket packet, CallbackInfo callback) {
		if (!cambium$serverThread() || !BridgeInteractionBridge.participant(player)) return;
		var action = packet.getAction();
		if (action == ServerboundPlayerActionPacket.Action.DROP_ITEM || action == ServerboundPlayerActionPacket.Action.DROP_ALL_ITEMS
				|| action == ServerboundPlayerActionPacket.Action.SWAP_ITEM_WITH_OFFHAND || action == ServerboundPlayerActionPacket.Action.STAB
				|| action == ServerboundPlayerActionPacket.Action.RELEASE_USE_ITEM && BridgeInteractionBridge.locked(player)) {
			player.stopUsingItem();
			player.inventoryMenu.sendAllDataToRemote();
			callback.cancel();
		}
	}

	@Inject(method = "handleContainerClick", at = @At("HEAD"), cancellable = true)
	private void cambium$inventory(ServerboundContainerClickPacket packet, CallbackInfo callback) {
		if (cambium$serverThread() && BridgeInteractionBridge.participant(player)) {
			player.containerMenu.sendAllDataToRemote();
			callback.cancel();
		}
	}

	@Inject(method = "handleSetCarriedItem", at = @At("HEAD"), cancellable = true)
	private void cambium$slot(ServerboundSetCarriedItemPacket packet, CallbackInfo callback) {
		if (cambium$serverThread() && BridgeInteractionBridge.locked(player)) {
			player.connection.send(new ClientboundSetHeldSlotPacket(player.getInventory().getSelectedSlot()));
			callback.cancel();
		}
	}

	@Inject(method = "handleUseItem", at = @At("HEAD"), cancellable = true)
	private void cambium$use(ServerboundUseItemPacket packet, CallbackInfo callback) {
		if (cambium$serverThread() && !BridgeInteractionBridge.useAllowed(player, packet.hand())) {
			player.connection.ackBlockChangesUpTo(packet.sequence());
			callback.cancel();
		}
	}

	@Inject(method = "handleAttack", at = @At("HEAD"), cancellable = true)
	private void cambium$attack(ServerboundAttackPacket packet, CallbackInfo callback) {
		if (cambium$serverThread() && !BridgeInteractionBridge.attackAllowed(player, packet.entityId())) callback.cancel();
	}

	@Inject(method = "handleInteract", at = @At("HEAD"), cancellable = true)
	private void cambium$interact(ServerboundInteractPacket packet, CallbackInfo callback) {
		if (cambium$serverThread() && BridgeInteractionBridge.participant(player)) callback.cancel();
	}
}