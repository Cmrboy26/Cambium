package net.cambium;

import net.minecraft.core.BlockPos;
import net.minecraft.network.protocol.game.ServerboundPlayerActionPacket;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.InteractionResult;
import net.minecraft.world.item.BlockItem;
import net.minecraft.world.phys.BlockHitResult;

public final class BridgeInteractionBridge {
	private BridgeInteractionBridge() {}

	private static LiveBridge active(ServerPlayer player) {
		var session = SimulationReplay.active;
		return session != null && session.level == player.level() && session.live instanceof LiveBridge bridge ? bridge : null;
	}

	public static boolean participant(ServerPlayer player) {
		LiveBridge bridge = active(player);
		return bridge != null && bridge.participant(player);
	}
	public static boolean locked(ServerPlayer player) {
		LiveBridge bridge = active(player);
		return bridge != null && bridge.participant(player) && bridge.locked();
	}
	public static boolean attackAllowed(ServerPlayer player, int target) {
		LiveBridge bridge = active(player);
		return bridge == null || bridge.attackAllowed(player, target);
	}
	public static boolean breakBlock(ServerPlayer player, BlockPos position, ServerboundPlayerActionPacket.Action action) {
		LiveBridge bridge = active(player);
		if (bridge == null) return false;
		if (bridge.participant(player)) {
			bridge.humanBreak(action, position);
			return true;
		}
		return false;
	}
	public static InteractionResult useBlock(ServerPlayer player, InteractionHand hand, BlockHitResult hit) {
		LiveBridge bridge = active(player);
		if (bridge == null) return null;
		if (bridge.participant(player)) {
			if (bridge.humanPlace(player, hand, hit)) return InteractionResult.SUCCESS_SERVER;
			if (bridge.humanUseAllowed(player, hand)) return player.gameMode.useItem(player, player.level(), player.getItemInHand(hand), hand);
			return InteractionResult.FAIL;
		}
		if (bridge.contains(hit.getBlockPos()) || bridge.contains(hit.getBlockPos().relative(hit.getDirection()))) {
			var item = player.getItemInHand(hand).getItem();
			if (item instanceof BlockItem block && block.getBlock().defaultBlockState().hasBlockEntity()) return InteractionResult.FAIL;
			if (!player.getItemInHand(hand).isEmpty() && !(item instanceof BlockItem)) return InteractionResult.FAIL;
		}
		return null;
	}
	public static boolean useAllowed(ServerPlayer player, InteractionHand hand) {
		LiveBridge bridge = active(player);
		return bridge == null || !bridge.participant(player) || bridge.humanUseAllowed(player, hand);
	}
}