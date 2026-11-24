package net.cambium;

import java.util.HashMap;
import java.util.Map;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.Identifier;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.AABB;

final class BridgeWorld {
	private final SimulationReplay.Session session;
	private final BridgeArena arena;
	private final Map<Integer, BlockState> edits = new HashMap<>();
	private final BlockState[] states = {
			Blocks.AIR.defaultBlockState(),
			BuiltInRegistries.BLOCK.getValue(Identifier.parse("minecraft:blue_wool")).defaultBlockState(),
			BuiltInRegistries.BLOCK.getValue(Identifier.parse("minecraft:red_wool")).defaultBlockState()
	};

	BridgeWorld(SimulationReplay.Session session, BridgeArena arena, Map<Integer, BlockState> previousEdits) {
		if (session == null || arena == null || previousEdits == null || session.originalBlocks == null || !session.buildArena) {
			throw new IllegalStateException("Bridge requires a snapshotted checkpoint arena");
		}
		this.session = session;
		this.arena = arena;
		for (var entry : arena.overlay().entrySet()) write(entry.getKey(), entry.getValue());
		for (var entry : previousEdits.entrySet()) {
			BlockPos position = session.environmentPosition(entry.getKey());
			if (!session.level.hasChunkAt(position) || session.level.getBlockEntity(position) != null) {
				throw new IllegalStateException("Bridge terrain cell is unavailable: " + position);
			}
			rememberEdit(entry.getKey(), entry.getValue(), position);
			session.level.setBlock(position, entry.getValue(), Block.UPDATE_CLIENTS | Block.UPDATE_KNOWN_SHAPE);
		}
	}

	Map<Integer, Integer> overlay() { return arena.overlay(); }
	Map<Integer, BlockState> edits() { return Map.copyOf(edits); }
	void restoreBase() { arena.restoreBase(); }
	float miningProgress(int actor) { return arena.miningProgress(actor); }
	int miningDelay(int actor) { return arena.miningDelay(actor); }
	void cancelMining(int actor) { arena.cancelMining(actor); }

	void verify() {
		Map<Integer, Integer> overlay = arena.overlay();
		for (int index = 0; index < session.trace.environment.cells.length; index++) {
			BlockPos position = session.environmentPosition(index);
			int owner = overlay.getOrDefault(index, 0);
			BlockState expected = owner != 0 ? states[owner] : edits.getOrDefault(index,
					session.trace.environment.cells[index] == 1 ? Blocks.STONE.defaultBlockState() : states[0]);
			if (!session.level.hasChunkAt(position) || session.level.getBlockEntity(position) != null) {
				throw new IllegalStateException("Bridge arena cell is unloaded or contains a block entity at " + position);
			}
			BlockState actual = session.level.getBlockState(position);
			if (!actual.equals(expected)) rememberEdit(index, actual, position);
		}
	}

	private void rememberEdit(int index, BlockState state, BlockPos position) {
		if (state.hasBlockEntity() || !state.getFluidState().isEmpty()) {
			throw new IllegalStateException("Bridge arena edits cannot contain block entities or fluids at " + position);
		}
		arena.editBase(index, !state.getCollisionShape(session.level, position).isEmpty());
		edits.put(index, state);
	}

	boolean place(int actor, BlockPos target, BlockPos destination, int remaining, AABB[] worldPlayers) {
		BlockPos origin = BlockPos.containing(session.origin);
		BlockPos localTarget = target.subtract(origin), localDestination = destination.subtract(origin);
		AABB[] players = new AABB[worldPlayers.length];
		for (int index = 0; index < players.length; index++) players[index] = worldPlayers[index].move(-origin.getX(), -origin.getY(), -origin.getZ());
		if (!arena.place(actor, localTarget.getX(), localTarget.getY(), localTarget.getZ(),
				localDestination.getX(), localDestination.getY(), localDestination.getZ(), remaining, players)) return false;
		write(arena.cell(localDestination.getX(), localDestination.getY(), localDestination.getZ()), actor + 1);
		return true;
	}

	boolean mine(int actor, BlockPos target, boolean onGround) {
		BlockPos local = target.subtract(BlockPos.containing(session.origin));
		if (!arena.mine(actor, local.getX(), local.getY(), local.getZ(), onGround)) return false;
		write(arena.cell(local.getX(), local.getY(), local.getZ()), 0);
		return true;
	}

	void clearOverlay() {
		var previous = arena.overlay();
		arena.clear();
		for (int index : previous.keySet()) write(index, 0);
	}

	private void write(int index, int owner) {
		BlockPos position = session.environmentPosition(index);
		if (!session.level.hasChunkAt(position) || session.level.getBlockEntity(position) != null) {
			throw new IllegalStateException("Bridge terrain cell is unavailable: " + position);
		}
		session.level.setBlock(position, states[owner], Block.UPDATE_CLIENTS | Block.UPDATE_KNOWN_SHAPE);
		if (!session.level.getBlockState(position).equals(states[owner])) throw new IllegalStateException("Could not change Bridge terrain at " + position);
	}
}