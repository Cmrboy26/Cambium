package net.cambium;

import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.damagesource.DamageSource;

interface LiveSession extends AutoCloseable {
	void reset();
	default void afterEnvironmentBuilt() {}
	default void afterActorSpawned(int index) {}
	void tick(ServerPlayer viewer);
	void spawnHuman();
	boolean isHumanSlot(int index);
	void setBot(ReplayPlayer bot);
	void updateTargets();
	boolean ownsProjectile(Entity owner);
	boolean allowDeath(LivingEntity entity);
	boolean allowDamage(LivingEntity entity, DamageSource source);
	void selectHuman(ServerPlayer viewer, int slot);
	void restoreHuman();
	void restoreHuman(ServerPlayer player);
	void cancelForLifecycle();
	void cancelRequest();
	String status();
	String deploymentSummary();
	@Override void close();
}
