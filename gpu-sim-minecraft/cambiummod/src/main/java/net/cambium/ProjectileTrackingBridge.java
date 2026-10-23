package net.cambium;

import net.minecraft.world.entity.projectile.arrow.Arrow;

public final class ProjectileTrackingBridge {

	// Source: minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/Arrow.java, owner constructor (29-34);
	// minecraft-26.3-src/net/minecraft/world/item/BowItem.java, releaseUsing preserves this shooter.
	public static void track(Arrow arrow) {
		SimulationReplay.trackArrow(arrow);
	}

	private ProjectileTrackingBridge() {}
}
