package net.cambium;

import net.minecraft.world.entity.projectile.arrow.Arrow;
import net.minecraft.world.entity.projectile.throwableitemprojectile.ThrownEgg;
import net.minecraft.world.entity.projectile.throwableitemprojectile.Snowball;

public final class ProjectileTrackingBridge {

	// Source: minecraft-26.3-src/net/minecraft/world/entity/projectile/arrow/Arrow.java, owner constructor (29-34);
	// minecraft-26.3-src/net/minecraft/world/item/BowItem.java, releaseUsing preserves this shooter.
	public static void track(Arrow arrow) {
		SimulationReplay.trackArrow(arrow);
	}

	public static void track(ThrownEgg egg) {
		SimulationReplay.trackEgg(egg);
	}

	public static void track(Snowball snowball) {
		SimulationReplay.trackSnowball(snowball);
	}

	public static void track(net.minecraft.world.entity.projectile.throwableitemprojectile.ThrownEnderpearl pearl) {
		SimulationReplay.trackPearl(pearl);
	}

	public static void track(net.minecraft.world.entity.projectile.hurtingprojectile.windcharge.WindCharge wind) {
		SimulationReplay.trackWind(wind);
	}

	public static boolean syntheticPearlOwner(net.minecraft.world.entity.projectile.throwableitemprojectile.ThrownEnderpearl pearl) {
		return pearl.getOwner() instanceof ReplayPlayer && SimulationReplay.ownsPearl(pearl);
	}

	public static boolean suppressHatch(ThrownEgg egg) {
		return SimulationReplay.ownsEgg(egg);
	}

	private ProjectileTrackingBridge() {}
}
