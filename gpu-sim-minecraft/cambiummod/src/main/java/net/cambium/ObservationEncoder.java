package net.cambium;

final class ObservationEncoder {

	// Adapter data mirrors LivingEntity#getUseItem/getTicksUsingItem and BowItem#getPowerForTime; it is not simulator bow physics.
	record BowState(int arrowCount, boolean usingBow, int drawTicks, float power, boolean equipped) {}
	record ArrowState(double x, double y, double z, double velocityX, double velocityY, double velocityZ) {}
	record State(double x, double y, double z, float health, float yaw, float pitch, boolean alive,
			double velocityX, double velocityY, double velocityZ, BowState bow) {
		State(double x, double y, double z, float health, float yaw, float pitch, boolean alive,
				double velocityX, double velocityY, double velocityZ) {
			this(x, y, z, health, yaw, pitch, alive, velocityX, velocityY, velocityZ,
					new BowState(0, false, 0, 0, false));
		}
	}
	record Weapon(boolean present, boolean axe, double damage, double speed) {}
	record Equipment(boolean shield, double damage, double speed, double armor, double toughness) {}

	static void equipment(float[] catalogue, Equipment own, Equipment opponent, Weapon[] ownSlots) {
		for (int actor = 0; actor < 2; actor++) {
			Equipment state = actor == 0 ? own : opponent;
			catalogue[33 + actor] = state.shield() ? 1 : 0;
			catalogue[35 + actor] = equipmentFraction(state.damage(), 10);
			catalogue[37 + actor] = equipmentFraction(state.speed(), 4);
			catalogue[39 + actor] = equipmentFraction(state.armor(), 20);
			catalogue[41 + actor] = equipmentFraction(state.toughness(), 12);
		}
		for (int slot = 0; slot < 9; slot++) {
			int start = 43 + slot * 4;
			java.util.Arrays.fill(catalogue, start, start + 4, 0);
			Weapon weapon = slot < ownSlots.length ? ownSlots[slot] : null;
			if (weapon == null || !weapon.present()) continue;
			catalogue[start] = 1;
			catalogue[start + 1] = weapon.axe() ? 1 : 0;
			catalogue[start + 2] = equipmentFraction(weapon.damage(), 10);
			catalogue[start + 3] = equipmentFraction(weapon.speed(), 4);
		}
	}

	static void velocityObservation(float[] catalogue, State source, State target) {
		catalogue[79] = (float) source.velocityX();
		catalogue[80] = (float) source.velocityY();
		catalogue[81] = (float) source.velocityZ();
		catalogue[82] = (float) (target.velocityX() - source.velocityX());
		catalogue[83] = (float) (target.velocityY() - source.velocityY());
		catalogue[84] = (float) (target.velocityZ() - source.velocityZ());
	}

	static void movementObservation(float[] catalogue, boolean sourceGrounded, boolean targetGrounded,
			boolean sourceSprinting, boolean targetSprinting) {
		catalogue[85] = sourceGrounded ? 1 : 0;
		catalogue[86] = targetGrounded ? 1 : 0;
		catalogue[87] = sourceSprinting ? 1 : 0;
		catalogue[88] = targetSprinting ? 1 : 0;
	}

	// Vanilla reference: minecraft-26.3-src/net/minecraft/world/item/BowItem.java, getPowerForTime (74-82), and
	// minecraft-26.3-src/net/minecraft/world/entity/LivingEntity.java, getTicksUsingItem (3597-3598). Arrow deltas here
	// are relative to the actor's sampled movement; projectile selection belongs to the live adapter.
	static void bowObservation(float[] catalogue, State source, State target, ArrowState arrow) {
		if (catalogue.length < 102) throw new IllegalArgumentException("Bow observations need catalogue indices 89..101");
		BowState ownBow = source.bow();
		BowState opponentBow = target.bow();
		catalogue[89] = Math.clamp(ownBow.arrowCount() / 64.0F, 0.0F, 1.0F);
		catalogue[90] = ownBow.usingBow() ? 1 : 0;
		catalogue[91] = ownBow.usingBow() ? Math.clamp(ownBow.power(), 0.0F, 1.0F) : 0;
		catalogue[92] = ownBow.equipped() ? 1 : 0;
		catalogue[93] = opponentBow.equipped() ? 1 : 0;
		catalogue[94] = opponentBow.usingBow() ? 1 : 0;
		for (int index = 95; index <= 101; index++) catalogue[index] = 0;
		if (arrow == null) return;
		catalogue[95] = 1;
		catalogue[96] = (float) ((arrow.x() - source.x()) / 10.0);
		catalogue[97] = (float) ((arrow.y() - source.y()) / 10.0);
		catalogue[98] = (float) ((arrow.z() - source.z()) / 10.0);
		catalogue[99] = (float) (arrow.velocityX() - source.velocityX());
		catalogue[100] = (float) (arrow.velocityY() - source.velocityY());
		catalogue[101] = (float) (arrow.velocityZ() - source.velocityZ());
	}

	private static float equipmentFraction(double value, double maximum) {
		return (float) Math.clamp(value / maximum, 0.0, 1.0);
	}

	static float[] observe(State source, State target, float[] rays) {
		if (rays.length != 8) throw new IllegalArgumentException("Expected eight wall rays");
		double relativeX = target.x - source.x;
		double relativeY = target.y - source.y;
		double relativeZ = target.z - source.z;
		double distance = Math.sqrt(relativeX * relativeX + relativeY * relativeY + relativeZ * relativeZ);
		float[] result = new float[22];
		result[0] = source.health / 20.0F;
		result[1] = (float) (wrap(source.yaw) / 180.0);
		result[2] = source.pitch / 90.0F;
		result[3] = target.alive ? 1 : 0;
		result[4] = target.health / 20.0F;
		result[5] = (float) (relativeX / 10.0);
		result[6] = (float) (relativeY / 10.0);
		result[7] = (float) (relativeZ / 10.0);
		result[8] = (float) (wrap(target.yaw) / 180.0);
		result[9] = target.pitch / 90.0F;
		System.arraycopy(rays, 0, result, 10, 8);
		result[18] = (float) Math.min(distance / 10.0, 1.0);
		if (distance > 0.01) {
			double radians = 0.017453292519943295;
			double yaw = source.yaw * radians;
			double pitch = source.pitch * radians;
			result[19] = (float) ((-Math.sin(yaw) * Math.cos(pitch) * relativeX - Math.sin(pitch) * relativeY
					+ Math.cos(yaw) * Math.cos(pitch) * relativeZ) / distance);
			result[20] = (float) (wrap(Math.atan2(-relativeX, relativeZ) / radians - source.yaw) / 180.0);
			result[21] = (float) ((Math.atan2(-relativeY, Math.max(Math.hypot(relativeX, relativeZ), 0.01)) / radians - source.pitch) / 90.0);
		}
		for (float value : result) if (!Float.isFinite(value)) throw new IllegalArgumentException("Non-finite live observation");
		return result;
	}

	private static double wrap(double angle) {
		return angle - 360.0 * Math.floor((angle + 180.0) / 360.0);
	}

	private ObservationEncoder() {}
}