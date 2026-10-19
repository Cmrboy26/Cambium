package net.cambium;

final class ObservationEncoder {

	record State(double x, double y, double z, float health, float yaw, float pitch, boolean alive) {}

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