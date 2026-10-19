package net.cambium;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.StringJoiner;

final class MovementCsv implements AutoCloseable {

	record State(double x, double y, double z, double vx, double vy, double vz,
			float yaw, float pitch, boolean onGround, boolean horizontalCollision,
			boolean verticalCollision, int jumpDelay, boolean sprinting,
			boolean crouching, boolean flying, int pose, boolean previousForward,
			boolean previousSneak, int sprintTriggerTime, int sprintWindow,
			int foodLevel, boolean mayFly, double fallDistance, boolean minorHorizontalCollision) {
		boolean continues(State previous) {
			return x == previous.x && y == previous.y && z == previous.z
					&& vx == previous.vx && vy == previous.vy && vz == previous.vz
					&& onGround == previous.onGround && jumpDelay == previous.jumpDelay && pose == previous.pose;
		}
	}

	private final BufferedWriter writer;
	private long rows;

	MovementCsv(Path path) throws IOException {
		writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8, StandardOpenOption.CREATE_NEW);
		try {
			writer.write("tick,input,forward,backward,left,right,jump,sneak,sprint,x,y,z,yaw,pitch,on_ground,"
					+ "vx,vy,vz,horizontal_collision,vertical_collision,jump_delay,sprinting,crouching,flying,"
					+ "before_x,before_y,before_z,before_vx,before_vy,before_vz,before_on_ground,before_jump_delay,"
					+ "pose,previous_forward,previous_sneak,sprint_trigger_time,sprint_window,food_level,may_fly,fall_distance,"
					+ "minor_horizontal_collision\n");
		} catch (IOException error) {
			writer.close();
			throw error;
		}
	}

	void append(int input, State before, State after) throws IOException {
		StringJoiner row = new StringJoiner(",");
		row.add(Long.toString(rows));
		row.add(Integer.toString(input));
		for (int bit = 0; bit < 7; bit++) {
			row.add(Integer.toString((input >> bit) & 1));
		}
		row.add(Double.toString(after.x)).add(Double.toString(after.y)).add(Double.toString(after.z));
		row.add(Float.toString(before.yaw)).add(Float.toString(before.pitch)).add(binary(after.onGround));
		row.add(Double.toString(after.vx)).add(Double.toString(after.vy)).add(Double.toString(after.vz));
		row.add(binary(after.horizontalCollision)).add(binary(after.verticalCollision));
		row.add(Integer.toString(after.jumpDelay)).add(binary(after.sprinting));
		row.add(binary(after.crouching)).add(binary(after.flying));
		row.add(Double.toString(before.x)).add(Double.toString(before.y)).add(Double.toString(before.z));
		row.add(Double.toString(before.vx)).add(Double.toString(before.vy)).add(Double.toString(before.vz));
		row.add(binary(before.onGround)).add(Integer.toString(before.jumpDelay));
		row.add(Integer.toString(after.pose)).add(binary(after.previousForward)).add(binary(after.previousSneak));
		row.add(Integer.toString(after.sprintTriggerTime)).add(Integer.toString(after.sprintWindow));
		row.add(Integer.toString(after.foodLevel)).add(binary(after.mayFly)).add(Double.toString(after.fallDistance));
		row.add(binary(after.minorHorizontalCollision));
		writer.write(row.toString());
		writer.write('\n');
		rows++;
	}

	long rows() {
		return rows;
	}

	private static String binary(boolean value) {
		return value ? "1" : "0";
	}

	@Override
	public void close() throws IOException {
		writer.close();
	}
}