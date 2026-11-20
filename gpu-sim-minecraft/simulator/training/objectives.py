"""Task-neutral objective geometry for host-staged training."""

import numpy as np

from simulator.training.item_progress import item_progress_observations


def actor_task_observations(actor_state, targets, eating_duration=32, distance_scale=50.0,
                            use_cooldown_duration=4, mining_cooldown_duration=5):
    state = np.asarray(actor_state, dtype=np.float64)
    if state.ndim < 1 or state.shape[-1] != 9 or not np.isfinite(state).all():
        raise ValueError("Invalid actor task state")
    ticks = state[..., 5]
    if np.any(ticks != np.floor(ticks)):
        raise ValueError("Eating progress ticks must be integers")
    progress = item_progress_observations(ticks.astype(np.int64), eating_duration, state[..., 6])
    geometry = objective_observations(state[..., :3], state[..., 3], state[..., 4], targets, distance_scale)
    delays = state[..., 7:9]
    durations = np.asarray([use_cooldown_duration, mining_cooldown_duration], dtype=np.float64)
    if (durations.shape != (2,) or not np.isfinite(durations).all() or np.any(durations <= 0)
            or np.any(delays < 0) or np.any(delays > durations) or np.any(delays != np.floor(delays))):
        raise ValueError("Invalid item cooldown state")
    return np.concatenate((progress, geometry, (delays / durations).astype(np.float32)), axis=-1)


def objective_observations(positions, yaw, pitch, targets, distance_scale=50.0):
    positions = np.asarray(positions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)
    pitch = np.asarray(pitch, dtype=np.float64)
    if (positions.ndim < 1 or positions.shape[-1] != 3 or targets.shape != positions.shape
            or yaw.shape != positions.shape[:-1] or pitch.shape != yaw.shape
            or not all(np.isfinite(value).all() for value in (positions, targets, yaw, pitch))
            or isinstance(distance_scale, (bool, np.bool_))
            or not isinstance(distance_scale, (int, float, np.integer, np.floating))
            or not np.isfinite(distance_scale) or distance_scale <= 0):
        raise ValueError("Invalid objective geometry")
    relative = targets - positions
    horizontal = np.hypot(relative[..., 0], relative[..., 2])
    distance = np.hypot(horizontal, relative[..., 1])
    if not np.isfinite(distance).all():
        raise ValueError("Objective distance must be finite")
    desired_yaw = np.degrees(np.arctan2(-relative[..., 0], relative[..., 2]))
    yaw_error = np.where(horizontal > 0, (desired_yaw - yaw + 180) % 360 - 180, 0)
    desired_pitch = np.degrees(np.arctan2(-relative[..., 1], horizontal))
    pitch_error = np.where(distance > 0, desired_pitch - pitch, 0)
    return np.stack((yaw_error / 180, pitch_error / 90, np.minimum(distance / distance_scale, 1)), axis=-1).astype(np.float32)


def objective_progress_rewards(previous_distance, current_distance, best_distance,
                               distance_scale=50.0, toward_weight=0.0, away_weight=0.0):
    values = [np.asarray(value, dtype=np.float64) for value in (previous_distance, current_distance, best_distance)]
    if any(not np.isfinite(value).all() or np.any(value < 0) for value in values):
        raise ValueError("Objective distances must be finite and nonnegative")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
           or not np.isfinite(value) or value < 0
           for value in (toward_weight, away_weight)):
        raise ValueError("Objective progress weights must be finite and nonnegative")
    if (isinstance(distance_scale, (bool, np.bool_))
            or not isinstance(distance_scale, (int, float, np.integer, np.floating))
            or not np.isfinite(distance_scale) or distance_scale <= 0):
        raise ValueError("Objective distance scale must be finite and positive")
    try:
        previous, current, best = np.broadcast_arrays(*values)
    except ValueError:
        raise ValueError("Objective distance shapes must broadcast") from None
    if np.any(best > previous):
        raise ValueError("Best objective distance cannot exceed the previous distance")
    closest = np.minimum(best, current)
    toward = (best - closest) / distance_scale * toward_weight
    away = -np.maximum(current - previous, 0) / distance_scale * away_weight
    if not np.isfinite(toward).all() or not np.isfinite(away).all():
        raise ValueError("Objective progress rewards must be finite")
    return toward, away, closest