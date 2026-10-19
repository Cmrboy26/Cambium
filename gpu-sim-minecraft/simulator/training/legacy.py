"""Frozen convenience presets; new experiments use semantic policy descriptors."""

ACTION_NAMES = ("movement", "jump", "sprint", "attack", "yaw", "pitch")
ACTION_SIZES = (9, 2, 2, 2, 9, 5)
SCHEMA = {
    "id": "minecraft_sim_duel_22_v1",
    "minecraft": "26.3",
    "observation_dim": 22,
    "observations": ["health/20", "yaw_wrapped/180", "pitch/90", "opponent_alive",
                     "opponent_health/20", "relative_x/10", "relative_y/10", "relative_z/10",
                     "opponent_yaw_wrapped/180", "opponent_pitch/90",
                     *[f"eye_wall_ray_world_yaw_{angle}/50" for angle in range(0, 360, 45)],
                     "min(distance/10,1)", "feet_relative_aim_dot", "yaw_error/180", "pitch_error/90"],
    "action_names": list(ACTION_NAMES),
    "action_sizes": list(ACTION_SIZES),
    "movement": ["neutral", "forward", "forward_right", "right", "backward_right",
                 "backward", "backward_left", "left", "forward_left"],
    "yaw_bins": [-20, -10, -5, -2, 0, 2, 5, 10, 20],
    "pitch_bins": [-5, -2, 0, 2, 5],
    "ticks_per_action": 1,
    "reward_profile": "accepted_hits_win_loss_timeout_v1",
    "termination": "death_or_exit_xz_bounds_or_y_below_min_minus_4",
    "timeout": "truncated_bootstrap_final_observation",
}
SCHEMA_V2 = {
    **SCHEMA,
    "id": "minecraft_sim_duel_31_v2",
    "observation_dim": 31,
    "observations": [*SCHEMA["observations"], "weapon_is_axe", "opponent_weapon_is_axe",
                     "using_shield", "opponent_using_shield", "shield_disabled",
                     "opponent_shield_disabled_observed", "min(shield_cooldown/100,1)",
                     "attack_charge", "using_shield*min(shield_use_ticks/5,1)"],
    "action_names": [*ACTION_NAMES, "weapon", "shield"],
    "action_sizes": [*ACTION_SIZES, 2, 2],
    "weapon": ["diamond_sword", "diamond_axe"],
    "shield": ["release", "hold_use"],
    "loadout": "unarmored_sword_axe_shield",
    "opponent_disabled": "own_observed_axe_disable_event_then_100_tick_countdown",
}
SCHEMAS = {"v1": SCHEMA, "v2": SCHEMA_V2}
