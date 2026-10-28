"""Metric definitions shared by the dashboard and sweep charts."""

import math


def definition(label, unit, scope="training", field=None):
    result = {"label": label, "unit": unit, "scope": scope, "scale": 100 if unit == "%" else 1}
    if field is not None:
        result["field"] = field
    return result


CATALOG = {
    "win_rate": definition("Benchmark win rate", "%", "evaluation", "win_rate"),
    "self_launch_rate": definition("Self-launch rate", "%", "evaluation", "self_launch_rate"),
    "mace_smash_rate": definition("Mace smash rate", "%", "evaluation", "mace_smash_rate"),
    "launch_smash_rate": definition("Launch-to-smash rate", "%", "evaluation", "launch_smash_rate"),
    "evaluation_reward": definition("Benchmark episode return", "reward / episode", "evaluation", "reward_mean"),
    "training_win_rate": definition("Training win rate", "%", "episode"),
    "training_loss_rate": definition("Training loss rate", "%", "episode"),
    "training_timeout_rate": definition("Training timeout rate", "%", "episode"),
    "training_draw_rate": definition("Training draw rate", "%", "episode"),
    "training_loss_or_draw_rate": definition("Training loss or draw rate", "%", "episode"),
    "training_return": definition("Training episode return", "reward / episode", "episode"),
    "training_length": definition("Episode length", "ticks / episode", "episode"),
    "training_win_length": definition("Winning episode length", "ticks / win", "episode"),
    "training_loss_length": definition("Losing episode length", "ticks / loss", "episode"),
    "training_episodes": definition("Completed episodes", "episodes", "episode"),
    "reward_mean": definition("Training reward per tick", "reward / tick"),
    "reward_total": definition("Rollout total reward", "reward"),
    "entropy": definition("Policy entropy", "nats"),
    "policy_loss": definition("Policy loss", "loss"),
    "value_loss": definition("Value loss", "loss"),
    "loss": definition("Total PPO loss", "loss"),
    "player_transitions_per_second": definition("Learner throughput", "transitions / second"),
    "collection_seconds": definition("Collection duration", "seconds"),
    "update_seconds": definition("Optimization duration", "seconds"),
    "evaluation_seconds": definition("Evaluation duration", "seconds"),
    "replay_seconds": definition("Replay duration", "seconds"),
    "iteration_seconds": definition("Iteration duration", "seconds"),
}
EPISODE_FIELDS = tuple(name for name, entry in CATALOG.items() if entry["scope"] == "episode")
AXES = {"elapsed_seconds": "Elapsed training time (seconds)", "samples": "Learner transitions", "update": "PPO update"}


def episode_summary(row, aggregation="recent", opponent=None):
    if aggregation not in ("rollout", "recent"):
        raise ValueError("Unsupported episode aggregation")
    source = row.get("training_rollout", {}) if aggregation == "rollout" else row
    return source.get("training_opponents", {}).get(opponent, {}) if opponent else source


def metric_value(row, metric, suite=None, opponent=None, aggregation="recent"):
    entry = CATALOG[metric]
    source = row
    if entry["scope"] == "evaluation":
        source = row.get("evaluations", {}).get(suite, {})
    elif entry["scope"] == "episode":
        source = episode_summary(row, aggregation, opponent)
    value = source.get(entry.get("field", metric))
    if value is None and metric == "training_loss_or_draw_rate" and all(
            name in source for name in ("training_loss_rate", "training_draw_rate")):
        value = source["training_loss_rate"] + source["training_draw_rate"]
    return None if value is None else value * entry["scale"]


def recover_rollout(row, config):
    if not isinstance(config, dict) or not isinstance(config.get("metadata"), dict):
        return None
    if config["metadata"].get("learner") != "opponent_pool":
        return None
    try:
        stages = config["experiment"]["stages"]
        stage = stages[row["stage_index"]]
        if stage["name"] != row["stage"]:
            return None
        weights = stage["rewards"]["weights"]
        outcomes = []
        for name in ("won_duel", "death"):
            weight = weights[name]
            if type(weight) not in (float, int) or not math.isfinite(weight) or not weight:
                return None
            value = row["reward_types"][name] / weight
            if value < 0 or not math.isclose(value, round(value), rel_tol=0, abs_tol=1e-5):
                return None
            outcomes.append(round(value))
        wins, losses_or_draws = outcomes
        if wins + losses_or_draws != row["terminated_duels"]:
            return None
        episodes = row["terminated_duels"] + row["timeouts"]
        result = {"training_episodes": episodes, "source": "recovered_rewards", "training_opponents": {}}
        if episodes:
            result.update(training_win_rate=wins / episodes,
                          training_loss_or_draw_rate=losses_or_draws / episodes,
                          training_timeout_rate=row["timeouts"] / episodes)
        return result
    except (KeyError, IndexError, TypeError, ValueError, OverflowError):
        return None


def elapsed_rows(rows):
    elapsed = 0
    for row in rows:
        if row.get("elapsed_seconds") is not None:
            elapsed = row["elapsed_seconds"]
        else:
            duration = row.get("iteration_seconds")
            if duration is None and all(row.get(key) is not None for key in ("collection_seconds", "update_seconds")):
                duration = row["collection_seconds"] + row["update_seconds"]
            elapsed = elapsed + duration if elapsed is not None and duration is not None else None
        yield {**row, "elapsed_seconds": elapsed}