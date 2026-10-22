"""Test-owned plans independent of editable experiment curricula."""

from simulator.training.environment import ROOT

TEST_SETUPS = ROOT / "simulator/tests/setups"


def example_plan():
    return {
        "format": "minecraft_sim_training_plan", "version": 2,
        "name": "test-curriculum", "seed": 42,
        "policy": str(TEST_SETUPS / "policy.json"),
        "environment": str(TEST_SETUPS / "environment.json"),
        "opponents": {
            "stationary": {"type": "scripted", "controller": "stationary_target"},
            "fighter": {"type": "scripted", "controller": "basic_melee", "options": {"strafe": True}},
        },
        "mixtures": {
            "target": [{"id": "stationary", "weight": 1}],
            "combat": [{"id": "stationary", "weight": 1}, {"id": "fighter", "weight": 3}],
            "held_out": [{"id": "fighter", "weight": 1}],
        },
        "evaluations": {"held_out_combat": {
            "environment": str(TEST_SETUPS / "environment.json"),
            "every_updates": 50, "episodes": 64, "seed": 1729, "opponents": "held_out",
        }},
        "stages": [
            {"name": "aim", "opponents": "target",
             "controls": {"fixed": {"movement": "neutral", "jump": False, "sprint": False,
                                     "attack": False, "weapon": 0, "shield": False}},
             "task_override": {"rewards": {"good_aim": 5, "aim_hold": 2}},
             "advance": {"updates": 300}},
            {"name": "combat", "opponents": "combat", "controls": {"fixed": {}},
             "advance": {"any": [
                 {"updates": 1500},
                 {"evaluation": {"suite": "held_out_combat", "metric": "win_rate", "at_least": 0.75}},
             ]}},
        ],
    }