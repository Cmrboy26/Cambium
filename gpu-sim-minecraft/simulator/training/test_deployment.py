"""Public experiment entry point and concrete deployment boundary checks."""

import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import urlopen

import numpy as np
import torch

from backend.fast_rl_model import ActorCritic
from simulator.training.contract import from_legacy, resolve_scenario, validate_contract
from simulator.training.environment import Environment, REWARD_NAMES, ROOT, SCHEMA
from simulator.training.research import SampledWorlds
from simulator.training.serve import Policy, make_server
from simulator.training.train import arguments, main


class DeploymentChecks(unittest.TestCase):
    def test_legacy_training_flags_are_rejected(self):
        for flag in ("--schema", "--environment", "--seed"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments([flag, "v1"])

    def test_http_summary_and_empty_weapon_slots(self):
        contract = validate_contract({"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
            "ticks_per_action": 1, "observations": [{"feature": "target.distance_fraction"}],
            "actions": [{"name": "weapon", "values": list(range(9))}]})
        scenario = json.loads((ROOT / "simulator/environments/duel.json").read_text())
        scenario["profile"] = "configured"
        for player in scenario["players"]:
            player["loadout"] = {"weapons": ["minecraft:iron_sword"] + ["minecraft:air"] * 8, "selected": 7}
        summary = {"name": "Experiment", "stage": "Combat", "stage_index": 3,
                   "environment": "Optional gear", "arenas": ["Flat"], "controllers": "shared_learner",
                   "sampling": "fixed_checkpoint_episode"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            torch.save({"format": "minecraft_sim_ppo", "version": 1, "update": 1,
                "metadata": {"schema": contract, "hidden_dim": 8, "max_ticks": 5, "players_per_environment": 2,
                    "learner": "opponent_pool", "scenario": scenario, "controls": {"fixed": {"weapon": 7}},
                    "experiment": summary},
                "model": ActorCritic(1, 8, movement_bins=9, action_sizes={"weapon": 9}).state_dict()}, path)
            policy = Policy(path, deterministic=True)
            with make_server(policy, 0) as server:
                worker = threading.Thread(target=server.serve_forever, daemon=True)
                worker.start()
                try:
                    with urlopen(f"http://127.0.0.1:{server.server_port}/info", timeout=3) as response:
                        info = json.load(response)
                    self.assertEqual(info["experiment"], summary)
                    self.assertEqual(info["environment"]["players"][0]["loadout"]["selected"], 7)
                    actions = policy.act({"protocol": 1, "session": "deployment", "tick": 0,
                        "contract_hash": info["contract_hash"], "observations": [[0], [0]]})["actions"]
                    self.assertEqual(actions, [[7], [7]])
                finally:
                    server.shutdown()
                    worker.join(timeout=3)


class ExperimentRuntimeChecks(unittest.TestCase):
    def test_public_training_resume_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--updates", "1", "--envs", "2", "--rollout-steps", "4", "--threads", "1",
                      "--max-ticks", "3", "--hidden-dim", "16", "--epochs", "1", "--minibatch-size", "4",
                      "--output", str(output), "--replay-every", "1"])
            configuration = json.loads((output / "config.json").read_text())
            self.assertNotIn("plan", configuration)
            self.assertEqual(configuration["experiment"]["format"], "minecraft_sim_experiment")
            self.assertEqual(configuration["experiment"]["learner"]["hidden_dim"], 16)
            saved = torch.load(output / "latest.pt", weights_only=True)
            self.assertEqual(saved["metadata"]["experiment"]["controllers"], "shared_learner")
            self.assertEqual(saved["metadata"]["max_ticks"], 3)
            self.assertEqual(saved["research"]["world_count"], 2)
            policy = Policy(output / "latest.pt")
            self.assertEqual(policy.info["experiment"]["name"], configuration["experiment"]["name"])
            replay = next((output / "replays").glob("*.jsonl"))
            header = json.loads(replay.read_text().splitlines()[0])
            self.assertEqual(header["environment"]["max_ticks"], 3)
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--resume", str(output / "latest.pt"), "--updates", "1", "--threads", "1"])
            resumed = torch.load(output / "latest.pt", weights_only=True)
            self.assertEqual(resumed["update"], 2)
            self.assertEqual(resumed["research"]["world_count"], 2)
            self.assertEqual(resumed["research"]["rollout_steps"], 4)
            self.assertEqual(resumed["research"]["max_ticks_override"], 3)
            rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["update"] for row in rows], [1, 2])
            self.assertFalse(rows[-1]["completed_curriculum"])

    def test_episode_limit_reconfiguration(self):
        with Environment(1, "cpu", max_ticks=2) as environment:
            environment.set_max_ticks(4)
            actions = np.zeros((1, 2, environment.action_count), dtype=np.int32)
            actions[:, :, 4:6] = [4, 2]
            for tick in range(4):
                _, _, terminated, truncated = environment.step(actions)
                self.assertFalse(terminated[0])
                self.assertEqual(bool(truncated[0]), tick == 3)
            self.assertEqual(environment.episode_scenario()["max_ticks"], 4)
            for invalid in (0, True, 1000001):
                with self.assertRaises(ValueError):
                    environment.set_max_ticks(invalid)

    def test_stage_task_changes_reuse_native_scene(self):
        scene = resolve_scenario(json.loads((ROOT / "simulator/environments/duel.json").read_text()))
        scenario = {"arenas": [{"environment": scene, "weight": 1}], "spawns": None, "equipment": None}
        worlds = SampledWorlds(scenario, from_legacy(SCHEMA), 2, "cpu", None, None, 1)
        try:
            native = worlds.groups[0][1]
            stage = {"scenario": deepcopy(scenario),
                     "rewards": {"mode": "replace", "weights": dict.fromkeys(REWARD_NAMES, 0.0)}}
            stage["scenario"]["arenas"][0]["environment"]["max_ticks"] = 7
            stage["rewards"]["weights"]["aim_hold"] = 2.0
            worlds.configure(stage)
            worlds.reset()
            self.assertIs(worlds.groups[0][1], native)
            self.assertEqual(worlds.max_ticks, 7)
            self.assertEqual(native.weights["aim_hold"], 2.0)
            self.assertEqual(worlds.episode(0)[0]["max_ticks"], 7)
        finally:
            worlds.close()


if __name__ == "__main__":
    unittest.main()
