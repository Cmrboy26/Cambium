import ctypes
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import weakref

import numpy as np
import torch

from simulator.training.contract import FEATURES, bridge_observation_descriptor
from simulator.training.environment import Environment, REWARD_NAMES, ROOT


class CUDAObservationChecks(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CPU_LIBRARY")
                         and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA, MCSIM_TEST_CPU_LIBRARY and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_bridge_shared_combat_rewards_match_cpu_and_cuda(self):
        scene = json.loads((ROOT / "simulator/tests/reset_environment.json").read_text(encoding="utf-8"))
        scene["players"][1]["health"] = 1
        contract = {"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
                    "ticks_per_action": 1, "observations": [{"feature": "self.health_fraction"}],
                    "actions": [{"name": "attack", "values": [False, True]}]}
        bridge = {"mode": {"name": "bridge", "version": 1},
                  "goals": [{"min": [0, 1, 2.4], "max": [1, 3, 2.9]},
                            {"min": [0, 1, -1.1], "max": [1, 3, -0.8]}],
                  "void_y": -2, "match": {"goals_to_win": 2, "max_ticks": 2, "cage_ticks": 0,
                                           "persist_blocks_on_goal": True, "arrow_refill_ticks": 0},
                  "task_rewards": {"score": 0, "win": 0, "loss": 0, "draw": -5}}
        weights = {"doing_damage": 20, "taking_damage": -20, "death": -3,
                   "episode_timeout": -4, "no_damage_penalty": -2, "win_speed": 0.1}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "combat.json"
            path.write_text(json.dumps(scene), encoding="utf-8")
            options = {"environment": path, "contract": contract, "max_ticks": 10,
                       "bridge": bridge, "weights": weights}
            with Environment(2, device="cpu", library=os.environ["MCSIM_TEST_CPU_LIBRARY"], **options) as cpu, \
                 Environment(2, device="cuda", library=os.environ["MCSIM_TEST_CUDA_LIBRARY"], **options) as cuda:
                actions = np.zeros((2, 2, 1), dtype=np.int32)
                actions[0, 0, 0] = 1
                for tick in range(2):
                    if tick:
                        actions.fill(0)
                    expected, actual = cpu.step(actions), cuda.step(actions)
                    for first, second in zip(expected, actual, strict=True):
                        np.testing.assert_allclose(first, second, rtol=0, atol=1e-6)
                    self.assertEqual(cpu.bridge_state(), cuda.bridge_state())
                    self.assertEqual(cpu.step_reward_totals(), cuda.step_reward_totals())

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CPU_LIBRARY")
                         and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA, MCSIM_TEST_CPU_LIBRARY and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_objective_rewards_and_actor_state_match_cpu_and_cuda(self):
        contract = {"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
                    "ticks_per_action": 1, "observations": [{"feature": "relative.z"}],
                    "actions": [{"name": "movement", "values": ["neutral", "forward", "backward"]}]}
        bridge = {"mode": {"name": "bridge", "version": 1},
                  "goals": [{"min": [0, 1, 2.4], "max": [1, 3, 2.9]},
                            {"min": [0, 1, -1.1], "max": [1, 3, -0.8]}],
                  "void_y": -2, "match": {"goals_to_win": 2, "max_ticks": 100, "cage_ticks": 0,
                                           "persist_blocks_on_goal": True, "arrow_refill_ticks": 0},
                  "task_rewards": {"score": 2, "win": 5, "loss": -3,
                                   "opponent_score": -1, "toward": 0.25, "away": 0.1}}
        path = ROOT / "simulator/tests/reset_environment.json"
        with Environment(2, device="cpu", environment=path, contract=contract, max_ticks=100, bridge=bridge,
                         library=os.environ["MCSIM_TEST_CPU_LIBRARY"]) as cpu, \
             Environment(2, device="cuda", environment=path, contract=contract, max_ticks=100, bridge=bridge,
                         library=os.environ["MCSIM_TEST_CUDA_LIBRARY"]) as cuda:
            actions = np.zeros((2, 2, 1), dtype=np.int32)
            actions[0, 0, 0] = 1
            actions[1, 1, 0] = 2
            for _ in range(50):
                expected, actual = cpu.step(actions), cuda.step(actions)
                for first, second in zip(expected, actual, strict=True):
                    np.testing.assert_allclose(first, second, rtol=0, atol=1e-5)
                np.testing.assert_allclose(cpu.actor_task_state(), cuda.actor_task_state(), rtol=0, atol=1e-5)
                np.testing.assert_allclose(cpu.bridge_reward_terms(), cuda.bridge_reward_terms(), rtol=0, atol=1e-7)
                self.assertEqual(cpu.bridge_state(), cuda.bridge_state())
                if expected[2].any() or expected[3].any():
                    break

    def configured_cuda(self, environment):
        prepare = environment.native.mcsim_prepare_cuda_observations
        prepare.argtypes = [ctypes.c_void_p]
        prepare.restype = ctypes.c_int
        self.assertEqual(prepare(environment._handle), 0, environment.native.mcsim_last_error())
        function = environment.native.mcsim_observe_cuda_sync
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        function.restype = ctypes.c_int
        output = torch.empty((environment.count, 2, environment.observation_dim), device="cuda", dtype=torch.float32)
        self.assertEqual(function(environment._handle, output.data_ptr(), output.numel(),
                                  torch.cuda.current_stream().cuda_stream), 0, environment.native.mcsim_last_error())
        return output.cpu().numpy()

    def compare_configured(self, environment):
        expected = environment.observe()
        np.testing.assert_allclose(self.configured_cuda(environment), expected, rtol=0, atol=1e-6)
        if environment._cuda_observations is None:
            environment.register_cuda_observations()
        output = environment.observe_cuda_async()
        np.testing.assert_allclose(output.cpu().numpy(), expected, rtol=0, atol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CPU_LIBRARY")
                         and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA, MCSIM_TEST_CPU_LIBRARY and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_bridge_caged_aim_matches_cpu_and_cuda(self):
        scene = json.loads((ROOT / "simulator/tests/reset_environment.json").read_text(encoding="utf-8"))
        for player in scene["players"]:
            player["yaw"] = 0
            player["pitch"] = 0
            player["loadout"] = {"weapons": ["minecraft:diamond_sword"],
                                 "selected": 0, "arrows": 0, "offhand": "minecraft:air"}
        contract = {"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
                    "ticks_per_action": 1, "observations": [{"feature": "self.health_fraction"}],
                    "actions": [{"name": "yaw", "values": [0, 20, -30]},
                                {"name": "pitch", "values": [0, 5, -6]}]}
        kit = {"slots": ["sword"], "blocks": 0, "golden_apples": 0}
        bridge = {"mode": {"name": "bridge", "version": 1},
                  "goals": [{"min": [0, 1, -2.9], "max": [1, 3, -2.4]},
                            {"min": [0, 1, 2.4], "max": [1, 3, 2.9]}],
                  "void_y": -2, "match": {"goals_to_win": 2, "max_ticks": 100, "cage_ticks": 2,
                                           "persist_blocks_on_goal": True, "arrow_refill_ticks": 0},
                  "kits": [kit, kit], "task_rewards": {"score": 2, "win": 5, "loss": -3}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bridge-caged-aim.json"
            path.write_text(json.dumps(scene), encoding="utf-8")
            with Environment(1, device="cpu", environment=path, contract=contract, max_ticks=100,
                             bridge=bridge, library=os.environ["MCSIM_TEST_CPU_LIBRARY"]) as cpu, \
                    Environment(1, device="cuda", environment=path, contract=contract, max_ticks=100,
                                bridge=bridge, library=os.environ["MCSIM_TEST_CUDA_LIBRARY"]) as cuda:
                actions = np.array([[[1, 1], [2, 2]]], dtype=np.int32)
                cpu.step(actions)
                cuda.step(actions)
                expected = np.array([[[20.0, 5.0], [-30.0, -6.0]]])
                np.testing.assert_array_equal(cpu.actor_task_state()[:, :, 3:5], expected)
                np.testing.assert_array_equal(cuda.actor_task_state()[:, :, 3:5], expected)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CPU_LIBRARY")
                         and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA, MCSIM_TEST_CPU_LIBRARY and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_bridge_lifecycle_matches_cpu_and_cuda(self):
        scene = json.loads((ROOT / "simulator/tests/reset_environment.json").read_text(encoding="utf-8"))
        for player in scene["players"]:
            player["loadout"] = {"weapons": ["minecraft:diamond_sword", "minecraft:air"],
                                 "selected": 0, "arrows": 0, "offhand": "minecraft:air"}
        contract = {"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
                    "ticks_per_action": 1,
                    "observations": [{"feature": "relative.z"}, {"feature": "self.health_fraction"},
                                     {"feature": "opponent.health_fraction"}, {"feature": "self.on_ground"}],
                    "actions": [{"name": "movement", "values": ["neutral", "forward"]},
                                {"name": "weapon", "values": [0, 1]}]}
        kit = {"slots": ["sword", "golden_apple"], "blocks": 0, "golden_apples": 2}
        bridge = {"mode": {"name": "bridge", "version": 1},
                  "goals": [{"min": [0, 1, -2.9], "max": [1, 3, -2.4]},
                            {"min": [0, 1, 2.4], "max": [1, 3, 2.9]}],
                  "void_y": -2, "match": {"goals_to_win": 2, "max_ticks": 100, "cage_ticks": 2,
                                           "persist_blocks_on_goal": True, "arrow_refill_ticks": 0},
                  "kits": [kit, kit], "task_rewards": {"score": 2, "win": 5, "loss": -3}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bridge-lifecycle.json"
            for scorer in (0, 1):
                scene["players"][scorer]["position"][0] = 0.5
                scene["players"][1 - scorer]["position"][0] = -1.5
                path.write_text(json.dumps(scene), encoding="utf-8")
                with self.subTest(scorer=scorer), \
                        Environment(2, device="cpu", environment=path, contract=contract, max_ticks=100,
                                    seed=7, bridge=bridge, library=os.environ["MCSIM_TEST_CPU_LIBRARY"]) as cpu, \
                        Environment(2, device="cuda", environment=path, contract=contract, max_ticks=100,
                                    seed=7, bridge=bridge, library=os.environ["MCSIM_TEST_CUDA_LIBRARY"]) as cuda:
                    all_worlds = np.array([True, True])
                    selected_world = np.array([True, False])

                    def compare_state():
                        states = cpu.bridge_state()
                        self.assertEqual(cuda.bridge_state(), states)
                        for player in (0, 1):
                            self.assertEqual(cuda.bridge_kits(player), cpu.bridge_kits(player))
                        np.testing.assert_allclose(cuda.observe(), cpu.observe(), rtol=0, atol=1e-6,
                                                   equal_nan=False)
                        for mask in (all_worlds, selected_world):
                            np.testing.assert_array_equal(cuda.losses(mask), cpu.losses(mask))
                        return states

                    initial_observations = cpu.observe()
                    initial_states = compare_state()
                    self.assertEqual([state["cage_remaining"] for state in initial_states], [2, 2])
                    np.testing.assert_array_equal(cpu.losses(all_worlds), np.zeros((2, 2), dtype=bool))
                    actions = np.zeros((2, 2, 2), dtype=np.int32)
                    actions[:, :, 1] = 1
                    actions[0, scorer, 0] = 1
                    scores_seen = 0
                    for tick in range(50):
                        cpu_result = cpu.step(actions)
                        cuda_result = cuda.step(actions)
                        for expected, actual in zip(cpu_result[:2], cuda_result[:2]):
                            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6, equal_nan=False)
                        for expected, actual in zip(cpu_result[2:], cuda_result[2:]):
                            np.testing.assert_array_equal(actual, expected)
                        observations, rewards, terminated, truncated = cpu_result
                        states = compare_state()
                        np.testing.assert_array_equal(truncated, [False, False])
                        self.assertEqual(states[1]["scores"], [0, 0])
                        self.assertFalse(terminated[1])
                        if tick < 2:
                            self.assertEqual(states[0]["cage_remaining"], 1 - tick)
                            np.testing.assert_allclose(observations, initial_observations, rtol=0, atol=1e-6,
                                                       equal_nan=False)
                        if states[0]["scored"][scorer]:
                            scores_seen += 1
                            expected_scores = [0, 0]
                            expected_scores[scorer] = scores_seen
                            self.assertEqual(states[0]["scores"], expected_scores)
                            self.assertEqual(states[0]["scored"], [scorer == 0, scorer == 1])
                            expected_rewards = np.zeros((2, 2), dtype=np.float32)
                            expected_rewards[0, scorer] = 2 if scores_seen == 1 else 7
                            if scores_seen == 2:
                                expected_rewards[0, 1 - scorer] = -3
                            np.testing.assert_array_equal(rewards, expected_rewards)
                            if scores_seen == 1:
                                self.assertTrue(states[0]["round_reset"])
                                self.assertEqual(states[0]["cage_remaining"], 2)
                                self.assertFalse(terminated[0])
                                np.testing.assert_allclose(observations[0], initial_observations[0],
                                                           rtol=0, atol=1e-6, equal_nan=False)
                                for player in (0, 1):
                                    self.assertEqual(cpu.bridge_kits(player)[0]["selected_slot"], 0)
                                    self.assertEqual(cpu.bridge_kits(player)[1]["selected_slot"], 1)
                            else:
                                self.assertFalse(states[0]["round_reset"])
                                self.assertTrue(terminated[0])
                                self.assertEqual(states[0]["outcome"], f"team{scorer}")
                                break
                        else:
                            np.testing.assert_array_equal(rewards, np.zeros((2, 2), dtype=np.float32))
                            self.assertFalse(terminated[0])
                    else:
                        self.fail("Bridge match did not finish within 50 ticks")
                    self.assertEqual(scores_seen, 2)
                    expected_losses = np.zeros((2, 2), dtype=bool)
                    expected_losses[0, 1 - scorer] = True
                    np.testing.assert_array_equal(cpu.losses(all_worlds), expected_losses)
                    other_state = states[1]
                    other_kits = [cpu.bridge_kits(player)[1] for player in (0, 1)]
                    other_observation = observations[1].copy()
                    cpu_reset = cpu.reset(selected_world)
                    cuda_reset = cuda.reset(selected_world)
                    np.testing.assert_allclose(cuda_reset, cpu_reset, rtol=0, atol=1e-6, equal_nan=False)
                    reset_states = compare_state()
                    self.assertEqual(reset_states[0], initial_states[0])
                    self.assertEqual(reset_states[1], other_state)
                    self.assertEqual([cpu.bridge_kits(player)[1] for player in (0, 1)], other_kits)
                    np.testing.assert_allclose(cpu_reset[0], initial_observations[0], rtol=0, atol=1e-6,
                                               equal_nan=False)
                    np.testing.assert_allclose(cpu_reset[1], other_observation, rtol=0, atol=1e-6,
                                               equal_nan=False)
                    np.testing.assert_array_equal(cpu.losses(all_worlds), np.zeros((2, 2), dtype=bool))

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CPU_LIBRARY")
                         and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA, MCSIM_TEST_CPU_LIBRARY and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_bridge_death_and_time_cap_trajectories_match_cpu_and_cuda(self):
        contract = {"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
                    "ticks_per_action": 1,
                    "observations": [{"feature": "relative.y"}, {"feature": "relative.z"},
                                     {"feature": "self.health_fraction"},
                                     {"feature": "opponent.health_fraction"},
                                     {"feature": "self.velocity.y"}, {"feature": "self.on_ground"}],
                    "actions": [{"name": "movement", "values": ["neutral", "backward"]},
                                {"name": "weapon", "values": [0, 1]},
                                {"name": "attack", "values": [False, True]}]}
        kit = {"slots": ["sword", "golden_apple"], "blocks": 0, "golden_apples": 2}
        bridge = {"mode": {"name": "bridge", "version": 1},
                  "goals": [{"min": [-2.5, 1, -2.5], "max": [-2, 3, -2]},
                            {"min": [-2.5, 1, 2], "max": [-2, 3, 2.5]}],
                  "void_y": -2, "match": {"goals_to_win": 2, "max_ticks": 100, "cage_ticks": 0,
                                           "persist_blocks_on_goal": True, "arrow_refill_ticks": 0},
                  "kits": [kit, kit], "task_rewards": {"score": 2, "win": 5, "loss": -3}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bridge-death-parity.json"
            for kind in ("melee", "void", "time_cap", "external_cap", "score_time_cap"):
                for target in (0, 1):
                    scene = json.loads((ROOT / "simulator/tests/reset_environment.json").read_text(encoding="utf-8"))
                    scene["players"][0]["position"][2] = -0.5
                    for player in scene["players"]:
                        player["velocity"][1] = -0.0784000015258789
                        player["loadout"] = {"weapons": ["minecraft:diamond_sword", "minecraft:air"],
                                             "selected": 0, "arrows": 0, "offhand": "minecraft:air"}
                    rules = {**bridge, "match": dict(bridge["match"])}
                    if kind == "melee":
                        scene["players"][target]["health"] = 1
                    elif kind in ("time_cap", "external_cap"):
                        scene["blocks"] = [{"from": [0, 0, -3], "to": [3, 0, 3],
                                            "block": "minecraft:stone"}]
                        scene["players"][target]["position"][0] = -1.5
                        scene["players"][target]["velocity"][1] = -0.3
                        scene["players"][target]["on_ground"] = False
                        scene["players"][1 - target]["position"][0] = 1.5
                        rules["void_y"] = 0.5
                        if kind == "time_cap":
                            rules["match"]["max_ticks"] = 2
                    elif kind == "score_time_cap":
                        scene["players"][0]["velocity"][2] = 0.3
                        scene["players"][1]["velocity"][2] = -0.3
                        rules["goals"] = [{"min": [0, 1, 0.95], "max": [1, 3, 1.05]},
                                          {"min": [0, 1, -0.05], "max": [1, 3, 0.05]}]
                        rules["match"]["max_ticks"] = 1
                    path.write_text(json.dumps(scene), encoding="utf-8")
                    external_limit = 2 if kind in ("time_cap", "external_cap") else 1 if kind == "score_time_cap" else 100
                    with self.subTest(kind=kind, target=target), \
                            Environment(2, device="cpu", environment=path, contract=contract, max_ticks=external_limit,
                                        seed=7, bridge=rules, library=os.environ["MCSIM_TEST_CPU_LIBRARY"]) as cpu, \
                            Environment(2, device="cuda", environment=path, contract=contract, max_ticks=external_limit,
                                        seed=7, bridge=rules, library=os.environ["MCSIM_TEST_CUDA_LIBRARY"]) as cuda:
                        def compare_state():
                            states = cpu.bridge_state()
                            self.assertEqual(cuda.bridge_state(), states)
                            for player in (0, 1):
                                self.assertEqual(cuda.bridge_kits(player), cpu.bridge_kits(player))
                            np.testing.assert_allclose(cuda.observe(), cpu.observe(), rtol=0, atol=1e-6,
                                                       equal_nan=False)
                            return states

                        initial_observations = cpu.observe()
                        initial_states = compare_state()
                        initial_kits = [cpu.bridge_kits(player) for player in (0, 1)]
                        actions = np.zeros((2, 2, 3), dtype=np.int32)
                        actions[:, :, 1] = 1
                        if kind == "melee":
                            actions[:, 1 - target, 1] = 0
                        elif kind == "void":
                            actions[0, target, 0] = 1
                        for tick in range(80):
                            if kind == "melee" and tick == 1:
                                actions[0, 1 - target, 2] = 1
                            cpu_result = cpu.step(actions)
                            cuda_result = cuda.step(actions)
                            for expected, actual in zip(cpu_result[:2], cuda_result[:2]):
                                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6, equal_nan=False)
                            for expected, actual in zip(cpu_result[2:], cuda_result[2:]):
                                np.testing.assert_array_equal(actual, expected)
                            observations, rewards, terminated, truncated = cpu_result
                            states = compare_state()
                            np.testing.assert_allclose(observations, cpu.observe(), rtol=0, atol=1e-6,
                                                       equal_nan=False)
                            expected_reward = 2 if kind == "score_time_cap" else 0
                            np.testing.assert_array_equal(rewards, np.full((2, 2), expected_reward, dtype=np.float32))
                            np.testing.assert_array_equal(truncated, [kind == "external_cap" and tick == 1] * 2)
                            for state in states:
                                self.assertEqual(state["scores"], [1, 1] if kind == "score_time_cap" else [0, 0])
                                self.assertEqual(state["scored"], [kind == "score_time_cap"] * 2)
                                self.assertFalse(state["round_reset"])
                            if states[0]["died"][target] or kind == "score_time_cap":
                                break
                            np.testing.assert_array_equal(terminated, [False, False])
                            self.assertEqual(states[0]["respawn"], [False, False])
                        else:
                            self.fail(f"{kind} did not cause death within 80 ticks")

                        expected_deaths = ([False, False] if kind == "score_time_cap"
                                           else [target == 0, target == 1])
                        self.assertEqual(states[0]["died"], expected_deaths)
                        self.assertEqual(states[0]["ticks"], tick + 1)
                        if kind == "time_cap":
                            self.assertEqual(tick, 1)
                            np.testing.assert_array_equal(terminated, [True, True])
                            for state in states:
                                self.assertEqual(state["outcome"], "draw")
                                self.assertEqual(state["died"], expected_deaths)
                                self.assertEqual(state["respawn"], [False, False])
                            self.assertEqual(cpu.bridge_kits(target)[0]["selected_slot"], 1)
                            self.assertLess(observations[0, target, 4], initial_observations[0, target, 4])
                        elif kind == "external_cap":
                            self.assertEqual(tick, 1)
                            np.testing.assert_array_equal(terminated, [False, False])
                            for state in states:
                                self.assertEqual(state["outcome"], "ongoing")
                                self.assertEqual(state["died"], expected_deaths)
                                self.assertEqual(state["respawn"], [False, False])
                            self.assertEqual(cpu.bridge_kits(target)[0]["selected_slot"], 1)
                            self.assertLess(observations[0, target, 4], initial_observations[0, target, 4])
                        elif kind == "score_time_cap":
                            self.assertEqual(tick, 0)
                            np.testing.assert_array_equal(terminated, [True, True])
                            for state in states:
                                self.assertEqual(state["outcome"], "draw")
                                self.assertEqual(state["died"], [False, False])
                                self.assertEqual(state["respawn"], [False, False])
                        else:
                            np.testing.assert_array_equal(terminated, [False, False])
                            self.assertEqual(states[0]["outcome"], "ongoing")
                            self.assertEqual(states[0]["respawn"], expected_deaths)
                            self.assertEqual(states[1]["died"], [False, False])
                            self.assertEqual(states[1]["respawn"], [False, False])
                            self.assertEqual(cpu.bridge_kits(target)[0], initial_kits[target][0])
                            self.assertEqual(cpu.bridge_kits(target)[1]["selected_slot"], 1)
                            self.assertEqual(cpu.bridge_kits(1 - target)[0]["selected_slot"],
                                             0 if kind == "melee" else 1)
                            self.assertEqual(observations[0, target, 2], initial_observations[0, target, 2])
                            self.assertEqual(observations[0, target, 4], initial_observations[0, target, 4])
                            self.assertEqual(observations[0, target, 5], initial_observations[0, target, 5])

                        other_state = states[1]
                        other_kits = [cpu.bridge_kits(player)[1] for player in (0, 1)]
                        other_observation = observations[1].copy()
                        mask = np.array([True, False])
                        cpu_reset = cpu.reset(mask)
                        cuda_reset = cuda.reset(mask)
                        np.testing.assert_allclose(cuda_reset, cpu_reset, rtol=0, atol=1e-6, equal_nan=False)
                        reset_states = compare_state()
                        self.assertEqual(reset_states[0], initial_states[0])
                        self.assertEqual(reset_states[1], other_state)
                        for player in (0, 1):
                            self.assertEqual(cpu.bridge_kits(player)[0], initial_kits[player][0])
                            self.assertEqual(cpu.bridge_kits(player)[1], other_kits[player])
                        np.testing.assert_allclose(cpu_reset[0], initial_observations[0], rtol=0, atol=1e-6,
                                                   equal_nan=False)
                        np.testing.assert_allclose(cpu_reset[1], other_observation, rtol=0, atol=1e-6,
                                                   equal_nan=False)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_bridge_respawn_clears_observed_shield_disable_only_in_affected_world(self):
        scene = json.loads((ROOT / "simulator/tests/reset_environment.json").read_text(encoding="utf-8"))
        scene["players"][0]["position"][2] = -0.5
        for player in scene["players"]:
            player["velocity"][1] = -0.0784000015258789
            player["loadout"] = {"weapons": ["minecraft:diamond_sword", "minecraft:diamond_axe"],
                                 "selected": 0, "arrows": 0, "offhand": "minecraft:shield"}
        contract = {"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
                    "ticks_per_action": 1, "requires": ["movement", "melee", "equipment", "shield"],
                    "observations": [{"feature": "self.shield_disabled"},
                                     {"feature": "opponent.shield_disabled_observed"},
                                     {"feature": "self.shield_cooldown_fraction"},
                                     {"feature": "self.shield_using"}],
                    "actions": [{"name": "movement", "values": ["neutral", "backward"]},
                                {"name": "weapon", "values": [0, 1]},
                                {"name": "shield", "values": [False, True]},
                                {"name": "attack", "values": [False, True]}]}
        bridge = {"mode": {"name": "bridge", "version": 1},
                  "goals": [{"min": [-2.5, 1, -2.5], "max": [-2, 3, -2]},
                            {"min": [-2.5, 1, 2], "max": [-2, 3, 2.5]}],
                  "void_y": -2, "match": {"goals_to_win": 2, "max_ticks": 150, "cage_ticks": 0,
                                           "persist_blocks_on_goal": True, "arrow_refill_ticks": 0},
                  "task_rewards": {"score": 2, "win": 5, "loss": -3}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bridge-shield.json"
            path.write_text(json.dumps(scene), encoding="utf-8")
            for target in (0, 1):
                attacker = 1 - target
                with self.subTest(target=target), Environment(
                        3, device="cuda", environment=path, contract=contract, max_ticks=150,
                        bridge=bridge, library=os.environ.get("MCSIM_TEST_CUDA_LIBRARY")) as environment:
                    self.compare_configured(environment)
                    actions = np.zeros((3, 2, 4), dtype=np.int32)
                    actions[:2, attacker, 1] = 1
                    actions[:, target, 2] = 1
                    for _ in range(6):
                        raised, _, terminated, truncated = environment.step(actions)
                    np.testing.assert_array_equal(raised[:, target, 3], 1)
                    self.assertFalse(terminated.any() or truncated.any())
                    actions[:2, attacker, 3] = 1
                    disabled, _, terminated, truncated = environment.step(actions)
                    self.assertFalse(terminated.any() or truncated.any())
                    np.testing.assert_array_equal(disabled[:, target, 0], [1, 1, 0])
                    np.testing.assert_array_equal(disabled[:, attacker, 1], [1, 1, 0])
                    np.testing.assert_allclose(disabled[:2, target, 2], 0.99, rtol=0, atol=1e-6)
                    self.compare_configured(environment)

                    actions[:, :, 3] = 0
                    actions[:, target, 2] = 0
                    actions[:2, target, 1] = 1
                    actions[:, attacker, 2] = 1
                    for _ in range(6):
                        raised, _, terminated, truncated = environment.step(actions)
                    np.testing.assert_array_equal(raised[:, attacker, 3], 1)
                    self.assertFalse(terminated.any() or truncated.any())
                    actions[:2, target, 3] = 1
                    disabled, _, terminated, truncated = environment.step(actions)
                    self.assertFalse(terminated.any() or truncated.any())
                    np.testing.assert_array_equal(disabled[:, :, 0], [[1, 1], [1, 1], [0, 0]])
                    np.testing.assert_array_equal(disabled[:, :, 1], [[1, 1], [1, 1], [0, 0]])
                    np.testing.assert_allclose(disabled[:2, attacker, 2], 0.99, rtol=0, atol=1e-6)
                    self.compare_configured(environment)

                    actions[:, :, 3] = 0
                    actions[:, attacker, 2] = 0
                    actions[0, target, 0] = 1
                    for _ in range(80):
                        observed, _, terminated, truncated = environment.step(actions)
                        states = environment.bridge_state()
                        self.assertFalse(terminated.any() or truncated.any())
                        self.assertFalse(any(states[world]["round_reset"] for world in range(3)))
                        self.assertFalse(any(states[world]["respawn"][player]
                                             for world in (1, 2) for player in (0, 1)))
                        if states[0]["respawn"][target]:
                            break
                    else:
                        self.fail("Target did not respawn after walking into the void")
                    self.assertEqual(states[0]["died"], [target == 0, target == 1])
                    self.assertEqual(states[0]["respawn"], [target == 0, target == 1])
                    np.testing.assert_array_equal(observed[:, target, 0], [0, 1, 0])
                    np.testing.assert_array_equal(observed[:, attacker, 0], [1, 1, 0])
                    np.testing.assert_array_equal(observed[:, :, 1], [[0, 0], [1, 1], [0, 0]])
                    self.assertEqual(observed[0, target, 2], 0)
                    self.assertGreater(observed[0, attacker, 2], 0)
                    self.assertGreater(observed[1, target, 2], 0)
                    np.testing.assert_allclose(environment.observe(), observed, rtol=0, atol=1e-6)
                    self.compare_configured(environment)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_bridge_building_device_observations_follow_edits_and_resets(self):
        scene = json.loads((ROOT / "simulator/tests/reset_environment.json").read_text(encoding="utf-8"))
        scene["blocks"].append({"from": [0, 2, 1], "to": [0, 2, 1], "block": "minecraft:stone"})
        scene["players"][1]["position"][0] = -2.5
        for player in scene["players"]:
            player["velocity"][1] = -0.0784000015258789
            player["loadout"] = {"weapons": ["minecraft:diamond_sword", "minecraft:air", "minecraft:air"],
                                 "selected": 1, "arrows": 0, "offhand": "minecraft:air"}
        contract = {"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
                    "ticks_per_action": 1, "observations": [{"feature": "arena.ray_world_0"}],
                    "actions": [{"name": "weapon", "values": [0, 1, 2]},
                                {"name": "use_item", "values": [False, True]},
                                {"name": "attack", "values": [False, True]}]}
        kit = {"slots": ["sword", "blocks", "pickaxe"], "blocks": 4, "golden_apples": 0}
        bridge = {"mode": {"name": "bridge", "version": 1},
                  "goals": [{"min": [0, 1, 2.4], "max": [1, 3, 2.9]},
                            {"min": [0, 1, -1.1], "max": [1, 3, -0.8]}],
                  "void_y": -2, "match": {"goals_to_win": 2, "max_ticks": 100, "cage_ticks": 0,
                                           "persist_blocks_on_goal": True, "arrow_refill_ticks": 0},
                  "kits": [kit, kit], "task_rewards": {"score": 2, "win": 5, "loss": -3},
                  "building": {"zones": [{"min": [-3, 0, -3], "max": [4, 4, 4]}],
                               "block_range": 4.5, "entity_range": 3.0,
                               "hardness": 1.0, "mining_speed": 15.0, "correct_tool": True}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bridge-building.json"
            path.write_text(json.dumps(scene), encoding="utf-8")
            with Environment(2, device="cuda", environment=path, contract=contract, max_ticks=100,
                             bridge=bridge, library=os.environ.get("MCSIM_TEST_CUDA_LIBRARY")) as environment:
                def check(expected):
                    np.testing.assert_allclose(environment.observe()[:, 0, 0], expected, rtol=0, atol=1e-6)
                    np.testing.assert_allclose(environment.observe_base_cuda().cpu().numpy()[:, 0, 10],
                                               expected, rtol=0, atol=1e-6)
                    self.compare_configured(environment)

                check([0.05, 0.05])
                actions = np.zeros((2, 2, 3), dtype=np.int32)
                actions[0, 0] = [1, 1, 0]
                environment.step(actions)
                self.assertTrue(environment.bridge_state()[0]["placed"][0])
                check([0.03, 0.05])
                environment.reset(np.array([False, True]))
                check([0.03, 0.05])
                actions[0, 0] = [2, 0, 1]
                for tick in range(3):
                    environment.step(actions)
                    if tick == 1:
                        self.assertAlmostEqual(environment.actor_task_state()[0, 0, 6], 0.5)
                        self.assertEqual(environment.actor_task_state()[1, 0, 6], 0)
                self.assertTrue(environment.bridge_state()[0]["broke"][0])
                check([0.05, 0.05])
                actions[0, 0] = [1, 1, 0]
                environment.step(actions)
                self.assertTrue(environment.bridge_state()[0]["placed"][0])
                check([0.03, 0.05])
                environment.reset(np.array([True, False]))
                check([0.05, 0.05])

    def test_cpu_environment_rejects_cuda_observations(self):
        with Environment(1, library=os.environ.get("MCSIM_TEST_CPU_LIBRARY")) as environment:
            with self.assertRaisesRegex(ValueError, "require a CUDA environment"):
                environment.observe_base_cuda()
            function = environment.native.mcsim_observe_base_cuda_sync
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
            function.restype = ctypes.c_int
            self.assertEqual(function(environment._handle, None, 44, None), -1)
            with self.assertRaisesRegex(ValueError, "require a CUDA environment"):
                environment.register_cuda_observations()
            with self.assertRaisesRegex(RuntimeError, "Register a CUDA observation buffer"):
                environment.observe_cuda_async()

    def test_resident_bridge_requires_portable_descriptor_and_capability(self):
        with Environment(1, library=os.environ.get("MCSIM_TEST_CPU_LIBRARY")) as environment:
            with mock.patch.object(environment, "bridge", {}):
                with self.assertRaisesRegex(ValueError, "portable observation descriptor"):
                    environment.register_cuda_resident(bridge_observation_descriptor(2))
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    environment.register_cuda_resident(bridge_observation_descriptor(3))
                self.assertIsNone(environment._cuda_resident_buffers)
                self.assertFalse(environment._cuda_resident_ready)

    def test_resident_bridge_observation_status_is_reported(self):
        with Environment(1, library=os.environ.get("MCSIM_TEST_CPU_LIBRARY")) as environment:
            def status(_handle, pointer):
                ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 8
                return 0

            function = mock.Mock(side_effect=status)
            with mock.patch.object(environment, "_cuda_actions", object()), \
                 mock.patch.object(environment, "_cuda_action_device", "cuda"), \
                 mock.patch.object(environment.native, "mcsim_cuda_action_status", function), \
                 mock.patch.object(torch.cuda, "device", return_value=nullcontext()):
                with self.assertRaisesRegex(ValueError, "Invalid CUDA Bridge observation"):
                    environment.wait_cuda_actions()

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CPU_LIBRARY")
                         and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA, MCSIM_TEST_CPU_LIBRARY and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_resident_bridge_adapter_matches_cpu_through_reset(self):
        from simulator.training.research import bridge_observation_inputs

        contract = {"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
                    "ticks_per_action": 1,
                    "observations": [{"feature": "relative.z"}, {"feature": "self.health_fraction"}],
                    "actions": [{"name": "movement", "values": ["neutral", "forward"]}]}
        bridge = {"mode": {"name": "bridge", "version": 1},
                  "goals": [{"min": [0, 1, -2.9], "max": [1, 3, -2.4]},
                            {"min": [0, 1, 2.4], "max": [1, 3, 2.9]}],
                  "void_y": -2, "match": {"goals_to_win": 1, "max_ticks": 100, "cage_ticks": 0,
                                           "persist_blocks_on_goal": True, "arrow_refill_ticks": 0},
                  "task_rewards": {"score": 2, "win": 5, "loss": -3,
                                   "opponent_score": -1, "toward": 0.25, "away": 0.1}}
        scene = json.loads((ROOT / "simulator/tests/reset_environment.json").read_text(encoding="utf-8"))
        scene["players"][0]["position"][2] = 2.0
        scene["players"][1]["position"][0] = -1.5
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "resident-bridge.json"
        path.write_text(json.dumps(scene), encoding="utf-8")
        descriptor = bridge_observation_descriptor(3)
        options = {"environment": path, "contract": contract, "max_ticks": 20, "seed": 7, "bridge": bridge}
        with Environment(2, device="cpu", library=os.environ["MCSIM_TEST_CPU_LIBRARY"], **options) as cpu, \
             Environment(2, device="cuda", library=os.environ["MCSIM_TEST_CUDA_LIBRARY"], **options) as cuda:
            cuda.register_cuda_observations()
            actions = cuda.register_cuda_actions()
            outputs = cuda.register_cuda_resident(descriptor)
            stream = torch.cuda.Stream()

            def compare(actual, scalar):
                expected = bridge_observation_inputs(cpu, scalar, descriptor)
                for name in expected:
                    np.testing.assert_allclose(actual[name].cpu().numpy(), expected[name], rtol=0, atol=1e-6)

            compare(cuda.observe_cuda_async(), cpu.observe())
            host_actions = np.zeros((2, 2, 1), dtype=np.int32)
            host_actions[0, 0, 0] = 1
            saw_termination = False
            for _ in range(20):
                expected, rewards, terminated, truncated = cpu.step(host_actions)
                states = cpu.bridge_state()
                terms = cpu.bridge_reward_terms()
                losses = cpu.losses(np.ones(2, dtype=bool))
                with torch.cuda.stream(stream):
                    actions.copy_(torch.as_tensor(host_actions, device=actions.device))
                    actual, step_outputs = cuda.step_cuda_resident_async()
                    captured = {name: value.clone() for name, value in step_outputs.items() if name != "reset_mask"}
                    final = {name: value.clone() for name, value in actual.items()}
                stream.synchronize()
                cuda.wait_cuda_actions()
                compare(final, expected)
                np.testing.assert_allclose(captured["rewards"].cpu().numpy(), rewards, rtol=0, atol=1e-6)
                np.testing.assert_array_equal(captured["terminated"].cpu().numpy(), terminated)
                np.testing.assert_array_equal(captured["truncated"].cpu().numpy(), truncated)
                np.testing.assert_array_equal(captured["losses"].cpu().numpy(), losses)
                np.testing.assert_array_equal(captured["bridge_scored"].cpu().numpy(),
                                              [state["scored"] for state in states])
                np.testing.assert_allclose(captured["bridge_reward_terms"].cpu().numpy(), terms, rtol=0, atol=1e-6)
                done = terminated | truncated
                if done.any():
                    saw_termination |= bool(terminated.any())
                    reset_expected = cpu.reset(done)
                    with torch.cuda.stream(stream):
                        outputs["reset_mask"].copy_(torch.as_tensor(done, device=actions.device, dtype=torch.uint8))
                        reset = {name: value.clone() for name, value in cuda.reset_cuda_resident_async().items()}
                    stream.synchronize()
                    cuda.wait_cuda_actions()
                    compare(reset, reset_expected)
            self.assertTrue(saw_termination)
            observations = cuda.observe_cuda_async()
            observations["match"] = torch.empty_like(observations["match"])
            with self.assertRaisesRegex(RuntimeError, "Bridge observations changed"):
                cuda.step_cuda_resident_async()

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CPU_LIBRARY")
                         and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA, MCSIM_TEST_CPU_LIBRARY and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_resident_bridge_building_and_items_match_cpu(self):
        from simulator.training.research import bridge_observation_inputs

        scene = json.loads((ROOT / "simulator/tests/reset_environment.json").read_text(encoding="utf-8"))
        scene["blocks"].append({"from": [0, 2, 1], "to": [0, 2, 1], "block": "minecraft:stone"})
        scene["players"][1]["position"][0] = -2.5
        scene["players"][0]["health"] = 10
        for player in scene["players"]:
            player["velocity"][1] = -0.0784000015258789
            player["loadout"] = {"weapons": ["minecraft:diamond_sword", "minecraft:air", "minecraft:air",
                                             "minecraft:air"],
                                 "selected": 1, "arrows": 0, "offhand": "minecraft:air"}
        contract = {"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
                    "ticks_per_action": 1, "observations": [{"feature": "arena.ray_world_0"},
                                                             {"feature": "self.health_fraction"}],
                    "actions": [{"name": "weapon", "values": [0, 1, 2, 3]},
                                {"name": "use_item", "values": [False, True]},
                                {"name": "attack", "values": [False, True]}]}
        kit = {"slots": ["sword", "blocks", "pickaxe", "golden_apple"], "blocks": 4, "golden_apples": 2}
        bridge = {"mode": {"name": "bridge", "version": 1},
                  "goals": [{"min": [0, 1, 2.4], "max": [1, 3, 2.9]},
                            {"min": [0, 1, -1.1], "max": [1, 3, -0.8]}],
                  "void_y": -2, "match": {"goals_to_win": 2, "max_ticks": 100, "cage_ticks": 0,
                                           "persist_blocks_on_goal": True, "arrow_refill_ticks": 0},
                  "kits": [kit, kit], "task_rewards": {"score": 2, "win": 5, "loss": -3},
                  "building": {"zones": [{"min": [-3, 0, -3], "max": [4, 4, 4]}],
                               "block_range": 4.5, "entity_range": 3.0,
                               "hardness": 1.0, "mining_speed": 15.0, "correct_tool": True}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resident-building.json"
            path.write_text(json.dumps(scene))
            options = {"environment": path, "contract": contract, "max_ticks": 100, "seed": 7, "bridge": bridge}
            with Environment(2, device="cpu", library=os.environ["MCSIM_TEST_CPU_LIBRARY"], **options) as cpu, \
                 Environment(2, device="cuda", library=os.environ["MCSIM_TEST_CUDA_LIBRARY"], **options) as cuda:
                descriptor = bridge_observation_descriptor(3)
                cuda.register_cuda_observations()
                actions = cuda.register_cuda_actions()
                outputs = cuda.register_cuda_resident(descriptor)
                host_actions = np.zeros((2, 2, 3), dtype=np.int32)

                def compare(actual, scalar):
                    expected = bridge_observation_inputs(cpu, scalar, descriptor)
                    for name in expected:
                        np.testing.assert_allclose(actual[name].cpu().numpy(), expected[name], rtol=0, atol=1e-6)

                def step():
                    scalar, rewards, terminated, truncated = cpu.step(host_actions)
                    actions.copy_(torch.as_tensor(host_actions, device=actions.device))
                    actual, step_outputs = cuda.step_cuda_resident_async()
                    cuda.wait_cuda_actions()
                    compare(actual, scalar)
                    np.testing.assert_allclose(step_outputs["rewards"].cpu().numpy(), rewards, rtol=0, atol=1e-6)
                    self.assertFalse(terminated.any() or truncated.any())
                    return cpu.bridge_state()

                host_actions[0, 0] = [1, 1, 0]
                self.assertTrue(step()[0]["placed"][0])
                mask = np.array([False, True])
                scalar = cpu.reset(mask)
                outputs["reset_mask"].copy_(torch.as_tensor(mask, device=actions.device, dtype=torch.uint8))
                compare(cuda.reset_cuda_resident_async(), scalar)
                host_actions[0, 0] = [2, 0, 1]
                for _ in range(3):
                    states = step()
                self.assertTrue(states[0]["broke"][0])
                host_actions[0, 0] = [3, 1, 0]
                for _ in range(35):
                    step()
                self.assertEqual(cpu.bridge_kits(0)[0]["golden_apples"], 1)
                mask = np.array([True, False])
                scalar = cpu.reset(mask)
                outputs["reset_mask"].copy_(torch.as_tensor(mask, device=actions.device, dtype=torch.uint8))
                compare(cuda.reset_cuda_resident_async(), scalar)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_device_observations_match_host_after_steps_and_resets(self):
        library = os.environ.get("MCSIM_TEST_CUDA_LIBRARY")
        with Environment(3, device="cuda", library=library, max_ticks=20) as environment:
            def compare():
                expected = environment.observe()[:, :, :22]
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    actual = environment.observe_base_cuda()
                self.assertEqual(actual.device.type, "cuda")
                self.assertEqual(actual.dtype, torch.float32)
                self.assertEqual(tuple(actual.shape), (3, 2, 22))
                np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=0, atol=1e-6)

            compare()
            actions = np.zeros((3, 2, 6), dtype=np.int32)
            actions[:, :, 4] = 4
            actions[:, :, 5] = 2
            actions[:, 0, 0] = 1
            for _ in range(3):
                environment.step(actions)
                compare()
            environment.reset(np.array([True, False, True]))
            compare()

            function = environment.native.mcsim_observe_base_cuda_sync
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
            function.restype = ctypes.c_int
            host = np.empty((3, 2, 22), dtype=np.float32)
            self.assertEqual(function(environment._handle, host.ctypes.data, host.size, None), -1)
            self.assertIn(b"CUDA device", environment.native.mcsim_last_error())
            self.assertEqual(function(environment._handle, None, host.size, None), -1)
            device_output = torch.empty((3, 2, 22), device="cuda", dtype=torch.float32)
            self.assertEqual(function(environment._handle, device_output.data_ptr(), device_output.numel() - 1, None), -1)

        with self.assertRaisesRegex(RuntimeError, "closed"):
            environment.observe_base_cuda()

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_device_observations_are_not_policy_projected(self):
        contract = {"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
                    "ticks_per_action": 1,
                    "observations": [{"feature": "opponent.health_fraction"},
                                     {"feature": "self.health_fraction"}],
                    "actions": [{"name": "jump", "values": [False, True]}]}
        with Environment(2, device="cuda", contract=contract,
                         library=os.environ.get("MCSIM_TEST_CUDA_LIBRARY")) as environment:
            players = [environment.episode_scenario(world)["players"] for world in range(2)]
            for definition in players:
                definition[0]["health"] = 9
                definition[1]["health"] = 17
            environment.reset_players(players, np.array([True, True]))
            projected = environment.observe()
            raw = environment.observe_base_cuda().cpu().numpy()
            self.assertEqual(projected.shape, (2, 2, 2))
            self.assertEqual(raw.shape, (2, 2, 22))
            self.assertNotEqual(projected[0, 0, 0], projected[0, 0, 1])
            np.testing.assert_array_equal(projected[:, :, 0], raw[:, :, 4])
            np.testing.assert_array_equal(projected[:, :, 1], raw[:, :, 0])

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_configured_layout_tracks_loadouts_bows_and_selective_resets(self):
        contract = {"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
                    "ticks_per_action": 1, "requires": ["bow"],
                    "observations": [{"feature": name} for name in FEATURES] + [
                        {"feature": "self.weapon_slot", "encoding": "one_hot", "categories": [1, 0], "scale": -2},
                        {"feature": "self.health_fraction", "clip": [0.2, 0.6], "scale": 3}],
                    "actions": [{"name": "weapon", "values": [0, 1]},
                                {"name": "use_item", "values": [False, True]},
                                {"name": "attack", "values": [False, True]}]}
        path = ROOT / "simulator/tests/reset_environment.json"
        with Environment(3, device="cuda", environment=path, contract=contract, max_ticks=80,
                         library=os.environ.get("MCSIM_TEST_CUDA_LIBRARY")) as environment:
            self.compare_configured(environment)
            definitions = [environment.episode_scenario(world)["players"] for world in range(3)]
            definitions[0][0]["loadout"] = {"weapons": ["minecraft:bow", "minecraft:air"], "arrows": 2}
            definitions[1][0]["loadout"] = {"weapons": ["minecraft:air", "minecraft:air"], "selected": 1}
            definitions[2] = None
            environment.reset_players(definitions, np.array([True, True, False]))
            self.compare_configured(environment)
            actions = np.zeros((3, 2, 3), dtype=np.int32)
            actions[0, 0, 1] = 1
            for _ in range(12):
                environment.step(actions)
                self.compare_configured(environment)
            actions[0, 0, 1] = 0
            actions[1, 0, 0] = 0
            environment.step(actions)
            self.compare_configured(environment)
            environment.reset(np.array([False, True, False]))
            self.compare_configured(environment)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_v2_layout_tracks_shield_disable(self):
        with tempfile.TemporaryDirectory() as directory:
            definition = json.loads((ROOT / "simulator/environments/duel.json").read_text())
            definition["players"][0]["position"][2] = -0.5
            definition["players"][1]["position"][2] = 1.5
            path = Path(directory) / "shield-duel.json"
            path.write_text(json.dumps(definition), encoding="utf-8")
            for prepared_first in (False, True):
                with self.subTest(prepared_first=prepared_first), Environment(
                        1, device="cuda", environment=path, schema="v2", max_ticks=120,
                        library=os.environ.get("MCSIM_TEST_CUDA_LIBRARY")) as environment:
                    if prepared_first:
                        self.compare_configured(environment)
                    actions = np.zeros((1, 2, 8), dtype=np.int32)
                    actions[:, :, 4] = 4
                    actions[:, :, 5] = 2
                    actions[0, 0, 6] = 1
                    actions[0, 1, 7] = 1
                    for _ in range(6):
                        environment.step(actions)
                        if prepared_first:
                            self.compare_configured(environment)
                    actions[0, 0, 3] = 1
                    observed, _, _, _ = environment.step(actions)
                    self.assertEqual(observed[0, 0, 27], 1.0)
                    self.compare_configured(environment)
                    actions[0, 0, 3] = 0
                    for _ in range(3):
                        environment.step(actions)
                        self.compare_configured(environment)
                    environment.reset()
                    self.compare_configured(environment)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_async_observation_streams_lifetime_and_validation(self):
        library = os.environ.get("MCSIM_TEST_CUDA_LIBRARY")
        with Environment(3, device="cuda", library=library, max_ticks=30) as environment:
            with self.assertRaisesRegex(RuntimeError, "Register a CUDA observation buffer"):
                environment.observe_cuda_async()
            shape = (3, 2, environment.observation_dim)
            invalid = [torch.empty(shape), torch.empty(shape, device="cuda", dtype=torch.float64),
                       torch.empty((3, environment.observation_dim, 2), device="cuda").transpose(1, 2),
                       torch.empty((3, 2, environment.observation_dim + 1), device="cuda"),
                       torch.empty(shape, device="cuda", requires_grad=True)]
            for output in invalid:
                with self.assertRaisesRegex(ValueError, "contiguous float32"):
                    environment.register_cuda_observations(output)
            output = environment.register_cuda_observations()
            with ThreadPoolExecutor(max_workers=1) as executor:
                with self.assertRaisesRegex(RuntimeError, "registration thread"):
                    executor.submit(environment.observe_cuda_async).result()
                with self.assertRaisesRegex(RuntimeError, "registration thread"):
                    executor.submit(environment.reset).result()
            reference = weakref.ref(output)
            del output
            gc.collect()
            self.assertIsNotNone(reference())
            producer = torch.cuda.Stream()
            consumer = torch.cuda.Stream()
            expected = environment.observe()
            with torch.cuda.stream(producer):
                output = environment.observe_cuda_async()
                same_stream = output.clone()
            environment.wait_cuda_observations()
            with torch.cuda.stream(consumer):
                consumer.wait_stream(producer)
                cross_stream = output.clone()
            np.testing.assert_allclose(same_stream.cpu().numpy(), expected, rtol=0, atol=1e-6)
            np.testing.assert_allclose(cross_stream.cpu().numpy(), expected, rtol=0, atol=1e-6)
            with torch.cuda.stream(producer):
                output = environment.observe_cuda_async()
            actions = np.zeros((3, 2, 6), dtype=np.int32)
            actions[:, :, 4] = 4
            actions[:, :, 5] = 2
            actions[:, 0, 0] = 1
            environment.step(actions)
            np.testing.assert_allclose(output.cpu().numpy(), expected, rtol=0, atol=1e-6)
            expected = environment.observe()
            with torch.cuda.stream(producer):
                output = environment.observe_cuda_async()
            environment.reset(np.array([True, False, True]))
            np.testing.assert_allclose(output.cpu().numpy(), expected, rtol=0, atol=1e-6)
            output.resize_((output.numel() + 1,))
            with self.assertRaisesRegex(RuntimeError, "buffer changed"):
                environment.observe_cuda_async()
            replacement = environment.register_cuda_observations()
            np.testing.assert_allclose(environment.observe_cuda_async().cpu().numpy(), environment.observe(),
                                       rtol=0, atol=1e-6)
            self.assertEqual(tuple(replacement.shape), shape)
            graph = torch.cuda.CUDAGraph()
            capture_marker = torch.empty(1, device="cuda")
            with self.assertRaisesRegex(RuntimeError, "do not support graph capture"):
                with torch.cuda.graph(graph, stream=producer):
                    capture_marker.zero_()
                    environment.observe_cuda_async()
            expected = environment.observe()
            with torch.cuda.stream(producer):
                pending = environment.observe_cuda_async()
            environment.close()
            environment.close()
            np.testing.assert_allclose(pending.cpu().numpy(), expected, rtol=0, atol=1e-6)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            environment.observe_cuda_async()

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY")
                         and torch.cuda.device_count() >= 2, "Two CUDA devices and a CUDA library are required")
    def test_async_observation_rejects_another_cuda_device(self):
        library = os.environ.get("MCSIM_TEST_CUDA_LIBRARY")
        with torch.cuda.device(0):
            environment = Environment(2, device="cuda", library=library)
        wrong = torch.empty((2, 2, environment.observation_dim), device="cuda:1", dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "differs from batch device"):
            environment.register_cuda_observations(wrong)
        with torch.cuda.device(0):
            environment.register_cuda_observations()
            expected = environment.observe()
            other_stream = torch.cuda.Stream(device=1)
            function = environment.native.mcsim_observe_cuda_async
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            function.restype = ctypes.c_int
            self.assertEqual(function(environment._handle, other_stream.cuda_stream), -1)
            self.assertIn(b"CUDA", environment.native.mcsim_last_error())
            output = environment.observe_cuda_async()
        with torch.cuda.device(1):
            environment.close()
        np.testing.assert_allclose(output.cpu().numpy(), expected, rtol=0, atol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_async_observation_trace_has_no_synchronization_or_transfers(self):
        library = os.environ.get("MCSIM_TEST_CUDA_LIBRARY")
        with Environment(2, device="cuda", library=library) as environment:
            environment.register_cuda_observations()
            environment.observe_cuda_async()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                with torch.profiler.record_function("mcsim_async_observation"):
                    environment.observe_cuda_async()
            names = []
            for event in profiler.events():
                parent = event.cpu_parent
                while parent is not None:
                    if parent.name == "mcsim_async_observation":
                        names.append(event.name)
                        break
                    parent = parent.cpu_parent
            self.assertIn("cudaLaunchKernel", names, "Profiler must capture native CUDA runtime calls")
            self.assertFalse(any(name.startswith(("cudaDeviceSynchronize", "cudaStreamSynchronize",
                                                  "cudaEventSynchronize", "cudaMemcpy")) for name in names), names)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_async_observation_supports_maximum_policy_width(self):
        contract = {"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
                    "ticks_per_action": 1,
                    "observations": [{"feature": "self.health_fraction", "scale": (-1 if index % 2 else 1)}
                                     for index in range(128)],
                    "actions": [{"name": "jump", "values": [False, True]}]}
        with Environment(2, device="cuda", contract=contract,
                         library=os.environ.get("MCSIM_TEST_CUDA_LIBRARY")) as environment:
            output = environment.register_cuda_observations()
            self.assertEqual(tuple(output.shape), (2, 2, 128))
            self.compare_configured(environment)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_async_observations_cover_non_aligned_world_batches(self):
        library = os.environ.get("MCSIM_TEST_CUDA_LIBRARY")
        with Environment(137, device="cuda", library=library, max_ticks=30) as environment:
            self.compare_configured(environment)
            actions = np.zeros((137, 2, 6), dtype=np.int32)
            actions[:, :, 4] = 4
            actions[:, :, 5] = 2
            actions[::2, 0, 0] = 1
            for _ in range(3):
                environment.step(actions)
                self.compare_configured(environment)
            environment.reset(np.arange(137) % 3 == 0)
            self.compare_configured(environment)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_async_close_releases_resources_when_wait_fails(self):
        library = os.environ.get("MCSIM_TEST_CUDA_LIBRARY")
        environment = Environment(1, device="cuda", library=library)
        environment.register_cuda_observations()
        environment.observe_cuda_async()
        with mock.patch.object(environment, "wait_cuda_observations", side_effect=RuntimeError("wait failed")):
            with self.assertRaisesRegex(RuntimeError, "wait failed"):
                environment.close()
        self.assertIsNone(environment._handle)
        self.assertIsNone(environment._cuda_observations)
        environment.close()


class CUDAActionChecks(unittest.TestCase):
    def test_cpu_environment_rejects_cuda_actions(self):
        with Environment(1, library=os.environ.get("MCSIM_TEST_CPU_LIBRARY")) as environment:
            with self.assertRaisesRegex(ValueError, "require a CUDA environment"):
                environment.register_cuda_actions()
            with self.assertRaisesRegex(RuntimeError, "Register a CUDA action buffer"):
                environment.decode_cuda_actions_async()

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_async_actions_validate_buffers_and_sticky_status(self):
        library = os.environ.get("MCSIM_TEST_CUDA_LIBRARY")
        with Environment(3, device="cuda", library=library) as environment:
            with self.assertRaisesRegex(RuntimeError, "observation buffer"):
                environment.register_cuda_actions()
            environment.register_cuda_observations()
            shape = (3, 2, environment.action_count)
            invalid = [torch.empty(shape), torch.empty(shape, device="cuda", dtype=torch.int64),
                       torch.empty((3, environment.action_count, 2), device="cuda", dtype=torch.int32).transpose(1, 2),
                       torch.empty((3, 2, environment.action_count + 1), device="cuda", dtype=torch.int32)]
            for actions in invalid:
                with self.assertRaisesRegex(ValueError, "contiguous int32"):
                    environment.register_cuda_actions(actions)
            actions = environment.register_cuda_actions()
            reference = weakref.ref(actions)
            del actions
            gc.collect()
            self.assertIsNotNone(reference())
            actions = reference()
            producer = torch.cuda.Stream()
            with torch.cuda.stream(producer):
                actions.zero_()
                actions[:, :, 4] = 4
                actions[:, :, 5] = 2
                environment.decode_cuda_actions_async()
            environment.wait_cuda_actions()
            with torch.cuda.stream(producer):
                actions[0, 0, 0] = -1
                environment.decode_cuda_actions_async()
            with self.assertRaisesRegex(ValueError, "Invalid CUDA action"):
                environment.wait_cuda_actions()
            with torch.cuda.stream(producer):
                actions[0, 0, 0] = 0
                environment.decode_cuda_actions_async()
            with self.assertRaisesRegex(ValueError, "Invalid CUDA action"):
                environment.wait_cuda_actions()
            environment.clear_cuda_action_status()
            with torch.cuda.stream(producer):
                environment.decode_cuda_actions_async()
            environment.wait_cuda_actions()
            actions.resize_((actions.numel() + 1,))
            with self.assertRaisesRegex(RuntimeError, "buffer changed"):
                environment.decode_cuda_actions_async()
            actions = environment.register_cuda_actions()
            actions.zero_()
            actions[:, :, 4] = 4
            actions[:, :, 5] = 2
            environment.decode_cuda_actions_async()
            environment.wait_cuda_actions()

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_cuda_actions_support_custom_head_order_and_selected_slots(self):
        contract = {"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
                    "ticks_per_action": 1, "observations": [{"feature": "self.weapon_slot"}],
                    "actions": [{"name": "yaw", "values": [10, 0, -10]},
                                {"name": "weapon", "values": [1, 0]},
                                {"name": "sneak", "values": [True, False]}]}
        path = ROOT / "simulator/tests/reset_environment.json"
        with Environment(2, device="cuda", environment=path, contract=contract,
                         library=os.environ.get("MCSIM_TEST_CUDA_LIBRARY")) as environment:
            environment.register_cuda_observations()
            actions = environment.register_cuda_actions()
            actions.zero_()
            environment.decode_cuda_actions_async()
            environment.wait_cuda_actions()
            actions[1, 1, 1] = 2
            environment.decode_cuda_actions_async()
            with self.assertRaisesRegex(ValueError, "Invalid CUDA action"):
                environment.wait_cuda_actions()
            environment.clear_cuda_action_status()
            actions[1, 1, 1] = 1
            environment.decode_cuda_actions_async()
            environment.wait_cuda_actions()

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_cuda_action_trace_has_no_synchronization_or_transfers(self):
        library = os.environ.get("MCSIM_TEST_CUDA_LIBRARY")
        with Environment(2, device="cuda", library=library) as environment:
            environment.register_cuda_observations()
            actions = environment.register_cuda_actions()
            actions.zero_()
            environment.decode_cuda_actions_async()
            environment.wait_cuda_actions()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                with torch.profiler.record_function("mcsim_async_actions"):
                    environment.decode_cuda_actions_async()
            names = []
            for event in profiler.events():
                parent = event.cpu_parent
                while parent is not None:
                    if parent.name == "mcsim_async_actions":
                        names.append(event.name)
                        break
                    parent = parent.cpu_parent
            self.assertIn("cudaLaunchKernel", names, "Profiler must capture native CUDA runtime calls")
            self.assertFalse(any(name.startswith(("cudaDeviceSynchronize", "cudaStreamSynchronize",
                                                  "cudaEventSynchronize", "cudaMemcpy")) for name in names), names)
            environment.wait_cuda_actions()


class CUDAResidentChecks(unittest.TestCase):
    def test_cpu_environment_rejects_resident_mode(self):
        with Environment(1, library=os.environ.get("MCSIM_TEST_CPU_LIBRARY")) as environment:
            with self.assertRaisesRegex(ValueError, "requires a CUDA environment"):
                environment.register_cuda_resident()
            with self.assertRaisesRegex(RuntimeError, "Register CUDA resident buffers"):
                environment.step_cuda_resident_async()

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_invalid_resident_action_and_reset_do_not_mutate_worlds(self):
        with Environment(3, device="cuda", library=os.environ["MCSIM_TEST_CUDA_LIBRARY"]) as environment:
            environment.register_cuda_observations()
            actions = environment.register_cuda_actions()
            outputs = environment.register_cuda_resident()
            initial = environment.observe_cuda_async().clone()
            actions.zero_()
            actions[:, :, 4] = 4
            actions[:, :, 5] = 2
            actions[1, 0, 0] = 999
            environment.step_cuda_resident_async()
            with self.assertRaisesRegex(ValueError, "Invalid CUDA action"):
                environment.wait_cuda_actions()
            environment.clear_cuda_action_status()
            torch.testing.assert_close(environment.observe_cuda_async(), initial)
            outputs["reset_mask"].fill_(2)
            environment.reset_cuda_resident_async()
            with self.assertRaisesRegex(ValueError, "Invalid CUDA reset mask"):
                environment.wait_cuda_actions()
            environment.clear_cuda_action_status()
            torch.testing.assert_close(environment.observe_cuda_async(), initial)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_native_resident_episode_counters_round_trip(self):
        with Environment(3, device="cuda", library=os.environ["MCSIM_TEST_CUDA_LIBRARY"]) as environment:
            environment.register_cuda_observations()
            environment.register_cuda_actions()
            buffers = environment.register_cuda_resident()
            read = environment.native.mcsim_cuda_reset_episodes
            write = environment.native.mcsim_set_cuda_reset_episodes
            for function in (read, write):
                function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
                function.restype = ctypes.c_int
            before = np.empty(3, dtype=np.uint64)
            self.assertEqual(read(environment._handle, before.ctypes.data, 3), 0)
            buffers["reset_mask"].copy_(torch.tensor([1, 0, 1], device="cuda", dtype=torch.uint8))
            environment.reset_cuda_resident_async()
            after = np.empty_like(before)
            self.assertEqual(read(environment._handle, after.ctypes.data, 3), 0)
            np.testing.assert_array_equal(after, before + np.array([1, 0, 1], dtype=np.uint64))
            self.assertEqual(write(environment._handle, before.ctypes.data, 3), 0)
            self.assertEqual(read(environment._handle, after.ctypes.data, 3), 0)
            np.testing.assert_array_equal(after, before)
            self.assertEqual(read(environment._handle, after.ctypes.data, 2), -1)
            expected = torch.tensor([0, -9223372036854775808, -1], dtype=torch.int64)
            environment.set_cuda_reset_episodes(expected)
            torch.testing.assert_close(environment.cuda_reset_episodes(), expected)
            buffers["reset_mask"].copy_(torch.tensor([1, 0, 1], device="cuda", dtype=torch.uint8))
            environment.reset_cuda_resident_async()
            with self.assertRaisesRegex(ValueError, "episode counter cannot be incremented"):
                environment.wait_cuda_actions()
            environment.clear_cuda_action_status()
            torch.testing.assert_close(environment.cuda_reset_episodes(), expected)
            with self.assertRaisesRegex(ValueError, "one entry per world"):
                environment.set_cuda_reset_episodes(torch.zeros(2, dtype=torch.int64))

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_resident_duel_behavior_events_match_cpu_without_reward_weights(self):
        definition = json.loads((ROOT / "simulator/environments/duel.json").read_text())
        definition["players"][0]["pitch"] = 90
        definition["players"][0]["loadout"] = {"weapons": ["minecraft:wind_charge"], "selected": 0,
                                               "wind_charges": 2, "offhand": "minecraft:air"}
        contract = {"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
                    "ticks_per_action": 1, "observations": [{"feature": "self.velocity.y"}],
                    "actions": [{"name": "use_item", "values": [False, True]}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wind.json"
            path.write_text(json.dumps(definition), encoding="utf-8")
            options = {"environment": path, "contract": contract, "max_ticks": 20, "seed": 7,
                       "weights": dict.fromkeys(REWARD_NAMES, 0.0)}
            with Environment(2, device="cpu", library=os.environ.get("MCSIM_TEST_CPU_LIBRARY"), **options) as reference, \
                 Environment(2, device="cuda", library=os.environ["MCSIM_TEST_CUDA_LIBRARY"], **options) as environment:
                environment.register_cuda_observations()
                actions = environment.register_cuda_actions()
                buffers = environment.register_cuda_resident(behavior_events=True)
                with self.assertRaisesRegex(RuntimeError, "behavior buffers already registered"):
                    environment.register_cuda_resident()
                host_actions = np.zeros((2, 2, 1), dtype=np.int32)
                host_actions[0, 0, 0] = 1
                accumulated = 0.0
                for _ in range(12):
                    reference.step(host_actions)
                    expected = reference.step_behavior_events()
                    actions.copy_(torch.as_tensor(host_actions, device=actions.device))
                    _, outputs = environment.step_cuda_resident_async()
                    actual = outputs["behavior_events"].cpu().numpy()
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
                    accumulated += float(actual[0, 0, 0])
                    host_actions.fill(0)
                environment.wait_cuda_actions()
                self.assertGreater(accumulated, 0.2)
                self.assertEqual(float(buffers["rewards"].abs().sum()), 0.0)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    mask = np.array([True, False])
                    reference.reset(mask)
                    buffers["reset_mask"].copy_(torch.as_tensor(mask, device=actions.device))
                    environment.reset_cuda_resident_async()
                    reference.step(host_actions)
                    actions.copy_(torch.as_tensor(host_actions, device=actions.device))
                    _, outputs = environment.step_cuda_resident_async()
                    np.testing.assert_allclose(outputs["behavior_events"].cpu().numpy(),
                                               reference.step_behavior_events(), rtol=0, atol=1e-6)
                    original = buffers["behavior_events"]
                    buffers["behavior_events"] = torch.empty_like(original)
                    with self.assertRaisesRegex(RuntimeError, "resident buffer changed"):
                        environment.step_cuda_resident_async()
                    buffers["behavior_events"] = original
                torch.cuda.current_stream().wait_stream(stream)

        with Environment(1, device="cuda", library=os.environ["MCSIM_TEST_CUDA_LIBRARY"]) as environment:
            environment.register_cuda_observations()
            environment.register_cuda_actions()
            with mock.patch.object(environment.native, "mcsim_register_cuda_reset_mask", return_value=-1), \
                 mock.patch.object(environment.native, "mcsim_last_error", return_value=b"forced reset registration failure"):
                with self.assertRaisesRegex(RuntimeError, "forced reset registration failure"):
                    environment.register_cuda_resident(behavior_events=True)
            with self.assertRaisesRegex(RuntimeError, "behavior buffers already registered"):
                environment.register_cuda_resident()

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_resident_duel_mace_smash_event_matches_cpu(self):
        definition = json.loads((ROOT / "simulator/environments/duel.json").read_text())
        definition["bounds"]["size"][1] = 10
        definition["players"][0].update(position=[0.5, 6, 0.5], velocity=[0, 0, 0], pitch=90,
                                        on_ground=False, attack_strength_ticks=40,
                                        loadout={"weapons": ["minecraft:mace"], "selected": 0})
        definition["players"][1]["position"] = [0.5, 1, 0.5]
        contract = {"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
                    "ticks_per_action": 1, "observations": [{"feature": "self.velocity.y"}],
                    "actions": [{"name": "attack", "values": [False, True]}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mace.json"
            path.write_text(json.dumps(definition), encoding="utf-8")
            options = {"environment": path, "contract": contract, "max_ticks": 20, "seed": 7,
                       "weights": dict.fromkeys(REWARD_NAMES, 0.0)}
            with Environment(1, device="cpu", library=os.environ.get("MCSIM_TEST_CPU_LIBRARY"), **options) as reference, \
                 Environment(1, device="cuda", library=os.environ["MCSIM_TEST_CUDA_LIBRARY"], **options) as environment:
                environment.register_cuda_observations()
                actions = environment.register_cuda_actions()
                environment.register_cuda_resident(behavior_events=True)
                host_actions = np.zeros((1, 2, 1), dtype=np.int32)
                smashed = False
                for tick in range(20):
                    host_actions[0, 0, 0] = tick % 2 == 0
                    _, _, terminated, truncated = reference.step(host_actions)
                    actions.copy_(torch.as_tensor(host_actions, device=actions.device))
                    _, outputs = environment.step_cuda_resident_async()
                    expected = reference.step_behavior_events()
                    np.testing.assert_allclose(outputs["behavior_events"].cpu().numpy(), expected, rtol=0, atol=1e-6)
                    smashed |= bool(expected[0, 0, 1])
                    if terminated[0] or truncated[0]:
                        break
                environment.wait_cuda_actions()
                self.assertTrue(smashed)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_resident_step_matches_cpu_and_preserves_final_observations(self):
        cpu_library = os.environ.get("MCSIM_TEST_CPU_LIBRARY")
        cuda_library = os.environ.get("MCSIM_TEST_CUDA_LIBRARY")
        with Environment(3, device="cpu", library=cpu_library, max_ticks=3, seed=7) as reference, \
                Environment(3, device="cuda", library=cuda_library, max_ticks=3, seed=7) as environment:
            initial = reference.observe()
            environment.register_cuda_observations()
            actions = environment.register_cuda_actions()
            buffers = environment.register_cuda_resident()
            producer = torch.cuda.Stream()
            np.testing.assert_allclose(environment.observe_cuda_async().cpu().numpy(), initial, rtol=0, atol=1e-6)
            with self.assertRaisesRegex(RuntimeError, "Host state is unavailable"):
                environment.observe()
            with self.assertRaisesRegex(RuntimeError, "Host state is unavailable"):
                environment.step(np.zeros((3, 2, 6), dtype=np.int32))
            host_actions = np.zeros((3, 2, 6), dtype=np.int32)
            host_actions[:, :, 4] = 4
            host_actions[:, :, 5] = 2
            host_actions[::2, 0, 0] = 1
            for _ in range(3):
                expected, rewards, terminated, truncated = reference.step(host_actions)
                losses = reference.losses(np.ones(3, dtype=bool))
                with torch.cuda.stream(producer):
                    actions.copy_(torch.as_tensor(host_actions, device=actions.device))
                    observed, outputs = environment.step_cuda_resident_async()
                    captured = {name: value.clone() for name, value in outputs.items() if name != "reset_mask"}
                    final = observed.clone()
                producer.synchronize()
                environment.wait_cuda_actions()
                np.testing.assert_allclose(final.cpu().numpy(), expected, rtol=0, atol=1e-6)
                np.testing.assert_allclose(captured["rewards"].cpu().numpy(), rewards, rtol=0, atol=1e-6)
                np.testing.assert_array_equal(captured["terminated"].cpu().numpy(), terminated)
                np.testing.assert_array_equal(captured["truncated"].cpu().numpy(), truncated)
                np.testing.assert_array_equal(captured["losses"].cpu().numpy(), losses)
                for world in range(3):
                    for player in range(2):
                        selected = np.zeros((3, 2), dtype=bool)
                        selected[world, player] = True
                        totals = reference.step_reward_totals(selected)
                        np.testing.assert_allclose(captured["contributions"][world, player].cpu().numpy(),
                                                   [totals[name] for name in REWARD_NAMES], rtol=0, atol=1e-6)
            self.assertTrue(truncated.all())
            mask = np.array([True, False, True])
            expected_reset = reference.reset(mask)
            with torch.cuda.stream(producer):
                buffers["reset_mask"].copy_(torch.as_tensor(mask, device=actions.device, dtype=torch.uint8))
                reset = environment.reset_cuda_resident_async().clone()
            producer.synchronize()
            environment.wait_cuda_actions()
            np.testing.assert_allclose(reset.cpu().numpy(), expected_reset, rtol=0, atol=1e-6)
            np.testing.assert_array_equal(buffers["truncated"].cpu().numpy(), truncated)
            with torch.cuda.stream(producer):
                environment.step_cuda_resident_async()
            with self.assertRaisesRegex(ValueError, "finished world"):
                environment.wait_cuda_actions()
            environment.clear_cuda_action_status()
            mask = np.array([False, True, False])
            expected_reset = reference.reset(mask)
            with torch.cuda.stream(producer):
                buffers["reset_mask"].copy_(torch.as_tensor(mask, device=actions.device, dtype=torch.uint8))
                reset = environment.reset_cuda_resident_async().clone()
            producer.synchronize()
            environment.wait_cuda_actions()
            np.testing.assert_allclose(reset.cpu().numpy(), expected_reset, rtol=0, atol=1e-6)
            buffers["rewards"] = torch.empty_like(buffers["rewards"])
            with self.assertRaisesRegex(RuntimeError, "resident buffer changed"):
                environment.step_cuda_resident_async()

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_resident_configured_projection_and_loss_draw_match_cpu(self):
        definition = json.loads((ROOT / "simulator/tests/reset_environment.json").read_text())
        definition["blocks"] = [{"from": [0, 0, -2], "to": [0, 0, -2], "block": "minecraft:stone"}]
        definition["policy"] = {"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
                                "ticks_per_action": 1,
                                "observations": [{"feature": "self.weapon_slot", "encoding": "one_hot",
                                                  "categories": [1, 0]},
                                                 {"feature": "opponent.health_fraction"}],
                                "actions": [{"name": "yaw", "values": [-5, 0, 5]},
                                            {"name": "jump", "values": [False, True]}]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fall.json"
            path.write_text(json.dumps(definition), encoding="utf-8")
            with Environment(2, "cpu", path, os.environ.get("MCSIM_TEST_CPU_LIBRARY"), max_ticks=100) as reference, \
                    Environment(2, "cuda", path, os.environ["MCSIM_TEST_CUDA_LIBRARY"], max_ticks=100) as device:
                players = [reference.episode_scenario(index)["players"] for index in range(2)]
                players[0][0]["position"] = [-1.5, 1, -1.5]
                players[0][1]["position"] = [1.5, 1, 1.5]
                players[1][1]["position"] = [1.5, 1, 1.5]
                mask = np.ones(2, dtype=bool)
                reference.reset_players(players, mask)
                device.reset_players(players, mask)
                device.register_cuda_observations()
                actions = device.register_cuda_actions()
                device.register_cuda_resident()
                host_actions = np.zeros((2, 2, 2), dtype=np.int32)
                host_actions[:, :, 0] = 1
                actions.copy_(torch.as_tensor(host_actions, device="cuda"))
                for _ in range(30):
                    expected, rewards, terminated, truncated = reference.step(host_actions)
                    actual, outputs = device.step_cuda_resident_async()
                    device.wait_cuda_actions()
                    np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=0, atol=1e-6)
                    np.testing.assert_allclose(outputs["rewards"].cpu().numpy(), rewards, rtol=0, atol=1e-6)
                    np.testing.assert_array_equal(outputs["terminated"].cpu().numpy(), terminated)
                    np.testing.assert_array_equal(outputs["truncated"].cpu().numpy(), truncated)
                    if terminated.all():
                        break
                self.assertTrue(terminated.all())
                np.testing.assert_array_equal(reference.losses(mask), [[True, True], [False, True]])
                np.testing.assert_array_equal(outputs["losses"].cpu().numpy(), [[1, 1], [0, 1]])

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_invalid_resident_registration_preserves_host_path(self):
        library = os.environ.get("MCSIM_TEST_CUDA_LIBRARY")
        with Environment(2, device="cuda", library=library) as environment:
            environment.register_cuda_observations()
            environment.register_cuda_actions()
            function = environment.native.mcsim_register_cuda_step_outputs
            function.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_size_t]
            function.restype = ctypes.c_int
            host = np.zeros(2, dtype=np.float32)
            self.assertEqual(function(environment._handle, host.ctypes.data, host.ctypes.data,
                                      host.ctypes.data, host.ctypes.data, host.ctypes.data, 2), -1)
            self.assertEqual(environment.observe().shape, (2, 2, 22))

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_resident_step_and_reset_trace_has_no_sync_or_transfers(self):
        library = os.environ.get("MCSIM_TEST_CUDA_LIBRARY")
        with Environment(2, device="cuda", library=library, max_ticks=1) as environment:
            environment.register_cuda_observations()
            actions = environment.register_cuda_actions()
            buffers = environment.register_cuda_resident()
            actions.zero_()
            actions[:, :, 4] = 4
            actions[:, :, 5] = 2
            buffers["reset_mask"].fill_(1)
            environment.step_cuda_resident_async()
            environment.wait_cuda_actions()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                with torch.profiler.record_function("mcsim_resident_step_reset"):
                    environment.reset_cuda_resident_async()
                    environment.step_cuda_resident_async()
            names = []
            for event in profiler.events():
                parent = event.cpu_parent
                while parent is not None:
                    if parent.name == "mcsim_resident_step_reset":
                        names.append(event.name)
                        break
                    parent = parent.cpu_parent
            self.assertGreaterEqual(names.count("cudaLaunchKernel"), 8,
                                    "Profiler must capture native CUDA step and reset kernels")
            self.assertFalse(any(name.startswith(("cudaDeviceSynchronize", "cudaStreamSynchronize",
                                                  "cudaEventSynchronize", "cudaMemcpy")) for name in names), names)
            environment.wait_cuda_actions()

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_resident_collector_matches_host_across_timeout_and_rollouts(self):
        from copy import deepcopy
        from backend.fast_rl_model import ActorCritic
        from simulator.training._test_plans import example_plan
        from simulator.training.plans import resolve_plan
        from simulator.training.policies import fixed_indices
        from simulator.training.ppo import optimize
        from simulator.training.research import OpponentPool, ResidentCollector, SampledWorlds, collect

        plan = resolve_plan(example_plan())
        stage = deepcopy(plan["stages"][0])
        stage["scenario"].pop("spawns", None)
        stage["scenario"].pop("equipment", None)
        count = 4
        cpu = SampledWorlds(stage["scenario"], plan["policy"], count, "cpu",
                            os.environ.get("MCSIM_TEST_CPU_LIBRARY"), 3, 42)
        gpu = SampledWorlds(stage["scenario"], plan["policy"], count, "cuda",
                            os.environ["MCSIM_TEST_CUDA_LIBRARY"], 3, 42)
        try:
            cpu.configure(stage)
            gpu.configure(stage)
            torch.manual_seed(7)
            sizes = dict(zip(cpu.action_names, plan["policy"]["action_sizes"]))
            host_model = ActorCritic(cpu.observation_dim, 16, action_sizes=sizes)
            device_model = ActorCritic(cpu.observation_dim, 16, action_sizes=sizes).cuda()
            device_model.load_state_dict(host_model.state_dict())
            host_pool = OpponentPool(plan["opponents"], plan["policy"], count, "cpu", 42, learner=host_model)
            device_pool = OpponentPool(plan["opponents"], plan["policy"], count, "cuda", 42, learner=device_model)
            neutral = host_pool.controllers["stationary"]._neutral
            controls = {"fixed": {head["name"]: head["values"][int(index)]
                                  for head, index in zip(plan["policy"]["actions"], neutral)}}
            host_pool.configure(stage["opponents"], controls)
            device_pool.configure(stage["opponents"], controls)
            fixed = fixed_indices(plan["policy"], controls)
            host_observations = cpu.observe()
            resident = ResidentCollector(gpu, device_pool, 42)
            host_pool.assignments[:] = resident.pool.assignments.cpu().numpy()
            host_pool.learner_slots[:] = resident.pool.learner_slots.cpu().numpy()
            for steps in (2, 1):
                expected, host_observations, endings, timeouts = collect(
                    host_model, cpu, host_pool, host_observations, steps, torch.device("cpu"), fixed)
                actual, _, device_endings, device_timeouts = resident.collect(device_model, steps, fixed)
                self.assertEqual((endings, timeouts), (device_endings, device_timeouts))
                for name in ("observations", "actions", "log_probs", "values", "next_values", "rewards",
                             "terminated", "truncated", "advantages", "returns"):
                    torch.testing.assert_close(actual[name].cpu(), expected[name], rtol=0, atol=2e-6)
                for name in REWARD_NAMES:
                    self.assertAlmostEqual(actual["reward_types"][name], expected["reward_types"][name], places=5)
                self.assertEqual(actual["episode_metrics"], expected["episode_metrics"])
            self.assertEqual(device_timeouts, count)
            host_optimizer = torch.optim.Adam(host_model.parameters(), lr=3e-4, eps=1e-5)
            device_optimizer = torch.optim.Adam(device_model.parameters(), lr=3e-4, eps=1e-5)
            host_stats = optimize(host_model, host_optimizer, expected, 1, count)
            device_stats = optimize(device_model, device_optimizer, actual, 1, count)
            for name in host_stats:
                self.assertAlmostEqual(host_stats[name], device_stats[name], places=5)
            for host_parameter, device_parameter in zip(host_model.parameters(), device_model.parameters(), strict=True):
                torch.testing.assert_close(device_parameter.cpu(), host_parameter, rtol=2e-5, atol=2e-6)
        finally:
            gpu.close()
            cpu.close()

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_resident_training_checkpoint_replay_stage_and_resume(self):
        import contextlib
        from copy import deepcopy
        import io
        from simulator.training._test_plans import example_plan
        from simulator.training.plans import resolve_plan
        from simulator.training.train import main

        plan = resolve_plan(example_plan())
        suite = plan["evaluations"]["held_out_combat"]
        suite.update(every_updates=1, episodes=2, worlds=2)
        plan["opponents"]["self"] = {"type": "self_play", "deterministic": False}
        plan["stages"] = [plan["stages"][0], deepcopy(plan["stages"][0])]
        for index, stage in enumerate(plan["stages"]):
            stage["name"] = f"resident-{index}"
            stage["advance"] = {"updates": 1 if index == 0 else 3}
            stage["opponents"] = [{"id": "stationary", "weight": 1}, {"id": "self", "weight": 1}]
            stage["scenario"].pop("spawns", None)
            stage["scenario"].pop("equipment", None)
            stage["evaluations"] = {}
        plan["scenario"] = deepcopy(plan["stages"][0]["scenario"])
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            path = root / "plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            output = root / "run"
            library = os.environ["MCSIM_TEST_CUDA_LIBRARY"]
            flags = ["--device", "cuda", "--rollout-backend", "cuda-resident", "--library", library,
                     "--envs", "4", "--rollout-steps", "3", "--max-ticks", "3", "--hidden-dim", "16",
                     "--epochs", "1", "--minibatch-size", "12", "--updates", "1",
                     "--replay-every-updates", "1"]
            self.assertEqual(main(["--plan", str(path), "--output", str(output), *flags]), 0)
            checkpoint = torch.load(output / "latest.pt", weights_only=True)
            self.assertEqual(checkpoint["research"]["rollout_backend"], "cuda-resident")
            self.assertEqual(checkpoint["research"]["stage_index"], 1)
            self.assertIn("resident_rng", checkpoint["research"])
            self.assertIn("resident_episodes", checkpoint["research"])
            self.assertEqual(len(list((output / "replays").glob("*.jsonl"))), 1)
            first_metrics = json.loads((output / "metrics.jsonl").read_text().splitlines()[0])
            self.assertEqual(first_metrics["evaluations"]["held_out_combat"]["episodes"], 2)
            with self.assertRaisesRegex(ValueError, "rollout backend differs"):
                main(["--resume", str(output / "latest.pt"), "--device", "cuda", "--library", library,
                      "--updates", "1"])
            self.assertEqual(main(["--resume", str(output / "latest.pt"), "--device", "cuda",
                                   "--rollout-backend", "cuda-resident", "--library", library,
                                   "--updates", "1", "--replay-every-updates", "1"]), 0)
            resumed = torch.load(output / "latest.pt", weights_only=True)
            self.assertEqual(resumed["update"], 2)
            self.assertEqual(resumed["research"]["rollout_backend"], "cuda-resident")
            torch.testing.assert_close(resumed["research"]["resident_episodes"],
                                       checkpoint["research"]["resident_episodes"] + 2)
            self.assertEqual(len(list((output / "replays").glob("*.jsonl"))), 2)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_resident_collector_decisions_have_no_host_transfers_or_waits(self):
        from backend.fast_rl_model import ActorCritic
        from simulator.training._test_plans import example_plan
        from simulator.training.plans import resolve_plan
        from simulator.training.research import OpponentPool, ResidentCollector, SampledWorlds

        plan = resolve_plan(example_plan())
        stage = plan["stages"][0]
        stage["scenario"].pop("spawns", None)
        stage["scenario"].pop("equipment", None)
        definitions = {"target": {"type": "scripted", "controller": "stationary_target"},
                       "self": {"type": "self_play"}}
        stage["opponents"] = [{"id": "target", "weight": 1}, {"id": "self", "weight": 1}]
        worlds = SampledWorlds(stage["scenario"], plan["policy"], 4, "cuda",
                               os.environ["MCSIM_TEST_CUDA_LIBRARY"], 2, 42)
        try:
            worlds.configure(stage)
            model = ActorCritic(worlds.observation_dim, 16,
                                action_sizes=dict(zip(worlds.action_names, plan["policy"]["action_sizes"]))).cuda()
            pool = OpponentPool(definitions, plan["policy"], 4, "cuda", 42, learner=model)
            pool.configure(stage["opponents"], {"fixed": {}})
            resident = ResidentCollector(worlds, pool, 42)
            resident.collect(model, 2, {})
            original_act, original_reset = resident.pool.act, resident.pool.reset
            scopes = []

            def start_scope(*args):
                scope = torch.profiler.record_function("mcsim_resident_decision")
                scope.__enter__()
                scopes.append(scope)
                return original_act(*args)

            def end_scope(*args):
                result = original_reset(*args)
                scopes.pop().__exit__(None, None, None)
                return result

            with mock.patch.object(resident.pool, "act", side_effect=start_scope), \
                    mock.patch.object(resident.pool, "reset", side_effect=end_scope), \
                    torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as profiler:
                resident.collect(model, 2, {})
            names = []
            operations = []
            for event in profiler.events():
                parent = event.cpu_parent
                chain = []
                while parent is not None:
                    if parent.name == "mcsim_resident_decision":
                        names.append(event.name)
                        if event.name.startswith(("cudaDeviceSynchronize", "cudaStreamSynchronize",
                                                  "cudaEventSynchronize", "aten::item",
                                                  "aten::_local_scalar_dense")):
                            operations.append((event.name, chain))
                        if event.name.startswith("cudaMemcpy"):
                            kernels = event.cpu_parent.kernels
                            if not kernels or any(not kernel.name.startswith("Memcpy DtoD") for kernel in kernels):
                                operations.append((event.name, chain, kernels))
                        break
                    chain.append(parent.name)
                    parent = parent.cpu_parent
            self.assertIn("cudaLaunchKernel", names, "Profiler must capture native CUDA calls")
            self.assertFalse(operations, operations)
        finally:
            worlds.close()

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_resident_collector_rejects_nonfinite_policy_before_optimization(self):
        from backend.fast_rl_model import ActorCritic
        from simulator.training._test_plans import example_plan
        from simulator.training.plans import resolve_plan
        from simulator.training.research import OpponentPool, ResidentCollector, SampledWorlds

        plan = resolve_plan(example_plan())
        stage = plan["stages"][0]
        stage["scenario"].pop("spawns", None)
        stage["scenario"].pop("equipment", None)
        worlds = SampledWorlds(stage["scenario"], plan["policy"], 2, "cuda",
                               os.environ["MCSIM_TEST_CUDA_LIBRARY"], 2, 42)
        try:
            worlds.configure(stage)
            model = ActorCritic(worlds.observation_dim, 16,
                                action_sizes=dict(zip(worlds.action_names, plan["policy"]["action_sizes"]))).cuda()
            pool = OpponentPool(plan["opponents"], plan["policy"], 2, "cuda", 42, learner=model)
            pool.configure(stage["opponents"], stage["controls"])
            resident = ResidentCollector(worlds, pool, 42)
            with torch.no_grad():
                next(model.parameters()).fill_(float("nan"))
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                resident.collect(model, 1, {})
        finally:
            worlds.close()


if __name__ == "__main__":
    unittest.main()
