"""Focused checks for bounded checkpoint evaluation helpers."""

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from simulator.training.contract import fingerprint, from_legacy
from simulator.training.environment import SCHEMA
from simulator.training.evaluate_checkpoints import (
    _evaluate_pairing,
    _freeze_file,
    _outcome,
    _require_matching_samples,
    _resolved_stage,
    _select_common_key,
    _slot_for_episode,
    _summarize,
    _wilson_interval,
    run,
)


class CheckpointSelectionChecks(unittest.TestCase):
    def test_training_opponent_weights_do_not_change_evaluation_conditions(self):
        policy = from_legacy(SCHEMA)
        stage = {"name": "combat", "controls": {"fixed": {}}, "scenario": {},
                 "rewards": {}, "opponents": [{"id": "self", "weight": 2}]}
        plan = {"policy": policy, "stages": [stage]}
        record = {"stage_index": 0, "stage_name": "combat", "stage_updates": 30,
                  "policy_sha256": fingerprint(policy), "controls": {"fixed": {}}, "max_ticks": 1200}
        _, first = _resolved_stage(plan, record)
        stage["opponents"][0]["weight"] = 6
        _, second = _resolved_stage(plan, record)
        self.assertEqual(first, second)
        stage["rewards"]["win"] = 2
        self.assertNotEqual(first, _resolved_stage(plan, record)[1])

    def test_selects_highest_exact_update_and_stage_intersection(self):
        first = {
            (20, "combat", 1, 10): {},
            (40, "combat", 1, 30): {},
            (60, "combat", 1, 50): {},
        }
        second = {
            (20, "combat", 1, 10): {},
            (40, "combat", 1, 30): {},
            (80, "combat", 1, 70): {},
        }
        self.assertEqual(_select_common_key({"a": first, "b": second}), (40, "combat", 1, 30))
        self.assertEqual(_select_common_key({"a": first, "b": second}, 20), (20, "combat", 1, 10))

    def test_refuses_same_global_update_with_incompatible_stage_identity(self):
        inventories = {
            "a": {(60, "combat", 1, 40): {}},
            "b": {(60, "pursuit", 0, 60): {}},
        }
        with self.assertRaisesRegex(ValueError, "No exact common retained update and stage identity"):
            _select_common_key(inventories)

    def test_refuses_requested_update_when_only_a_nearby_snapshot_exists(self):
        inventories = {
            "a": {(40, "combat", 1, 30): {}},
            "b": {(40, "combat", 1, 30): {}, (60, "combat", 1, 50): {}},
        }
        with self.assertRaisesRegex(ValueError, "Requested update 50 was not retained"):
            _select_common_key(inventories, 50)

    def test_refuses_candidate_checkpoints_with_different_training_sample_counts(self):
        with self.assertRaisesRegex(ValueError, "incompatible training sample counts"):
            _require_matching_samples([{"samples": 1200}, {"samples": 1192}])


class OutcomeAndStatisticsChecks(unittest.TestCase):
    def test_cuda_selection_passes_a_concrete_visible_device_index(self):
        manifest = {"experiment": {"policy": {}, "stages": [
            {"name": "combat", "scenario": {"arenas": [{"environment": {"max_ticks": 100}}]}}]},
            "stage": {"name": "combat", "index": 0}, "candidates": [{}]}
        with tempfile.TemporaryDirectory() as temporary, \
                patch("simulator.training.evaluate_checkpoints.torch.cuda.is_available", return_value=True), \
                patch("simulator.training.evaluate_checkpoints.torch.cuda.current_device", return_value=0), \
                patch("simulator.training.evaluate_checkpoints.torch.cuda.set_device") as select, \
                patch("simulator.training.evaluate_checkpoints._load_manifest",
                      return_value=(Path(temporary), manifest, "hash")), \
                patch("simulator.training.evaluate_checkpoints._load_candidate_model",
                      side_effect=RuntimeError("stop after device selection")):
            with self.assertRaisesRegex(RuntimeError, "stop after device selection"):
                run(temporary, episodes=2, worlds=1, device="cuda")
            select.assert_called_once_with(0)

    def test_timeout_is_not_a_loss_and_terminal_loss_flags_define_combat_outcomes(self):
        self.assertEqual(_outcome(False, True, [True, False], 0), "timeout")
        self.assertEqual(_outcome(True, False, [True, True], 1), "draw")
        self.assertEqual(_outcome(True, False, [False, True], 0), "win")
        self.assertEqual(_outcome(True, False, [True, False], 0), "loss")
        self.assertEqual(_outcome(True, True, [False, True], 0), "win")

    def test_episode_assignment_balances_learner_slots(self):
        assignments = [_slot_for_episode(index) for index in range(256)]
        self.assertEqual(assignments.count(0), 128)
        self.assertEqual(assignments.count(1), 128)

    def test_summary_keeps_seat_counts_and_wilson_win_interval(self):
        episodes = [
            {"outcome": "win", "learner_slot": 0, "length": 8, "return": 1.5},
            {"outcome": "loss", "learner_slot": 1, "length": 10, "return": -1.0},
            {"outcome": "timeout", "learner_slot": 0, "length": 12, "return": 0.0},
            {"outcome": "draw", "learner_slot": 1, "length": 11, "return": 0.0},
        ]
        result = _summarize(episodes)
        self.assertEqual((result["win"], result["loss"], result["draw"], result["timeout"]), (1, 1, 1, 1))
        self.assertEqual(result["learner_slot_counts"]["0"]["episodes"], 2)
        self.assertEqual(result["learner_slot_counts"]["1"]["episodes"], 2)
        self.assertEqual(result["win_rate"], 0.25)
        self.assertEqual(result["mean_episode_length"], 10.25)
        self.assertEqual(result["mean_return"], 0.125)
        lower, upper = result["win_rate_wilson_95"]
        self.assertLess(lower, result["win_rate"])
        self.assertGreater(upper, result["win_rate"])
        self.assertIn("independent Bernoulli", result["uncertainty_assumptions"])
        self.assertIsNone(_wilson_interval(0, 0))


class FrozenArtifactChecks(unittest.TestCase):
    def test_identical_artifacts_are_deduplicated_by_content_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.pt"
            second = root / "second.pt"
            payload = b"trusted checkpoint test payload"
            first.write_bytes(payload)
            second.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            first_path = _freeze_file(first, root, "artifacts", ".pt", digest)
            second_path = _freeze_file(second, root, "artifacts", ".pt", digest)
            self.assertEqual(first_path, second_path)
            self.assertEqual((root / first_path).read_bytes(), payload)
            self.assertTrue(first.is_file())
            self.assertTrue(second.is_file())


class PairingScheduleChecks(unittest.TestCase):
    def _run_with_durations(self, duration_batches):
        created_worlds = []

        class MockWorlds:
            def __init__(self, count, seed, durations):
                self.count = count
                self.seed = seed
                self.durations = np.asarray(durations, dtype=np.int64)
                self.ticks = np.zeros(count, dtype=np.int64)
                self.max_ticks = 10
                self.action_names = ("move",)
                self.action_count = 1
                self.closed = False
                self.needs_reset = np.zeros(count, dtype=bool)
                created_worlds.append(self)

            def configure(self, stage):
                pass

            def observe(self):
                return np.zeros((self.count, 2, 1), dtype=np.float32)

            def step(self, actions):
                if self.needs_reset.any():
                    raise RuntimeError("Reset finished environments before the next step")
                self.ticks += 1
                terminated = self.ticks >= self.durations
                self.needs_reset[:] = terminated
                rewards = np.ones((self.count, 2), dtype=np.float32)
                return self.observe(), rewards, terminated, np.zeros(self.count, dtype=bool)

            def reset(self, mask):
                self.ticks[mask] = 0
                self.needs_reset[mask] = False
                return self.observe()

            def losses(self, done):
                return np.zeros((self.count, 2), dtype=bool)

            def close(self):
                self.closed = True

        class MockController:
            def reset(self, mask):
                pass

            def act(self, observations, world_ids):
                return np.zeros((len(world_ids), 1), dtype=np.int64)

        remaining_durations = iter(duration_batches)

        def make_worlds(scenario, contract, count, device, library, max_ticks, seed):
            worlds = MockWorlds(count, seed, next(remaining_durations))
            self.assertEqual(count, len(worlds.durations))
            return worlds

        def policy_actions(model, observations, action_names, fixed):
            self.assertFalse(torch.is_grad_enabled())
            actions = torch.zeros((len(observations), 1), dtype=torch.int64, device=observations.device)
            return actions, None, None, None

        options = type("Options", (), {
            "device": "cpu", "worlds": 4, "library": None,
            "max_ticks": None, "bundle_root": Path("."),
        })()
        with patch("simulator.training.evaluate_checkpoints.SampledWorlds", side_effect=make_worlds), \
                patch("simulator.training.evaluate_checkpoints.create_controller", return_value=MockController()), \
                patch("simulator.training.evaluate_checkpoints.fixed_indices", return_value=None), \
                patch("simulator.training.evaluate_checkpoints.policy_actions", side_effect=policy_actions):
            summary = _evaluate_pairing(
                object(), {}, {"scenario": {}, "controls": {"fixed": {}}},
                {"type": "scripted", "definition": {}}, options, episodes=6, seed=700)
        return summary, created_worlds

    def test_variable_completion_order_keeps_fixed_episode_seeds_and_seats(self):
        first, first_worlds = self._run_with_durations(([1, 5, 2, 4], [3, 1]))
        second, second_worlds = self._run_with_durations(([5, 1, 4, 2], [1, 3]))

        def assignments(summary):
            return [(sample["episode_id"], sample["seed"], sample["learner_slot"])
                    for sample in summary["episode_samples"]]

        self.assertEqual(assignments(first), assignments(second))
        self.assertEqual([sample["episode_id"] for sample in first["episode_samples"]], list(range(6)))
        self.assertEqual([sample["seed"] for sample in first["episode_samples"]], [700, 700, 700, 700, 701, 701])
        self.assertEqual(first["learner_slot_counts"]["0"]["episodes"], 3)
        self.assertEqual(first["learner_slot_counts"]["1"]["episodes"], 3)
        self.assertNotEqual([sample["length"] for sample in first["episode_samples"]],
                            [sample["length"] for sample in second["episode_samples"]])
        for worlds in (first_worlds, second_worlds):
            self.assertEqual([world.count for world in worlds], [4, 2])
            self.assertEqual([world.seed for world in worlds], [700, 701])
            self.assertTrue(all(world.closed for world in worlds))


if __name__ == "__main__":
    unittest.main()
