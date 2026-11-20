import contextlib
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import torch

from backend.fast_rl_model import ActorCritic, BridgeActorCritic
from simulator.training.analyze_metrics import main as analyze_metrics
from simulator.training._test_plans import TEST_SETUPS, example_plan
from simulator.training.contract import bridge_observation_descriptor, from_legacy, resolve_players, resolve_scenario, validate_contract
from simulator.training.environment import Environment, ROOT, SCHEMA, SCHEMA_V2, load_library
from simulator.training.controllers import CheckpointController
from simulator.training.legacy import SCHEMA as LEGACY_SCHEMA, SCHEMA_V2 as LEGACY_V2
from simulator.training.plans import EpisodeSampler, plan_hash, reached, resolve_plan
from simulator.training.policies import fixed_indices, policy_actions, resident_policy_actions
from simulator.training.research import (BehaviorEvaluation, EpisodeMetrics, OpponentPool,
                                         ResidentOpponentPool, SampledWorlds, bridge_observation_inputs,
                                         collect, evaluate, export_replay,
                                         requires_new_worlds, research_metadata, resident_unsupported)
from simulator.training.serve import Policy
from simulator.training.train import arguments, main, optimize
from simulator.training.timing import CollectionTimer


def short_plan():
    value = example_plan()
    value["evaluations"] = {}
    for stage in value["stages"]:
        stage["advance"] = {"updates": 2}
    return value


class BridgeResearchChecks(unittest.TestCase):
    def test_evaluation_gated_bridge_curriculum_resume_and_completion(self):
        import simulator.training.research as research

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = json.loads((TEST_SETUPS / "environment.json").read_text())
            environment["arena"] = str(TEST_SETUPS / "arena.json")
            environment.pop("spawns")
            environment.pop("equipment")
            environment["task"] = {"max_ticks": 3, "rewards": {}}
            environment["bridge"] = {"mode": {"name": "bridge", "version": 1},
                "goals": [{"min": [-6, 1, -1], "max": [-5, 3, 1]},
                          {"min": [5, 1, -1], "max": [6, 3, 1]}],
                "void_y": 0, "match": {"goals_to_win": 2, "max_ticks": 3, "cage_ticks": 0,
                    "persist_blocks_on_goal": True, "arrow_refill_ticks": 0},
                "task_rewards": {"score": 2, "win": 5, "loss": -3}}
            environment_path = root / "environment.json"
            environment_path.write_text(json.dumps(environment))
            second_arena = json.loads((TEST_SETUPS / "arena.json").read_text())
            second_arena["blocks"].append({"from": [0, 3, 0], "to": [0, 3, 0], "block": "minecraft:stone"})
            second_arena_path = root / "second-arena.json"
            second_arena_path.write_text(json.dumps(second_arena))
            second_environment = deepcopy(environment)
            second_environment["arena"] = str(second_arena_path)
            second_environment_path = root / "second-environment.json"
            second_environment_path.write_text(json.dumps(second_environment))
            stages = []
            for name in ("first", "second"):
                suites = {f"{name}-{suffix}": {"every_updates": 1, "episodes": 2, "worlds": 2, "seed": seed}
                          for suffix, seed in (("a", 17), ("b", 19))}
                stages.append({"name": name, "opponents": "target", "evaluations": suites,
                    "advance": {"all": [{"evaluation": {"suite": suite, "metric": "win_rate", "at_least": 0.5}}
                                        for suite in suites]},
                        "snapshots": [{"name": name, "on_completion": True}]})
            plan = {"format": "minecraft_sim_training_plan", "version": 2, "name": "gated-bridge-test", "seed": 42,
                "policy": str(TEST_SETUPS / "policy.json"), "environment": str(environment_path),
                "bridge_observation": bridge_observation_descriptor(),
                "learner": {"hidden_dim": 8, "epochs": 1, "minibatch_size": 2},
                "opponents": {"target": {"type": "scripted", "controller": "stationary_target"}},
                "mixtures": {"target": [{"id": "target", "weight": 1}]}, "stages": stages}
            plan["stages"][1]["environment"] = str(second_environment_path)
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan))
            output = root / "run"
            common = ["--device", "cpu", "--threads", "1", "--library", self.library, "--updates", "1"]
            rates = {17: 1.0, 19: 0.0}

            def evaluation(model, resolved, suite, options, device, fixed, opponent_pool=None):
                return {"episodes": suite["episodes"], "win_rate": rates[suite["seed"]]}

            with patch.object(research, "evaluate", side_effect=evaluation), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--plan", str(plan_path), "--output", str(output), "--envs", "2",
                                       "--rollout-steps", "2", *common]), 0)
                first = torch.load(output / "latest.pt", weights_only=True)
                self.assertEqual(first["research"]["stage_index"], 0)
                self.assertFalse((output / "snapshots/first.pt").exists())
                rates[19] = 0.5
                self.assertEqual(main(["--resume", str(output / "latest.pt"), *common]), 0)
                second = torch.load(output / "latest.pt", weights_only=True)
                self.assertEqual(second["research"]["stage_index"], 1)
                self.assertEqual(second["research"]["evaluations"], {})
                self.assertEqual(second["metadata"]["scenario"]["blocks"], second_arena["blocks"])
                self.assertTrue((output / "snapshots/first.pt").exists())
                rates[19] = 0.0
                self.assertEqual(main(["--resume", str(output / "latest.pt"), *common]), 0)
                third = torch.load(output / "latest.pt", weights_only=True)
                self.assertEqual(third["research"]["stage_index"], 1)
                self.assertFalse((output / "snapshots/second.pt").exists())
                rates[19] = 0.5
                self.assertEqual(main(["--resume", str(output / "latest.pt"), *common]), 0)
            rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["stage"] for row in rows], ["first", "first", "second", "second"])
            self.assertEqual(rows[1]["curriculum_boundary"]["next_stage"], "second")
            self.assertTrue(rows[-1]["completed_curriculum"])
            self.assertTrue((output / "snapshots/second.pt").exists())

    def test_structured_plan_rejects_invalid_descriptor_and_duel(self):
        plan = short_plan()
        plan["opponents"] = {"stationary": plan["opponents"]["stationary"]}
        plan["mixtures"] = {"target": plan["mixtures"]["target"]}
        plan["stages"] = [plan["stages"][0]]
        plan["bridge_observation"] = {**bridge_observation_descriptor(), "version": 2}
        with self.assertRaisesRegex(ValueError, "descriptor"):
            resolve_plan(plan)
        plan["bridge_observation"] = bridge_observation_descriptor(2)
        with self.assertRaisesRegex(ValueError, "Bridge stages"):
            resolve_plan(plan)
        plan["bridge_observation"] = bridge_observation_descriptor()
        with self.assertRaisesRegex(ValueError, "Bridge stages"):
            resolve_plan(plan)

    def test_structured_trainer_checkpoint_resume_and_consumers(self):
        import simulator.training.research as research

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = json.loads((TEST_SETUPS / "environment.json").read_text())
            environment["arena"] = str(TEST_SETUPS / "arena.json")
            environment.pop("spawns")
            environment.pop("equipment")
            environment["task"] = {"max_ticks": 3, "rewards": {}}
            environment["bridge"] = {"mode": {"name": "bridge", "version": 1},
                "goals": [{"min": [-6, 1, -1], "max": [-5, 3, 1]},
                          {"min": [5, 1, -1], "max": [6, 3, 1]}],
                "void_y": 0, "match": {"goals_to_win": 2, "max_ticks": 3, "cage_ticks": 0,
                    "persist_blocks_on_goal": True, "arrow_refill_ticks": 0},
                "task_rewards": {"score": 2, "win": 5, "loss": -3,
                                 "opponent_score": -1, "toward": 0.25, "away": 0}}
            environment_path = root / "environment.json"
            environment_path.write_text(json.dumps(environment))
            plan = {"format": "minecraft_sim_training_plan", "version": 2, "name": "structured-test", "seed": 42,
                "policy": str(TEST_SETUPS / "policy.json"), "environment": str(environment_path),
                "bridge_observation": bridge_observation_descriptor(2),
                "learner": {"hidden_dim": 8, "epochs": 1, "minibatch_size": 2},
                "opponents": {"target": {"type": "scripted", "controller": "stationary_target"}},
                "mixtures": {"target": [{"id": "target", "weight": 1}]},
                "stages": [{"name": "bridge", "opponents": "target", "advance": {"updates": 5}}],
                "evaluations": {"held": {"stage": "bridge", "every_updates": 1, "episodes": 2, "seed": 9}},
                "snapshots": [{"name": "initial", "at_updates": 0}]}
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan))
            output = root / "run"
            common = ["--device", "cpu", "--threads", "1", "--library", self.library,
                      "--updates", "1", "--replay-every-updates", "1"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--plan", str(plan_path), "--output", str(output), "--envs", "2",
                                       "--rollout-steps", "2", *common]), 0)
            first = torch.load(output / "latest.pt", weights_only=True)
            snapshot = torch.load(output / "snapshots/initial.pt", weights_only=True)
            self.assertEqual(first["metadata"]["bridge_observation"], plan["bridge_observation"])
            self.assertEqual(first["metadata"]["reward_semantics"], "bridge_score_opponent_objective_progress_v2")
            self.assertEqual(snapshot["metadata"]["bridge_observation"], plan["bridge_observation"])
            self.assertFalse(torch.equal(first["model"]["voxel_encoder.0.weight"],
                                         snapshot["model"]["voxel_encoder.0.weight"]))
            metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            self.assertTrue(all(np.isfinite(metrics[0][name]) for name in ("loss", "policy_loss", "value_loss", "entropy")))
            self.assertEqual(metrics[0]["evaluations"]["held"]["episodes"], 2)
            replay = next((output / "replays").glob("*.jsonl"))
            header = json.loads(replay.read_text().splitlines()[0])
            self.assertEqual(header["training"]["bridge_observation"], plan["bridge_observation"])
            self.assertEqual(header["version"], 4)
            restored = {}
            original_collect = research.collect
            original_optimize = research.optimize
            original_load_pool = OpponentPool.load_state_dict

            def observed_collect(model, worlds, pool, *args, **kwargs):
                restored["model"] = deepcopy(model.state_dict())
                restored["rng"] = torch.get_rng_state().clone()
                restored["episodes"] = worlds.bridge_reset_episodes()
                return original_collect(model, worlds, pool, *args, **kwargs)

            def observed_optimize(model, optimizer, *args, **kwargs):
                restored["optimizer"] = deepcopy(optimizer.state_dict())
                return original_optimize(model, optimizer, *args, **kwargs)

            def observed_load_pool(pool, state):
                original_load_pool(pool, state)
                restored["pool"] = deepcopy(pool.state_dict())

            with patch.object(research, "collect", side_effect=observed_collect), \
                 patch.object(research, "optimize", side_effect=observed_optimize), \
                 patch.object(OpponentPool, "load_state_dict", new=observed_load_pool), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--resume", str(output / "latest.pt"), *common]), 0)
            for name, value in first["model"].items():
                torch.testing.assert_close(restored["model"][name], value)
            torch.testing.assert_close(restored["rng"], first["rng_state"])
            self.assertEqual(restored["episodes"], [value + 1 for value in first["research"]["bridge_episodes"]])
            self.assertEqual(restored["optimizer"]["param_groups"], first["optimizer"]["param_groups"])
            for key, state in first["optimizer"]["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(restored["optimizer"]["state"][key][name], value)
            self.assertEqual(restored["pool"], first["research"]["pool"])
            second = torch.load(output / "latest.pt", weights_only=True)
            self.assertEqual(second["update"], 2)
            self.assertEqual(second["research"]["resume_mode"], "fresh_episodes")
            resolved = second["research"]["plan"]
            reloaded = BridgeActorCritic(resolved["policy"]["observation_dim"],
                second["metadata"]["bridge_observation"], second["metadata"]["hidden_dim"],
                dict(zip(resolved["policy"]["action_names"], resolved["policy"]["action_sizes"])))
            reloaded.load_state_dict(second["model"], strict=True)
            consumer_options = SimpleNamespace(device="cpu", library=self.library, max_ticks=None)
            result = evaluate(reloaded, resolved, resolved["evaluations"]["held"], consumer_options,
                              torch.device("cpu"), {})
            self.assertEqual(result["episodes"], 2)
            path = export_replay(reloaded, resolved, resolved["stages"][0], consumer_options,
                header["environment"], header["training"]["opponent"], torch.device("cpu"), 2, 2,
                root / "reloaded-replay")
            self.assertEqual(json.loads(Path(path).read_text().splitlines()[0])["training"]["bridge_observation"],
                             plan["bridge_observation"])
            with self.assertRaisesRegex(ValueError, "Structured Bridge checkpoints"):
                Policy(output / "latest.pt")
            with self.assertRaisesRegex(ValueError, "Structured Bridge checkpoints"):
                CheckpointController(output / "latest.pt", resolved["policy"], 1)
            checkpoint = root / "fixed-baseline.pt"
            shutil.copyfile(output / "latest.pt", checkpoint)
            checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            fixed_plan = deepcopy(plan)
            fixed_plan["opponents"] = {"baseline": {"type": "checkpoint", "path": str(checkpoint),
                                                     "deterministic": True}}
            fixed_plan["mixtures"] = {"baseline": [{"id": "baseline", "weight": 1}]}
            fixed_plan["stages"][0]["opponents"] = "baseline"
            fixed_path = root / "fixed-plan.json"
            fixed_path.write_text(json.dumps(fixed_plan))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--plan", str(fixed_path), "--output", str(root / "fixed-run"),
                                       "--envs", "2", "--rollout-steps", "2", *common]), 0)
            fixed_metrics = json.loads((root / "fixed-run/metrics.jsonl").read_text().splitlines()[0])
            self.assertEqual(fixed_metrics["evaluations"]["held"]["episodes"], 2)
            fixed_replay = next((root / "fixed-run/replays").glob("*.jsonl"))
            fixed_header = json.loads(fixed_replay.read_text().splitlines()[0])
            self.assertEqual(fixed_header["training"]["opponent"]["controller"]["sha256"], checkpoint_hash)
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), checkpoint_hash)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--resume", str(root / "fixed-run/latest.pt"), *common]), 0)
            checkpoint.write_bytes(b"replaced checkpoint")
            with self.assertRaisesRegex(ValueError, "checkpoint changed|SHA256|changed"), contextlib.redirect_stdout(io.StringIO()):
                main(["--resume", str(root / "fixed-run/latest.pt"), *common])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--plan", str(plan_path), "--initialize", str(output / "latest.pt"),
                                       "--output", str(root / "initialized"), "--envs", "2",
                                       "--rollout-steps", "2", *common]), 0)
            scalar_plan = deepcopy(plan)
            scalar_plan.pop("bridge_observation")
            scalar_path = root / "scalar.json"
            scalar_path.write_text(json.dumps(scalar_plan))
            scalar_output = root / "scalar-run"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--plan", str(scalar_path), "--output", str(scalar_output), "--envs", "2",
                                       "--rollout-steps", "2", *common]), 0)
            scalar_checkpoint = scalar_output / "latest.pt"
            self.assertNotIn("bridge_observation", torch.load(scalar_checkpoint, weights_only=True)["metadata"])
            for selected_plan, checkpoint_path in ((plan_path, scalar_checkpoint),
                                                    (scalar_path, output / "latest.pt")):
                for flag in ("--initialize", "--resume"):
                    with self.subTest(plan=selected_plan.name, flag=flag), \
                         contextlib.redirect_stdout(io.StringIO()), \
                         self.assertRaisesRegex(ValueError, "structured Bridge model descriptor"):
                        main(["--plan", str(selected_plan), flag, str(checkpoint_path),
                              "--output", str(root / "incompatible"), *common])
            for invalid in (None, bridge_observation_descriptor(), {**plan["bridge_observation"], "version": 3},
                            {**plan["bridge_observation"], "old_field": True}):
                broken = deepcopy(first)
                broken["metadata"]["bridge_observation"] = invalid
                broken_path = root / "broken.pt"
                torch.save(broken, broken_path)
                for flag in ("--initialize", "--resume"):
                    with self.subTest(invalid=invalid, flag=flag), \
                         contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "descriptor"):
                        main(["--plan", str(plan_path), flag, str(broken_path),
                              "--output", str(root / "broken-run"), *common])
            with self.assertRaisesRegex(ValueError, "resident rollout unavailable"):
                main(["--plan", str(plan_path), "--output", str(root / "resident"),
                      "--rollout-backend", "cuda-resident", *common])

    def test_structured_mixed_self_play_train_save_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = json.loads((TEST_SETUPS / "environment.json").read_text())
            environment["arena"] = str(TEST_SETUPS / "arena.json")
            environment.pop("spawns")
            environment.pop("equipment")
            environment["task"] = {"max_ticks": 3, "rewards": {}}
            environment["bridge"] = {"mode": {"name": "bridge", "version": 1},
                "goals": [{"min": [-6, 1, -1], "max": [-5, 3, 1]},
                          {"min": [5, 1, -1], "max": [6, 3, 1]}],
                "void_y": 0, "match": {"goals_to_win": 2, "max_ticks": 3, "cage_ticks": 0,
                    "persist_blocks_on_goal": True, "arrow_refill_ticks": 0},
                "task_rewards": {"score": 2, "win": 5, "loss": -3}}
            environment_path = root / "environment.json"
            environment_path.write_text(json.dumps(environment))
            plan = {"format": "minecraft_sim_training_plan", "version": 2, "name": "mixed", "seed": 42,
                "policy": str(TEST_SETUPS / "policy.json"), "environment": str(environment_path),
                "bridge_observation": bridge_observation_descriptor(),
                "learner": {"hidden_dim": 8, "epochs": 1, "minibatch_size": 2},
                "opponents": {"target": {"type": "scripted", "controller": "stationary_target"},
                              "current": {"type": "self_play"},
                              "past": {"type": "frozen_self_play", "every_updates": 1}},
                "mixtures": {"mixed": [{"id": name, "weight": 1} for name in ("target", "current", "past")],
                             "past": [{"id": "past", "weight": 1}]},
                "stages": [{"name": "bridge", "opponents": "mixed", "advance": {"updates": 3}}],
                "evaluations": {"held": {"stage": "bridge", "overrides": {"opponents": "past"}, "every_updates": 1,
                                         "episodes": 4, "worlds": 2, "seed": 9}},
                "snapshots": [{"name": "initial", "at_updates": 0}]}
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan))
            output = root / "run"
            common = ["--device", "cpu", "--threads", "1", "--library", self.library,
                      "--updates", "1", "--replay-every-updates", "1"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--plan", str(plan_path), "--output", str(output), "--envs", "4",
                                       "--rollout-steps", "2", *common]), 0)
            first = torch.load(output / "latest.pt", weights_only=True)
            initial = torch.load(output / "snapshots/initial.pt", weights_only=True)
            self.assertFalse(torch.equal(first["model"]["voxel_encoder.0.weight"],
                                         initial["model"]["voxel_encoder.0.weight"]))
            self.assertEqual(first["research"]["pool"]["controllers"]["past"]["latest_update"], 1)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--resume", str(output / "latest.pt"), *common]), 0)
            final = torch.load(output / "latest.pt", weights_only=True)
            self.assertEqual((final["update"], final["research"]["resume_mode"]), (2, "fresh_episodes"))
            self.assertEqual(final["research"]["pool"]["controllers"]["past"]["latest_update"], 2)
            metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([entry["evaluations"]["held"]["opponents"][0]["controller"]["snapshot_update"]
                              for entry in metrics], [1, 2])
            self.assertEqual([[seat["episodes"] for seat in entry["evaluations"]["held"]["seats"]]
                              for entry in metrics], [[2, 2], [2, 2]])
            self.assertEqual(len(list((output / "replays").glob("*.jsonl"))), 2)

    def test_scalar_contract_rejects_structured_bridge_descriptor(self):
        policy = {"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
                  "ticks_per_action": 1, "observations": [{"feature": "relative.z"}],
                  "actions": [{"name": "movement", "values": ["neutral"]}],
                  "bridge_observation": {"version": 1}}
        with self.assertRaisesRegex(ValueError, "Structured Bridge observations"):
            validate_contract(policy)
        with self.assertRaisesRegex(ValueError, "Structured Bridge observations"):
            from_legacy({**LEGACY_SCHEMA, "bridge_observation": bridge_observation_descriptor()})

    def test_bridge_observation_inputs_are_actor_relative(self):
        worlds = self.worlds()
        environment = worlds.groups[0][1]
        scalar = environment.observe()
        state = {**environment.bridge_state()[0], "scores": [1, 2], "ticks": 25, "cage_remaining": 3}
        kits = [{"blocks": 16, "golden_apples": 2, "arrows": 1,
                 "arrow_refill_elapsed": 4, "selected_slot": 1},
                {"blocks": 48, "golden_apples": 3, "arrows": 0,
                 "arrow_refill_elapsed": 2, "selected_slot": 4}]
        environment.bridge["match"].update(cage_ticks=6, arrow_refill_ticks=8)
        with patch.object(environment, "bridge_state", return_value=[state]), \
             patch.object(environment, "bridge_kits", side_effect=lambda player: [kits[player]]):
            inputs = bridge_observation_inputs(environment, scalar)
        self.assertEqual(inputs["voxels"].shape, (1, 2, 5, 9, 9))
        self.assertEqual(inputs["voxels"].dtype, np.uint8)
        np.testing.assert_array_equal(inputs["voxels"], environment.bridge_local_voxels())
        np.testing.assert_array_equal(inputs["scalar"], scalar)
        np.testing.assert_allclose(inputs["match"][0, 0],
                                   [0.5, 1, 0.75, 0.5, 0.25, 2/64, 1/64, 0.5, 1, 1/8])
        np.testing.assert_allclose(inputs["match"][0, 1],
                                   [1, 0.5, 0.75, 0.5, 0.75, 3/64, 0, 0.25, 1, 0.5])
        with self.assertRaisesRegex(ValueError, "shape or dtype"):
            bridge_observation_inputs(environment, scalar.astype(np.float64))

    def test_bridge_observation_reads_native_own_kits(self):
        for player in self.scenario["arenas"][0]["environment"]["players"]:
            player["loadout"] = {"weapons": ["minecraft:diamond_sword", "minecraft:air", "minecraft:air"],
                                 "selected": 0, "offhand": "minecraft:air"}
        self.scenario["bridge"]["kits"] = [
            {"slots": ["sword", "blocks", "golden_apple"] + ["empty"] * 6, "blocks": 16, "golden_apples": 2},
            {"slots": ["sword", "blocks", "golden_apple"] + ["empty"] * 6, "blocks": 48, "golden_apples": 3}]
        worlds = self.worlds()
        inputs = bridge_observation_inputs(worlds.groups[0][1], worlds.observe())
        np.testing.assert_allclose(inputs["match"][0, :, 4:6], [[16/64, 2/64], [48/64, 3/64]])
        np.testing.assert_array_equal(inputs["match"][0, :, 8], [0, 0])

    def test_v2_samples_task_state_for_both_actors(self):
        worlds = self.worlds()
        environment = worlds.groups[0][1]
        scalar = worlds.observe()
        actor_state = np.array([[[0, 1, 0, 0, 0, 16, 0.25, 2, 1], [0, 1, 0, 180, 0, 8, 0.75, 4, 5]]], dtype=np.float64)
        with patch.object(environment, "actor_task_state", return_value=actor_state):
            inputs = worlds.bridge_observations(scalar, bridge_observation_descriptor(2))
        self.assertEqual(inputs["match"].shape, (1, 2, 17))
        np.testing.assert_allclose(inputs["match"][0, :, 10:12], [[0.5, 0.25], [0.75, 0.75]])
        np.testing.assert_allclose(inputs["match"][0, :, 15:], [[0.5, 0.2], [1, 1]])
        self.assertTrue(np.isfinite(inputs["match"]).all())

    def test_v2_rollout_uses_task_state_for_both_seats(self):
        self.scenario["bridge"]["task_rewards"].update(opponent_score=-1, toward=0.25, away=0)
        worlds = self.worlds(count=2)
        pool = self.pool(2)
        pool.learner_slots[:] = [0, 1]
        model = BridgeActorCritic(1, bridge_observation_descriptor(2), 8, {"movement": 3})
        rollout, _, _, _ = collect(model, worlds, pool, worlds.observe(), 3, torch.device("cpu"),
                                    {"movement": 1}, structured_bridge=True)
        self.assertEqual(rollout["observations"]["match"].shape, (3, 2, 17))
        self.assertTrue(torch.isfinite(rollout["observations"]["match"]).all())
        self.assertGreater(rollout["bridge_events"]["bridge_reward_toward"], 0)
        self.assertEqual(rollout["bridge_events"]["bridge_reward_away"], 0)

    def test_objective_rewards_score_and_round_reset_for_both_sides(self):
        self.scenario["bridge"]["task_rewards"].update(opponent_score=-1, toward=0.25, away=0)
        for scorer in (0, 1):
            with self.subTest(scorer=scorer):
                self.scenario["arenas"][0]["environment"]["players"][1 - scorer]["position"][0] = -1.5
                self.scenario["arenas"][0]["environment"]["players"][scorer]["position"][0] = 0.5
                worlds = self.worlds(count=2)
                environment = worlds.groups[0][1]
                actions = np.zeros((2, 2, 1), dtype=np.int64)
                actions[0, scorer, 0] = 1 if scorer == 0 else 2
                goals = 0
                for _ in range(80):
                    _, rewards, terminated, truncated = worlds.step(actions)
                    state = environment.bridge_state()[0]
                    terms = environment.bridge_reward_terms()
                    np.testing.assert_array_equal(terms[1], np.zeros((2, 4)))
                    np.testing.assert_array_equal(terms[:, :, 3], np.zeros((2, 2)))
                    if state["scored"][scorer]:
                        goals += 1
                        self.assertEqual(terms[0, scorer, 0], 2)
                        self.assertEqual(terms[0, 1 - scorer, 1], -1)
                        if state["round_reset"]:
                            np.testing.assert_array_equal(terms[0, :, 2:], np.zeros((2, 2)))
                        self.assertEqual(rewards[0, 1 - scorer], -1 if goals == 1 else -4)
                    if terminated[0] or truncated[0]:
                        break
                self.assertEqual(goals, 2)

    def test_bridge_observation_reads_native_score_perspectives(self):
        worlds = self.worlds()
        environment = worlds.groups[0][1]
        actions = np.zeros((1, 2, 1), dtype=np.int64)
        actions[0, 0, 0] = 1
        for _ in range(50):
            scalar, _, terminated, truncated = worlds.step(actions)
            if environment.bridge_state()[0]["scores"][0]:
                break
            self.assertFalse(terminated[0] or truncated[0])
        else:
            self.fail("Player 0 did not score")
        inputs = bridge_observation_inputs(environment, scalar)
        np.testing.assert_allclose(inputs["match"][0, :, :2], [[0.5, 0], [0, 0.5]])

    def test_bridge_model_is_separate_and_strict(self):
        descriptor = bridge_observation_descriptor()
        legacy = self.model()
        keys = set(legacy.state_dict())
        self.assertEqual(keys, {"shared.0.weight", "shared.0.bias", "shared.2.weight", "shared.2.bias",
                    "movement_head.weight", "movement_head.bias", "value_head.weight", "value_head.bias"})
        model = BridgeActorCritic(1, descriptor, 8, {"movement": 3})
        inputs = {"scalar": torch.zeros(2, 1), "match": torch.zeros(2, 10),
                  "voxels": torch.zeros(2, 5, 9, 9, dtype=torch.uint8)}
        logits, values = model(inputs)
        self.assertEqual(logits["movement"].shape, (2, 3))
        self.assertEqual(values.shape, (2,))
        (logits["movement"].sum() + values.sum()).backward()
        self.assertIsNotNone(model.voxel_encoder[0].weight.grad)
        self.assertEqual(keys, set(self.model().state_dict()))
        buffer = io.BytesIO()
        torch.save({"descriptor": descriptor, "model": model.state_dict()}, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        restored = BridgeActorCritic(1, saved["descriptor"], 8, {"movement": 3})
        restored.load_state_dict(saved["model"], strict=True)
        restored_logits, restored_values = restored(inputs)
        torch.testing.assert_close(restored_logits["movement"], logits["movement"])
        torch.testing.assert_close(restored_values, values)
        with self.assertRaises(RuntimeError):
            model.load_state_dict(legacy.state_dict(), strict=True)
        with self.assertRaisesRegex(ValueError, "descriptor"):
            BridgeActorCritic(1, {**descriptor, "version": 2}, 8, {"movement": 3})
        with self.assertRaisesRegex(ValueError, "descriptor"):
            BridgeActorCritic(1, {**descriptor, "version": True}, 8, {"movement": 3})
        with self.assertRaisesRegex(ValueError, "observation tensors"):
            model({**inputs, "voxels": inputs["voxels"].float()})

    def test_bridge_observation_samples_final_state_before_reset(self):
        for terminal in (False, True):
            with self.subTest(terminal=terminal):
                self.scenario["bridge"]["match"]["max_ticks"] = 1 if terminal else 100
                worlds = self.worlds(max_ticks=100 if terminal else 1)
                environment = worlds.groups[0][1]
                initial = bridge_observation_inputs(environment, worlds.observe())
                final_scalar, _, terminated, truncated = worlds.step(np.zeros((1, 2, 1), dtype=np.int64))
                self.assertEqual((bool(terminated[0]), bool(truncated[0])), (terminal, not terminal))
                final = bridge_observation_inputs(environment, final_scalar)
                np.testing.assert_allclose(final["match"][0, :, 2], 0 if terminal else 0.99)
                reset = bridge_observation_inputs(environment, worlds.reset(np.array([True])))
                np.testing.assert_allclose(initial["match"][0, :, 2], 1)
                np.testing.assert_allclose(reset["match"][0, :, 2], 1)
                np.testing.assert_allclose(final["match"][0, :, 2], 0 if terminal else 0.99)

    def test_bridge_ppo_optimizer_updates_voxel_encoder(self):
        torch.manual_seed(42)
        model = BridgeActorCritic(1, bridge_observation_descriptor(), 8, {"movement": 3})
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scalar = torch.arange(4, dtype=torch.float32).reshape(2, 2, 1)
        match = torch.arange(40, dtype=torch.float32).reshape(2, 2, 10) / 40
        voxels = (torch.arange(4 * 5 * 9 * 9).reshape(2, 2, 5, 9, 9) % 5).to(torch.uint8)
        observations = {"scalar": scalar, "match": match, "voxels": voxels}
        flat = {name: value.flatten(0, 1) for name, value in observations.items()}
        with torch.no_grad():
            actions, log_probs, values, _ = policy_actions(model, flat, model.action_names)
        rollout = {"observations": observations, "actions": actions.reshape(2, 2, 1),
                   "log_probs": log_probs.reshape(2, 2), "values": values.reshape(2, 2),
                   "advantages": torch.tensor([[1., -1.], [0.5, -0.5]]),
                   "returns": values.reshape(2, 2) + 1}
        original = model.voxel_encoder[0].weight.detach().clone()
        metrics = optimize(model, optimizer, rollout, epochs=2, minibatch_size=2)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertFalse(torch.equal(original, model.voxel_encoder[0].weight))

    def test_structured_bridge_collects_final_before_reset_for_both_seats(self):
        for terminal in (False, True):
            with self.subTest(terminal=terminal):
                torch.manual_seed(42)
                self.scenario["bridge"]["match"]["max_ticks"] = 1 if terminal else 100
                worlds = self.worlds(count=2, max_ticks=100 if terminal else 1)
                pool = self.pool(2)
                pool.learner_slots[:] = [0, 1]
                model = BridgeActorCritic(1, bridge_observation_descriptor(), 8, {"movement": 3})
                scalar = worlds.observe()
                initial = worlds.bridge_observations(scalar)
                seen = []

                def capture(module, inputs, outputs):
                    seen.append(({name: value.detach().clone() for name, value in inputs[0].items()},
                                 outputs[1].detach().clone()))

                handle = model.register_forward_hook(capture)
                try:
                    rollout, reset_scalar, endings, timeouts = collect(
                        model, worlds, pool, scalar, 1, torch.device("cpu"), {"movement": 0},
                        structured_bridge=True)
                finally:
                    handle.remove()
                self.assertEqual((endings, timeouts), (2, 0) if terminal else (0, 2))
                for name in ("scalar", "match", "voxels"):
                    np.testing.assert_array_equal(rollout["observations"][name][0].numpy(),
                                                  initial[name][[0, 1], [0, 1]])
                self.assertEqual(rollout["observations"]["voxels"].dtype, torch.uint8)
                self.assertGreater(torch.unique(rollout["observations"]["voxels"]).numel(), 1)
                np.testing.assert_allclose(seen[1][0]["match"][:, 2], 0 if terminal else 0.99)
                np.testing.assert_allclose(worlds.bridge_observations(reset_scalar)["match"][:, :, 2], 1)
                torch.testing.assert_close(rollout["next_values"][0], seen[1][1])
                expected = rollout["rewards"][0] - rollout["values"][0]
                if not terminal:
                    expected += 0.99 * seen[1][1]
                torch.testing.assert_close(rollout["advantages"][0], expected)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                metrics = optimize(model, optimizer, rollout, epochs=1, minibatch_size=2)
                self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_structured_bridge_collection_requires_explicit_scripted_opt_in(self):
        worlds = self.worlds()
        pool = self.pool(1)
        model = BridgeActorCritic(1, bridge_observation_descriptor(), 8, {"movement": 3})
        observations = worlds.observe()
        with self.assertRaisesRegex(ValueError, "explicit opt-in"):
            collect(model, worlds, pool, observations, 1, torch.device("cpu"), {})
        with self.assertRaisesRegex(ValueError, "current scalar"):
            collect(model, worlds, pool, observations + 1, 1, torch.device("cpu"), {}, structured_bridge=True)
        incompatible = BridgeActorCritic(1, bridge_observation_descriptor(), 8, {"movement": 2})
        with self.assertRaisesRegex(ValueError, "model and scalar policy are incompatible"):
            collect(incompatible, worlds, pool, observations, 1, torch.device("cpu"), {}, structured_bridge=True)
        model.descriptor["version"] = 2
        with self.assertRaisesRegex(ValueError, "descriptor"):
            collect(model, worlds, pool, observations, 1, torch.device("cpu"), {}, structured_bridge=True)
        model.descriptor = bridge_observation_descriptor()
        with self.assertRaisesRegex(ValueError, "shape or dtype"):
            worlds.bridge_observations(np.zeros((2, 2, 1), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            collect(model, worlds, pool, observations, 1, torch.device("cpu"), {}, structured_bridge=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "opponent.pt"
            scalar_model = self.model()
            torch.save({"format": "minecraft_sim_ppo", "version": 1,
                        "metadata": {"schema": self.contract, "hidden_dim": 8,
                                     "players_per_environment": 2, "learner": "opponent_pool"},
                        "model": scalar_model.state_dict()}, path)
            for definition in ({"type": "self_play"}, {"type": "frozen_self_play", "every_updates": 1},
                               {"type": "checkpoint", "path": str(path)}):
                with self.subTest(definition=definition):
                    unsupported = OpponentPool({"opponent": definition}, self.contract, 1, "cpu", 42,
                                               learner=scalar_model)
                    unsupported.configure([{"id": "opponent", "weight": 1}])
                    unsupported.reset(np.ones(1, dtype=bool))
                    with self.assertRaisesRegex(ValueError, "opponent model descriptor differs"):
                        collect(model, worlds, unsupported, observations, 1, torch.device("cpu"), {},
                                structured_bridge=True)
        pool.controllers["target"] = object()
        with self.assertRaisesRegex(ValueError, "supports scripted"):
            collect(model, worlds, pool, observations, 1, torch.device("cpu"), {}, structured_bridge=True)

    def test_structured_checkpoint_rejected_by_scalar_consumers(self):
        saved = {"format": "minecraft_sim_ppo", "version": 1,
                 "metadata": {"schema": self.contract, "contract": self.contract,
                              "players_per_environment": 2, "learner": "opponent_pool",
                              "hidden_dim": 8, "bridge_observation": bridge_observation_descriptor()},
                 "model": self.model().state_dict()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "structured.pt"
            torch.save(saved, path)
            with self.assertRaisesRegex(ValueError, "Structured Bridge checkpoints"):
                Policy(path)
            with self.assertRaisesRegex(ValueError, "Structured Bridge checkpoints"):
                CheckpointController(path, self.contract, 1)

    def setUp(self):
        torch.set_num_threads(1)
        scene = json.loads((ROOT / "simulator/tests/reset_environment.json").read_text(encoding="utf-8"))
        scene["max_ticks"] = 100
        self.contract = validate_contract({
            "format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3", "ticks_per_action": 1,
            "observations": [{"feature": "relative.z"}],
            "actions": [{"name": "movement", "values": ["neutral", "forward", "backward"]}]})
        self.scenario = {"arenas": [{"environment": scene, "weight": 1}], "bridge": {
            "mode": {"name": "bridge", "version": 1},
            "goals": [{"min": [0, 1, 2.4], "max": [1, 3, 2.9]},
                      {"min": [0, 1, -1.1], "max": [1, 3, -0.8]}],
            "void_y": -2, "match": {"goals_to_win": 2, "max_ticks": 100, "cage_ticks": 0,
                                     "persist_blocks_on_goal": True, "arrow_refill_ticks": 0},
            "task_rewards": {"score": 2, "win": 5, "loss": -3}}}
        self.stage = {"name": "bridge", "scenario": self.scenario,
                      "opponents": [{"id": "target", "weight": 1}], "controls": {"fixed": {}}}
        self.definitions = {"target": {"type": "scripted", "controller": "stationary_target"}}
        self.library = str(load_library(os.environ.get("MCSIM_TEST_CPU_LIBRARY"))._name)

    def worlds(self, count=1, max_ticks=100):
        worlds = SampledWorlds(self.scenario, self.contract, count, "cpu", self.library, max_ticks, 42)
        self.addCleanup(worlds.close)
        worlds.configure(self.stage)
        return worlds

    def pool(self, count):
        pool = OpponentPool(self.definitions, self.contract, count, "cpu", 42)
        pool.configure(self.stage["opponents"], self.stage["controls"])
        pool.reset(np.ones(count, dtype=bool))
        return pool

    def model(self):
        model = ActorCritic(self.contract["observation_dim"], 8,
                            action_sizes=dict(zip(self.contract["action_names"], self.contract["action_sizes"])))
        with torch.no_grad():
            model.value_head.weight.zero_()
            model.value_head.bias.fill_(3)
        return model

    def replay_inputs(self, slot=0, max_ticks=100):
        self.stage["controls"] = {"fixed": {"movement": "forward" if slot == 0 else "backward"}}
        worlds = self.worlds(max_ticks=max_ticks)
        pool = self.pool(1)
        pool.learner_slots[:] = slot
        plan = {"name": "fixture", "seed": 42, "policy": self.contract, "opponents": self.definitions,
                "stages": [self.stage]}
        options = SimpleNamespace(device="cpu", library=self.library, max_ticks=max_ticks)
        return plan, options, worlds.episode_scenario(0), pool.provenance(0)

    def dashboard_replay(self, directory):
        from simulator.training.dashboard import RunStore

        root = Path(directory)
        (root / "config.json").write_text("{}", encoding="utf-8")
        store = RunStore(runs=[root], discover=False)
        identity = store.list_runs()[0]["id"]
        recordings = store.list_replays(identity)
        self.assertEqual(len(recordings), 1)
        return store.read_replay(identity, recordings[0]["id"])

    def test_bridge_replay_complete_match_and_provenance(self):
        for slot in (0, 1):
            with self.subTest(slot=slot), tempfile.TemporaryDirectory() as directory:
                self.scenario["bridge"]["match"]["persist_blocks_on_goal"] = bool(slot)
                plan, options, scenario, provenance = self.replay_inputs(slot)
                original = deepcopy(scenario)
                model = self.model()
                path = Path(export_replay(model, plan, self.stage, options, scenario, provenance,
                                          torch.device("cpu"), 17, 4, Path(directory) / "replays"))
                header, *frames = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual((header["format"], header["version"], header["players"], header["ticks_per_second"]),
                                 ("minecraft_sim_replay", 4, 2, 20))
                self.assertEqual(header["bridge"], self.scenario["bridge"])
                self.assertNotIn("experimental_bridge", header["environment"])
                self.assertEqual(header["environment"]["players"], original["players"])
                self.assertEqual(header["environment"]["max_ticks"], 100)
                self.assertEqual(header["policy"], self.contract)
                self.assertEqual(header["training"]["opponent"], provenance)
                self.assertEqual((header["training"]["completed_rounds"], header["training"]["policy_update"]), (17, 4))
                self.assertEqual([frame["tick"] for frame in frames], list(range(len(frames))))
                self.assertEqual([frame["bridge"]["ticks"] for frame in frames], list(range(len(frames))))
                self.assertEqual(frames[0]["bridge"]["terrain_edits"], [])
                self.assertFalse(frames[0]["bridge"]["clear_overlay"])
                self.assertEqual(sum(frame["bridge"]["scored"][slot] for frame in frames), 2)
                self.assertTrue(any(frame["bridge"]["round_reset"] for frame in frames))
                self.assertEqual([frame["bridge"]["clear_overlay"] for frame in frames],
                                 [frame["bridge"]["round_reset"] and not bool(slot) for frame in frames])
                self.assertEqual(frames[-1]["bridge"]["outcome"], f"team{slot}")
                self.assertTrue(frames[-1]["terminated"] and frames[-1]["bridge"]["terminated"])
                self.assertFalse(frames[-1]["truncated"] or frames[-1]["bridge"]["truncated"])
                self.assertTrue(all(not frame.get("terminated", False) and not frame.get("truncated", False)
                                    for frame in frames[:-1]))
                self.assertEqual(scenario, original)
                self.assertTrue(model.training)
                self.assertEqual(list(path.parent.glob("*.part")), [])
                self.assertEqual(len(self.dashboard_replay(directory)["frames"]), len(frames))

    def test_bridge_replay_external_cutoff_and_match_clock_precedence(self):
        for match_limit, expected_terminal in ((100, False), (1, True)):
            with self.subTest(match_limit=match_limit), tempfile.TemporaryDirectory() as directory:
                self.scenario["bridge"]["match"]["max_ticks"] = match_limit
                plan, options, scenario, provenance = self.replay_inputs(max_ticks=1)
                path = export_replay(self.model(), plan, self.stage, options, scenario, provenance,
                                     torch.device("cpu"), 1, 1, directory)
                header, initial, final = [json.loads(line) for line in Path(path).read_text().splitlines()]
                self.assertEqual(header["environment"]["max_ticks"], 1)
                self.assertEqual((initial["tick"], final["tick"]), (0, 1))
                self.assertEqual((final["terminated"], final["truncated"]), (expected_terminal, not expected_terminal))
                self.assertEqual((final["bridge"]["terminated"], final["bridge"]["truncated"]),
                                 (expected_terminal, not expected_terminal))

    def test_bridge_replay_failure_cleanup_and_unsupported_tick_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, options, scenario, provenance = self.replay_inputs()
            model = self.model()
            with patch.object(Environment, "step", side_effect=RuntimeError("step failed")):
                with self.assertRaisesRegex(RuntimeError, "step failed"):
                    export_replay(model, plan, self.stage, options, scenario, provenance,
                                  torch.device("cpu"), 1, 1, directory)
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertTrue(model.training)
            plan["policy"] = {**plan["policy"], "ticks_per_action": 2}
            with self.assertRaisesRegex(ValueError, "ticks_per_action 1"):
                export_replay(model, plan, self.stage, options, scenario, provenance,
                              torch.device("cpu"), 1, 1, directory)
            self.assertEqual(list(Path(directory).iterdir()), [])
            plan["policy"] = self.contract
            options.max_ticks = 100001
            with self.assertRaisesRegex(ValueError, "max_ticks <= 100000"):
                export_replay(model, plan, self.stage, options, scenario, provenance,
                              torch.device("cpu"), 1, 1, directory)
            self.assertEqual(list(Path(directory).iterdir()), [])
            scenario["experimental_bridge"]["void_y"] = -3
            with self.assertRaisesRegex(ValueError, "differs from the stage"):
                export_replay(model, plan, self.stage, options, scenario, provenance,
                              torch.device("cpu"), 1, 1, directory)

    def test_bridge_replay_size_limit_removes_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, options, scenario, provenance = self.replay_inputs()
            model = self.model()
            dumps = json.dumps

            def oversized_frame(value, *args, **kwargs):
                if isinstance(value, dict) and "actors" in value:
                    return '"' + 'x' * (64 * 1024 * 1024) + '"'
                return dumps(value, *args, **kwargs)

            with patch("simulator.training.research.json.dumps", side_effect=oversized_frame):
                with self.assertRaisesRegex(RuntimeError, "64 MiB"):
                    export_replay(model, plan, self.stage, options, scenario, provenance,
                                  torch.device("cpu"), 1, 1, directory)
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertTrue(model.training)

    def test_bridge_replay_preserves_native_terrain_edits(self):
        scene = self.scenario["arenas"][0]["environment"]
        scene["blocks"].append({"from": [0, 2, 1], "to": [0, 2, 1], "block": "minecraft:stone"})
        scene["players"][1]["position"][0] = -2.5
        for player in scene["players"]:
            player["velocity"][1] = -0.0784000015258789
            player["loadout"] = {"weapons": ["minecraft:diamond_sword", "minecraft:air", "minecraft:air"],
                                 "selected": 1, "arrows": 0, "offhand": "minecraft:air"}
        self.contract = validate_contract({
            "format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3", "ticks_per_action": 1,
            "observations": [{"feature": "arena.ray_world_0"}],
            "actions": [{"name": "weapon", "values": [0, 1, 2]},
                        {"name": "use_item", "values": [False, True]},
                        {"name": "attack", "values": [False, True]}]})
        kit = {"slots": ["sword", "blocks", "pickaxe"] + ["empty"] * 6, "blocks": 4, "golden_apples": 0}
        self.scenario["bridge"].update(kits=[kit, kit], building={
            "zones": [{"min": [-3, 0, -3], "max": [4, 4, 4]}], "block_range": 4.5,
            "entity_range": 3.0, "hardness": 1.0, "mining_speed": 15.0, "correct_tool": True})
        with tempfile.TemporaryDirectory() as directory:
            worlds = self.worlds(max_ticks=2)
            pool = self.pool(1)
            pool.learner_slots[:] = 0
            self.stage["controls"] = {"fixed": {"weapon": 1, "use_item": True, "attack": False}}
            plan = {"name": "fixture", "seed": 42, "policy": self.contract, "opponents": self.definitions,
                    "stages": [self.stage]}
            options = SimpleNamespace(device="cpu", library=self.library, max_ticks=2)
            path = export_replay(self.model(), plan, self.stage, options, worlds.episode_scenario(0),
                                 pool.provenance(0), torch.device("cpu"), 1, 1, Path(directory) / "replays")
            _, initial, first, final = [json.loads(line) for line in Path(path).read_text().splitlines()]
            self.assertEqual(initial["bridge"]["terrain_edits"], [])
            self.assertTrue(first["bridge"]["placed"][0])
            self.assertEqual(first["bridge"]["terrain_edits"],
                             [{"player": 0, "action": "place", "position": [0, 2, 0]}])
            self.assertEqual(final["bridge"]["terrain_edits"], [])
            self.assertTrue(final["truncated"])
            self.assertEqual(self.dashboard_replay(directory)["frames"][1]["bridge"]["terrain_edits"],
                             first["bridge"]["terrain_edits"])

    def test_duel_replay_remains_version_three(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, options, scenario, provenance = self.replay_inputs(max_ticks=2)
            self.stage["scenario"] = {name: value for name, value in self.scenario.items() if name != "bridge"}
            scenario.pop("experimental_bridge")
            path = export_replay(self.model(), plan, self.stage, options, scenario, provenance,
                                 torch.device("cpu"), 1, 1, directory)
            header, *frames = [json.loads(line) for line in Path(path).read_text().splitlines()]
            self.assertEqual(header["version"], 3)
            self.assertNotIn("bridge", header)
            self.assertTrue(all("bridge" not in frame for frame in frames))
            self.assertTrue(frames[-1]["truncated"])

    def test_bridge_periodic_and_dashboard_requests_export(self):
        from simulator.training.replay_requests import create_request, read_request
        from simulator.training.research import collect as collect_training

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = self.scenario["arenas"][0]["environment"]
            arena = {"format": "minecraft_sim_arena", "version": 1, "minecraft": "26.3", "name": "fixture",
                     "rules": {"simulate_food": False, "natural_regeneration": False},
                     "bounds": scene["bounds"], "blocks": scene["blocks"]}
            bridge = deepcopy(self.scenario["bridge"])
            bridge["match"]["max_ticks"] = 2
            environment = {"format": "minecraft_sim_environment", "version": 2, "minecraft": "26.3",
                           "name": "fixture", "arena": "arena.json", "players": {"overrides": scene["players"]},
                           "task": {"max_ticks": 3, "rewards": {}}, "bridge": bridge}
            plan = {"format": "minecraft_sim_training_plan", "version": 2, "name": "fixture", "seed": 42,
                    "policy": "policy.json", "environment": "environment.json", "opponents": self.definitions,
                    "mixtures": {"target": [{"id": "target", "weight": 1}]},
                    "stages": [{"name": "bridge", "opponents": "target", "advance": {"updates": 1}}]}
            for name, value in (("arena", arena), ("environment", environment),
                                ("policy", self.contract), ("plan", plan)):
                (root / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
            for mode in ("updates", "episodes", "dashboard"):
                with self.subTest(mode=mode):
                    output = root / mode
                    flags = ["--plan", str(root / "plan.json"), "--library", self.library, "--envs", "1",
                             "--hidden-dim", "8", "--rollout-steps", "2", "--epochs", "1", "--minibatch-size", "2",
                             "--updates", "1", "--threads", "1", "--output", str(output)]

                    def collect_then_request(*args, **kwargs):
                        result = collect_training(*args, **kwargs)
                        create_request(output)
                        return result

                    with contextlib.ExitStack() as stack:
                        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                        if mode == "dashboard":
                            stack.enter_context(patch("simulator.training.research.collect", side_effect=collect_then_request))
                        else:
                            flags.extend(["--replay-every-updates" if mode == "updates" else "--replay-every", "1"])
                        self.assertEqual(main(flags), 0)
                    recordings = list((output / "replays").glob("*.jsonl"))
                    self.assertEqual(len(recordings), 1)
                    header, *frames = [json.loads(line) for line in recordings[0].read_text().splitlines()]
                    self.assertEqual(header["version"], 4)
                    self.assertEqual(header["bridge"], bridge)
                    self.assertTrue(frames[-1]["terminated"])
                    self.assertTrue(json.loads((output / "config.json").read_text())["options"]["dashboard_replay_requests"])
                    if mode == "dashboard":
                        request = read_request(output)
                        self.assertEqual(request["status"], "completed")
                        self.assertEqual(request["replay"], recordings[0].name)

    def test_scoring_outcomes_rewards_and_selective_reset(self):
        original = deepcopy(self.scenario)
        worlds = self.worlds(3)
        self.assertEqual(self.scenario, original)
        actions = np.zeros((3, 2, 1), dtype=np.int64)
        actions[0, 0, 0] = 1
        actions[1, 1, 0] = 2
        rewards_sum = np.zeros((3, 2))
        outcomes = {}
        slots = np.zeros(3, dtype=int)
        for _ in range(80):
            _, rewards, terminated, truncated = worlds.step(actions)
            rewards_sum += rewards
            done = terminated | truncated
            selected = np.flatnonzero(done)
            names = worlds.outcomes(done, terminated, slots) if len(selected) else []
            worlds.episode_metrics.step(rewards[:, 0], done, names, ["target"] * len(selected))
            for world, outcome in zip(selected, names, strict=True):
                outcomes[int(world)] = outcome
            if done.any():
                environment = worlds.groups[0][1]
                idle_before = environment.bridge_state()[2]
                worlds.reset(done)
                self.assertEqual(environment.bridge_state()[2], idle_before)
                actions[done] = 0
            if 0 in outcomes and 1 in outcomes:
                break
        self.assertEqual(outcomes, {0: "win", 1: "loss"})
        np.testing.assert_array_equal(rewards_sum[:2], [[9, -3], [-3, 9]])
        metrics = worlds.episode_metrics.snapshot()
        self.assertEqual(metrics["training_episodes"], 2)
        self.assertEqual(metrics["training_win_rate"], 0.5)
        self.assertEqual(metrics["training_loss_rate"], 0.5)
        self.assertEqual(metrics["training_return"], 3)
        self.assertIsNone(worlds.step_reward_totals(slots))
        changed = deepcopy(self.stage)
        changed["scenario"]["bridge"]["task_rewards"]["win"] = 10
        with self.assertRaisesRegex(ValueError, "new environment batch"):
            worlds.configure(changed)
        changed = deepcopy(self.stage)
        changed["rewards"] = {"weights": {"good_aim": 1}}
        with self.assertRaisesRegex(ValueError, "duel reward"):
            worlds.configure(changed)
        manifest = research_metadata(worlds.groups[0][1], 8, {})
        self.assertEqual(manifest["reward_semantics"], "bridge_score_win_loss_v1")
        self.assertNotIn("reward_weights", manifest)
        self.assertEqual(manifest["scenario"]["experimental_bridge"], original["bridge"])
        self.assertEqual(self.scenario, original)

    def test_stationary_opponent_collection_records_bridge_return(self):
        worlds = self.worlds()
        pool = self.pool(1)
        pool.learner_slots[:] = 0
        with patch.object(pool.controllers["target"], "act", wraps=pool.controllers["target"].act) as stationary:
            rollout, _, _, _ = collect(self.model(), worlds, pool, worlds.observe(), 30,
                                       torch.device("cpu"), {"movement": 1})
        self.assertTrue(stationary.called)
        self.assertNotIn("reward_types", rollout)
        self.assertGreaterEqual(rollout["episode_metrics"]["training_episodes"], 1)
        self.assertEqual(rollout["episode_metrics"]["training_win_rate"], 1)
        self.assertEqual(rollout["episode_metrics"]["training_return"], 9)
        self.assertGreaterEqual(rollout["bridge_events"]["bridge_goals_for"], 2)
        self.assertEqual(rollout["bridge_events"]["bridge_goals_against"], 0)
        self.assertGreaterEqual(rollout["bridge_events"]["bridge_wins"], 1)
        self.assertEqual(rollout["bridge_events"]["bridge_losses"], 0)

    def test_final_observation_bootstraps_cutoff_but_not_match_end(self):
        for terminal in (False, True):
            with self.subTest(terminal=terminal):
                self.scenario["bridge"]["match"]["max_ticks"] = 1 if terminal else 100
                worlds = self.worlds(max_ticks=100 if terminal else 1)
                pool = self.pool(1)
                rollout, _, endings, timeouts = collect(self.model(), worlds, pool, worlds.observe(), 1,
                                                        torch.device("cpu"), {"movement": 0})
                self.assertEqual((endings, timeouts), (1, 0) if terminal else (0, 1))
                self.assertEqual(rollout["next_values"].item(), 3)
                self.assertAlmostEqual(rollout["advantages"].item(), -3 if terminal else -0.03, places=5)
                metrics = rollout["episode_metrics"]
                self.assertEqual(metrics["training_draw_rate" if terminal else "training_timeout_rate"], 1)

    def evaluation_inputs(self, episodes=4, worlds=3, max_ticks=100):
        plan = {"policy": self.contract, "opponents": self.definitions}
        suite = {**self.stage, "episodes": episodes, "worlds": worlds, "seed": 1729}
        options = SimpleNamespace(device="cpu", library=self.library, max_ticks=max_ticks)
        return plan, suite, options

    def test_bridge_evaluation_complete_matches_both_seats_and_isolated_seed(self):
        class ScoringPolicy(torch.nn.Module):
            def forward(self, observations):
                direction = observations[:, 0]
                logits = torch.stack((torch.full_like(direction, -1000),
                                      torch.where(direction > 0, 1000, -1000),
                                      torch.where(direction < 0, 1000, -1000)), dim=1)
                return {"movement": logits}, torch.zeros_like(direction)

        final_states = []

        class ObservedWorlds(SampledWorlds):
            def step(self, actions):
                result = super().step(actions)
                self.final_observations = result[0].copy()
                self.finished_mask = result[2] | result[3]
                return result

            def bridge_states(self):
                states = super().bridge_states()
                np.testing.assert_array_equal(self.observe()[self.finished_mask],
                                              self.final_observations[self.finished_mask])
                final_states.extend(deepcopy(states[world]) for world in np.flatnonzero(self.finished_mask))
                return states

        plan, suite, options = self.evaluation_inputs(episodes=5, worlds=3)
        model = ScoringPolicy()
        training_pool = self.pool(3)
        before = deepcopy(training_pool.state_dict())
        assignments, slots = training_pool.assignments.copy(), training_pool.learner_slots.copy()
        rng = torch.random.get_rng_state().clone()
        with patch("simulator.training.research.SampledWorlds", ObservedWorlds), \
                patch("simulator.training.research._lost", side_effect=AssertionError("duel loss")), \
                patch.object(SampledWorlds, "step_behavior_events", side_effect=AssertionError("duel behavior")):
            first = evaluate(model, plan, suite, options, torch.device("cpu"), {}, training_pool)
            second = evaluate(model, plan, suite, options, torch.device("cpu"), {}, training_pool)
        self.assertEqual(first, second)
        self.assertEqual(training_pool.state_dict(), before)
        np.testing.assert_array_equal(training_pool.assignments, assignments)
        np.testing.assert_array_equal(training_pool.learner_slots, slots)
        torch.testing.assert_close(torch.random.get_rng_state(), rng)
        self.assertTrue(model.training)
        self.assertEqual((first["wins"], first["losses"], first["draws"], first["truncations"]), (5, 0, 0, 0))
        self.assertEqual((first["goals_for"], first["goals_against"]), (10, 0))
        self.assertEqual([seat["episodes"] for seat in first["seats"]], [3, 2])
        self.assertTrue(all(state["outcome"] in ("team0", "team1") and sum(state["scores"]) == 2
                            for state in final_states))
        self.assertNotIn("mace_smash_rate", first)

    def test_bridge_evaluation_draw_and_external_truncation(self):
        for match_limit, external_limit, key in ((1, 100, "draws"), (100, 1, "truncations"),
                                                  (1, 1, "draws")):
            with self.subTest(match_limit=match_limit, external_limit=external_limit):
                self.scenario["bridge"]["match"]["max_ticks"] = match_limit
                plan, suite, options = self.evaluation_inputs(episodes=6, worlds=2, max_ticks=external_limit)
                suite["controls"] = {"fixed": {"movement": "neutral"}}
                result = evaluate(self.model(), plan, suite, options, torch.device("cpu"), {})
                self.assertEqual(result[key], 6)
                self.assertEqual(sum(result[name] for name in ("wins", "losses", "draws", "truncations")), 6)
                self.assertEqual([seat["episodes"] for seat in result["seats"]], [3, 3])
                self.assertEqual((result["goals_for"], result["goals_against"]), (0, 0))

    def test_structured_bridge_scripted_evaluation_and_replay(self):
        torch.manual_seed(42)
        self.scenario["bridge"]["match"]["max_ticks"] = 2
        plan, suite, options = self.evaluation_inputs(episodes=4, worlds=2)
        plan["bridge_observation"] = bridge_observation_descriptor()
        suite["controls"] = {"fixed": {"movement": "neutral"}}
        model = BridgeActorCritic(1, bridge_observation_descriptor(), 8, {"movement": 3})
        worlds = self.worlds(count=2)
        pool = self.pool(2)
        pool.learner_slots[:] = [0, 1]
        rollout, _, _, _ = collect(model, worlds, pool, worlds.observe(), 2,
                       torch.device("cpu"), {"movement": 0}, structured_bridge=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        original = model.voxel_encoder[0].weight.detach().clone()
        metrics = optimize(model, optimizer, rollout, epochs=1, minibatch_size=2)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertFalse(torch.equal(original, model.voxel_encoder[0].weight))
        result = evaluate(model, plan, suite, options, torch.device("cpu"), {})
        self.assertEqual(result["draws"], 4)
        self.assertEqual([seat["episodes"] for seat in result["seats"]], [2, 2])
        with tempfile.TemporaryDirectory() as directory:
            replay_plan, replay_options, scenario, provenance = self.replay_inputs(slot=1)
            replay_plan["bridge_observation"] = bridge_observation_descriptor()
            path = export_replay(model, replay_plan, self.stage, replay_options, scenario, provenance,
                                 torch.device("cpu"), 0, 0, Path(directory) / "replays")
            header, *frames = [json.loads(line) for line in Path(path).read_text().splitlines()]
            self.assertEqual(header["version"], 4)
            self.assertEqual(header["training"]["bridge_observation"], bridge_observation_descriptor())
            self.assertEqual(header["training"]["opponent"]["learner_slot"], 1)
            self.assertTrue(frames[-1]["terminated"])
            self.assertEqual(len(self.dashboard_replay(directory)["frames"]), len(frames))
        plan["opponents"]["target"] = {"type": "self_play"}
        self.assertEqual(evaluate(model, plan, suite, options, torch.device("cpu"), {})["draws"], 4)

    def test_bridge_evaluation_truncation_retains_partial_score(self):
        self.scenario["bridge"]["match"]["goals_to_win"] = 5
        plan, suite, options = self.evaluation_inputs(episodes=1, worlds=1, max_ticks=5)
        suite["controls"] = {"fixed": {"movement": "forward"}}
        result = evaluate(self.model(), plan, suite, options, torch.device("cpu"), {})
        self.assertEqual((result["wins"], result["losses"], result["draws"], result["truncations"]), (0, 0, 0, 1))
        self.assertEqual((result["goals_for"], result["goals_against"]), (2, 0))
        self.assertEqual(result["reward_mean"], 4)

    def test_bridge_evaluation_stochastic_policy_uses_suite_seed(self):
        self.scenario["bridge"]["match"]["max_ticks"] = 6
        plan, suite, options = self.evaluation_inputs(episodes=6, worlds=3)
        model = self.model()
        traces = []

        def capture(*args, **kwargs):
            result = resident_policy_actions(*args, **kwargs)
            traces[-1].append(result[0].cpu().tolist())
            return result

        rng = torch.random.get_rng_state().clone()
        with patch("simulator.training.research.resident_policy_actions", side_effect=capture):
            for seed in (1729, 1729, 1730):
                suite["seed"] = seed
                traces.append([])
                evaluate(model, plan, suite, options, torch.device("cpu"), {})
        self.assertEqual(traces[0], traces[1])
        self.assertNotEqual(traces[0], traces[2])
        torch.testing.assert_close(torch.random.get_rng_state(), rng)

    def test_bridge_frozen_evaluation_uses_source_snapshot(self):
        self.scenario["bridge"]["match"]["max_ticks"] = 2
        self.definitions["past"] = {"type": "frozen_self_play", "every_updates": 1, "deterministic": True}
        plan, suite, options = self.evaluation_inputs(episodes=5, worlds=3)
        suite["opponents"] = [{"id": "past", "weight": 1}]
        model = self.model()
        source = OpponentPool(self.definitions, self.contract, 2, "cpu", 42, learner=model)
        source.configure(suite["opponents"], {"fixed": {"movement": "backward"}})
        source.capture(model, 3, {"fixed": {"movement": "backward"}}, force=True)
        frozen_weight = next(source.controllers["past"].snapshots[3][0].parameters()).detach().clone()
        source_rng = source.controllers["past"]._generator.get_state().clone()
        with torch.no_grad():
            next(model.parameters()).add_(1)
        seen = []
        original_act = OpponentPool.act

        def capture(pool, observations, actions):
            self.assertEqual(pool.controllers["past"].latest_update, 3)
            torch.testing.assert_close(next(pool.controllers["past"].snapshots[3][0].parameters()), frozen_weight)
            self.assertFalse(torch.equal(frozen_weight, next(model.parameters())))
            result = original_act(pool, observations, actions)
            seen.append(result[np.arange(pool.count), 1 - pool.learner_slots].copy())
            return result

        with self.assertRaisesRegex(ValueError, "requires a compatible source snapshot"):
            evaluate(model, plan, suite, options, torch.device("cpu"), {})
        with patch.object(OpponentPool, "act", capture):
            first = evaluate(model, plan, suite, options, torch.device("cpu"), {}, source)
            first_actions = deepcopy(seen)
            seen.clear()
            second = evaluate(model, plan, suite, options, torch.device("cpu"), {}, source)
        self.assertEqual(first, second)
        self.assertEqual([seat["episodes"] for seat in first["seats"]], [3, 2])
        self.assertTrue(all((actions == 2).all() for actions in first_actions))
        for left, right in zip(first_actions, seen, strict=True):
            np.testing.assert_array_equal(left, right)
        torch.testing.assert_close(source.controllers["past"]._generator.get_state(), source_rng)
        incompatible = OpponentPool({"past": {"type": "frozen_self_play", "every_updates": 2}},
                                    self.contract, 2, "cpu", 42, learner=model)
        with self.assertRaisesRegex(ValueError, "requires a compatible source snapshot"):
            evaluate(model, plan, suite, options, torch.device("cpu"), {}, incompatible)
        source.controllers["past"].snapshots.clear()
        with self.assertRaisesRegex(ValueError, "snapshot is unavailable"):
            evaluate(model, plan, suite, options, torch.device("cpu"), {}, source)

    def test_structured_frozen_evaluation_and_replay_use_pinned_snapshot(self):
        self.scenario["bridge"]["match"]["max_ticks"] = 2
        self.definitions["past"] = {"type": "frozen_self_play", "every_updates": 1}
        plan, suite, options = self.evaluation_inputs(episodes=5, worlds=3, max_ticks=3)
        plan["bridge_observation"] = bridge_observation_descriptor()
        suite["opponents"] = [{"id": "past", "weight": 1}]
        model = BridgeActorCritic(1, plan["bridge_observation"], 8, {"movement": 3})
        source = OpponentPool(self.definitions, self.contract, 3, "cpu", 42, learner=model)
        source.configure(suite["opponents"])
        source.capture(model, 0, {"fixed": {"movement": "forward"}}, force=True)
        source.reset(np.ones(3, dtype=bool))
        source.learner_slots[:] = [0, 0, 1]
        original_weight = next(source.controllers["past"].snapshots[0][0].parameters()).detach().clone()
        with torch.no_grad():
            next(model.parameters()).add_(1)
        source.capture(model, 1, {"fixed": {"movement": "backward"}})
        source.reset(np.array([True, True, False]))
        self.assertEqual(source.controllers["past"].snapshot_updates.tolist(), [1, 1, 0])
        result = evaluate(model, plan, suite, options, torch.device("cpu"), {}, source)
        self.assertEqual([seat["episodes"] for seat in result["seats"]], [3, 2])
        self.assertEqual(result["opponents"][0]["controller"]["snapshot_update"], 1)
        self.assertEqual(result["opponents"][0]["controller"]["fixed_actions"], {"movement": 2})
        self.assertEqual(len(result["opponents"][0]["controller"]["snapshot_sha256"]), 64)
        with self.assertRaisesRegex(ValueError, "compatible source snapshot"):
            evaluate(model, plan, suite, options, torch.device("cpu"), {})
        missing = source.controllers["past"].snapshots.pop(1)
        with self.assertRaisesRegex(ValueError, "snapshot is unavailable"):
            evaluate(model, plan, suite, options, torch.device("cpu"), {}, source)
        with self.assertRaisesRegex(ValueError, "snapshot is unavailable"):
            source.provenance(0)
        source.controllers["past"].snapshots[1] = missing
        source.controllers["past"].snapshots[1][0].descriptor["version"] = 2
        with self.assertRaisesRegex(ValueError, "compatible source snapshot"):
            evaluate(model, plan, suite, options, torch.device("cpu"), {}, source)
        source.controllers["past"].snapshots[1][0].descriptor["version"] = 1
        worlds = self.worlds(count=3, max_ticks=3)
        provenance = source.provenance(2)
        self.assertEqual(provenance["controller"]["fixed_actions"], {"movement": 1})
        stage = {**self.stage, "opponents": suite["opponents"]}
        plan.update(name="fixture", seed=42, stages=[stage])
        seen = []
        original_act = OpponentPool.act

        def capture(pool, observations, actions, bridge_inputs=None):
            controller = pool.controllers["past"]
            torch.testing.assert_close(next(controller.snapshots[0][0].parameters()), original_weight)
            self.assertEqual(controller.seed, provenance["controller"]["seed"])
            if not seen:
                stream = np.random.SeedSequence(controller.seed).spawn(3)[2]
                expected_seed = int(stream.generate_state(1, dtype=np.uint64)[0]) % 2**63
                expected = torch.Generator().manual_seed(expected_seed).get_state()
                torch.testing.assert_close(controller._actor_generators[0].get_state(), expected)
            result = original_act(pool, observations, actions, bridge_inputs)
            seen.append(result[0, 1 - pool.learner_slots[0], 0])
            return result

        with tempfile.TemporaryDirectory() as directory:
            arguments = (model, plan, stage, options, worlds.episode_scenario(2), provenance,
                         torch.device("cpu"), 0, 1, Path(directory))
            with patch.object(OpponentPool, "act", capture):
                path = export_replay(*arguments, opponent_pool=source)
            header = json.loads(Path(path).read_text().splitlines()[0])
            self.assertEqual(header["training"]["opponent"], provenance)
            self.assertTrue(seen and all(action == 1 for action in seen))
            for key, value, message in (("snapshot_update", 1, "assignment differs"),
                                        ("fixed_actions", {"movement": 2}, "controls differ"),
                                        ("seed", provenance["controller"]["seed"] + 1, "assignment differs")):
                invalid = deepcopy(provenance)
                invalid["controller"][key] = value
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                    export_replay(*arguments[:5], invalid, *arguments[6:], opponent_pool=source)
            wrong_seat = {**provenance, "learner_slot": 0}
            with self.assertRaisesRegex(ValueError, "assignment differs"):
                export_replay(*arguments[:5], wrong_seat, *arguments[6:], opponent_pool=source)
            with torch.no_grad():
                next(source.controllers["past"].snapshots[0][0].parameters()).add_(1)
            with self.assertRaisesRegex(ValueError, "snapshot differs from provenance"):
                export_replay(*arguments, opponent_pool=source)
            with torch.no_grad():
                next(source.controllers["past"].snapshots[0][0].parameters()).copy_(original_weight)
            with self.assertRaisesRegex(ValueError, "compatible source snapshot"):
                export_replay(*arguments)
            source.controllers["past"].snapshots[0][0].descriptor["version"] = 2
            with self.assertRaisesRegex(ValueError, "compatible source snapshot"):
                export_replay(*arguments, opponent_pool=source)

    def test_structured_current_evaluation_uses_suite_seed(self):
        self.scenario["bridge"]["match"]["max_ticks"] = 3
        self.definitions["current"] = {"type": "self_play"}
        plan, suite, options = self.evaluation_inputs(episodes=6, worlds=3)
        plan["bridge_observation"] = bridge_observation_descriptor()
        suite["opponents"] = [{"id": "current", "weight": 1}]
        model = BridgeActorCritic(1, plan["bridge_observation"], 8, {"movement": 3})
        traces = []
        original_act = OpponentPool.act

        def capture(pool, observations, actions, bridge_inputs=None):
            self.assertIsNotNone(bridge_inputs)
            result = original_act(pool, observations, actions, bridge_inputs)
            traces[-1].append(result[np.arange(pool.count), 1 - pool.learner_slots].copy())
            return result

        with patch.object(OpponentPool, "act", capture):
            for seed in (1729, 1729, 1730):
                suite["seed"] = seed
                traces.append([])
                result = evaluate(model, plan, suite, options, torch.device("cpu"), {})
                self.assertEqual([seat["episodes"] for seat in result["seats"]], [3, 3])
                self.assertEqual(result["opponents"][0]["controller"]["type"], "self_play")
        self.assertTrue(all(np.array_equal(left, right) for left, right in zip(traces[0], traces[1], strict=True)))
        self.assertTrue(any(not np.array_equal(left, right) for left, right in zip(traces[0], traces[2], strict=True)))
        plan["bridge_observation"] = {**plan["bridge_observation"], "version": 2}
        with self.assertRaisesRegex(ValueError, "descriptor"):
            evaluate(model, plan, suite, options, torch.device("cpu"), {})

    def test_structured_pool_routes_opponent_seats_for_mixed_worlds(self):
        definitions = {"target": {"type": "scripted", "controller": "stationary_target"},
                       "current": {"type": "self_play", "deterministic": True},
                       "past": {"type": "frozen_self_play", "every_updates": 1, "deterministic": True}}
        model = BridgeActorCritic(1, bridge_observation_descriptor(), 8, {"movement": 3})
        pool = OpponentPool(definitions, self.contract, 5, "cpu", 73, learner=model)
        pool.configure([{"id": name, "weight": 1} for name in definitions])
        pool.reset(np.ones(5, dtype=bool))
        pool.assignments[:] = [0, 1, 2, 1, 2]
        pool.learner_slots[:] = [0, 1, 0, 0, 1]
        scalar = np.array([[[10], [11]], [[20], [21]], [[30], [31]],
                           [[40], [41]], [[50], [51]]], dtype=np.float32)
        match = np.zeros((5, 2, 10), dtype=np.float32)
        match[:, :, 4] = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        voxels = np.zeros((5, 2, 5, 9, 9), dtype=np.uint8)
        voxels[1, 0, 0, 1, 2] = 2
        voxels[3, 1, 4, 5, 6] = 3
        voxels[2, 1, 1, 2, 3] = 4
        voxels[4, 0, 3, 4, 5] = 2
        inputs = {"scalar": scalar, "match": match, "voxels": voxels}
        current, past, target = (pool.controllers[name] for name in ("current", "past", "target"))
        with patch.object(current, "act", wraps=current.act) as current_act, \
             patch.object(past, "act", wraps=past.act) as past_act, \
             patch.object(target, "act", wraps=target.act) as target_act:
            actions = pool.act(scalar, np.zeros((5, 1), dtype=np.int64), inputs)
        np.testing.assert_array_equal(target_act.call_args.args[0], [[11]])
        for called, expected_scalar, expected_blocks, cells in (
                (current_act, [20, 41], [3, 8], [(0, 0, 1, 2, 2), (1, 4, 5, 6, 3)]),
                (past_act, [31, 50], [6, 9], [(0, 1, 2, 3, 4), (1, 3, 4, 5, 2)])):
            batch = called.call_args.args[0]
            np.testing.assert_array_equal(batch["scalar"][:, 0], expected_scalar)
            np.testing.assert_array_equal(batch["match"][:, 4], expected_blocks)
            self.assertEqual(np.count_nonzero(batch["voxels"]), 2)
            for row, up, right, forward, category in cells:
                self.assertEqual(batch["voxels"][row, up, right, forward], category)
        np.testing.assert_array_equal(actions[np.arange(5), pool.learner_slots], np.zeros((5, 1)))
        worlds = self.worlds(count=5)
        rollout, _, _, _ = collect(model, worlds, pool, worlds.observe(), 1,
                        torch.device("cpu"), {}, structured_bridge=True)
        self.assertEqual(rollout["observations"]["voxels"].shape, (1, 5, 5, 9, 9))

    def test_bridge_self_play_uses_opponent_seat_observations(self):
        observations = np.arange(64, dtype=np.float32).reshape(32, 2, 1)
        learner_actions = np.zeros((32, 1), dtype=np.int64)
        for definition in ({"type": "self_play", "deterministic": True},
                           {"type": "frozen_self_play", "every_updates": 1, "deterministic": True}):
            with self.subTest(definition=definition):
                model = self.model()
                pool = OpponentPool({"opponent": definition}, self.contract, 32, "cpu", 73, learner=model)
                pool.configure([{"id": "opponent", "weight": 1}], {"fixed": {}})
                pool.capture(model, 0, {"fixed": {}}, force=True)
                pool.reset(np.ones(32, dtype=bool))
                self.assertEqual(set(pool.learner_slots.tolist()), {0, 1})
                controller = pool.controllers["opponent"]
                with patch.object(controller, "act", wraps=controller.act) as act:
                    result = pool.act(observations, learner_actions)
                np.testing.assert_array_equal(act.call_args.args[0],
                                              observations[np.arange(32), 1 - pool.learner_slots])
                np.testing.assert_array_equal(result[np.arange(32), pool.learner_slots], learner_actions)

    def test_bridge_frozen_snapshot_pinning_and_pool_resume(self):
        definition = {"past": {"type": "frozen_self_play", "every_updates": 1, "deterministic": True}}
        model = self.model()
        pool = OpponentPool(definition, self.contract, 2, "cpu", 42, learner=model)
        pool.configure([{"id": "past", "weight": 1}], {"fixed": {}})
        pool.capture(model, 0, {"fixed": {"movement": "forward"}}, force=True)
        pool.reset(np.ones(2, dtype=bool))
        pool.capture(model, 1, {"fixed": {"movement": "backward"}})
        pool.reset(np.array([True, False]))
        self.assertEqual([pool.provenance(world)["controller"]["snapshot_update"] for world in range(2)], [1, 0])
        observations = np.zeros((2, 2, 1), dtype=np.float32)
        learner_actions = np.zeros((2, 1), dtype=np.int64)
        joint = pool.act(observations, learner_actions)
        np.testing.assert_array_equal(joint[np.arange(2), 1 - pool.learner_slots, 0], [2, 1])
        saved = deepcopy(pool.state_dict())
        resumed = OpponentPool(definition, self.contract, 2, "cpu", 42, learner=model)
        resumed.configure([{"id": "past", "weight": 1}], {"fixed": {}})
        resumed.load_state_dict(saved)
        self.assertEqual(resumed.controllers["past"].snapshot_updates.tolist(), [1, 0])
        for candidate in (pool, resumed):
            candidate.reset(np.ones(2, dtype=bool))
        np.testing.assert_array_equal(pool.learner_slots, resumed.learner_slots)
        self.assertEqual(resumed.controllers["past"].snapshot_updates.tolist(), [1, 1])
        np.testing.assert_array_equal(pool.act(observations, learner_actions), resumed.act(observations, learner_actions))

    def test_bridge_current_self_play_rng_resume_and_seeded_evaluation(self):
        self.scenario["bridge"]["match"]["max_ticks"] = 3
        self.definitions.update(current={"type": "self_play"},
                                unused={"type": "frozen_self_play", "every_updates": 1})
        plan, suite, options = self.evaluation_inputs(episodes=6, worlds=3)
        suite["opponents"] = [{"id": "current", "weight": 1}]
        model = self.model()
        first = evaluate(model, plan, suite, options, torch.device("cpu"), {})
        second = evaluate(model, plan, suite, options, torch.device("cpu"), {})
        self.assertEqual(first, second)
        self.assertEqual([seat["episodes"] for seat in first["seats"]], [3, 3])
        pool = OpponentPool(self.definitions, self.contract, 3, "cpu", 42, learner=model)
        pool.configure(suite["opponents"], suite["controls"])
        pool.reset(np.ones(3, dtype=bool))
        saved = deepcopy(pool.state_dict())
        resumed = OpponentPool(self.definitions, self.contract, 3, "cpu", 42, learner=model)
        resumed.configure(suite["opponents"], suite["controls"])
        resumed.load_state_dict(saved)
        observations = np.zeros((3, 2, 1), dtype=np.float32)
        learner_actions = np.zeros((3, 1), dtype=np.int64)
        for candidate in (pool, resumed):
            candidate.reset(np.ones(3, dtype=bool))
        np.testing.assert_array_equal(pool.learner_slots, resumed.learner_slots)
        for _ in range(5):
            np.testing.assert_array_equal(pool.act(observations, learner_actions), resumed.act(observations, learner_actions))

    def test_bridge_self_play_checkpoints_resume_and_replay(self):
        for opponent_type in ("self_play", "frozen_self_play"):
            with self.subTest(opponent_type=opponent_type), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                scene = self.scenario["arenas"][0]["environment"]
                arena = {"format": "minecraft_sim_arena", "version": 1, "minecraft": "26.3", "name": "fixture",
                         "rules": {"simulate_food": False, "natural_regeneration": False},
                         "bounds": scene["bounds"], "blocks": scene["blocks"]}
                bridge = deepcopy(self.scenario["bridge"])
                bridge["match"]["max_ticks"] = 2
                environment = {"format": "minecraft_sim_environment", "version": 2, "minecraft": "26.3",
                               "name": "fixture", "arena": "arena.json", "players": {"overrides": scene["players"]},
                               "task": {"max_ticks": 3, "rewards": {}}, "bridge": bridge}
                definition = {"type": opponent_type, "deterministic": True}
                if opponent_type == "frozen_self_play":
                    definition["every_updates"] = 1
                plan = {"format": "minecraft_sim_training_plan", "version": 2, "name": "fixture", "seed": 42,
                        "policy": "policy.json", "environment": "environment.json", "opponents": {"opponent": definition},
                        "mixtures": {"opponent": [{"id": "opponent", "weight": 1}]},
                        "stages": [{"name": "bridge", "opponents": "opponent", "advance": {"updates": 2}}],
                        "snapshots": [{"name": "initial", "at_updates": 0}, {"name": "final", "on_completion": True}],
                        "evaluations": {"held": {"stage": "bridge", "every_updates": 1,
                                                 "episodes": 4, "worlds": 2, "seed": 1729}}}
                for name, value in (("arena", arena), ("environment", environment),
                                    ("policy", self.contract), ("plan", plan)):
                    (root / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
                output = root / "run"
                flags = ["--plan", str(root / "plan.json"), "--library", self.library, "--envs", "2",
                         "--hidden-dim", "8", "--rollout-steps", "2", "--epochs", "1", "--minibatch-size", "2",
                         "--updates", "1", "--threads", "1", "--output", str(output), "--replay-every-updates", "1"]
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(flags), 0)
                first = torch.load(output / "latest.pt", weights_only=True)
                initial = torch.load(output / "snapshots/initial.pt", weights_only=True)
                self.assertEqual((initial["update"], initial["samples"]), (0, 0))
                CheckpointController(output / "snapshots/initial.pt", self.contract, 1)
                self.assertEqual(first["research"]["pool"]["controllers"]["opponent"]["provenance"]["type"], opponent_type)
                if opponent_type == "frozen_self_play":
                    for name, value in initial["model"].items():
                        torch.testing.assert_close(first["research"]["pool"]["controllers"]["opponent"]
                                                   ["snapshots"][0]["model"][name], value)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(["--resume", str(output / "latest.pt"), "--library", self.library,
                                           "--updates", "1", "--threads", "1", "--output", str(output),
                                           "--replay-every-updates", "1"]), 0)
                final = torch.load(output / "snapshots/final.pt", weights_only=True)
                saved = torch.load(output / "latest.pt", weights_only=True)
                self.assertEqual((final["update"], saved["update"]), (2, 2))
                CheckpointController(output / "snapshots/final.pt", self.contract, 1)
                self.assertEqual(saved["research"]["resume_mode"], "fresh_episodes")
                self.assertEqual(len(saved["research"]["bridge_episodes"]), 2)
                self.assertEqual([json.loads(line)["evaluations"]["held"]["episodes"]
                                  for line in (output / "metrics.jsonl").read_text().splitlines()], [4, 4])
                for recording in (output / "replays").glob("*.jsonl"):
                    header, *frames = [json.loads(line) for line in recording.read_text().splitlines()]
                    self.assertEqual(header["version"], 4)
                    self.assertEqual(header["training"]["opponent"]["controller"]["type"], opponent_type)
                    self.assertTrue(frames[-1]["terminated"] or frames[-1]["truncated"])
                self.assertEqual(len(list((output / "replays").glob("*.jsonl"))), 2)
                resolved = saved["research"]["plan"]
                stage = resolved["stages"][0]
                model = self.model()
                source = OpponentPool(resolved["opponents"], self.contract, 2, "cpu", 42, learner=model)
                source.configure(stage["opponents"], stage["controls"])
                source.load_state_dict(first["research"]["pool"])
                options = SimpleNamespace(device="cpu", library=self.library, max_ticks=3)
                for name, checkpoint in (("initial", initial), ("final", final)):
                    model.load_state_dict(checkpoint["model"])
                    provenance = source.provenance(0)
                    if opponent_type == "frozen_self_play":
                        provenance["controller"]["snapshot_update"] = 0
                        with self.assertRaisesRegex(ValueError, "requires a compatible source snapshot"):
                            export_replay(model, resolved, stage, options, checkpoint["metadata"]["scenario"], provenance,
                                          torch.device("cpu"), 0, checkpoint["update"], root / "missing")
                    path = export_replay(model, resolved, stage, options, checkpoint["metadata"]["scenario"], provenance,
                                         torch.device("cpu"), 0, checkpoint["update"], root / name, source)
                    header, *frames = [json.loads(line) for line in Path(path).read_text().splitlines()]
                    self.assertEqual((header["version"], header["training"]["policy_update"]), (4, checkpoint["update"]))
                    self.assertTrue(frames[-1]["terminated"] or frames[-1]["truncated"])

    def test_bridge_fixed_lane_evaluation_and_training_plan(self):
        scene = self.scenario["arenas"][0]["environment"]
        scene["players"][0]["position"][0] = -0.5
        scene["players"][1]["position"][0] = 1.5
        bridge = deepcopy(self.scenario["bridge"])
        bridge["goals"] = [{"min": [-1, 1, -2.9], "max": [2, 3, -2.4]},
                           {"min": [-1, 1, 2.4], "max": [2, 3, 2.9]}]
        bridge["match"].update(goals_to_win=1, max_ticks=100)
        contract = validate_contract({
            "format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3", "ticks_per_action": 1,
            "observations": [{"feature": "self.yaw_fraction"}],
            "actions": [{"name": "movement", "values": ["neutral", "forward"]},
                        {"name": "yaw", "values": [-2, 0, 2]}]})
        definitions = {"lane": {"type": "scripted", "controller": "bridge_fixed_lane"}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arena = {"format": "minecraft_sim_arena", "version": 1, "minecraft": "26.3", "name": "lane",
                     "rules": {"simulate_food": False, "natural_regeneration": False},
                     "bounds": scene["bounds"], "blocks": scene["blocks"]}
            environment = {"format": "minecraft_sim_environment", "version": 2, "minecraft": "26.3",
                           "name": "lane", "arena": "arena.json", "players": {"overrides": scene["players"]},
                           "task": {"max_ticks": 100, "rewards": {}}, "bridge": bridge}
            plan = {"format": "minecraft_sim_training_plan", "version": 2, "name": "lane", "seed": 42,
                    "policy": "policy.json", "environment": "environment.json", "opponents": definitions,
                    "mixtures": {"lane": [{"id": "lane", "weight": 1}]},
                    "stages": [{"name": "bridge", "opponents": "lane", "advance": {"updates": 1},
                                "controls": {"fixed": {"movement": "neutral", "yaw": 0}}}],
                    "evaluations": {"held": {"stage": "bridge", "episodes": 2, "worlds": 2,
                                             "seed": 1729, "every_updates": 1}}}
            for name, value in (("arena", arena), ("environment", environment),
                                ("policy", contract), ("plan", plan)):
                (root / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
            resolved = resolve_plan(plan, root)
            model = ActorCritic(contract["observation_dim"], 8,
                                action_sizes=dict(zip(contract["action_names"], contract["action_sizes"])))
            options = SimpleNamespace(device="cpu", library=self.library, max_ticks=100)
            result = evaluate(model, resolved, resolved["evaluations"]["held"], options, torch.device("cpu"), {})
            self.assertEqual((result["wins"], result["losses"], result["draws"], result["truncations"]), (0, 2, 0, 0))
            self.assertEqual((result["goals_for"], result["goals_against"]), (0, 2))
            self.assertEqual([seat["losses"] for seat in result["seats"]], [1, 1])
            invalid_suite = deepcopy(resolved["evaluations"]["held"])
            invalid_suite["scenario"]["arenas"][0]["environment"]["players"][0]["yaw"] = 90
            with self.assertRaisesRegex(ValueError, "aligned"):
                evaluate(model, resolved, invalid_suite, options, torch.device("cpu"), {})
            output = root / "run"
            flags = ["--plan", str(root / "plan.json"), "--library", self.library, "--envs", "2",
                     "--hidden-dim", "8", "--rollout-steps", "2", "--epochs", "1", "--minibatch-size", "2",
                     "--updates", "1", "--threads", "1", "--output", str(output)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(flags), 0)
            metrics = json.loads((output / "metrics.jsonl").read_text().splitlines()[0])
            self.assertEqual(metrics["evaluations"]["held"]["losses"], 2)
            self.assertEqual(metrics["evaluation_episodes"], 2)

    def test_unsupported_paths_are_explicit(self):
        plan = {"stages": [self.stage], "opponents": self.definitions, "policy": self.contract}
        self.assertIn("Bridge", resident_unsupported(plan, "cuda"))
        plan, suite, options = self.evaluation_inputs()
        for definition in ({"type": "scripted", "controller": "basic_melee"},
                           {"type": "scripted", "controller": "bridge_fixed_lane", "options": {"speed": 1}}):
            plan["opponents"] = {"target": definition}
            with self.subTest(definition=definition), self.assertRaisesRegex(ValueError, "Bridge requires"):
                evaluate(self.model(), plan, suite, options, torch.device("cpu"), {})
        plan["opponents"] = self.definitions
        suite["scenario"] = {**self.scenario, "spawns": {"sample": "episode"}}
        with self.assertRaisesRegex(ValueError, "fixed scenario"):
            evaluate(self.model(), plan, suite, options, torch.device("cpu"), {})
        sampled = deepcopy(self.scenario)
        sampled["spawns"] = {"sample": "episode"}
        with self.assertRaisesRegex(ValueError, "fixed spawns"):
            SampledWorlds(sampled, self.contract, 1, "cpu", self.library, 100, 42)

    def test_stage_reuse_respects_bridge_conditions_and_legacy_duels(self):
        bridge = deepcopy(self.scenario)
        self.assertFalse(requires_new_worlds(bridge, deepcopy(bridge)))
        changed = deepcopy(bridge)
        changed["bridge"]["task_rewards"]["score"] = 4
        self.assertTrue(requires_new_worlds(bridge, changed))
        duel = deepcopy(bridge)
        duel.pop("bridge")
        self.assertTrue(requires_new_worlds(duel, bridge))
        self.assertTrue(requires_new_worlds(bridge, duel))
        changed_duel = deepcopy(duel)
        changed_duel["arenas"][0]["environment"]["players"][0]["yaw"] = 45
        self.assertFalse(requires_new_worlds(duel, changed_duel))

    def test_bridge_checkpoint_reset_counters_continue_on_resume(self):
        original = self.worlds(2)
        original.reset(np.array([True, False]))
        saved = original.bridge_reset_episodes()
        resumed = self.worlds(2)
        resumed.restore_bridge_reset_episodes(saved)
        np.testing.assert_array_equal(resumed.bridge_reset_episodes(), np.array(saved) + 1)
        original.reset()
        np.testing.assert_array_equal(resumed.observe(), original.observe())
        np.testing.assert_array_equal(resumed.bridge_reset_episodes(), original.bridge_reset_episodes())
        before = resumed.bridge_reset_episodes()
        with self.assertRaisesRegex(ValueError, "reset episodes"):
            resumed.restore_bridge_reset_episodes([1, -1])
        self.assertEqual(resumed.bridge_reset_episodes(), before)


class ResidentPolicyChecks(unittest.TestCase):
    def test_device_sampling_matches_policy_log_probabilities_and_fixed_controls(self):
        torch.manual_seed(7)
        model = ActorCritic(4, 8, action_sizes={"jump": 2, "yaw": 3})
        observations = torch.randn(32, 4)
        fixed = {"jump": 1}
        first = torch.Generator().manual_seed(19)
        second = torch.Generator().manual_seed(19)
        actions, log_probs, values, valid = resident_policy_actions(
            model, observations, model.action_names, fixed, first)
        repeated, _, _, _ = resident_policy_actions(model, observations, model.action_names, fixed, second)
        _, expected_log_probs, expected_values, _ = policy_actions(model, observations, model.action_names,
                                                                  fixed, actions)
        torch.testing.assert_close(actions, repeated)
        torch.testing.assert_close(actions[:, 0], torch.ones(32, dtype=torch.long))
        torch.testing.assert_close(log_probs, expected_log_probs)
        torch.testing.assert_close(values, expected_values)
        self.assertTrue(valid.all())

    def test_nonfinite_policy_outputs_are_reported_without_invalid_indices(self):
        class InvalidPolicy(torch.nn.Module):
            def forward(self, observations):
                return {"jump": torch.tensor([[float("nan"), 0.0]])}, torch.tensor([float("nan")])

        actions, log_probs, values, valid = resident_policy_actions(
            InvalidPolicy(), torch.zeros((1, 1)), ("jump",), generator=torch.Generator().manual_seed(1))
        self.assertFalse(valid.item())
        self.assertTrue(torch.isfinite(log_probs).all())
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(0 <= actions.item() < 2)


class ResidentOpponentChecks(unittest.TestCase):
    def test_backend_selection_rejects_unsupported_plans(self):
        plan = resolve_plan(short_plan())
        self.assertIn("requires --device cuda", resident_unsupported(plan, "cpu"))
        self.assertIn("per-episode spawns", resident_unsupported(plan, "cuda"))
        for stage in plan["stages"]:
            stage["scenario"].pop("spawns", None)
        self.assertIn("unsupported opponent", resident_unsupported(plan, "cuda"))
        plan["stages"] = plan["stages"][:1]
        self.assertIsNone(resident_unsupported(plan, "cuda"))
        self.assertEqual(arguments(["--rollout-backend", "auto"]).rollout_backend, "auto")

    def test_stationary_and_self_play_assembly_and_rng_restore(self):
        contract = resolve_plan(short_plan())["policy"]
        model = ActorCritic(contract["observation_dim"], 16,
                            action_sizes=dict(zip(contract["action_names"], contract["action_sizes"])))
        definitions = {"target": {"type": "scripted", "controller": "stationary_target"},
                       "self": {"type": "self_play", "deterministic": True}}
        pool = OpponentPool(definitions, contract, 256, "cpu", 17, learner=model)
        fixed = {"jump": False}
        pool.configure([{"id": "target", "weight": 1}, {"id": "self", "weight": 1}], {"fixed": fixed})
        resident = ResidentOpponentPool(pool, "cpu", 17)
        resident.reset(torch.ones(256, dtype=torch.bool))
        self.assertTrue(70 < int((resident.assignments == 0).sum()) < 186)
        self.assertTrue(70 < int(resident.learner_slots.sum()) < 186)
        observations = torch.randn(256, 2, contract["observation_dim"])
        fixed = fixed_indices(contract, {"fixed": fixed})
        current, learner, _, _, joint, valid = resident.act(model, observations, contract["action_names"], fixed)
        indices = torch.arange(256)
        torch.testing.assert_close(current, observations[indices, resident.learner_slots])
        torch.testing.assert_close(joint[indices, resident.learner_slots], learner)
        target = resident.assignments == pool.names.index("target")
        expected = torch.as_tensor(pool.controllers["target"]._neutral).expand(int(target.sum()), -1)
        torch.testing.assert_close(joint[indices[target], 1 - resident.learner_slots[target]], expected)
        self_play = ~target
        other = observations[indices[self_play], 1 - resident.learner_slots[self_play]]
        expected_self, _, _, _ = resident_policy_actions(model, other, contract["action_names"], fixed,
                                 deterministic=True)
        torch.testing.assert_close(joint[indices[self_play], 1 - resident.learner_slots[self_play]], expected_self)
        self.assertTrue(valid.all())
        state = resident.state_dict()
        resident.reset(torch.ones(256, dtype=torch.bool))
        expected_assignments, expected_slots = resident.assignments.clone(), resident.learner_slots.clone()
        resident.load_state_dict(state)
        resident.reset(torch.ones(256, dtype=torch.bool))
        torch.testing.assert_close(resident.assignments, expected_assignments)
        torch.testing.assert_close(resident.learner_slots, expected_slots)
        state = resident.state_dict()
        selected = torch.zeros(256, dtype=torch.bool)
        selected[0] = True
        resident.reset(selected)
        after = resident.state_dict()
        torch.testing.assert_close(after["episodes"][1:], state["episodes"][1:])
        self.assertEqual(after["episodes"][0], state["episodes"][0] + 1)
        reference = ResidentOpponentPool(pool, "cpu", 17)
        reference.load_state_dict(state)
        other_worlds = ~selected
        for _ in range(5):
            reference.reset(other_worlds)
        reference.reset(selected)
        self.assertEqual(reference.assignments[0], resident.assignments[0])
        self.assertEqual(reference.learner_slots[0], resident.learner_slots[0])

    def test_unsupported_opponent_rejected(self):
        plan = resolve_plan(short_plan())
        pool = OpponentPool(plan["opponents"], plan["policy"], 1, "cpu", 1)
        pool.configure([{"id": "fighter", "weight": 1}])
        with self.assertRaisesRegex(ValueError, "does not support"):
            ResidentOpponentPool(pool, "cpu", 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_unspecified_cuda_device_accepts_cuda_zero_mask(self):
        contract = resolve_plan(short_plan())["policy"]
        model = ActorCritic(contract["observation_dim"], 16,
                            action_sizes=dict(zip(contract["action_names"], contract["action_sizes"]))).cuda()
        pool = OpponentPool({"target": {"type": "scripted", "controller": "stationary_target"}},
                            contract, 8, "cuda", 3, learner=model)
        pool.configure([{"id": "target", "weight": 1}])
        resident = ResidentOpponentPool(pool, "cuda", 3)
        resident.reset(torch.ones(8, dtype=torch.bool, device="cuda"))
        reference = ResidentOpponentPool(pool, "cpu", 3)
        reference.reset(torch.ones(8, dtype=torch.bool))
        torch.testing.assert_close(resident.assignments.cpu(), reference.assignments)
        torch.testing.assert_close(resident.learner_slots.cpu(), reference.learner_slots)
        result = resident.act(model, torch.zeros((8, 2, contract["observation_dim"]), device="cuda"),
                              contract["action_names"], {})
        self.assertTrue(result[-1].all().item())


class EpisodeMetricChecks(unittest.TestCase):
    def test_launch_smash_requires_same_player_and_uninterrupted_flight(self):
        behavior = BehaviorEvaluation(4)
        slots = np.array([0, 1, 0, 0])
        events = np.zeros((4, 2, 3), dtype=np.float32)
        events[:, :, 2] = 1
        events[0, 0, 0] = events[1, 1, 0] = events[2, 0, 0] = 0.3
        events[3, 1, 0] = 0.3
        behavior.step(events, slots)
        events.fill(0)
        behavior.step(events, slots)
        events[1, 1, 2] = 1
        behavior.step(events, slots)
        behavior.remaining[2] = 1
        events.fill(0)
        behavior.step(events, slots)
        events[:, :, 1] = 1
        behavior.step(events, slots)
        np.testing.assert_array_equal(behavior.launch_smash, [True, False, False, False])
        np.testing.assert_array_equal(behavior.launched, [True, True, True, False])
        behavior.reset(np.array([True, False, False, False]))
        self.assertFalse(behavior.launch_smash[0])
        self.assertEqual(behavior.remaining[0], 0)

    def test_rollout_aggregates_include_every_completion_and_span_boundaries(self):
        metrics = EpisodeMetrics(1, window=2)
        metrics.step(np.array([10.]), np.array([False]), [], [])
        metrics.begin_rollout()
        for outcome, opponent in (("win", "sword"), ("loss", "sword"), ("timeout", "axe"),
                                  ("win", "axe"), ("draw", "sword")):
            metrics.step(np.array([2.]), np.array([True]), [outcome], [opponent])
        result = metrics.rollout_snapshot()
        self.assertEqual(metrics.snapshot()["training_episodes"], 2)
        self.assertEqual(result["training_episodes"], 5)
        self.assertEqual(result["training_return"], 4)
        self.assertEqual(result["training_length"], 1.2)
        self.assertEqual(result["training_win_rate"], .4)
        self.assertEqual(result["training_opponents"]["sword"]["training_episodes"], 3)
        self.assertEqual(result["training_opponents"]["axe"]["training_win_rate"], .5)
        metrics.begin_rollout()
        self.assertEqual(metrics.rollout_snapshot()["training_episodes"], 0)
        self.assertNotIn("training_return", metrics.rollout_snapshot())
        self.assertEqual(metrics.snapshot()["training_episodes"], 2)

    def test_complete_episodes_span_rollouts_and_track_opponents(self):
        metrics = EpisodeMetrics(2)
        metrics.step(np.array([1., 10.]), np.array([False, False]), [], [])
        self.assertEqual(metrics.snapshot()["training_episodes"], 0)
        self.assertNotIn("training_return", metrics.snapshot())
        metrics.step(np.array([2., 20.]), np.array([True, False]), ["win"], ["sword"])
        self.assertEqual(metrics.snapshot()["training_return"], 3.)
        metrics.step(np.array([4., 30.]), np.array([True, True]), ["timeout", "loss"], ["axe", "sword"])
        result = metrics.snapshot()
        self.assertEqual(result["training_episodes"], 3)
        self.assertAlmostEqual(result["training_return"], 67 / 3)
        self.assertEqual(result["training_length"], 2)
        self.assertEqual(result["training_win_length"], 2)
        self.assertEqual(result["training_loss_length"], 3)
        self.assertEqual(result["training_opponents"]["sword"]["training_return"], 31.5)
        self.assertEqual(result["training_opponents"]["axe"]["training_timeout_rate"], 1)
        np.testing.assert_array_equal(metrics.returns, [0, 0])

    def test_window_resume_and_stage_reset(self):
        metrics = EpisodeMetrics(1, window=2)
        for outcome in ("loss", "win", "draw"):
            metrics.step(np.array([2.]), np.array([True]), [outcome], ["past"])
        self.assertEqual(metrics.snapshot()["training_loss_rate"], 0)
        self.assertEqual(metrics.snapshot()["training_draw_rate"], .5)
        resumed = EpisodeMetrics(1, list(metrics.history), window=2)
        self.assertEqual(resumed.snapshot(), metrics.snapshot())
        metrics.step(np.array([99.]), np.array([False]), [], [])
        metrics.reset()
        self.assertEqual(metrics.snapshot()["training_episodes"], 0)
        np.testing.assert_array_equal(metrics.returns, [0])
        np.testing.assert_array_equal(metrics.lengths, [0])


class PlanChecks(unittest.TestCase):
    def test_resolve_players_keeps_independent_mutable_outputs(self):
        shared = {"weapons": ["minecraft:iron_sword", "minecraft:air", "minecraft:air"],
                  "armor": {"helmet": "minecraft:turtle_helmet"}}
        players = [{"position": [-2, 1, 0], "loadout": shared}, {"position": [2, 1, 0], "loadout": shared}]
        before = deepcopy(players)
        resolved = resolve_players(players)
        self.assertEqual(players, before)
        self.assertEqual(resolved[0]["loadout"], resolved[1]["loadout"])
        resolved[0]["loadout"]["weapons"][0] = "minecraft:diamond_axe"
        resolved[0]["loadout"]["armor"]["helmet"] = "minecraft:air"
        resolved[0]["position"][0] = 99
        self.assertEqual(players, before)
        self.assertEqual(resolved[1]["loadout"]["weapons"][0], "minecraft:iron_sword")
        self.assertEqual(resolved[1]["loadout"]["armor"]["helmet"], "minecraft:turtle_helmet")
        for profile in ("configured", "unarmored_diamond_sword", "unarmored_sword_axe_shield"):
            defaults = resolve_players([{}, {}], profile)
            defaults[0]["loadout"]["weapons"][0] = "minecraft:air"
            self.assertEqual(defaults[1]["loadout"]["weapons"][0], "minecraft:diamond_sword")

    def test_resolve_players_preserves_equipment_validation(self):
        thrown = resolve_players([{"loadout": {"weapons": ["minecraft:snowball", "minecraft:egg"],
                                                "snowballs": 16, "eggs": 2}}, {}])[0]["loadout"]
        self.assertEqual((thrown["snowballs"], thrown["eggs"]), (16, 2))
        pearl = resolve_players([{"loadout": {"weapons": ["minecraft:ender_pearl"], "ender_pearls": 8}}, {}])[0]["loadout"]
        self.assertEqual(pearl["ender_pearls"], 8)
        wind = resolve_players([{"loadout": {"weapons": ["minecraft:mace", "minecraft:wind_charge"], "wind_charges": 64}}, {}])[0]["loadout"]
        self.assertEqual(wind["wind_charges"], 64)
        self.assertNotIn("snowballs", resolve_players([{}, {}])[0]["loadout"])
        for loadout in ({"weapons": ["minecraft:egg"], "eggs": 0},
                        {"weapons": ["minecraft:snowball"], "snowballs": 17},
                        {"weapons": ["minecraft:air"], "eggs": 1}):
            with self.assertRaises(ValueError):
                resolve_players([{"loadout": loadout}, {}])
        for material in ("wooden", "stone", "copper", "iron", "golden", "diamond", "netherite"):
            for kind in ("sword", "axe"):
                item = f"minecraft:{material}_{kind}"
                resolved = resolve_players([{"loadout": {"weapons": [item]}}, {}])
                self.assertEqual(resolved[0]["loadout"]["weapons"], [item])
        bow = resolve_players([{"loadout": {"weapons": ["minecraft:bow"], "arrows": 17}}, {}])
        self.assertEqual(bow[0]["loadout"]["weapons"], ["minecraft:bow"])
        self.assertEqual(bow[0]["loadout"]["arrows"], 17)
        self.assertEqual(bow[1]["loadout"]["arrows"], 0)
        self.assertEqual(resolve_players([{}, {}])[0]["loadout"]["arrows"], 0)
        for material in ("leather", "copper", "chainmail", "iron", "golden", "diamond", "netherite"):
            armor = {slot: f"minecraft:{material}_{slot}" for slot in ("helmet", "chestplate", "leggings", "boots")}
            resolved = resolve_players([{"loadout": {"armor": armor}}, {}])
            self.assertEqual(resolved[0]["loadout"]["armor"], armor)
        invalid = [None, {"weapons": ["minecraft:banana"]}, {"weapons": ["minecraft:iron_sword"] * 2},
                   {"weapons": []}, {"weapons": ["minecraft:air"] * 10}, {"selected": True}, {"selected": 1},
                   {"offhand": "minecraft:banana"}, {"armor": {"head": "minecraft:air"}},
                   {"armor": {"boots": "minecraft:turtle_helmet"}}, {"arrows": 1}, {"arrows": True},
                   {"arrows": -1},
                   {"arrows": 65}, {"arrows": 1.5}, {"unknown": True}]
        for loadout in invalid:
            with self.subTest(loadout=loadout), self.assertRaises(ValueError):
                resolve_players([{"loadout": loadout}, {}])
        with self.assertRaises(ValueError):
            resolve_players([{"position": [float("nan"), 1, 0]}, {}])

    def test_collection_timer_cuda_events_are_opt_in(self):
        with patch("torch.cuda.Event") as event, patch("torch.cuda.current_stream"), \
                patch("torch.cuda.synchronize") as synchronize:
            event.return_value.elapsed_time.return_value = 12.5
            disabled = CollectionTimer("cuda")
            with disabled.phase("compute", cuda=True):
                pass
            event.assert_not_called()
            synchronize.assert_not_called()
            enabled = CollectionTimer("cuda", True)
            with enabled.phase("compute", cuda=True):
                pass
            synchronize.assert_not_called()
            report = enabled.finish()
            self.assertEqual(event.call_count, 2)
            synchronize.assert_called_once_with(torch.device("cuda"))
            self.assertEqual(report["cuda_stream_seconds"], {"compute": 0.0125})

    def test_legacy_exports_and_roundtrip(self):
        self.assertIs(SCHEMA, LEGACY_SCHEMA)
        self.assertIs(SCHEMA_V2, LEGACY_V2)
        self.assertEqual(from_legacy(SCHEMA)["observation_dim"], 22)
        self.assertEqual(from_legacy(SCHEMA_V2)["observation_dim"], 31)
        plan = resolve_plan(example_plan())
        self.assertEqual(plan_hash(plan), plan_hash(resolve_plan(plan)))
        self.assertLessEqual(plan["policy"]["observation_dim"], 128)
        self.assertEqual(plan["policy"]["observation_dim"], 78)
        self.assertIn("self.has_shield", [entry["feature"] for entry in plan["policy"]["observations"]])
        self.assertNotIn("opponent.weapon_slot", [entry["feature"] for entry in plan["policy"]["observations"]])
        self.assertFalse(any(entry["feature"].startswith("opponent.weapon_slot_")
                             for entry in plan["policy"]["observations"]))

    def test_advance_exact_budgets_and_boolean_groups(self):
        progress = {"updates": 3, "episodes": 8, "transitions": 24}
        self.assertFalse(reached({"updates": 4}, progress, {}))
        self.assertTrue(reached({"updates": 3}, progress, {}))
        evaluation = {"suite": "held", "metric": "win_rate", "at_least": 0.75}
        self.assertFalse(reached({"evaluation": evaluation}, progress, {}))
        self.assertTrue(reached({"any": [{"updates": 4}, {"evaluation": evaluation}]},
                                progress, {"held": {"win_rate": 0.75}}))
        self.assertFalse(reached({"all": [{"updates": 4}, {"episodes": 8}]}, progress, {}))

    def test_reject_unknown_and_invalid_conditions(self):
        mutations = [
            lambda plan: plan.update(scenario={"block_edits": True}),
            lambda plan: plan["stages"][0]["controls"]["fixed"].update(jump=0),
            lambda plan: plan["stages"][0].update(advance={"updates": True}),
            lambda plan: plan["stages"][0].update(advance={"updates": 3, "episodes": 2}),
            lambda plan: plan["stages"][0].update(equipment={"sample": "sometimes"}),
            lambda plan: plan["stages"][0].update(task_override={"rewards": {"unknown": 2}}),
            lambda plan: plan["stages"][0].update(equipment={"loadout": {"offhand": {
                "choices": [{"value": "minecraft:air", "weight": -1}]}}}),
            lambda plan: plan["stages"][0].update(equipment={"loadout": {"offhand": {
                "choices": [{"value": "minecraft:shield", "weight": 1},
                            {"value": "minecraft:banana", "weight": 0}]}}}),
            lambda plan: plan["stages"][0].update(spawns={"pitch": {"uniform": [-100, 100]}}),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                value = short_plan()
                mutation(value)
                with self.assertRaises(ValueError):
                    resolve_plan(value)

    def test_sampler_validates_scenarios_only_on_configuration(self):
        scenario = resolve_plan(short_plan())["scenario"]
        with patch("simulator.training.plans.resolve_scenario", wraps=resolve_scenario) as validate:
            sampler = EpisodeSampler(scenario, 2, 42)
            initial_calls = validate.call_count
            self.assertGreater(initial_calls, 0)
            for world in (0, 1, 0, 1):
                players = sampler.players(world)
                base = scenario["arenas"][sampler.arenas[world]]["environment"]
                self.assertEqual(players, resolve_scenario({**base, "players": players})["players"])
            self.assertEqual(validate.call_count, initial_calls)
            sampler.configure(scenario)
            self.assertGreater(validate.call_count, initial_calls)
        invalid = deepcopy(scenario)
        invalid["arenas"][0]["environment"]["rules"]["natural_regeneration"] = True
        before = sampler.state_dict()
        with self.assertRaises(ValueError):
            sampler.configure(invalid)
        self.assertEqual(sampler.state_dict(), before)

    def test_sampler_still_validates_sampled_players(self):
        scenario = resolve_plan(short_plan())["scenario"]
        scenario["spawns"]["sample"] = "world"
        scenario["equipment"]["sample"] = "world"
        sampler = EpisodeSampler(scenario, 1, 42)
        sampler.players(0)
        saved = sampler.state_dict()
        sampler.cache[0]["equipment"][0]["offhand"] = "minecraft:banana"
        with self.assertRaisesRegex(ValueError, "offhand"):
            sampler.players(0)
        sampler.load_state_dict(saved)
        sampler.cache[0]["spawns"][0]["yaw"] = float("nan")
        with self.assertRaises(ValueError):
            sampler.players(0)

    def test_component_rng_and_world_cache(self):
        scenario = resolve_plan(short_plan())["scenario"]
        first, second = EpisodeSampler(scenario, 8, 42), EpisodeSampler(scenario, 8, 42)
        for world in range(8):
            a, b = first.players(world), second.players(world)
            self.assertEqual(a, b)
            again = first.players(world)
            self.assertEqual(again, second.players(world))
            self.assertEqual([p["loadout"] for p in a], [p["loadout"] for p in again])
            self.assertNotEqual([p["position"] for p in a], [p["position"] for p in again])
        state = first.state_dict()
        second.load_state_dict(state)
        self.assertEqual(first.players(3), second.players(3))
        changed = deepcopy(scenario)
        changed["spawns"]["yaw"] = 0
        second.configure(changed)
        self.assertEqual(first.gear_rng.bit_generator.state, second.gear_rng.bit_generator.state)

    def test_sampler_rejects_impossible_region(self):
        scenario = resolve_plan(short_plan())["scenario"]
        scenario["spawns"]["region"] = {"min": [0, 0, 0], "max": [0, 0, 0]}
        with self.assertRaisesRegex(ValueError, "collision-free"):
            EpisodeSampler(scenario, 1, 42).players(0)

    def test_sampled_height_has_consistent_ground_state(self):
        scenario = resolve_plan(short_plan())["scenario"]
        sampler = EpisodeSampler(scenario, 1, 42)
        self.assertTrue(all(player["on_ground"] for player in sampler.players(0)))
        scenario["spawns"]["region"]["min"][1] = 1.5
        scenario["spawns"]["region"]["max"][1] = 1.5
        sampler.configure(scenario)
        self.assertTrue(all(not player["on_ground"] for player in sampler.players(0)))

    def test_fixed_distribution_probability_entropy_and_gradients(self):
        contract = from_legacy(SCHEMA)
        model = ActorCritic(22, 8, 9)
        observations = torch.zeros(4, 22)
        fixed = fixed_indices(contract, {"fixed": {"attack": False, "yaw": 0}})
        actions, log_probs, values, entropy = policy_actions(model, observations, model.action_names, fixed)
        self.assertTrue(torch.equal(actions[:, 3], torch.zeros(4, dtype=torch.long)))
        self.assertTrue(torch.equal(actions[:, 4], torch.full((4,), 4)))
        _, recomputed, _, repeated_entropy = policy_actions(model, observations, model.action_names, fixed, actions)
        torch.testing.assert_close(log_probs, recomputed)
        torch.testing.assert_close(entropy, repeated_entropy)
        (-log_probs.mean() - entropy.mean() + values.mean()).backward()
        self.assertIsNone(model.attack_head.weight.grad)
        self.assertIsNone(model.yaw_head.weight.grad)


class SnapshotChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.directory = ROOT / "simulator/training/runs" / f"test-snapshots-{uuid4().hex}"
        self.directory.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.directory)
        arena = {"format": "minecraft_sim_arena", "version": 1, "minecraft": "26.3", "name": "test-arena",
                 "requires": ["movement", "melee"],
                 "rules": {"simulate_food": False, "natural_regeneration": False},
                 "bounds": {"min": [-5, 0, -5], "size": [11, 4, 11]},
                 "blocks": [{"from": [-5, 0, -5], "to": [5, 0, 5], "block": "minecraft:stone"}]}
        environment = {"format": "minecraft_sim_environment", "version": 2, "minecraft": "26.3",
                       "name": "test-environment", "arena": "arena.json",
                       "players": {"shared": {"loadout": {"weapons": ["minecraft:iron_sword"]}},
                                   "overrides": [{"position": [-2, 1, 0]}, {"position": [2, 1, 0]}]},
                       "task": {"max_ticks": 2, "rewards": {"episode_timeout": -1}}}
        self.raw = {"format": "minecraft_sim_training_plan", "version": 2, "name": "snapshot-test", "seed": 42,
                    "policy": "policy.json", "environment": "environment.json",
                    "opponents": {"past": {"type": "frozen_self_play", "every_updates": 2}},
                    "mixtures": {"past": [{"id": "past", "weight": 1}]},
                    "stages": [{"name": "aim", "opponents": "past", "controls": {"fixed": {"attack": False}},
                                "advance": {"updates": 2}},
                               {"name": "combat", "opponents": "past", "advance": {"updates": 2}}]}
        for name, value in (("arena", arena), ("environment", environment), ("policy", from_legacy(SCHEMA))):
            (self.directory / f"{name}.json").write_text(json.dumps(value))
        self.path = self.directory / "plan.json"
        self.path.write_text(json.dumps(self.raw))
        self.library = str(load_library(os.environ.get("MCSIM_TEST_CPU_LIBRARY"))._name)
        self.flags = ["--plan", str(self.path), "--library", self.library, "--envs", "4", "--hidden-dim", "8",
                      "--rollout-steps", "3", "--epochs", "1", "--minibatch-size", "8", "--save-every", "1"]

    def test_auto_backend_records_host_fallback_and_strict_backend_rejects(self):
        output = self.directory / "fallback"
        with contextlib.redirect_stdout(io.StringIO()) as log:
            self.assertEqual(main([*self.flags, "--updates", "1", "--output", str(output),
                                   "--rollout-backend", "auto"]), 0)
        config = json.loads((output / "config.json").read_text())
        self.assertEqual(config["options"]["effective_rollout_backend"], "host")
        self.assertIn("requires --device cuda", config["options"]["rollout_fallback_reason"])
        self.assertIn("using host collection", log.getvalue())
        with self.assertRaisesRegex(ValueError, "requires --device cuda"):
            main([*self.flags, "--updates", "1", "--output", str(self.directory / "strict"),
                  "--rollout-backend", "cuda-resident"])

    def test_sampler_validation_keeps_cache_and_template_ownership(self):
        scenario = resolve_plan(self.raw, self.directory)["scenario"]
        scenario["spawns"] = {"sample": "world", "region": {"min": [-4, 1, -4], "max": [4, 1, 4]},
                              "minimum_separation": 1, "yaw": {"uniform": [-180, 180]}}
        scenario["equipment"] = {"sample": "world", "players": "independent",
            "loadout": {"weapons": ["minecraft:iron_sword", "minecraft:iron_axe"],
                        "offhand": "minecraft:shield", "armor": {"helmet": "minecraft:turtle_helmet"}}}
        sampler = EpisodeSampler(scenario, 2, 42)
        players = sampler.players(0)
        before = sampler.state_dict()
        template = deepcopy(sampler.scenario)
        players[0]["position"][0] = 99
        players[0]["loadout"]["weapons"][0] = "minecraft:diamond_axe"
        players[0]["loadout"]["armor"]["helmet"] = "minecraft:air"
        self.assertEqual(sampler.state_dict(), before)
        self.assertEqual(sampler.scenario, template)
        repeated = sampler.players(0)
        self.assertEqual(repeated[0]["loadout"]["weapons"][0], "minecraft:iron_sword")
        self.assertEqual(repeated[0]["loadout"]["armor"]["helmet"], "minecraft:turtle_helmet")
        sampler.cache[0]["equipment"][0]["offhand"] = "minecraft:banana"
        with self.assertRaisesRegex(ValueError, "offhand"):
            sampler.players(0)
        sampler.load_state_dict(before)
        sampler.cache[0]["spawns"][0]["yaw"] = float("nan")
        with self.assertRaises(ValueError):
            sampler.players(0)

    # Source: train.arguments defines mutually exclusive episode and global-update replay schedules.
    def test_update_replay_argument_validation(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for flags in (["--replay-every-updates", "-1"],
                          ["--replay-every", "1", "--replay-every-updates", "2"],
                          ["--replay-dir", "unused"]):
                with self.subTest(flags=flags), self.assertRaises(SystemExit):
                    arguments(flags)
        options = arguments(["--replay-every-updates", "30", "--replay-dir", "unused"])
        self.assertEqual(options.replay_every, 0)
        self.assertEqual(options.replay_every_updates, 30)

    # Source: research.run schedules once per global update boundary, including across stage changes and resume.
    def test_update_replays_ignore_episode_count_and_resume_without_duplicates(self):
        output = self.directory / "update-replays"
        with contextlib.redirect_stdout(io.StringIO()):
            main([*self.flags, "--updates", "3", "--output", str(output), "--replay-every-updates", "2"])
        recordings = list((output / "replays").glob("*.jsonl"))
        self.assertEqual(len(recordings), 1)
        with recordings[0].open() as source:
            self.assertEqual(json.loads(source.readline())["training"]["policy_update"], 2)
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--resume", str(output / "latest.pt"), "--library", self.library,
                  "--updates", "1", "--replay-every-updates", "2"])
        headers = []
        for path in (output / "replays").glob("*.jsonl"):
            with path.open() as source:
                headers.append(json.loads(source.readline())["training"])
        self.assertEqual(sorted(header["policy_update"] for header in headers), [2, 4])
        self.assertTrue(all(header["completed_rounds"] > header["policy_update"] for header in headers))

    def test_dashboard_replay_request_waits_for_episode_completion(self):
        from simulator.training.replay_requests import create_request, read_request
        from simulator.training.research import collect as collect_training

        output = self.directory / "dashboard-request"

        def collect_then_request(*args, **kwargs):
            result = collect_training(*args, **kwargs)
            create_request(output)
            return result

        with patch("simulator.training.research.collect", side_effect=collect_then_request), \
                contextlib.redirect_stdout(io.StringIO()):
            main([*self.flags, "--updates", "1", "--output", str(output)])
        request = read_request(output)
        self.assertEqual(request["status"], "completed")
        self.assertGreater(request["completed_rounds"], 0)
        self.assertEqual(request["policy_update"], 1)
        self.assertTrue((output / "replays" / request["replay"]).is_file())
        config = json.loads((output / "config.json").read_text())
        self.assertTrue(config["options"]["dashboard_replay_requests"])

    def test_dashboard_analytics_request_profiles_only_the_next_rollout(self):
        from simulator.training.analytics_requests import create_request, read_request
        from simulator.training.research import collect as collect_training

        output = self.directory / "dashboard-analytics-request"
        profile_modes = []

        def collect_then_request(*args, **kwargs):
            profile_modes.append(kwargs.get("profile", False))
            result = collect_training(*args, **kwargs)
            if len(profile_modes) == 1:
                create_request(output)
            return result

        with patch("simulator.training.research.collect", side_effect=collect_then_request), \
                contextlib.redirect_stdout(io.StringIO()):
            main([*self.flags, "--updates", "3", "--output", str(output)])
        request = read_request(output)
        self.assertEqual(profile_modes, [False, True, False])
        self.assertEqual(request["status"], "completed")
        self.assertEqual(request["update"], 2)
        self.assertEqual(request["stage"], "aim")
        self.assertEqual(request["profile"]["version"], 1)
        self.assertEqual(request["optimization_profile"]["version"], 1)
        self.assertGreater(request["optimization_profile"]["counts"]["minibatches"], 0)
        self.assertIn("optimizer_step", request["optimization_profile"]["seconds"])
        rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual(request["stage"], rows[1]["stage"])
        self.assertNotIn("collection_profile", rows[0])
        self.assertEqual(rows[1]["collection_profile"], request["profile"])
        self.assertNotIn("optimization_profile", rows[0])
        self.assertEqual(rows[1]["optimization_profile"], request["optimization_profile"])
        self.assertNotIn("optimization_profile", rows[2])
        self.assertNotIn("collection_profile", rows[2])
        config = json.loads((output / "config.json").read_text())
        self.assertTrue(config["options"]["dashboard_analytics_requests"])

    def test_progress_trace_is_opt_in_and_reports_step_and_optimization_boundaries(self):
        for enabled in (False, True):
            output = self.directory / f"trace-{enabled}"
            messages = io.StringIO()
            with patch.dict("os.environ", {"MCSIM_TRACE_PROGRESS": "1" if enabled else "0"}), \
                    contextlib.redirect_stdout(messages):
                main([*self.flags, "--updates", "1", "--output", str(output)])
            trace = [line for line in messages.getvalue().splitlines() if line.startswith("TRACE ")]
            if not enabled:
                self.assertEqual(trace, [])
                continue
            self.assertIn("TRACE update=1 phase=collection_begin", trace)
            self.assertIn("TRACE rollout step=1/3 phase=environment_step_begin", trace)
            self.assertIn("TRACE rollout step=1/3 phase=environment_step_done", trace)
            self.assertIn("TRACE rollout phase=gae_begin", trace)
            self.assertIn("TRACE update=1 phase=optimization_begin", trace)
            self.assertIn("TRACE update=1 phase=optimization_done", trace)

    def test_progress_trace_keeps_step_start_when_native_step_fails(self):
        messages = io.StringIO()
        with patch.dict("os.environ", {"MCSIM_TRACE_PROGRESS": "1"}), \
                patch("simulator.training.research.SampledWorlds.step", side_effect=RuntimeError("step failed")), \
                contextlib.redirect_stdout(messages), self.assertRaisesRegex(RuntimeError, "step failed"):
            main([*self.flags, "--updates", "1", "--output", str(self.directory / "trace-failure")])
        trace = [line for line in messages.getvalue().splitlines() if line.startswith("TRACE ")]
        self.assertEqual(trace[-1], "TRACE rollout step=1/3 phase=environment_step_begin")

    def test_named_saves_rolling_opponent_resume_evaluation_and_replay(self):
        self.raw["snapshots"] = [{"name": "final", "on_completion": True}]
        self.raw["stages"][0]["snapshots"] = [{"name": "aim-1", "at_updates": 1},
                                              {"name": "aim-end", "on_completion": True}]
        self.raw["stages"][1]["snapshots"] = [{"name": "combat-1", "at_updates": 1}]
        self.raw["evaluations"] = {"held": {"stage": "aim", "every_updates": 1, "episodes": 4, "seed": 17}}
        self.path.write_text(json.dumps(self.raw))
        plan = resolve_plan(self.raw, self.directory)
        self.assertEqual(plan_hash(plan), plan_hash(resolve_plan(plan)))
        output = self.directory / "run"
        with contextlib.redirect_stdout(io.StringIO()):
            main([*self.flags, "--updates", "1", "--output", str(output), "--replay-every", "4"])
        first = torch.load(output / "latest.pt", weights_only=True)
        frozen = first["research"]["pool"]["controllers"]["past"]
        self.assertEqual(frozen["latest_update"], 0)
        aim_one = torch.load(output / "snapshots/aim-1.pt", weights_only=True)
        for name, value in first["model"].items():
            torch.testing.assert_close(aim_one["model"][name], value, rtol=0, atol=0)
        staged_output = self.directory / "snapshot-start-combat"
        with contextlib.redirect_stdout(io.StringIO()):
            main([*self.flags, "--updates", "1", "--output", str(staged_output), "--initialize",
                  str(output / "snapshots/aim-1.pt"), "--start-stage", "combat"])
        staged = torch.load(staged_output / "latest.pt", weights_only=True)
        self.assertEqual(staged["research"]["stage_index"], 1)
        self.assertEqual(staged["research"]["progress"]["updates"], 1)
        for name, value in aim_one["model"].items():
            torch.testing.assert_close(staged["research"]["pool"]["controllers"]["past"]["snapshots"][0]["model"][name],
                                       value, rtol=0, atol=0)
        for path in self.directory.glob("*.json"):
            path.unlink()
        for _ in range(3):
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--resume", str(output / "latest.pt"), "--library", self.library,
                      "--updates", "1", "--output", str(output)])
        saved = torch.load(output / "latest.pt", weights_only=True)
        self.assertTrue(saved["research"]["completed"])
        self.assertEqual(saved["research"]["pool"]["controllers"]["past"]["latest_update"], 4)
        self.assertEqual({path.stem for path in (output / "snapshots").glob("*.pt")},
                         {"aim-1", "aim-end", "combat-1", "final"})
        for name, update, stage, local in (("aim-1", 1, "aim", 1), ("aim-end", 2, "aim", 2),
                                            ("combat-1", 3, "combat", 1), ("final", 4, "combat", 2)):
            snapshot = torch.load(output / f"snapshots/{name}.pt", weights_only=True)
            self.assertEqual(snapshot["update"], update)
            self.assertEqual(snapshot["metadata"]["experiment"]["stage"], stage)
            self.assertEqual(snapshot["metadata"]["snapshot"]["stage_updates"], local)
            self.assertEqual(snapshot["metadata"]["controls"], {"fixed": {"attack": False} if stage == "aim" else {}})
            CheckpointController(output / f"snapshots/{name}.pt", plan["policy"], 1)
        for name, value in saved["model"].items():
            torch.testing.assert_close(saved["research"]["pool"]["controllers"]["past"]["snapshots"][4]["model"][name], value)
        rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([row["stage_progress"]["updates"] for row in rows], [1, 2, 1, 2])
        self.assertTrue(all(row["evaluations"]["held"]["episodes"] == 4 for row in rows))
        with next((output / "replays").glob("*.jsonl")).open() as source:
            header = json.loads(source.readline())
        self.assertEqual(header["training"]["opponent"]["controller"]["snapshot_update"], 0)
        self.assertEqual(aim_one["metadata"]["controls"], {"fixed": {"attack": False}})

    def test_initial_and_stage_entry_snapshots_survive_resume(self):
        self.raw["snapshots"] = [{"name": "initial", "at_updates": 0}]
        self.raw["stages"][0]["snapshots"] = [{"name": "aim-start", "at_updates": 0}]
        self.raw["stages"][1]["snapshots"] = [{"name": "combat-start", "at_updates": 0}]
        self.path.write_text(json.dumps(self.raw))
        output = self.directory / "stage-entry"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([*self.flags, "--updates", "2", "--output", str(output)]), 0)
        initial = torch.load(output / "snapshots/initial.pt", weights_only=True)
        aim = torch.load(output / "snapshots/aim-start.pt", weights_only=True)
        combat = torch.load(output / "snapshots/combat-start.pt", weights_only=True)
        checkpoint = torch.load(output / "latest.pt", weights_only=True)
        self.assertEqual((initial["update"], aim["update"], combat["update"]), (0, 0, 2))
        self.assertEqual(combat["metadata"]["snapshot"]["stage_updates"], 0)
        self.assertEqual(combat["metadata"]["experiment"]["stage"], "combat")
        for name, value in initial["model"].items():
            torch.testing.assert_close(aim["model"][name], value)
            torch.testing.assert_close(checkpoint["research"]["pool"]["controllers"]["past"]
                                       ["snapshots"][2]["model"][name], combat["model"][name])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--resume", str(output / "latest.pt"), "--library", self.library,
                                   "--updates", "1", "--output", str(output)]), 0)
        resumed = torch.load(output / "snapshots/combat-start.pt", weights_only=True)
        self.assertEqual(resumed["update"], 2)
        for name, value in combat["model"].items():
            torch.testing.assert_close(resumed["model"][name], value)

    def test_collection_profile_preserves_rollout_and_rng(self):
        raw = deepcopy(self.raw)
        raw["opponents"] = {"target": {"type": "scripted", "controller": "stationary_target"}}
        raw["mixtures"] = {"past": [{"id": "target", "weight": 1}]}
        raw["stages"][0]["spawns"] = {"sample": "episode",
            "region": {"min": [-4, 1, -4], "max": [4, 1, 4]}, "minimum_separation": 1}
        results = []
        for profile in (False, True):
            torch.manual_seed(123)
            plan = resolve_plan(raw, self.directory)
            worlds = SampledWorlds(plan["scenario"], plan["policy"], 4, "cpu", self.library, 2, 42)
            try:
                worlds.configure(plan["stages"][0])
                pool = OpponentPool(plan["opponents"], plan["policy"], 4, "cpu", 42)
                pool.configure(plan["stages"][0]["opponents"])
                pool.reset(np.ones(4, dtype=bool))
                model = ActorCritic(worlds.observation_dim, 8,
                    action_sizes=dict(zip(worlds.action_names, plan["policy"]["action_sizes"])))
                fixed = fixed_indices(plan["policy"], plan["stages"][0]["controls"])
                result = collect(model, worlds, pool, worlds.observe(), 3, torch.device("cpu"), fixed,
                                 profile=profile)
                results.append((result, torch.get_rng_state(), worlds.sampler.state_dict(), pool.state_dict()))
            finally:
                worlds.close()
        (plain, plain_rng, plain_sampler, plain_pool), (timed, timed_rng, timed_sampler, timed_pool) = results
        self.assertNotIn("collection_profile", plain[0])
        report = timed[0].pop("collection_profile")
        for name, value in plain[0].items():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(value, timed[0][name]), name)
            else:
                self.assertEqual(value, timed[0][name])
        np.testing.assert_array_equal(plain[1], timed[1])
        self.assertEqual(plain[2:], timed[2:])
        self.assertTrue(torch.equal(plain_rng, timed_rng))
        self.assertEqual(plain_sampler, timed_sampler)
        self.assertEqual(plain_pool, timed_pool)
        self.assertEqual(report["counts"]["reset_worlds"], timed[2] + timed[3])
        self.assertEqual(report["counts"]["reset_calls"], 1)
        self.assertGreaterEqual(report["counts"]["spawn_attempts"], report["counts"]["reset_worlds"])
        self.assertEqual(report["cuda_stream_seconds"], {})
        self.assertGreater(report["seconds"]["environment_reset"], 0)
        self.assertGreater(report["seconds"]["gae"], 0)
        self.assertGreater(report["native_seconds"]["reset_validate"], 0)
        self.assertGreater(report["native_seconds"]["step_decode"], 0)
        self.assertEqual(report["counts"]["native_batches"], 1)
        self.assertLessEqual(sum(value for name, value in report["seconds"].items() if "." not in name),
                             report["total_seconds"])
        for phase in ("sample", "payload", "json_encode", "native", "copy_outputs"):
            self.assertGreater(report["seconds"][f"environment_reset.{phase}"], 0)

    def test_collection_profile_cadence_and_training_parity(self):
        self.assertEqual(arguments(["--plan", str(self.path)]).profile_every, 0)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            arguments(["--profile-every", "-1"])
        checkpoints = []
        for interval in (0, 2):
            output = self.directory / f"profile-{interval}"
            with contextlib.redirect_stdout(io.StringIO()):
                main([*self.flags, "--updates", "2", "--output", str(output), "--profile-every", str(interval)])
            rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["update"] for row in rows if "collection_profile" in row],
                             [2] if interval else [])
            checkpoints.append(torch.load(output / "latest.pt", weights_only=True))
            self.assertEqual(json.loads((output / "config.json").read_text())["options"]["profile_every"], interval)
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                analyze_metrics([str(output), "--timings", "--skip-updates", "0", "--json"])
            report = json.loads(captured.getvalue())
            self.assertEqual(report["profiled_updates"], 1 if interval else 0)
            self.assertEqual(report["groups"]["unprofiled"]["updates"], 1 if interval else 2)
        for name, value in checkpoints[0]["model"].items():
            self.assertTrue(torch.equal(value, checkpoints[1]["model"][name]), name)
        self.assertTrue(torch.equal(checkpoints[0]["rng_state"], checkpoints[1]["rng_state"]))
        self.assertEqual(checkpoints[0]["research"]["sampler"], checkpoints[1]["research"]["sampler"])
        output = self.directory / "profile-2"
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--resume", str(output / "latest.pt"), "--library", self.library, "--updates", "1",
                  "--output", str(output), "--profile-every", "3"])
        rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        self.assertIn("collection_profile", rows[-1])

    def test_episode_metrics_collection_resume_and_stage_boundaries(self):
        output = self.directory / "episode-metrics"
        with contextlib.redirect_stdout(io.StringIO()):
            main([*self.flags, "--updates", "1", "--output", str(output)])
        first = json.loads((output / "metrics.jsonl").read_text())
        self.assertEqual(first["training_episodes"], 4)
        self.assertEqual(first["training_timeout_rate"], 1)
        self.assertEqual(first["training_length"], 2)
        self.assertEqual(first["training_return"], -1)
        saved = torch.load(output / "latest.pt", weights_only=True)
        self.assertEqual(len(saved["research"]["episode_history"]), 4)
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--resume", str(output / "latest.pt"), "--library", self.library,
                  "--updates", "2", "--output", str(output)])
        rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([row["training_episodes"] for row in rows], [4, 8, 4])
        self.assertEqual([row["training_length"] for row in rows], [2, 2, 2])
        self.assertEqual(rows[-1]["training_opponents"]["past"]["training_timeout_rate"], 1)

    def test_native_losses_match_legacy_terminal_state_with_zero_rewards(self):
        plan = resolve_plan(self.raw, self.directory)
        worlds = SampledWorlds(plan["scenario"], plan["policy"], 4, "cpu", self.library, 2, 42)
        self.addCleanup(worlds.close)
        environment = worlds.groups[0][1]
        players = [deepcopy(environment.scenario["players"]) for _ in range(4)]
        for world, slot in ((1, 0), (2, 1), (3, 0), (3, 1)):
            players[world][slot].update(position=[5.5, 1, -2 if slot == 0 else 2], velocity=[20, 0, 0])
        environment.reset_players(players, np.ones(4, dtype=bool))
        environment.set_rewards({})
        environment.step(np.zeros((4, 2, environment.action_count), dtype=np.int64))
        native = environment.losses(np.ones(4, dtype=bool))
        with patch.object(environment.native, "mcsim_losses", None):
            fallback = environment.losses(np.ones(4, dtype=bool))
        np.testing.assert_array_equal(native, [[False, False], [True, False], [False, True], [True, True]])
        np.testing.assert_array_equal(native, fallback)
        environment.set_max_ticks(1)
        observations = environment.reset_players(players, np.ones(4, dtype=bool))
        pool = OpponentPool({"past": {"type": "scripted", "controller": "stationary_target"}},
                    plan["policy"], 4, "cpu", 42)
        pool.configure(plan["stages"][0]["opponents"])
        pool.reset(np.ones(4, dtype=bool))
        pool.learner_slots.fill(0)
        model = ActorCritic(worlds.observation_dim, 8,
            action_sizes=dict(zip(worlds.action_names, plan["policy"]["action_sizes"])))
        with patch.object(worlds, "episode", side_effect=AssertionError("Collection must not serialize replays")):
            rollout, _, endings, timeouts = collect(model, worlds, pool, observations, 1, torch.device("cpu"), {})
        self.assertEqual((endings, timeouts), (3, 1))
        for outcome in ("win", "loss", "draw", "timeout"):
            self.assertEqual(rollout["episode_metrics"][f"training_{outcome}_rate"], .25)

    def test_collection_profile_cleanup_on_error(self):
        plan = resolve_plan(self.raw, self.directory)
        worlds = SampledWorlds(plan["scenario"], plan["policy"], 4, "cpu", self.library, 2, 42)
        self.addCleanup(worlds.close)
        timer = CollectionTimer("cpu", True)
        with self.assertRaisesRegex(RuntimeError, "probe"):
            with worlds.profile_collection(timer):
                worlds.observe()
                raise RuntimeError("probe")
        environment = worlds.groups[0][1]
        saved = json.loads(environment.native.mcsim_profile(environment._handle))
        worlds.observe()
        self.assertEqual(saved, json.loads(environment.native.mcsim_profile(environment._handle)))
        self.assertGreater(timer.native_seconds["observe"], 0)

    def test_collection_profile_requires_new_native_api_only_when_enabled(self):
        plan = resolve_plan(self.raw, self.directory)
        worlds = SampledWorlds(plan["scenario"], plan["policy"], 4, "cpu", self.library, 2, 42)
        self.addCleanup(worlds.close)
        with patch.object(worlds.groups[0][1].native, "mcsim_set_profiling", None):
            with worlds.profile_collection(CollectionTimer("cpu")):
                worlds.observe()
            with self.assertRaisesRegex(RuntimeError, "Rebuild minecraft_sim_training"):
                with worlds.profile_collection(CollectionTimer("cpu", True)):
                    self.fail("Old native libraries cannot produce native profiles")

    def test_retained_saves_are_opt_in_and_initialize_seeds_frozen_policy(self):
        output = self.directory / "source"
        with contextlib.redirect_stdout(io.StringIO()):
            main([*self.flags, "--updates", "1", "--output", str(output)])
        self.assertFalse((output / "snapshots").exists())
        source = torch.load(output / "latest.pt", weights_only=True)
        target = self.directory / "initialized"
        with contextlib.redirect_stdout(io.StringIO()):
            main([*self.flags, "--updates", "1", "--output", str(target), "--initialize", str(output / "latest.pt")])
        saved = torch.load(target / "latest.pt", weights_only=True)
        frozen = saved["research"]["pool"]["controllers"]["past"]["snapshots"][0]
        for name, value in source["model"].items():
            torch.testing.assert_close(frozen["model"][name], value, rtol=0, atol=0)
        self.assertEqual(frozen["fixed"], {"attack": 0})
        staged = self.directory / "initialized-combat"
        with contextlib.redirect_stdout(io.StringIO()):
            main([*self.flags, "--updates", "1", "--output", str(staged), "--initialize",
                  str(output / "latest.pt"), "--start-stage", "combat"])
        staged_research = torch.load(staged / "latest.pt", weights_only=True)["research"]
        self.assertEqual(staged_research["stage_index"], 1)
        self.assertEqual(staged_research["plan"]["stages"][staged_research["stage_index"]]["name"], "combat")
        self.assertEqual(staged_research["progress"]["updates"], 1)


class ResearchChecks(unittest.TestCase):
    def test_save_every_retains_full_checkpoint_history(self):
        plan = short_plan()
        for stage in plan["stages"]:
            stage["advance"] = {"updates": 10}
        self.path.write_text(json.dumps(plan))
        output = self.directory / "checkpoint-history"
        flags = self.flags[:self.flags.index("--save-every")]
        with contextlib.redirect_stdout(io.StringIO()):
            main([*flags, "--updates", "5", "--output", str(output), "--save-every", "2"])
        checkpoints = sorted((output / "checkpoints").glob("update-*.pt"))
        self.assertEqual([path.name for path in checkpoints],
                         ["update-00000002.pt", "update-00000004.pt", "update-00000005.pt"])
        for path, update in zip(checkpoints, (2, 4, 5)):
            self.assertEqual(torch.load(path, weights_only=True)["update"], update)
        self.assertEqual(torch.load(output / "latest.pt", weights_only=True)["update"], 5)

    def test_learner_overrides_reach_training_metadata_and_resume(self):
        from simulator.training.ppo import compute_gae
        values = {"gamma": 0.8, "gae_lambda": 0.7, "clip_eps": 0.15,
                  "vf_coef": 0.3, "ent_coef": 0.025, "max_grad_norm": 0.8}
        flags = [item for name, value in values.items() for item in ("--" + name.replace("_", "-"), str(value))]
        output = self.directory / "learner-settings"
        with patch("simulator.training.research.compute_gae", wraps=compute_gae) as gae, \
                patch("simulator.training.research.optimize", wraps=optimize) as optimizer, \
                contextlib.redirect_stdout(io.StringIO()):
            main([*self.flags, *flags, "--updates", "1", "--output", str(output)])
        self.assertEqual(gae.call_args.args[-2:], (values["gamma"], values["gae_lambda"]))
        settings = optimizer.call_args.kwargs["learner"]
        for key, value in values.items():
            self.assertEqual(settings[key], value)
        config = json.loads((output / "config.json").read_text())
        checkpoint = torch.load(output / "latest.pt", weights_only=True)
        self.assertEqual(config["metadata"]["learner_config"], settings)
        self.assertEqual(checkpoint["metadata"]["learner_config"], settings)
        self.assertEqual(checkpoint["research"]["plan"]["learner"], settings)
        for key, value in values.items():
            self.assertEqual(config["options"][key], value)
            self.assertEqual(checkpoint["config"][key], value)
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--resume", str(output / "latest.pt"), "--library", self.library, "--updates", "1"])
        resumed = torch.load(output / "latest.pt", weights_only=True)
        self.assertEqual(resumed["metadata"]["learner_config"], settings)
        with self.assertRaisesRegex(ValueError, "differs from checkpoint"), contextlib.redirect_stdout(io.StringIO()):
            main(["--resume", str(output / "latest.pt"), "--library", self.library, "--gamma", "0.9"])
        for key, value in (("gamma", "1.1"), ("gae-lambda", "-0.1"), ("clip-eps", "0"),
                           ("vf-coef", "-1"), ("ent-coef", "nan"), ("max-grad-norm", "0")):
            with self.subTest(key=key), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments(["--" + key, value])

    def setUp(self):
        torch.set_num_threads(1)
        self.directory = ROOT / "simulator/training/runs" / f"test-research-{uuid4().hex}"
        self.directory.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.directory)
        self.path = self.directory / "plan.json"
        self.library = load_library()._build_info["library_path"]
        self.path.write_text(json.dumps(short_plan()))
        self.flags = ["--plan", str(self.path), "--library", self.library, "--envs", "4",
                      "--hidden-dim", "8", "--rollout-steps", "3", "--epochs", "1",
                      "--minibatch-size", "8", "--max-ticks", "2", "--save-every", "1"]

    def test_configured_ppo_coefficients_control_loss_and_gradient_clipping(self):
        from simulator.training.compiler import LEARNER_DEFAULTS
        model = ActorCritic(3, 8, action_sizes={"attack": 2})
        observations = torch.zeros(2, 3)
        actions = torch.tensor([[0], [1]])
        with torch.no_grad():
            logits, values = model(observations)
            log_probs = torch.distributions.Categorical(logits=logits["attack"]).log_prob(actions[:, 0])
        rollout = {"observations": observations.unsqueeze(1), "actions": actions.unsqueeze(1),
                   "log_probs": (log_probs - np.log(1.5)).unsqueeze(1), "values": (values + 1).unsqueeze(1),
                   "advantages": torch.tensor([[1.], [-1.]]), "returns": values.unsqueeze(1)}
        optimizer = torch.optim.SGD(model.parameters(), lr=0)
        for clipping, value_coef, entropy_coef, norm in ((.1, .25, .03, .8), (.3, .75, 0, .2)):
            settings = {**LEARNER_DEFAULTS, "clip_eps": clipping, "vf_coef": value_coef,
                        "ent_coef": entropy_coef, "max_grad_norm": norm}
            with patch("torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as clip:
                result = optimize(model, optimizer, rollout, 1, 2, learner=settings)
            self.assertAlmostEqual(result["policy_loss"], (.5 - clipping) / 2, places=5)
            self.assertAlmostEqual(result["value_loss"], .5 * (1 - clipping) ** 2, places=5)
            self.assertAlmostEqual(result["loss"], result["policy_loss"] + value_coef * result["value_loss"]
                                   - entropy_coef * result["entropy"], places=5)
            self.assertEqual(clip.call_args.args[1], norm)

    def test_legacy_learner_defaults_resume_without_allowing_parameter_changes(self):
        from simulator.training.compiler import LEARNER_DEFAULTS
        output = self.directory / "legacy-learner"
        with contextlib.redirect_stdout(io.StringIO()):
            main([*self.flags, "--updates", "1", "--output", str(output)])
        checkpoint = torch.load(output / "latest.pt", weights_only=True)
        checkpoint["metadata"].pop("learner_config")
        for name in set(LEARNER_DEFAULTS) - {"hidden_dim", "lr", "epochs", "minibatch_size"}:
            checkpoint["research"]["plan"]["learner"].pop(name)
            checkpoint["config"].pop(name)
        checkpoint["research"]["plan_hash"] = plan_hash(checkpoint["research"]["plan"])
        torch.save(checkpoint, output / "latest.pt")
        with self.assertRaisesRegex(ValueError, "differs from checkpoint"), contextlib.redirect_stdout(io.StringIO()):
            main(["--resume", str(output / "latest.pt"), "--library", self.library, "--gamma", "0.5"])
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--resume", str(output / "latest.pt"), "--library", self.library, "--updates", "1"])
        resumed = torch.load(output / "latest.pt", weights_only=True)
        self.assertEqual(resumed["metadata"]["learner_config"]["gamma"], LEARNER_DEFAULTS["gamma"])

    def test_stage_local_evaluation_cadence_resume_advancement_and_timing(self):
        plan = short_plan()
        plan["stages"][0]["evaluations"] = {"aim_accuracy": {
            "every_updates": 1, "worlds": 3, "episodes": 5, "seed": 1729,
        }}
        plan["stages"][0]["advance"] = {"evaluation": {
            "suite": "aim_accuracy", "metric": "reward_mean", "at_least": 0,
        }}
        plan["stages"][1]["evaluations"] = {"combat_check": {
            "every_updates": 2, "worlds": 3, "episodes": 5, "seed": 1730,
        }}
        self.path.write_text(json.dumps(plan))
        output = self.directory / "local-evaluations"
        with contextlib.redirect_stdout(io.StringIO()):
            main([*self.flags, "--updates", "1", "--output", str(output)])
        checkpoint = torch.load(output / "latest.pt", weights_only=True)
        self.assertEqual(checkpoint["research"]["stage_index"], 1)
        self.assertEqual(checkpoint["research"]["evaluations"], {})
        self.path.unlink()
        for _ in range(2):
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--resume", str(output / "latest.pt"), "--library", self.library,
                      "--updates", "1", "--output", str(output)])
        rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([set(row["evaluations"]) for row in rows],
                         [{"aim_accuracy"}, set(), {"combat_check"}])
        self.assertEqual([row["evaluation_episodes"] for row in rows], [5, 0, 5])
        self.assertEqual(rows[0]["evaluations"]["aim_accuracy"]["worlds"], 3)
        self.assertEqual(rows[2]["evaluations"]["combat_check"]["snapshot_update"], 3)
        self.assertEqual(rows[2]["evaluations"]["combat_check"]["stage_index"], 1)
        self.assertTrue(rows[-1]["completed_curriculum"])
        self.assertEqual(rows[-1]["samples"], 36)
        for row in rows:
            for key in ("collection_seconds", "update_seconds", "evaluation_seconds", "replay_seconds"):
                self.assertGreaterEqual(row[key], 0)
            self.assertGreaterEqual(row["iteration_seconds"], sum(row[key] for key in (
                "collection_seconds", "update_seconds", "evaluation_seconds", "replay_seconds")))
            self.assertAlmostEqual(row["player_transitions_per_second"] * row["iteration_seconds"], 12)
            self.assertNotIn("cuda_peak_allocated_bytes", row)

    def test_self_play_training_resume_evaluation_and_replay(self):
        plan = short_plan()
        plan["opponents"]["policy"] = {"type": "self_play"}
        plan["mixtures"]["self_play"] = [{"id": "policy", "weight": 1}]
        for stage in plan["stages"]:
            stage["opponents"] = "self_play"
        plan["evaluations"] = {"self_play": {
            "environment": plan["environment"], "every_updates": 1, "episodes": 4,
            "seed": 1729, "opponents": "self_play"}}
        self.path.write_text(json.dumps(plan))
        output = self.directory / "self-play"
        with contextlib.redirect_stdout(io.StringIO()):
            main([*self.flags, "--updates", "1", "--output", str(output), "--replay-every", "4"])
        checkpoint = torch.load(output / "latest.pt", weights_only=True)
        controller = checkpoint["research"]["pool"]["controllers"]["policy"]
        self.assertEqual(controller["provenance"]["type"], "self_play")
        self.assertIn("rng_state", controller)
        self.assertNotIn("model", controller)
        replay = next((output / "replays").glob("*.jsonl"))
        with replay.open() as source:
            header = json.loads(source.readline())
        self.assertEqual(header["training"]["opponent"]["controller"]["type"], "self_play")
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--resume", str(output / "latest.pt"), "--library", self.library,
                  "--updates", "1", "--output", str(output)])
        resumed = torch.load(output / "latest.pt", weights_only=True)
        self.assertEqual(resumed["update"], 2)
        self.assertEqual(resumed["samples"], 24)
        rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        self.assertTrue(all(row["evaluations"]["self_play"]["episodes"] == 4 for row in rows))
        self.assertTrue(all(np.isfinite(row["loss"]) for row in rows))
        for row in rows:
            np.testing.assert_allclose(sum(row["reward_types"].values()), row["reward_total"],
                                       rtol=1e-6, atol=2e-5)

    def test_fixed_loadout_training_evaluation_and_replay(self):
        plan = example_plan()
        plan["stages"] = [plan["stages"][1]]
        plan["stages"][0]["advance"] = {"updates": 1}
        plan["stages"][0]["equipment"] = {
            "sample": "episode", "players": "independent",
            "loadout": {"weapons": ["minecraft:iron_sword", "minecraft:iron_axe"],
                        "selected": 0, "offhand": "minecraft:shield"},
        }
        plan["evaluations"] = {"combat_check": {
            "stage": "combat", "every_updates": 1, "episodes": 4, "seed": 1729,
        }}
        self.path.write_text(json.dumps(plan))
        output = self.directory / "fixed-loadout-smoke"
        with contextlib.redirect_stdout(io.StringIO()):
            main([*self.flags, "--updates", "1", "--output", str(output), "--replay-every", "4",
                  "--max-ticks", "32", "--rollout-steps", "32"])
        checkpoint = torch.load(output / "latest.pt", weights_only=True)
        self.assertTrue(checkpoint["research"]["completed"])
        self.assertEqual(checkpoint["metadata"]["controls"], {"fixed": {}})
        self.assertEqual(checkpoint["metadata"]["experiment"]["name"], plan["name"])
        metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["stage"], "combat")
        self.assertEqual(metrics[0]["evaluations"]["combat_check"]["episodes"], 4)
        self.assertTrue(np.isfinite(metrics[0]["loss"]))
        np.testing.assert_allclose(sum(metrics[0]["reward_types"].values()),
                                   metrics[0]["reward_total"], rtol=1e-6, atol=2e-5)
        from simulator.training.dashboard import RunStore
        store = RunStore(self.directory / "unused", [output])
        rows = store.read(store.list_runs()[0]["id"])["metrics"]
        self.assertEqual(rows[0]["reward_types"], metrics[0]["reward_types"])
        replay = next((output / "replays").glob("*.jsonl"))
        with replay.open() as source:
            header = json.loads(source.readline())
        self.assertEqual(header["training"]["stage"], "combat")
        for player in header["environment"]["players"]:
            self.assertEqual(player["loadout"]["weapons"][:2], ["minecraft:iron_sword", "minecraft:iron_axe"])
            self.assertEqual(player["loadout"]["offhand"], "minecraft:shield")

    def test_reward_breakdown_tracks_learner_sides_across_groups_and_resets(self):
        plan = resolve_plan(short_plan())
        stage = plan["stages"][0]
        scenario = stage["scenario"]
        scenario.pop("spawns", None)
        scenario.pop("equipment", None)
        scene = scenario["arenas"][0]["environment"]
        for index, player in enumerate(scene["players"]):
            player.update(position=[0, 1, -1 if index == 0 else 1], yaw=0, pitch=0)
        scenario["arenas"].append(deepcopy(scenario["arenas"][0]))
        stage["rewards"]["weights"] = dict.fromkeys(stage["rewards"]["weights"], 0.0)
        stage["rewards"]["weights"]["good_aim"] = 6
        stage["controls"]["fixed"].update(yaw=0, pitch=0)
        worlds = SampledWorlds(scenario, plan["policy"], 8, "cpu", self.library, 2, 42)
        self.addCleanup(worlds.close)
        self.assertEqual(len(worlds.groups), 2)
        worlds.configure(stage)
        pool = OpponentPool(plan["opponents"], plan["policy"], 8, "cpu", 42)
        pool.configure(stage["opponents"])
        pool.reset(np.ones(8, dtype=bool))
        model = ActorCritic(worlds.observation_dim, 8,
                            action_sizes=dict(zip(worlds.action_names, plan["policy"]["action_sizes"])))
        fixed = fixed_indices(plan["policy"], stage["controls"])
        observations = worlds.observe()
        selected = []
        original_act = pool.act

        def record_sides(observations, actions):
            selected.append(pool.learner_slots.copy())
            return original_act(observations, actions)

        for _ in range(2):
            selected.clear()
            with patch.object(pool, "act", side_effect=record_sides):
                rollout, observations, _, _ = collect(
                    model, worlds, pool, observations, 3, torch.device("cpu"), fixed)
            expected = 3 * sum(int((slots == 0).sum()) for slots in selected)
            self.assertGreater(expected, 0)
            self.assertLess(expected, 8 * 3 * 3)
            self.assertEqual(rollout["reward_types"]["good_aim"], expected)
            np.testing.assert_allclose(sum(rollout["reward_types"].values()),
                                       rollout["rewards"].sum(dtype=torch.float64).item(), atol=1e-6)

    def test_bridge_does_not_request_duel_reward_breakdown(self):
        worlds = SampledWorlds.__new__(SampledWorlds)
        worlds.groups = [(np.array([0]), SimpleNamespace(bridge={}))]
        self.assertIsNone(worlds.step_reward_totals(np.array([0])))

    def test_missing_native_breakdown_warns_without_fabricating_contributions(self):
        output = self.directory / "old-library"
        messages = io.StringIO()
        with patch("simulator.training.research.Environment.step_reward_totals", return_value=None), \
                contextlib.redirect_stdout(messages):
            main([*self.flags, "--updates", "1", "--output", str(output)])
        self.assertIn("Reward breakdown unavailable: rebuild minecraft_sim_training", messages.getvalue())
        row = json.loads((output / "metrics.jsonl").read_text().splitlines()[0])
        self.assertNotIn("reward_types", row)
        self.assertTrue(np.isfinite(row["reward_total"]))

    def test_reset_reuses_observations_for_reset_groups(self):
        plan = resolve_plan(short_plan())
        stage = plan["stages"][0]
        stage["scenario"]["arenas"].append(deepcopy(stage["scenario"]["arenas"][0]))
        worlds = SampledWorlds(stage["scenario"], plan["policy"], 8, "cpu", self.library, 2, 42)
        self.addCleanup(worlds.close)
        self.assertEqual(len(worlds.groups), 2)
        worlds.configure(stage)
        partial = np.zeros(8, dtype=bool)
        partial[worlds.groups[0][0][0]] = True
        for mask in (np.zeros(8, dtype=bool), partial, np.ones(8, dtype=bool)):
            with self.subTest(mask=mask.tolist()), contextlib.ExitStack() as stack:
                spies = [(ids,
                    stack.enter_context(patch.object(environment, "observe", wraps=environment.observe)),
                    stack.enter_context(patch.object(environment, "reset_players", wraps=environment.reset_players)))
                    for ids, environment in worlds.groups]
                result = worlds.reset(mask)
                for ids, observe, reset in spies:
                    selected = mask[ids].any()
                    self.assertEqual(observe.call_count, 0 if selected else 1)
                    self.assertEqual(reset.call_count, 1 if selected else 0)
            np.testing.assert_array_equal(result, worlds.observe())

    def test_learner_only_collection_and_selective_resets(self):
        plan = resolve_plan(short_plan())
        worlds = SampledWorlds(plan["scenario"], plan["policy"], 4, "cpu", self.library, 2, 42)
        self.addCleanup(worlds.close)
        worlds.configure(plan["stages"][0])
        pool = OpponentPool(plan["opponents"], plan["policy"], 4, "cpu", 42)
        pool.configure(plan["stages"][0]["opponents"])
        pool.reset(np.ones(4, dtype=bool))
        model = ActorCritic(worlds.observation_dim, 8,
                            action_sizes=dict(zip(worlds.action_names, plan["policy"]["action_sizes"])))
        fixed = fixed_indices(plan["policy"], plan["stages"][0]["controls"])
        addresses = [environment._handle for _, environment in worlds.groups]
        for _, environment in worlds.groups:
            with self.assertRaisesRegex(ValueError, "Reset mask"):
                environment.reset_players([None] * environment.count, np.full(environment.count, 0.5))
        before, _ = worlds.episode(0)
        other = worlds.observe()[1:].copy()
        worlds.reset(np.array([True, False, False, False]))
        after, _ = worlds.episode(0)
        self.assertEqual([p["loadout"] for p in before["players"]], [p["loadout"] for p in after["players"]])
        np.testing.assert_array_equal(other, worlds.observe()[1:])
        rollout, _, _, _ = collect(model, worlds, pool, worlds.observe(), 3, torch.device("cpu"), fixed)
        self.assertEqual(rollout["actions"].shape, (3, 4, 8))
        self.assertEqual(rollout["observations"].shape[1], 4)
        self.assertEqual(addresses, [environment._handle for _, environment in worlds.groups])
        loss = optimize(model, torch.optim.Adam(model.parameters()), rollout, 1, 8)
        self.assertTrue(np.isfinite(loss["loss"]))
        worlds.configure(plan["stages"][1])
        self.assertEqual(addresses, [environment._handle for _, environment in worlds.groups])

    def test_four_stage_training_evaluation_and_replay(self):
        plan = example_plan()
        aim, combat = plan["stages"]
        attack = {**deepcopy(combat), "name": "attack",
                  "controls": {"fixed": {"jump": False, "sprint": False, "weapon": 0, "shield": False}}}
        pursuit = {**deepcopy(combat), "name": "pursuit",
                   "controls": {"fixed": {"jump": False, "weapon": 0, "shield": False}}}
        plan["stages"] = [aim, attack, pursuit, combat]
        for stage in plan["stages"]:
            stage["advance"] = {"updates": 1}
        plan["evaluations"]["held_out_combat"].update(every_updates=1, episodes=4)
        self.path.write_text(json.dumps(plan))
        output = self.directory / "four-stage-smoke"
        with contextlib.redirect_stdout(io.StringIO()):
            main([*self.flags, "--updates", "4", "--output", str(output), "--replay-every", "4",
                  "--max-ticks", "32", "--rollout-steps", "32"])
        checkpoint = torch.load(output / "latest.pt", weights_only=True)
        self.assertTrue(checkpoint["research"]["completed"])
        self.assertEqual(checkpoint["update"], 4)
        metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([row["stage"] for row in metrics], [stage["name"] for stage in plan["stages"]])
        self.assertEqual(metrics[-1]["evaluations"]["held_out_combat"]["episodes"], 4)
        replay_stages = set()
        for path in (output / "replays").glob("*.jsonl"):
            with path.open() as replay:
                replay_stages.add(json.loads(replay.readline())["training"]["stage"])
        self.assertEqual(replay_stages, set(row["stage"] for row in metrics))

    def test_training_resume_transition_replay_and_inference(self):
        output = self.directory / "run"
        with contextlib.redirect_stdout(io.StringIO()):
            main([*self.flags, "--updates", "1", "--output", str(output), "--replay-every", "4"])
        checkpoint = torch.load(output / "latest.pt", weights_only=True)
        self.assertEqual(checkpoint["research"]["progress"]["updates"], 1)
        self.assertEqual(checkpoint["samples"], 12)
        replay = next((output / "replays").glob("*.jsonl"))
        frames = [json.loads(line) for line in replay.read_text().splitlines()]
        self.assertEqual(frames[0]["training"]["stage"], "aim")
        self.assertIn("loadout", frames[0]["environment"]["players"][0])
        self.assertTrue(frames[-1]["terminated"] or frames[-1]["truncated"])
        self.path.unlink()
        flags = self.flags[2:]
        with contextlib.redirect_stdout(io.StringIO()):
            main([*flags, "--updates", "1", "--output", str(output), "--resume", str(output / "latest.pt")])
        checkpoint = torch.load(output / "latest.pt", weights_only=True)
        self.assertEqual(checkpoint["research"]["stage_index"], 1)
        self.assertEqual(checkpoint["research"]["progress"]["updates"], 0)
        self.assertEqual(checkpoint["update"], 2)
        policy = Policy(output / "latest.pt", deterministic=True)
        response = policy.act({"protocol": 1, "session": "research", "tick": 0,
            "contract_hash": policy.info["contract_hash"],
            "observations": np.zeros((2, policy.schema["observation_dim"])).tolist()})
        self.assertEqual(np.asarray(response["actions"]).shape, (2, 8))
        with contextlib.redirect_stdout(io.StringIO()):
            main([*flags, "--updates", "5", "--output", str(output), "--resume", str(output / "latest.pt")])
        checkpoint = torch.load(output / "latest.pt", weights_only=True)
        self.assertTrue(checkpoint["research"]["completed"])
        self.assertEqual(checkpoint["update"], 4)
        with self.assertRaisesRegex(ValueError, "complete"):
            main([*flags, "--updates", "1", "--output", str(output), "--resume", str(output / "latest.pt")])

    def test_multiscene_groups_and_held_out_evaluation(self):
        plan = resolve_plan(short_plan())
        arena = plan["scenario"]["arenas"][0]["environment"]
        second = deepcopy(arena)
        second["name"] = "second-scene"
        plan["scenario"]["arenas"].append({"environment": second, "weight": 1})
        worlds = SampledWorlds(plan["scenario"], plan["policy"], 16, "cpu", self.library, 2, 42)
        self.addCleanup(worlds.close)
        self.assertEqual(len(worlds.groups), 2)
        model = ActorCritic(worlds.observation_dim, 8,
                            action_sizes=dict(zip(worlds.action_names, plan["policy"]["action_sizes"])))
        suite = {"scenario": plan["scenario"], "episodes": 3, "seed": 12, "controls": {"fixed": {}},
                 "opponents": [{"id": "stationary"}]}
        options = arguments(self.flags)
        result = evaluate(model, plan, suite, options, torch.device("cpu"), {})
        self.assertEqual(result["episodes"], 3)
        self.assertEqual(result["win_rate"], 0)
        self.assertTrue(np.isfinite(result["reward_mean"]))

    def test_heldout_styles_never_train_and_preserve_roaming_rng(self):
        raw = short_plan()
        raw["opponents"]["walker"] = {"type": "scripted", "controller": "random_walk",
                                       "options": {"change_interval": 2}}
        raw["mixtures"]["roaming"] = [{"id": "walker", "weight": 1}]
        styles = ("evasive", "aggressive", "patient")
        for style in styles:
            raw["opponents"][style] = {"type": "scripted", "controller": "tactical_melee",
                                       "options": {"style": style}}
            raw["mixtures"][style] = [{"id": style, "weight": 1}]
        for stage in raw["stages"]:
            stage["opponents"] = "roaming"
        self.path.write_text(json.dumps(raw))
        baseline = self.directory / "unmonitored"
        monitored = self.directory / "heldout-styles"
        with contextlib.redirect_stdout(io.StringIO()):
            main([*self.flags, "--updates", "2", "--output", str(baseline)])
        raw["evaluations"] = {style: {
            "stage": "combat", "every_updates": 1, "worlds": 2, "episodes": 3, "seed": 100 + index,
            "overrides": {"opponents": style},
        } for index, style in enumerate(styles)}
        resolved = resolve_plan(raw)
        for suite in resolved["evaluations"].values():
            self.assertEqual(suite["controls"], resolved["stages"][1]["controls"])
            self.assertNotEqual(suite["controls"], resolved["stages"][0]["controls"])
        self.path.write_text(json.dumps(raw))
        with contextlib.redirect_stdout(io.StringIO()):
            main([*self.flags, "--updates", "2", "--output", str(monitored)])
        first = torch.load(baseline / "latest.pt", weights_only=True)
        second = torch.load(monitored / "latest.pt", weights_only=True)
        for name in first["model"]:
            torch.testing.assert_close(first["model"][name], second["model"][name], rtol=0, atol=0)
        self.assertEqual(first["research"]["pool"], second["research"]["pool"])
        for line in (monitored / "metrics.jsonl").read_text().splitlines():
            row = json.loads(line)
            self.assertEqual(set(row["evaluations"]), set(styles))
            self.assertEqual(set(row["training_opponents"]), {"walker"})
            for result in row["evaluations"].values():
                self.assertEqual(result["episodes"], 3)
                self.assertEqual(result["worlds"], 2)
                self.assertGreaterEqual(result["win_rate"], 0)
                self.assertLessEqual(result["win_rate"], 1)

    def test_evaluation_counts_real_combat_wins(self):
        plan = resolve_plan(short_plan())
        definition = plan["scenario"]["arenas"][0]["environment"]
        for index, player in enumerate(definition["players"]):
            player.update(position=[0.5, 1.0, 0.5 + 2 * index], yaw=180 * index,
                          pitch=0, health=1, attack_strength_ticks=20)
        plan["scenario"] = {"arenas": [{"environment": definition, "weight": 1}]}
        contract = plan["policy"]
        model = ActorCritic(contract["observation_dim"], 8,
                            action_sizes=dict(zip(contract["action_names"], contract["action_sizes"])))
        suite = {"scenario": plan["scenario"], "episodes": 3, "seed": 12,
                 "opponents": [{"id": "stationary"}],
                 "controls": {"fixed": {"movement": "neutral", "jump": False, "sprint": False,
                                        "attack": True, "yaw": 0, "pitch": 0, "weapon": 0, "shield": False}}}
        options = arguments([*self.flags, "--max-ticks", "30"])
        result = evaluate(model, plan, suite, options, torch.device("cpu"), {})
        self.assertEqual(result["win_rate"], 1)

    def test_plan_changes_rejected_but_weight_initialization_supported(self):
        output = self.directory / "first"
        with contextlib.redirect_stdout(io.StringIO()):
            main([*self.flags, "--updates", "1", "--output", str(output)])
        changed = short_plan()
        changed["stages"][0]["advance"]["updates"] = 5
        self.path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "differs"):
            main([*self.flags, "--updates", "1", "--resume", str(output / "latest.pt"),
                  "--output", str(output)])
        with contextlib.redirect_stdout(io.StringIO()):
            main([*self.flags, "--updates", "1", "--initialize", str(output / "latest.pt"),
                  "--output", str(self.directory / "initialized")])

    def test_evaluation_replays_do_not_change_training_and_evaluation_advances(self):
        baseline, monitored = self.directory / "baseline", self.directory / "monitored"
        with contextlib.redirect_stdout(io.StringIO()):
            main([*self.flags, "--updates", "2", "--output", str(baseline)])
        changed = short_plan()
        changed["evaluations"] = {
            "held": {"every_updates": 1, "episodes": 3, "seed": 101,
                     "environment": changed["environment"], "opponents": "target"}
        }
        changed["stages"][0]["evaluations"] = {
            "local": {"every_updates": 1, "worlds": 5, "episodes": 5, "seed": 102}
        }
        self.path.write_text(json.dumps(changed))
        with contextlib.redirect_stdout(io.StringIO()):
            main([*self.flags, "--updates", "2", "--output", str(monitored), "--replay-every", "4"])
        first = torch.load(baseline / "latest.pt", weights_only=True)
        second = torch.load(monitored / "latest.pt", weights_only=True)
        for name in first["model"]:
            torch.testing.assert_close(first["model"][name], second["model"][name], rtol=0, atol=0)
        changed["stages"][0]["advance"] = {
            "evaluation": {"suite": "held", "metric": "win_rate", "at_least": 0}
        }
        self.path.write_text(json.dumps(changed))
        target = self.directory / "evaluation-advance"
        with contextlib.redirect_stdout(io.StringIO()):
            main([*self.flags, "--updates", "1", "--output", str(target)])
        result = torch.load(target / "latest.pt", weights_only=True)
        self.assertEqual(result["research"]["stage_index"], 1)

    def test_frozen_checkpoint_pool_training_and_identity_guard(self):
        source = self.directory / "source"
        with contextlib.redirect_stdout(io.StringIO()):
            main([*self.flags, "--updates", "1", "--output", str(source)])
        raw = short_plan()
        raw["opponents"]["frozen"] = {"type": "checkpoint", "path": str(source / "latest.pt")}
        raw["mixtures"]["frozen_and_fighter"] = [{"id": "frozen", "weight": 1}, {"id": "fighter", "weight": 1}]
        for stage in raw["stages"]:
            stage["opponents"] = "frozen_and_fighter"
        self.path.write_text(json.dumps(raw))
        output = self.directory / "pool"
        with contextlib.redirect_stdout(io.StringIO()):
            main([*self.flags, "--updates", "2", "--output", str(output)])
        saved = torch.load(output / "latest.pt", weights_only=True)
        self.assertEqual(saved["samples"], 24)
        self.assertEqual(len(saved["research"]["plan"]["opponents"]["frozen"]["sha256"]), 64)
        with (source / "latest.pt").open("ab") as changed:
            changed.write(b" ")
        with self.assertRaisesRegex(ValueError, "checkpoint changed"):
            main([*self.flags[2:], "--updates", "1", "--resume", str(output / "latest.pt"),
                  "--output", str(output)])


if __name__ == "__main__":
    unittest.main()
