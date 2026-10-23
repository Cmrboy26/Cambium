import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import torch

from backend.fast_rl_model import ActorCritic
from simulator.training.analyze_metrics import main as analyze_metrics
from simulator.training._test_plans import example_plan
from simulator.training.contract import from_legacy, resolve_players, resolve_scenario
from simulator.training.environment import ROOT, SCHEMA, SCHEMA_V2, load_library
from simulator.training.controllers import CheckpointController
from simulator.training.legacy import SCHEMA as LEGACY_SCHEMA, SCHEMA_V2 as LEGACY_V2
from simulator.training.plans import EpisodeSampler, plan_hash, reached, resolve_plan
from simulator.training.policies import fixed_indices, policy_actions
from simulator.training.research import EpisodeMetrics, OpponentPool, SampledWorlds, collect, evaluate
from simulator.training.serve import Policy
from simulator.training.train import arguments, main, optimize
from simulator.training.timing import CollectionTimer


def short_plan():
    value = example_plan()
    value["evaluations"] = {}
    for stage in value["stages"]:
        stage["advance"] = {"updates": 2}
    return value


class EpisodeMetricChecks(unittest.TestCase):
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
        self.library = str(load_library()._name)
        self.flags = ["--plan", str(self.path), "--library", self.library, "--envs", "4", "--hidden-dim", "8",
                      "--rollout-steps", "3", "--epochs", "1", "--minibatch-size", "8", "--save-every", "1"]

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


class ResearchChecks(unittest.TestCase):
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
            log_probs, values, _ = model.evaluate_policy_actions(observations, actions)
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
