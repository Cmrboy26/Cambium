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
from simulator.training.contract import from_legacy, resolve_scenario
from simulator.training.environment import ROOT, SCHEMA, SCHEMA_V2, load_library
from simulator.training.legacy import SCHEMA as LEGACY_SCHEMA, SCHEMA_V2 as LEGACY_V2
from simulator.training.plans import EpisodeSampler, load_plan, plan_hash, reached, resolve_plan
from simulator.training.policies import fixed_indices, policy_actions
from simulator.training.research import OpponentPool, SampledWorlds, collect, evaluate
from simulator.training.serve import Policy
from simulator.training.train import arguments, main, optimize


PRESET = ROOT / "simulator/setups/curricula/gear-curriculum.json"
SHIELD_AXE_PRESET = ROOT / "simulator/setups/curricula/shield-axe-curriculum.json"
AXE_SHIELD_FIGHTING_PRESET = ROOT / "simulator/setups/curricula/axe-shield-fighting.json"


def preset(path=PRESET):
    value = json.loads(path.read_text(encoding="utf-8"))
    for key in ("policy", "environment"):
        value[key] = str((path.parent / value[key]).resolve())
    for suite in value["evaluations"].values():
        suite["environment"] = str((path.parent / suite["environment"]).resolve())
    return value


def short_plan():
    value = preset()
    value["evaluations"] = {}
    for stage in value["stages"]:
        stage["advance"] = {"updates": 2}
    return value


class PlanChecks(unittest.TestCase):
    def test_axe_shield_fighting_has_fixed_gear_and_direct_combat(self):
        plan = load_plan(AXE_SHIELD_FIGHTING_PRESET)
        self.assertEqual(len(plan["stages"]), 1)
        stage = plan["stages"][0]
        self.assertEqual(stage["name"], "combat")
        self.assertEqual(stage["advance"], {"updates": 1500})
        self.assertEqual(stage["controls"], {"fixed": {}})
        self.assertEqual(plan["opponents"]["policy"], {"type": "self_play"})
        self.assertTrue(all(opponent["type"] == "scripted" and opponent["controller"] == "basic_melee"
                            for name, opponent in plan["opponents"].items() if name != "policy"))
        self.assertEqual(stage["opponents"], [
            {"id": "shield_fighter", "weight": 1}, {"id": "axe_fighter", "weight": 1},
            {"id": "shield_axe_fighter", "weight": 2}, {"id": "policy", "weight": 4}])
        evaluation = plan["evaluations"]["axe_shield_combat"]
        self.assertEqual((evaluation["every_updates"], evaluation["episodes"]), (50, 64))
        self.assertEqual(evaluation["opponents"], stage["opponents"][:-1])
        for condition in (stage, evaluation):
            scenario = condition["scenario"]
            self.assertNotIn("equipment", scenario)
            self.assertEqual(scenario["spawns"]["sample"], "episode")
            self.assertEqual(scenario["arenas"][0]["environment"]["max_ticks"], 300)
            sampler = EpisodeSampler(scenario, 8, plan["seed"])
            initial_positions = []
            next_positions = []
            for world in range(8):
                for positions in (initial_positions, next_positions):
                    players = sampler.players(world)
                    positions.append([player["position"] for player in players])
                    self.assertGreaterEqual(np.linalg.norm(np.subtract(
                        players[0]["position"], players[1]["position"])), 2)
                    for player in players:
                        self.assertEqual(player["loadout"]["weapons"],
                                         ["minecraft:iron_sword", "minecraft:iron_axe"] + ["minecraft:air"] * 7)
                        self.assertEqual(player["loadout"]["selected"], 0)
                        self.assertEqual(player["loadout"]["offhand"], "minecraft:shield")
                        self.assertTrue(all(item == "minecraft:air" for item in player["loadout"]["armor"].values()))
            self.assertNotEqual(initial_positions, next_positions)

    def test_shield_axe_curriculum_episode_limits(self):
        plan = resolve_plan(preset(SHIELD_AXE_PRESET))
        self.assertEqual([stage["scenario"]["arenas"][0]["environment"]["max_ticks"]
                  for stage in plan["stages"]], [100, 300, 600, 1200])
        self.assertEqual(plan["evaluations"]["shield_axe_combat"]["scenario"]["arenas"][0]
                 ["environment"]["max_ticks"], 1200)

    def test_shield_axe_curriculum_budgets_and_spawn_equipment(self):
        plan = resolve_plan(preset(SHIELD_AXE_PRESET))
        self.assertEqual([stage["advance"]["updates"] for stage in plan["stages"]], [100, 200, 200, 1000])
        self.assertEqual(sum(stage["advance"]["updates"] for stage in plan["stages"]), 1500)
        self.assertEqual(plan["opponents"]["policy"], {"type": "self_play"})
        mixture = plan["stages"][-1]["opponents"]
        self.assertEqual(mixture[-1], {"id": "policy", "weight": 10})
        self.assertEqual(sum(entry["weight"] for entry in mixture[:-1]), 10)
        self.assertEqual(plan["evaluations"]["shield_axe_combat"]["opponents"], mixture[:-1])
        for stage in plan["stages"][:-1]:
            self.assertNotIn("policy", [entry["id"] for entry in stage["opponents"]])
        scenario = plan["stages"][1]["scenario"]
        sampler = EpisodeSampler(scenario, 16, 42)
        for world in range(16):
            players = sampler.players(world)
            distance = np.linalg.norm(np.subtract(players[0]["position"], players[1]["position"]))
            self.assertGreaterEqual(distance, 2.0)
            self.assertLess(distance, 2.7)
            for player in players:
                self.assertEqual(player["loadout"]["weapons"][0], "minecraft:iron_sword")
                self.assertIn(player["loadout"]["weapons"][1], ("minecraft:air", "minecraft:iron_axe"))
                self.assertTrue(all(item == "minecraft:air" for item in player["loadout"]["armor"].values()))
        model_slots = plan["policy"]["action_sizes"][plan["policy"]["action_names"].index("weapon")]
        self.assertEqual(model_slots, 9)
        for stage in plan["stages"]:
            self.assertEqual(stage["scenario"]["arenas"][0]["environment"]["blocks"],
                             plan["scenario"]["arenas"][0]["environment"]["blocks"])
            self.assertEqual(stage["scenario"]["equipment"], plan["scenario"]["equipment"])
        self.assertNotIn("movement", plan["stages"][1]["controls"]["fixed"])

    def test_legacy_exports_and_roundtrip(self):
        self.assertIs(SCHEMA, LEGACY_SCHEMA)
        self.assertIs(SCHEMA_V2, LEGACY_V2)
        self.assertEqual(from_legacy(SCHEMA)["observation_dim"], 22)
        self.assertEqual(from_legacy(SCHEMA_V2)["observation_dim"], 31)
        plan = resolve_plan(preset())
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


class ResearchChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.directory = ROOT / "simulator/training/runs" / f"test-research-{uuid4().hex}"
        self.directory.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.directory)
        self.path = self.directory / "plan.json"
        research_library = ROOT / "simulator/build-research/minecraft_sim_training.so"
        self.library = load_library(research_library if research_library.exists() else None)._build_info["library_path"]
        self.path.write_text(json.dumps(short_plan()))
        self.flags = ["--plan", str(self.path), "--library", self.library, "--envs", "4",
                      "--hidden-dim", "8", "--rollout-steps", "3", "--epochs", "1",
                      "--minibatch-size", "8", "--max-ticks", "2", "--save-every", "1"]

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

    def test_axe_shield_fighting_training_evaluation_and_replay(self):
        plan = preset(AXE_SHIELD_FIGHTING_PRESET)
        plan["stages"][0]["advance"] = {"updates": 1}
        plan["evaluations"]["axe_shield_combat"].update(every_updates=1, episodes=4)
        self.path.write_text(json.dumps(plan))
        output = self.directory / "axe-shield-fighting-smoke"
        with contextlib.redirect_stdout(io.StringIO()):
            main([*self.flags, "--updates", "1", "--output", str(output), "--replay-every", "4",
                  "--max-ticks", "32", "--rollout-steps", "32"])
        checkpoint = torch.load(output / "latest.pt", weights_only=True)
        self.assertTrue(checkpoint["research"]["completed"])
        self.assertEqual(checkpoint["metadata"]["controls"], {"fixed": {}})
        self.assertEqual(checkpoint["metadata"]["experiment"]["name"], "axe-shield-fighting")
        metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["stage"], "combat")
        self.assertEqual(metrics[0]["evaluations"]["axe_shield_combat"]["episodes"], 4)
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

    def test_shield_axe_preset_all_stages_smoke(self):
        plan = preset(SHIELD_AXE_PRESET)
        for stage in plan["stages"]:
            stage["advance"] = {"updates": 1}
        plan["evaluations"]["shield_axe_combat"].update(every_updates=1, episodes=4)
        self.path.write_text(json.dumps(plan))
        output = self.directory / "shield-axe-smoke"
        with contextlib.redirect_stdout(io.StringIO()):
            main([*self.flags, "--updates", "4", "--output", str(output), "--replay-every", "4",
                  "--max-ticks", "32", "--rollout-steps", "32"])
        checkpoint = torch.load(output / "latest.pt", weights_only=True)
        self.assertTrue(checkpoint["research"]["completed"])
        self.assertEqual(checkpoint["update"], 4)
        metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([row["stage"] for row in metrics], ["aim", "attack_timing", "pursuit", "shield_axe_combat"])
        self.assertEqual(metrics[-1]["evaluations"]["shield_axe_combat"]["episodes"], 4)
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
