from copy import deepcopy
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np

from simulator.training.environment import REWARD_NAMES, ROOT
from simulator.training.plans import EpisodeSampler, load_plan, plan_hash, resolve_plan, scene_key


class ExperimentChecks(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / "simulator/training/runs" / f"test-experiments-{uuid4().hex}"
        self.directory.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.directory)
        for name in ("arenas", "environments", "policies", "curricula"):
            (self.directory / name).mkdir()
        self.arena_path = self.directory / "arenas/flat.json"
        self.environment_path = self.directory / "environments/duel.json"
        self.policy_path = self.directory / "policies/combat.json"
        self.plan_path = self.directory / "curricula/train.json"
        self.arena = json.loads((ROOT / "simulator/setups/arenas/flat-duel.json").read_text())
        self.environment = json.loads((ROOT / "simulator/setups/environments/gear-duel.json").read_text())
        self.policy = json.loads((ROOT / "simulator/setups/policies/equipment-combat.json").read_text())
        self.raw = json.loads((ROOT / "simulator/setups/curricula/gear-curriculum.json").read_text())
        self.environment["arena"] = "../arenas/flat.json"
        self.raw["environment"] = "../environments/duel.json"
        self.raw["policy"] = "../policies/combat.json"
        self.raw.pop("learner")
        for suite in self.raw["evaluations"].values():
            suite["environment"] = "../environments/duel.json"
        self.write()

    def write(self):
        for path, value in ((self.arena_path, self.arena), (self.environment_path, self.environment),
                            (self.policy_path, self.policy), (self.plan_path, self.raw)):
            path.write_text(json.dumps(value), encoding="utf-8")

    def resolve(self):
        return resolve_plan(self.raw, base_dir=self.plan_path.parent)

    def test_explicit_files_and_canonical_runtime_contract(self):
        resolved = self.resolve()
        self.assertEqual(resolved["format"], "minecraft_sim_experiment")
        self.assertEqual(resolved["version"], 1)
        self.assertEqual(resolved["name"], "gear-curriculum")
        self.assertEqual(resolved["learner"],
                         {"hidden_dim": 256, "lr": 0.0003, "epochs": 4, "minibatch_size": 1024})
        self.assertEqual(resolved["policy"]["observation_dim"], 78)
        self.assertEqual(resolved["policy"]["action_sizes"], [9, 2, 2, 2, 9, 5, 9, 2])
        self.assertEqual(resolved["stages"][0]["opponents"], [{"id": "stationary", "weight": 1}])
        self.assertEqual(resolved["scenario"], resolved["stages"][0]["scenario"])
        for condition in [*resolved["stages"], *resolved["evaluations"].values()]:
            self.assertEqual(condition["environment_name"], self.environment["name"])
            self.assertEqual(condition["arena_names"], [self.arena["name"]])
            self.assertEqual(condition["rewards"]["mode"], "replace")
            self.assertEqual(set(condition["rewards"]["weights"]), set(REWARD_NAMES))
            scene = condition["scenario"]["arenas"][0]["environment"]
            self.assertEqual(scene["version"], 1)
            self.assertEqual(scene["max_ticks"], self.environment["task"]["max_ticks"])
            self.assertNotIn("policy", scene)
            for player in scene["players"]:
                self.assertEqual(len(player["loadout"]["weapons"]), 9)
            self.assertEqual(len(condition["scenario"]["equipment"]["loadout"]["weapons"]), 9)
        self.assertEqual(resolved["evaluations"]["held_out_combat"]["seed"], 1729)
        self.assertEqual(self.policy["actions"][-2]["values"], list(range(9)))

    def test_reference_directories_and_once_only_source_reads(self):
        original = Path.read_text
        calls = []
        def read(path, *args, **kwargs):
            calls.append(path.resolve())
            return original(path, *args, **kwargs)
        with patch.object(Path, "read_text", read):
            result = load_plan(self.plan_path)
        self.assertEqual(result, self.resolve())
        self.assertEqual(calls.count(self.arena_path), 1)
        self.assertEqual(calls.count(self.environment_path), 1)
        self.assertEqual(calls.count(self.policy_path), 1)
        with self.assertRaises(FileNotFoundError):
            resolve_plan(self.raw)

    def test_resolved_roundtrip_has_no_source_file_dependency(self):
        resolved = self.resolve()
        for path in (self.arena_path, self.environment_path, self.policy_path, self.plan_path):
            path.unlink()
        restored = resolve_plan(json.loads(json.dumps(resolved)), base_dir=self.directory / "unrelated")
        self.assertEqual(plan_hash(resolved), plan_hash(restored))
        self.assertEqual(resolved, restored)

    def test_rewards_never_import_global_or_previous_stage_defaults(self):
        self.environment["task"]["rewards"] = {"doing_damage": 10, "blocked_with_shield": 2}
        self.raw["stages"][0]["task_override"] = {"rewards": {"good_aim": 5}}
        self.raw["stages"].append({
            "name": "zero", "opponents": "target", "advance": {"updates": 1},
            "task_override": {"max_ticks": 77, "rewards": {}},
        })
        self.write()
        resolved = self.resolve()
        aim, combat, zero = resolved["stages"]
        self.assertEqual(aim["rewards"]["weights"]["good_aim"], 5)
        self.assertEqual(aim["rewards"]["weights"]["doing_damage"], 0)
        self.assertEqual(aim["rewards"]["weights"]["episode_timeout"], 0)
        self.assertEqual(combat["rewards"]["weights"]["doing_damage"], 10)
        self.assertEqual(combat["rewards"]["weights"]["good_aim"], 0)
        self.assertTrue(all(value == 0 for value in zero["rewards"]["weights"].values()))
        self.assertEqual(zero["scenario"]["arenas"][0]["environment"]["max_ticks"], 77)
        self.environment["task"].pop("rewards")
        self.raw["stages"][0].pop("task_override")
        self.write()
        self.assertTrue(all(value == 0 for value in self.resolve()["stages"][0]["rewards"]["weights"].values()))

    def test_stage_sampling_patches_arrays_replace_and_null_disables(self):
        stage = self.raw["stages"][0]
        stage["spawns"] = {"region": {"min": [-1, 1, -1]}, "yaw": 0}
        stage["equipment"] = {"loadout": {"weapons": ["minecraft:stone_sword"],
                                          "offhand": "minecraft:air"}}
        self.raw["stages"][1].update(spawns=None, equipment=None)
        resolved = self.resolve()
        first, second = resolved["stages"]
        self.assertEqual(first["scenario"]["spawns"]["region"],
                         {"min": [-1, 1, -1], "max": [4.5, 1, 4.5]})
        self.assertEqual(first["scenario"]["spawns"]["minimum_separation"], 2)
        self.assertEqual(first["scenario"]["spawns"]["yaw"], 0)
        slots = first["scenario"]["equipment"]["loadout"]["weapons"]
        self.assertEqual(slots, ["minecraft:stone_sword", *(["minecraft:air"] * 8)])
        self.assertNotIn("spawns", second["scenario"])
        self.assertNotIn("equipment", second["scenario"])
        self.assertEqual(self.environment["spawns"]["region"]["min"], [-4.5, 1, -4.5])

    def test_per_player_defaults_and_policy_are_independent(self):
        self.environment["equipment"] = None
        self.environment["spawns"] = None
        self.environment["players"]["overrides"][1]["loadout"] = {
            "weapons": ["minecraft:wooden_sword"], "selected": 8,
            "armor": {"helmet": "minecraft:turtle_helmet"},
        }
        self.write()
        result = self.resolve()
        players = EpisodeSampler(result["scenario"], 1, 12).players(0)
        self.assertEqual(players[0]["loadout"]["weapons"][0], "minecraft:iron_sword")
        self.assertEqual(players[1]["loadout"]["weapons"],
                         ["minecraft:wooden_sword", *(["minecraft:air"] * 8)])
        self.assertEqual(players[1]["loadout"]["selected"], 8)
        self.assertEqual(players[1]["loadout"]["armor"]["helmet"], "minecraft:turtle_helmet")
        self.assertEqual(result["policy"]["actions"][-2]["values"], list(range(9)))
        self.assertEqual(self.policy["actions"][-2]["values"], list(range(9)))

    def test_world_equipment_is_independent_and_stable_across_task_changes(self):
        resolved = self.resolve()
        first, second = resolved["stages"]
        self.assertEqual(scene_key(first["scenario"]), scene_key(second["scenario"]))
        sampler = EpisodeSampler(first["scenario"], 64, 101)
        gear = [[player["loadout"] for player in sampler.players(world)] for world in range(64)]
        self.assertTrue(any(players[0] != players[1] for players in gear))
        state = deepcopy(sampler.gear_rng.bit_generator.state)
        sampler.configure(second["scenario"])
        self.assertEqual(gear, [[player["loadout"] for player in sampler.players(world)] for world in range(64)])
        self.assertEqual(state, sampler.gear_rng.bit_generator.state)
        changed = deepcopy(second["scenario"])
        changed["arenas"][0]["environment"]["players"][0]["loadout"]["armor"]["helmet"] = "minecraft:turtle_helmet"
        sampler.configure(changed)
        self.assertTrue(all(cache["equipment"] is None for cache in sampler.cache))
        self.assertEqual(sampler.players(0)[0]["loadout"]["armor"]["helmet"], "minecraft:turtle_helmet")

    def test_literal_sampling_consumes_no_rng(self):
        self.environment["equipment"]["loadout"] = {
            "weapons": ["minecraft:iron_sword", "minecraft:air"],
            "selected": 8, "offhand": "minecraft:air",
        }
        self.environment["spawns"] = {"sample": "episode", "yaw": 0, "pitch": 0}
        self.write()
        sampler = EpisodeSampler(self.resolve()["scenario"], 2, 1)
        before = sampler.state_dict()["rng"]
        players = sampler.players(0)
        sampler.players(1)
        self.assertEqual(before, sampler.state_dict()["rng"])
        self.assertTrue(all(len(player["loadout"]["weapons"]) == 9 for player in players))

    def test_tight_spawn_region_does_not_exhaust_random_retries(self):
        self.environment["spawns"] = {
            "sample": "episode", "region": {"min": [-0.1, 1, -1.3], "max": [0.1, 1, 1.3]},
            "minimum_separation": 2.6,
        }
        self.write()
        scenario = self.resolve()["scenario"]
        sampler = EpisodeSampler(scenario, 1, 42)
        for _ in range(4):
            players = sampler.players(0)
            positions = np.asarray([player["position"] for player in players])
            self.assertGreaterEqual(np.linalg.norm(positions[0] - positions[1]), 2.6)
            self.assertTrue(np.all(positions >= [-0.1, 1, -1.3]))
            self.assertTrue(np.all(positions <= [0.1, 1, 1.3]))
            self.assertTrue(all(sampler._valid_position(position, scenario["arenas"][0]["environment"])
                                for position in positions))
        clone = EpisodeSampler(scenario, 1, 999)
        clone.load_state_dict(sampler.state_dict())
        self.assertEqual(sampler.players(0), clone.players(0))
        self.assertEqual(sampler.state_dict(), clone.state_dict())

    def test_impossible_spawn_region_still_fails(self):
        self.environment["spawns"] = {
            "sample": "episode", "region": {"min": [-0.1, 1, -1.3], "max": [0.1, 1, 1.3]},
            "minimum_separation": 3,
        }
        self.write()
        sampler = EpisodeSampler(self.resolve()["scenario"], 1, 42)
        with self.assertRaisesRegex(ValueError, "collision-free separated spawns"):
            sampler.players(0)

    def test_spawn_corner_fallback_rejects_blocked_positions(self):
        self.environment["spawns"] = {
            "sample": "episode", "region": {"min": [-0.1, 1, -1.3], "max": [0.1, 1, 1.3]},
            "minimum_separation": 2,
        }
        self.write()
        scenario = self.resolve()["scenario"]
        scenario["arenas"][0]["environment"]["blocks"].append({
            "from": [-1, 1, -2], "to": [0, 2, 1], "block": "minecraft:stone",
        })
        sampler = EpisodeSampler(scenario, 1, 42)
        with self.assertRaisesRegex(ValueError, "corner fallback.*world 0.*minimum_separation 2"):
            sampler.players(0)

    def test_checkpoint_opponents_are_the_only_reopened_runtime_files(self):
        checkpoint = self.directory / "opponent.pt"
        checkpoint.write_bytes(b"trusted-checkpoint-identity")
        self.raw["opponents"]["frozen"] = {"type": "checkpoint", "path": "../opponent.pt"}
        self.raw["mixtures"]["target"] = [{"id": "frozen", "weight": 1}]
        resolved = self.resolve()
        self.assertEqual(resolved["opponents"]["frozen"]["path"], str(checkpoint))
        self.assertEqual(len(resolved["opponents"]["frozen"]["sha256"]), 64)
        self.environment_path.unlink()
        self.assertEqual(resolved, resolve_plan(resolved))
        checkpoint.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "checkpoint changed"):
            resolve_plan(resolved)

    def test_rejects_legacy_and_untyped_authoring(self):
        mutations = [
            lambda raw: raw.update(version=1),
            lambda raw: raw.update(policy={"preset": "equipment_v3"}),
            lambda raw: raw.update(scenario={}),
            lambda raw: raw.update(extends="other.json"),
            lambda raw: raw["stages"][0].update(opponents=[{"id": "stationary"}]),
            lambda raw: raw["stages"][0].update(rewards={"mode": "environment"}),
            lambda raw: raw["stages"][0].update(task_override={"rules": {}}),
            lambda raw: raw["stages"][0].update(task_override={"rewards": {"won_duel": True}}),
            lambda raw: raw["stages"][0].update(task_override={"max_ticks": True}),
            lambda raw: raw["stages"][0].update(equipment={"loadout": {"weapons": [
                "minecraft:iron_sword", "minecraft:iron_sword"]}}),
            lambda raw: raw["stages"][0].update(spawns={"minimum_separation": 0.1}),
            lambda raw: raw["evaluations"]["held_out_combat"].pop("environment"),
            lambda raw: raw["evaluations"]["held_out_combat"].pop("seed"),
            lambda raw: raw.update(learner={"lr": float("inf")}),
        ]
        original = deepcopy(self.raw)
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.raw = deepcopy(original)
                mutation(self.raw)
                with self.assertRaises(ValueError):
                    self.resolve()
        self.raw = original

    def test_rejects_cross_concept_fields_unknown_capabilities_and_bad_player_states(self):
        cases = [
            (self.arena, lambda value: value.update(policy={})),
            (self.arena, lambda value: value.update(players={})),
            (self.arena, lambda value: value.update(requires=["teleport"])),
            (self.arena, lambda value: value["rules"].update(block_edits=True)),
            (self.arena, lambda value: value["rules"].update(natural_regeneration=True)),
            (self.arena, lambda value: value["bounds"].update(size=[129, 4, 15])),
            (self.environment, lambda value: value.update(policy={})),
            (self.environment, lambda value: value.update(reward_weights={})),
            (self.environment, lambda value: value["players"]["shared"].update(pitch=91)),
            (self.environment, lambda value: value["players"]["shared"].update(health=False)),
            (self.environment, lambda value: value["players"]["shared"].update(attack_strength_ticks=-1)),
            (self.environment, lambda value: value["players"]["overrides"][0].update(position=[0, 0, 0])),
            (self.environment, lambda value: value["players"]["overrides"][0].update(position=[100, 1, 0])),
            (self.policy, lambda value: value.update(requires=["block_edits"])),
            (self.policy, lambda value: value["actions"][-2].update(values=[0, 9])),
        ]
        originals = [deepcopy(value) for value in (self.arena, self.environment, self.policy)]
        for target, mutate in cases:
            with self.subTest(mutation=mutate):
                mutate(target)
                self.write()
                with self.assertRaises(ValueError):
                    self.resolve()
                for value, original in zip((self.arena, self.environment, self.policy), originals):
                    value.clear()
                    value.update(deepcopy(original))
        self.write()

    def test_resolved_validation_rejects_authoring_references_and_tampering(self):
        resolved = self.resolve()
        mutations = [
            lambda value: value.update(environment=str(self.environment_path)),
            lambda value: value["stages"][0]["scenario"]["arenas"][0].update(base=str(self.environment_path)),
            lambda value: value["stages"][0]["rewards"].update(mode="environment"),
            lambda value: value["stages"][0]["rewards"]["weights"].pop("death"),
            lambda value: value["stages"][0]["scenario"]["arenas"][0]["environment"].update(
                reward_weights={"death": 1}),
            lambda value: value["stages"][0]["scenario"]["arenas"][0]["environment"].update(policy={}),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                value = deepcopy(resolved)
                mutate(value)
                with self.assertRaises(ValueError):
                    resolve_plan(value)

    def test_sampler_state_validation_is_atomic(self):
        sampler = EpisodeSampler(self.resolve()["scenario"], 2, 12)
        sampler.players(0)
        good = sampler.state_dict()
        mutations = [
            lambda value: value.update(arenas=[2, 0]),
            lambda value: value.update(rng=value["rng"][:2]),
            lambda value: value.update(cache=[{}, {}]),
            lambda value: value["cache"][0].update(equipment=[{}]),
            lambda value: value["rng"].__setitem__(1, {"invalid": True}),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                bad = deepcopy(good)
                mutate(bad)
                with self.assertRaises(ValueError):
                    sampler.load_state_dict(bad)
                self.assertEqual(good, sampler.state_dict())
        clone = EpisodeSampler(self.resolve()["scenario"], 2, 999)
        clone.load_state_dict(good)
        self.assertEqual(sampler.players(1), clone.players(1))
        np.testing.assert_array_equal(sampler.arenas, clone.arenas)


if __name__ == "__main__":
    unittest.main()
