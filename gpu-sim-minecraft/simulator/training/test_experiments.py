from copy import deepcopy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np

from simulator.training._test_plans import TEST_SETUPS, example_plan
from simulator.training.compiler import LEARNER_DEFAULTS, _learner, _opponents, _policy, _snapshots
from simulator.training.contract import FEATURES, categorical_values, fingerprint, from_legacy, project_observations, validate_contract
from simulator.training.environment import REWARD_NAMES, ROOT, SCHEMA, SCHEMA_V2
from simulator.training.plans import EpisodeSampler, load_plan, plan_hash, resolve_plan, scene_key
from simulator.training.validate_plans import main as validate_plans


class ObservationEncodingChecks(unittest.TestCase):
    def test_existing_policy_files_keep_legacy_values_dimensions_and_fingerprints(self):
        paths = sorted((ROOT / "simulator/setups/policies").rglob("*.json"))
        self.assertTrue(paths)
        fixture = json.loads((ROOT / "contracts/policy_fixtures.json").read_text())
        from simulator.training.contract import FEATURES

        checked = 0
        for path in paths:
            raw = json.loads(path.read_text())
            if any(set(entry) - {"feature", "scale"} for entry in raw["observations"]):
                continue
            with self.subTest(path=path):
                expected = deepcopy(raw)
                for entry in expected["observations"]:
                    entry["scale"] = float(entry.get("scale", 1))
                expected["requires"] = sorted(set(expected.get("requires", [])))
                expected["observation_dim"] = len(expected["observations"])
                expected["action_names"] = [head["name"] for head in expected["actions"]]
                expected["action_sizes"] = [len(head["values"]) for head in expected["actions"]]
                self.assertEqual(_policy(raw), expected)
                old_hash = hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(",", ":"),
                                                     allow_nan=False).encode()).hexdigest()
                self.assertEqual(fingerprint(raw), old_hash)
                self.assertEqual(project_observations(fixture["catalogue"], expected),
                                 [fixture["catalogue"][FEATURES.index(entry["feature"])] * entry["scale"]
                                  for entry in expected["observations"]])
                explicit = deepcopy(raw)
                for entry in explicit["observations"]:
                    entry["encoding"] = "scalar"
                self.assertEqual(fingerprint(explicit), old_hash)
                checked += 1
        self.assertGreater(checked, 0)

    def test_shared_projection_and_invalid_encoding_fixtures(self):
        fixture = json.loads((ROOT / "contracts/policy_fixtures.json").read_text())
        for sample in fixture["projection_cases"]:
            with self.subTest(sample=sample["name"]):
                parsed = _policy({**fixture["policy"], "observations": sample["observations"]})
                self.assertEqual(parsed["observation_dim"], len(sample["expected"]))
                self.assertEqual(validate_contract(parsed), parsed)
                np.testing.assert_allclose(project_observations(fixture["catalogue"], parsed),
                                           sample["expected"], atol=fixture["tolerance"], rtol=0)
        for change in fixture["invalid"]:
            raw = deepcopy(fixture["policy"])
            path = change["pointer"].strip("/").split("/")
            parent = raw
            for key in path[:-1]:
                parent = parent[int(key)] if isinstance(parent, list) else parent[key]
            parent[int(path[-1]) if isinstance(parent, list) else path[-1]] = change["value"]
            with self.subTest(change=change), self.assertRaises(ValueError):
                _policy(raw)

    def test_expansion_limits_and_stale_dimensions_are_rejected(self):
        fixture = json.loads((ROOT / "contracts/policy_fixtures.json").read_text())
        entry = {"feature": "self.weapon_slot", "encoding": "one_hot", "categories": list(range(9))}
        with self.assertRaisesRegex(ValueError, "Expanded"):
            validate_contract({**fixture["policy"], "observations": [entry] * 15})
        with self.assertRaisesRegex(ValueError, "observation_dim"):
            validate_contract({**fixture["policy"], "observations": [entry], "observation_dim": 1})
        self.assertNotEqual(fingerprint({**fixture["policy"], "observations": [entry]}),
                            fingerprint({**fixture["policy"], "observations": [
                                {**entry, "categories": list(reversed(range(9)))}]}))

    def test_bow_projection_is_seat_local_scaled_and_alias_safe(self):
        descriptors = [
            {"feature": "opponent.bow_equipped", "encoding": "one_hot", "categories": [1, 0], "scale": -2},
            {"feature": "self.bow_equipped", "encoding": "one_hot", "categories": [0, 1]},
            {"feature": "self.arrow_count_fraction", "scale": 4},
            {"feature": "self.bow_using", "encoding": "one_hot", "categories": [0, 1], "scale": 3},
            {"feature": "relative.x", "scale": -2},
            {"feature": "arrow.present", "encoding": "one_hot", "categories": [1, 0], "scale": -4},
        ]
        contract = validate_contract({
            "format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
            "ticks_per_action": 1, "observations": descriptors,
            "requires": ["bow"],
            "actions": [{"name": "use_item", "values": [True, False]},
                        {"name": "yaw", "values": [-5, 0, 5]}],
        })
        self.assertEqual(contract["action_names"], ["use_item", "yaw"])
        for feature in ("self.bow_using", "self.bow_equipped", "opponent.bow_equipped",
                        "opponent.bow_using", "arrow.present"):
            self.assertEqual(categorical_values(feature), (0, 1))
        self.assertEqual(categorical_values("self.arrow_count_fraction"), ())

        catalogue = np.zeros((2, len(FEATURES)), dtype=np.float32)
        values = {
            "opponent.bow_equipped": [0, 1], "self.bow_equipped": [1, 0],
            "self.arrow_count_fraction": [0.5, 0], "self.bow_using": [1, 0],
            "relative.x": [0.2, -0.2], "arrow.present": [1, 0],
        }
        for feature, entries in values.items():
            catalogue[:, FEATURES.index(feature)] = entries
        projected = np.stack([project_observations(row, contract) for row in catalogue])
        np.testing.assert_allclose(projected, [
            [0, -2, 0, 1, 2, 0, 3, -0.4, -4, 0],
            [-2, 0, 1, 0, 0, 3, 0, 0.4, 0, -4],
        ])
        self.assertEqual(from_legacy(SCHEMA)["observation_dim"], 22)
        self.assertEqual(from_legacy(SCHEMA_V2)["action_names"], SCHEMA_V2["action_names"])

        invalid_ammo = deepcopy(contract)
        invalid_ammo.pop("observation_dim")
        invalid_ammo["observations"] = [{"feature": "self.arrow_count_fraction",
                                          "encoding": "one_hot", "categories": [0, 1]}]
        with self.assertRaisesRegex(ValueError, "one_hot"):
            validate_contract(invalid_ammo)
        legacy_alias = deepcopy(contract)
        legacy_alias.pop("observation_dim")
        legacy_alias.pop("action_names")
        legacy_alias.pop("action_sizes")
        legacy_alias["actions"] = [{"name": "shield", "values": [False, True]}]
        self.assertEqual(validate_contract(legacy_alias)["action_names"], ["shield"])
        conflict = deepcopy(contract)
        conflict.pop("observation_dim")
        conflict.pop("action_names")
        conflict.pop("action_sizes")
        conflict["actions"].append({"name": "shield", "values": [False, True]})
        with self.assertRaisesRegex(ValueError, "both shield and use_item"):
            validate_contract(conflict)


class SnapshotRuleChecks(unittest.TestCase):
    def test_learner_hyperparameters_validate_and_roundtrip(self):
        self.assertEqual(_learner({}), LEARNER_DEFAULTS)
        values = {"gamma": 0.9, "gae_lambda": 0.8, "clip_eps": 0.1,
                  "vf_coef": 0.25, "ent_coef": 0.02, "max_grad_norm": 1.0}
        raw = example_plan()
        raw["learner"] = values
        resolved = resolve_plan(raw)
        self.assertEqual(resolved["learner"], {**LEARNER_DEFAULTS, **values})
        self.assertEqual(resolve_plan(resolved), resolved)
        for key in values:
            for value in (True, float("nan"), float("inf"), -0.1):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    _learner({key: value})
        for key, value in (("gamma", 1.1), ("gae_lambda", 1.1), ("clip_eps", 0),
                           ("clip_eps", 1), ("max_grad_norm", 0)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                _learner({key: value})
        for key in ("gamma", "gae_lambda", "vf_coef", "ent_coef"):
            self.assertEqual(_learner({key: 0})[key], 0)

    def test_shared_plan_uses_test_owned_setup_files(self):
        raw = example_plan()
        references = [raw["environment"], raw["policy"],
                      *(suite["environment"] for suite in raw["evaluations"].values())]
        for reference in references:
            with self.subTest(reference=reference):
                self.assertTrue(Path(reference).is_relative_to(ROOT / "simulator/tests"))
                self.assertTrue(Path(reference).is_file())
        original = Path.read_text

        def read_test_file(path, *args, **kwargs):
            self.assertFalse(path.resolve().is_relative_to(ROOT / "simulator/setups"))
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", read_test_file):
            resolve_plan(raw)

    def test_optional_combat_rewards_compile_and_stage_overrides_replace(self):
        additions = {"missed_attack": -0.1, "shield_disabled": 2.0, "win_speed": 20.0}
        scene = json.loads((ROOT / "simulator/environments/duel.json").read_text())
        arena = {"format": "minecraft_sim_arena", "version": 1, "minecraft": "26.3",
                 "name": "reward-arena", "bounds": scene["bounds"], "blocks": scene["blocks"]}
        environment = {"format": "minecraft_sim_environment", "version": 2, "minecraft": "26.3",
                       "name": "reward-duel", "arena": "arena.json",
                       "players": {"shared": {"loadout": {
                           "weapons": ["minecraft:diamond_sword", "minecraft:diamond_axe"],
                           "offhand": "minecraft:shield"}}, "overrides": scene["players"]},
                       "task": {"max_ticks": 100, "rewards": additions}}
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for name, value in (("arena.json", arena), ("environment.json", environment),
                                ("policy.json", from_legacy(SCHEMA_V2))):
                (base / name).write_text(json.dumps(value), encoding="utf-8")
            raw = example_plan()
            raw.update(environment="environment.json", policy="policy.json")
            raw["evaluations"]["held_out_combat"]["environment"] = "environment.json"
            resolved = resolve_plan(raw, base_dir=base)
            for name, value in additions.items():
                self.assertEqual(resolved["stages"][0]["rewards"]["weights"][name], 0.0)
                self.assertEqual(resolved["stages"][1]["rewards"]["weights"][name], value)
                self.assertEqual(resolved["evaluations"]["held_out_combat"]["rewards"]["weights"][name], value)
            self.assertEqual(resolve_plan(resolved), resolved)
            raw["stages"][0]["task_override"]["rewards"] = additions
            weights = resolve_plan(raw, base_dir=base)["stages"][0]["rewards"]["weights"]
            for name, value in additions.items():
                self.assertEqual(weights[name], value)
            self.assertEqual(weights["good_aim"], 0.0)

    def test_snapshot_rules_and_frozen_opponent_validation(self):
        names = set()
        _snapshots([{"name": "stage-100", "at_updates": 100}], names)
        _snapshots([{"name": "final", "on_completion": True}], names)
        self.assertEqual(names, {"stage-100", "final"})
        definition = {"past": {"type": "frozen_self_play", "every_updates": 100}}
        self.assertEqual(_opponents(definition, Path.cwd()), definition)
        invalid = [None, {}, [{"name": "../outside", "at_updates": 1}],
                   [{"name": "stage-100", "at_updates": 1}],
                   [{"name": "zero", "at_updates": 0}],
                   [{"name": "boolean", "at_updates": True}],
                   [{"name": "no", "on_completion": False}],
                   [{"name": "both", "at_updates": 1, "on_completion": True}],
                   [{"name": "none"}]]
        for rules in invalid:
            with self.subTest(rules=rules), self.assertRaises(ValueError):
                _snapshots(rules, names.copy())
        for interval in (None, True, 0, -1, 1.5):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                _opponents({"past": {"type": "frozen_self_play", "every_updates": interval}}, Path.cwd())


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
        self.arena = json.loads((TEST_SETUPS / "arena.json").read_text())
        self.environment = json.loads((TEST_SETUPS / "environment.json").read_text())
        self.policy = json.loads((TEST_SETUPS / "policy.json").read_text())
        self.raw = example_plan()
        self.environment["arena"] = "../arenas/flat.json"
        self.raw["environment"] = "../environments/duel.json"
        self.raw["policy"] = "../policies/combat.json"
        for suite in self.raw["evaluations"].values():
            suite["environment"] = "../environments/duel.json"
        self.write()

    def write(self):
        for path, value in ((self.arena_path, self.arena), (self.environment_path, self.environment),
                            (self.policy_path, self.policy), (self.plan_path, self.raw)):
            path.write_text(json.dumps(value), encoding="utf-8")

    def resolve(self):
        return resolve_plan(self.raw, base_dir=self.plan_path.parent)

    def test_arena_fall_damage_rule_roundtrip(self):
        self.arena["rules"] = {"simulate_fall_damage": True}
        self.write()
        resolved = self.resolve()
        for condition in [*resolved["stages"], *resolved["evaluations"].values()]:
            rules = condition["scenario"]["arenas"][0]["environment"]["rules"]
            self.assertEqual(rules, {"simulate_food": False, "natural_regeneration": False,
                                     "simulate_fall_damage": True})
        self.assertEqual(resolve_plan(resolved), resolved)
        self.arena["rules"]["simulate_fall_damage"] = False
        self.write()
        self.assertNotIn("simulate_fall_damage", self.resolve()["scenario"]["arenas"][0]["environment"]["rules"])
        for value in (1, "true", None):
            self.arena["rules"]["simulate_fall_damage"] = value
            self.write()
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.resolve()

    def test_explicit_files_and_canonical_runtime_contract(self):
        resolved = self.resolve()
        self.assertEqual(resolved["format"], "minecraft_sim_experiment")
        self.assertEqual(resolved["version"], 1)
        self.assertEqual(resolved["name"], "test-curriculum")
        self.assertEqual(resolved["learner"],
                         {"hidden_dim": 256, "lr": 0.0003, "epochs": 4, "minibatch_size": 1024,
                          "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2,
                          "vf_coef": 0.5, "ent_coef": 0.01, "max_grad_norm": 0.5})
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

    def test_opt_in_validator_reports_invalid_files_without_stopping(self):
        invalid_json = self.directory / "curricula/broken.json"
        invalid_json.write_text("{", encoding="utf-8")
        invalid_plan = self.directory / "curricula/invalid.json"
        invalid_plan.write_text(json.dumps({**self.raw, "policy": "missing.json"}), encoding="utf-8")
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            status = validate_plans([str(self.plan_path), str(invalid_json), str(invalid_plan)])
        self.assertEqual(status, 1)
        self.assertIn(f"OK {self.plan_path}", output.getvalue())
        self.assertIn("Validated 3 plan(s); 2 failed.", output.getvalue())
        self.assertIn(str(invalid_json), errors.getvalue())
        self.assertIn(str(invalid_plan), errors.getvalue())

    def test_opt_in_validator_accepts_directories_and_empty_selections(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = validate_plans([str(self.plan_path.parent), str(self.plan_path)])
        self.assertEqual(status, 0)
        self.assertIn("Validated 1 plan(s); 0 failed.", output.getvalue())
        self.plan_path.unlink()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = validate_plans([str(self.plan_path.parent)])
        self.assertEqual(status, 0)
        self.assertIn("Validated 0 plan(s); 0 failed.", output.getvalue())

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

    def test_local_evaluation_inherits_resolved_stage_and_roundtrips(self):
        stage = self.raw["stages"][0]
        stage["task_override"]["max_ticks"] = 100
        stage["evaluations"] = {"aim_accuracy": {
            "every_updates": 25, "worlds": 64, "episodes": 64, "seed": 1729,
        }}
        stage["advance"] = {"evaluation": {"suite": "aim_accuracy", "metric": "reward_mean", "at_least": 6}}
        resolved = self.resolve()
        aim = resolved["stages"][0]
        suite = aim["evaluations"]["aim_accuracy"]
        for key in ("scenario", "controls", "opponents", "rewards", "environment_name", "arena_names"):
            self.assertEqual(suite[key], aim[key])
        self.assertEqual(suite["worlds"], 64)
        self.assertEqual(resolve_plan(resolved), resolved)
        self.assertNotIn("evaluations", resolved["stages"][1])

    def test_evaluation_overrides_merge_controls_but_replace_rewards(self):
        stage = self.raw["stages"][0]
        stage["task_override"]["max_ticks"] = 100
        stage["evaluations"] = {"aim_accuracy": {
            "every_updates": 25, "episodes": 64, "seed": 1729,
            "overrides": {"controls": {"fixed": {"attack": True}},
                          "spawns": {"yaw": 0}, "equipment": None,
                          "task_override": {"rewards": {"aim_hold": 1}}},
        }}
        resolved = self.resolve()
        aim = resolved["stages"][0]
        suite = aim["evaluations"]["aim_accuracy"]
        self.assertTrue(suite["controls"]["fixed"]["attack"])
        self.assertFalse(aim["controls"]["fixed"]["attack"])
        self.assertEqual(suite["controls"]["fixed"]["movement"], "neutral")
        self.assertEqual(suite["scenario"]["spawns"]["yaw"], 0)
        self.assertNotIn("equipment", suite["scenario"])
        self.assertEqual(suite["scenario"]["arenas"][0]["environment"]["max_ticks"], 100)
        self.assertEqual(suite["rewards"]["weights"]["aim_hold"], 1)
        self.assertEqual(suite["rewards"]["weights"]["good_aim"], 0)
        self.assertEqual(aim["rewards"]["weights"]["good_aim"], 5)

    def test_global_evaluation_can_reference_fixed_stage_with_overrides(self):
        self.raw["evaluations"] = {"benchmark": {
            "stage": "aim", "every_updates": 50, "episodes": 64, "seed": 1729,
            "overrides": {"opponents": "held_out", "environment": "../environments/duel.json"},
        }}
        self.raw["stages"][0]["task_override"]["max_ticks"] = 100
        self.raw["stages"][1]["advance"] = {"updates": 1500}
        resolved = self.resolve()
        suite = resolved["evaluations"]["benchmark"]
        self.assertEqual(suite["controls"], resolved["stages"][0]["controls"])
        self.assertEqual(suite["rewards"], resolved["stages"][0]["rewards"])
        self.assertEqual(suite["scenario"]["arenas"][0]["environment"]["max_ticks"], 100)
        self.assertEqual(suite["opponents"], [{"id": "fighter", "weight": 1}])

    def test_environment_override_preserves_stage_task_and_controls(self):
        alternate = deepcopy(self.environment)
        alternate["name"] = "alternate"
        alternate["task"] = {"max_ticks": 800, "rewards": {"death": -99}}
        alternate["spawns"]["yaw"] = 0
        (self.environment_path.parent / "alternate.json").write_text(json.dumps(alternate))
        self.raw["stages"][0]["evaluations"] = {"alternate_aim": {
            "every_updates": 25, "episodes": 64, "seed": 1729,
            "overrides": {"environment": "../environments/alternate.json"},
        }}
        resolved = self.resolve()
        aim = resolved["stages"][0]
        suite = aim["evaluations"]["alternate_aim"]
        self.assertEqual(suite["environment_name"], "alternate")
        self.assertEqual(suite["scenario"]["spawns"]["yaw"], 0)
        self.assertEqual(suite["rewards"], aim["rewards"])
        self.assertEqual(suite["controls"], aim["controls"])
        self.assertEqual(suite["scenario"]["arenas"][0]["environment"]["max_ticks"],
                         aim["scenario"]["arenas"][0]["environment"]["max_ticks"])

    def test_evaluation_controls_can_release_one_or_all_fixed_heads(self):
        stage = self.raw["stages"][0]
        stage["evaluations"] = {"aim_accuracy": {
            "every_updates": 25, "episodes": 64, "seed": 1729,
            "overrides": {"controls": {"fixed": {"attack": None}}},
        }}
        suite = self.resolve()["stages"][0]["evaluations"]["aim_accuracy"]
        self.assertNotIn("attack", suite["controls"]["fixed"])
        self.assertEqual(suite["controls"]["fixed"]["movement"], "neutral")
        stage["evaluations"]["aim_accuracy"]["overrides"]["controls"]["fixed"] = None
        suite = self.resolve()["stages"][0]["evaluations"]["aim_accuracy"]
        self.assertEqual(suite["controls"], {"fixed": {}})
        self.assertFalse(stage["controls"]["fixed"]["attack"])

    def test_evaluation_references_scopes_and_world_counts_are_validated(self):
        suite = {"every_updates": 25, "episodes": 64, "seed": 1729}
        mutations = [
            lambda plan: plan["evaluations"].update(broken={**suite, "stage": "missing"}),
            lambda plan: plan["stages"][0].update(evaluations={"local": {**suite, "worlds": 0}}),
            lambda plan: plan["stages"][0].update(evaluations={"local": {**suite, "worlds": True}}),
            lambda plan: plan["stages"][0].update(evaluations={"local": {**suite, "stage": "combat"}}),
            lambda plan: plan["stages"][0].update(evaluations={"local": {**suite, "controls": {"fixed": {}}}}),
            lambda plan: plan["stages"][0].update(evaluations={"held_out_combat": suite}),
            lambda plan: plan["stages"][1].update(advance={"evaluation": {
                "suite": "local", "metric": "reward_mean", "at_least": 1}}),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                original = deepcopy(self.raw)
                mutation(self.raw)
                with self.assertRaises(ValueError):
                    self.resolve()
                self.raw = original

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
        stage["equipment"] = {"loadout": {"weapons": ["minecraft:stone_sword", "minecraft:bow"],
                                          "offhand": "minecraft:air", "arrows": 7}}
        self.raw["stages"][1].update(spawns=None, equipment=None)
        resolved = self.resolve()
        first, second = resolved["stages"]
        self.assertEqual(first["scenario"]["spawns"]["region"],
                         {"min": [-1, 1, -1], "max": [4.5, 1, 4.5]})
        self.assertEqual(first["scenario"]["spawns"]["minimum_separation"], 2)
        self.assertEqual(first["scenario"]["spawns"]["yaw"], 0)
        slots = first["scenario"]["equipment"]["loadout"]["weapons"]
        self.assertEqual(slots, ["minecraft:stone_sword", "minecraft:bow", *(["minecraft:air"] * 7)])
        self.assertEqual(first["scenario"]["equipment"]["loadout"]["arrows"], 7)
        players = EpisodeSampler(first["scenario"], 1, 17).players(0)
        self.assertEqual([player["loadout"]["arrows"] for player in players], [7, 7])
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

    def test_sampled_players_do_not_alias_templates_or_component_caches(self):
        for sample in ("world", "episode"):
            with self.subTest(sample=sample):
                self.environment["equipment"]["sample"] = sample
                self.environment["equipment"]["loadout"]["armor"] = {"helmet": "minecraft:turtle_helmet"}
                self.write()
                sampler = EpisodeSampler(self.resolve()["scenario"], 2, 42)
                players = sampler.players(0)
                expected_scenario = deepcopy(sampler.scenario)
                expected_state = sampler.state_dict()
                for player in players:
                    player["position"][0] += 100
                    player["velocity"][0] = 100
                    player["loadout"]["weapons"][0] = "minecraft:air"
                    player["loadout"]["armor"]["helmet"] = "minecraft:air"
                self.assertEqual(sampler.scenario, expected_scenario)
                self.assertEqual(sampler.state_dict(), expected_state)
                following = sampler.players(0)
                self.assertTrue(all(player["loadout"]["armor"]["helmet"] == "minecraft:turtle_helmet"
                                    for player in following))
                self.assertTrue(all(player["loadout"]["weapons"][0] == "minecraft:iron_sword"
                                    for player in following))

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

    def test_spawn_sampling_matches_previous_positions_and_rng_state(self):
        class PreviousDraws:
            def __init__(self, generator):
                self.generator = generator

            def uniform(self, low, high, size=None):
                if size == (2, 3):
                    return np.array([self.generator.uniform(low, high), self.generator.uniform(low, high)])
                return self.generator.uniform(low, high, size=size)

            def __getattr__(self, name):
                return getattr(self.generator, name)

        def previous_separated(positions, minimum):
            delta = np.abs(np.subtract(positions[0], positions[1]))
            return ((delta[0] >= 0.6 or delta[2] >= 0.6 or delta[1] >= 1.8)
                    and np.linalg.norm(delta) >= minimum)

        for minimum in (2, 2.6):
            with self.subTest(minimum=minimum):
                self.environment["spawns"] = {
                    "sample": "episode", "region": {"min": [-0.1, 1, -1.3], "max": [0.1, 1, 1.3]},
                    "minimum_separation": minimum, "yaw": {"uniform": [-180, 180]},
                    "pitch": {"uniform": [-20, 20]},
                }
                self.write()
                scenario = self.resolve()["scenario"]
                optimized = EpisodeSampler(scenario, 4, 42)
                previous = EpisodeSampler(scenario, 4, 42)
                previous.spawn_rng = PreviousDraws(previous.spawn_rng)
                with patch.object(previous, "_separated", side_effect=previous_separated):
                    for world in (0, 1, 2, 3, 0, 2):
                        self.assertEqual(optimized.players(world), previous.players(world))
                        self.assertEqual(optimized.state_dict(), previous.state_dict())

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
