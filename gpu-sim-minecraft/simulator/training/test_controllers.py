"""Focused controller checks; no native simulator or two-player serving API."""

from copy import deepcopy
import hashlib
import io
from pathlib import Path
import shutil
import sys
from types import ModuleType
import unittest
from unittest import mock
from uuid import uuid4

import numpy as np
import torch

from backend.fast_rl_model import ActorCritic
from simulator.training.contract import fingerprint, from_legacy, validate_contract
from simulator.training.controllers import (
    BasicMeleeController, CheckpointController, Controller, CONTROLLER_REGISTRY,
    StationaryTargetController, checkpoint_sha256, create_controller, register_controller,
)
from simulator.training.environment import SCHEMA


def policy(observations=None, actions=None):
    return validate_contract({
        "format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
        "ticks_per_action": 1,
        "observations": observations or [
            {"feature": "target.distance_fraction"},
            {"feature": "target.yaw_error"},
            {"feature": "target.pitch_error"},
        ],
        "actions": actions or [
            {"name": "movement", "values": ["neutral", "forward", "right", "left",
                                           "forward_right", "forward_left"]},
            {"name": "attack", "values": [False, True]},
            {"name": "yaw", "values": [-20, -5, 0, 5, 20]},
            {"name": "pitch", "values": [-5, 0, 5]},
        ],
    })


def observations(contract, count=3, **features):
    result = np.zeros((count, contract["observation_dim"]), dtype=np.float32)
    for index, descriptor in enumerate(contract["observations"]):
        result[:, index] = np.asarray(features.get(descriptor["feature"], 0)) * descriptor["scale"]
    return result


def decoded(contract, actions, name):
    column = contract["action_names"].index(name)
    return np.asarray(contract["actions"][column]["values"])[actions[:, column]]


class ScriptedChecks(unittest.TestCase):
    def test_stationary_legacy_and_reordered_neutral_values(self):
        legacy = from_legacy(SCHEMA)
        target = create_controller({"type": "scripted", "controller": "stationary_target"},
                                   legacy, 4)
        np.testing.assert_array_equal(target.act(np.zeros((4, 22))),
                                      np.tile([0, 0, 0, 0, 4, 2], (4, 1)))
        contract = policy(actions=[
            {"name": "pitch", "values": [5, -2]},
            {"name": "attack", "values": [True, False]},
            {"name": "movement", "values": ["forward", "neutral"]},
            {"name": "weapon", "values": [4, 2, 0]},
            {"name": "yaw", "values": [10, 0, -10]},
            {"name": "jump", "values": [True]},
        ])
        target = StationaryTargetController(contract, 4)
        self.assertIsInstance(target, Controller)
        np.testing.assert_array_equal(target.act(observations(contract, 2)),
                                      [[1, 1, 1, 2, 1, 0]] * 2)
        self.assertEqual(target.act(observations(contract, 0)).shape, (0, 6))
        description = target.describe
        description["options"]["injected"] = True
        self.assertEqual(target.provenance["options"], {})
        target.load_state_dict(target.state_dict())

    def test_basic_aim_movement_and_attack_timing(self):
        contract = policy()
        controller = BasicMeleeController(contract, 3)
        obs = observations(contract, **{"target.distance_fraction": [0.2, 0.4, 0.3],
                                       "target.yaw_error": [-1, 1 / 36, 1],
                                       "target.pitch_error": [1 / 18, -1 / 18, 0]})
        actions = controller.act(obs)
        np.testing.assert_array_equal(decoded(contract, actions, "movement"),
                                      ["neutral", "forward", "neutral"])
        np.testing.assert_array_equal(decoded(contract, actions, "attack"), [True, False, True])
        np.testing.assert_array_equal(decoded(contract, actions, "yaw"), [-20, 5, 20])
        np.testing.assert_array_equal(decoded(contract, actions, "pitch"), [5, -5, 0])
        for _ in range(12):
            self.assertFalse(decoded(contract, controller.act(obs), "attack").any())
        np.testing.assert_array_equal(decoded(contract, controller.act(obs), "attack"),
                                      [True, False, True])

    def test_reordered_scaled_features_and_heads(self):
        contract = policy(observations=[
            {"feature": "self.health_fraction", "scale": 5},
            {"feature": "target.pitch_error", "scale": -4},
            {"feature": "target.distance_fraction", "scale": 3},
            {"feature": "target.yaw_error", "scale": -2},
        ], actions=[
            {"name": "yaw", "values": [20, 0, -5]},
            {"name": "pitch", "values": [0, -5, 5]},
            {"name": "attack", "values": [True, False]},
            {"name": "movement", "values": ["forward", "neutral"]},
        ])
        controller = BasicMeleeController(contract, 2)
        obs = observations(contract, 2, **{"target.distance_fraction": [0.1, 0.8],
                                           "target.yaw_error": [-5 / 180, 20 / 180],
                                           "target.pitch_error": [5 / 90, -5 / 90]})
        np.testing.assert_array_equal(controller.act(obs), [[2, 2, 0, 1], [0, 1, 1, 0]])

    def test_subset_indices_selective_reset_and_roundtrip(self):
        contract = policy()
        controller = BasicMeleeController(contract, 5, options={"attack_interval": 3})
        obs = observations(contract, 2, **{"target.distance_fraction": 0.2})
        controller.act(obs, indices=np.array([4, 1]))
        self.assertEqual(controller.state_dict()["ticks"], [0, 1, 0, 0, 1])
        controller.reset(np.array([False, True, False, False, False]))
        actions = controller.act(obs, indices=np.array([1, 4]))
        np.testing.assert_array_equal(decoded(contract, actions, "attack"), [True, False])
        saved = controller.state_dict()
        clone = BasicMeleeController(contract, 5, options={"attack_interval": 3})
        clone.load_state_dict(saved)
        np.testing.assert_array_equal(controller.act(obs, indices=[4, 0]),
                                      clone.act(obs, indices=[4, 0]))
        saved["ticks"][0] = 20
        self.assertNotEqual(controller.state_dict()["ticks"][0], 20)
        controller.reset(np.ones(5, dtype=bool))
        self.assertEqual(controller.state_dict()["ticks"], [0] * 5)

    def test_strafe_sprint_and_omitted_heads(self):
        contract = policy(actions=[*policy()["actions"],
                                   {"name": "sprint", "values": [True, False]}])
        controller = BasicMeleeController(contract, 2, options={
            "strafe": True, "strafe_interval": 1, "sprint": True})
        obs = observations(contract, 2, **{"target.distance_fraction": [0.2, 0.6]})
        first = controller.act(obs)
        second = controller.act(obs)
        np.testing.assert_array_equal(decoded(contract, first, "movement"), ["right", "forward_right"])
        np.testing.assert_array_equal(decoded(contract, second, "movement"), ["left", "forward_left"])
        np.testing.assert_array_equal(decoded(contract, first, "sprint"), [False, True])
        aim_only = policy(actions=[{"name": "yaw", "values": [-10, 0, 10]}])
        self.assertEqual(BasicMeleeController(aim_only, 2).act(observations(aim_only, 2)).shape, (2, 1))

    def test_explicit_equipment_slots_and_shield_features(self):
        contract = policy(observations=[*policy()["observations"],
            {"feature": "self.attack_charge", "scale": -2},
            {"feature": "self.shield_using"},
            {"feature": "self.shield_disabled"},
            {"feature": "opponent.shield_using", "scale": 3},
            {"feature": "opponent.shield_disabled_observed"},
        ], actions=[*policy()["actions"],
            {"name": "weapon", "values": [5, 3]},
            {"name": "shield", "values": [True, False]},
        ])
        controller = BasicMeleeController(contract, 3, options={
            "axe_slot": 5, "sword_slot": 3, "shield": True})
        obs = observations(contract, **{
            "target.distance_fraction": 0.2, "self.attack_charge": 0.2,
            "self.shield_disabled": [0, 1, 0],
            "opponent.shield_using": 1, "opponent.shield_disabled_observed": [0, 0, 1]})
        first = controller.act(obs)
        np.testing.assert_array_equal(decoded(contract, first, "weapon"), [5, 5, 3])
        self.assertFalse(decoded(contract, first, "shield").any())
        second = controller.act(obs)
        np.testing.assert_array_equal(decoded(contract, second, "shield"), [True, False, True])

    def test_required_features_and_options_are_not_silently_ignored(self):
        for feature in ("target.distance_fraction", "target.yaw_error", "target.pitch_error"):
            for remove in (False, True):
                entries = deepcopy(policy()["observations"])
                entries = [entry for entry in entries if entry["feature"] != feature] if remove else [
                    {**entry, "scale": 0} if entry["feature"] == feature else entry for entry in entries]
                with self.subTest(feature=feature, remove=remove), self.assertRaisesRegex(ValueError, feature):
                    BasicMeleeController(policy(observations=entries), 1)
        for options in ({"unknown": 1}, {"attack_interval": 0}, {"attack_interval": True},
                        {"strafe_interval": -1}, {"strafe": "yes"}, {"sprint": 1},
                        {"melee_distance": float("nan")}, {"melee_distance": 10},
                        {"melee_distance": True}, {"shield": True}, {"axe_slot": 1},
                        {"axe_slot": 0, "sword_slot": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                BasicMeleeController(policy(), 2, options=options)
        with self.assertRaises(ValueError):
            StationaryTargetController(policy(), 1, options={"strafe": True})
        with self.assertRaises(ValueError):
            StationaryTargetController(policy(actions=[{"name": "movement", "values": ["forward"]}]), 1)

    def test_invalid_batches_resets_and_states_do_not_mutate_ticks(self):
        contract = policy()
        controller = BasicMeleeController(contract, 2)
        obs = observations(contract, 2)
        for invalid in (obs.tolist(), np.zeros((3, 3)), np.zeros((2, 2)), np.zeros(3),
                        np.full((2, 3), np.nan), np.full((2, 3), "bad")):
            with self.subTest(observations=invalid), self.assertRaises(ValueError):
                controller.act(invalid)
        for indices in ([1, 1], [0, 2], [-1, 0], [0], [0.0, 1.0], [True, False]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                controller.act(obs, indices)
        for mask in ([True], [0, 1], [[True, False]]):
            with self.assertRaises(ValueError):
                controller.reset(mask)
        state = controller.state_dict()
        for invalid in ({}, {**state, "ticks": [-1, 0]}, {**state, "ticks": [True, 0]},
                        {**state, "ticks": [0]}, {**state, "extra": 0}):
            with self.assertRaises(ValueError):
                controller.load_state_dict(invalid)
        self.assertEqual(controller.state_dict()["ticks"], [0, 0])

    def test_registered_and_explicit_import_factories_receive_contract(self):
        contract = policy()
        received = []

        def factory(**kwargs):
            received.append(kwargs)
            return StationaryTargetController(kwargs["contract"], kwargs["count"], kwargs["seed"])

        name = f"test_{uuid4().hex}"
        register_controller(name, factory)
        self.addCleanup(CONTROLLER_REGISTRY.pop, name)
        instance = create_controller({"type": "scripted", "controller": name, "options": {"custom": 4}},
                                     contract, 2, seed=7)
        self.assertIsInstance(instance, Controller)
        self.assertEqual(received[-1], {"contract": contract, "count": 2, "seed": 7, "options": {"custom": 4}})
        module = ModuleType("controller_test_plugin")
        module.factory = factory
        with mock.patch.dict(sys.modules, controller_test_plugin=module):
            instance = create_controller({"type": "scripted", "controller": "controller_test_plugin:factory"},
                                         contract, 1, seed=9)
            self.assertEqual(instance.seed, 9)
        with self.assertRaises(ValueError):
            register_controller(name, factory)
        with self.assertRaises(ValueError):
            register_controller("not:a:name", factory)
        with self.assertRaises(ValueError):
            register_controller("not_callable", None)

    def test_invalid_definitions(self):
        for definition in (None, [], {}, {"type": "bad"},
                           {"type": "scripted", "controller": "missing"},
                           {"type": "scripted", "controller": "os:system('bad')"},
                           {"type": "scripted", "controller": "os:path"},
                           {"type": "scripted", "controller": "a:b:c"},
                           {"type": "scripted", "controller": "stationary_target", "options": []},
                           {"type": "scripted", "controller": "stationary_target", "extra": True},
                           {"type": "checkpoint"}, {"type": "checkpoint", "path": ""},
                           {"type": "checkpoint", "path": "policy.pt", "options": {}}):
            with self.subTest(definition=definition), self.assertRaises(ValueError):
                create_controller(definition, policy(), 2)
        for count, seed in ((0, 1), (True, 1), (2, True), (2, -1)):
            with self.assertRaises(ValueError):
                create_controller({"type": "scripted", "controller": "stationary_target"},
                                  policy(), count, seed=seed)


class CheckpointChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.directory = Path.cwd() / f".controller-check-{uuid4().hex}"
        self.directory.mkdir()
        self.addCleanup(shutil.rmtree, self.directory)
        self.path = self.directory / "policy.pt"
        self.contract = policy()
        with torch.random.fork_rng(devices=[]):
            model = ActorCritic(self.contract["observation_dim"], 8,
                                action_sizes=dict(zip(self.contract["action_names"],
                                                      self.contract["action_sizes"])))
        self.saved = {"format": "minecraft_sim_ppo", "version": 1,
                      "metadata": {"contract": deepcopy(self.contract), "schema": deepcopy(self.contract),
                                   "hidden_dim": 8, "players_per_environment": 2,
                                   "learner": "both_shared_policy"},
                      "model": model.state_dict()}
        torch.save(self.saved, self.path)

    def test_checkpoint_actions_arbitrary_batches_and_eval(self):
        controller = create_controller({"type": "checkpoint", "path": str(self.path),
                                        "deterministic": True}, self.contract, 7)
        self.assertFalse(controller.model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in controller.model.parameters()))
        for count in (0, 1, 3, 7):
            obs = observations(self.contract, count)
            actions = controller.act(obs, indices=np.arange(count)[::-1])
            self.assertEqual(actions.shape, (count, 4))
            self.assertEqual(actions.dtype, np.int64)
            self.assertTrue((actions >= 0).all())
            self.assertTrue((actions < self.contract["action_sizes"]).all())
            np.testing.assert_array_equal(actions, controller.act(obs))
        controller.reset(np.ones(7, dtype=bool))
        with self.assertRaises(ValueError):
            controller.act(observations(self.contract, 8))

    def test_rng_independence_and_checkpointable_sampling_roundtrip(self):
        learner_rng = torch.get_rng_state().clone()
        self.addCleanup(torch.set_rng_state, learner_rng)
        controller = CheckpointController(self.path, self.contract, 6, seed=12)
        self.assertTrue(torch.equal(learner_rng, torch.get_rng_state()))
        obs = observations(self.contract, 6)
        controller.act(obs)
        self.assertTrue(torch.equal(learner_rng, torch.get_rng_state()))
        state = controller.state_dict()
        archive = io.BytesIO()
        torch.save(state, archive)
        archive.seek(0)
        state = torch.load(archive, weights_only=True)
        clone = CheckpointController(self.path, self.contract, 6, seed=12)
        clone.load_state_dict(state)
        for _ in range(5):
            expected = controller.act(obs)
            torch.rand(13)
            before = torch.get_rng_state().clone()
            np.testing.assert_array_equal(expected, clone.act(obs))
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
        snapshot = controller.state_dict()
        snapshot["rng_state"].zero_()
        self.assertFalse(torch.equal(snapshot["rng_state"], controller.state_dict()["rng_state"]))

    def test_pool_and_legacy_checkpoint_saved_controls_constrain_inference(self):
        self.saved["metadata"]["controls"] = {
            "fixed": {"movement": "forward", "yaw": -5, "attack": False}}
        for name, index in (("movement", 1), ("yaw", 1), ("attack", 0)):
            self.saved["model"][f"{name}_head.weight"].zero_()
            self.saved["model"][f"{name}_head.bias"].fill_(1000)
            self.saved["model"][f"{name}_head.bias"][index] = -1000
        obs = observations(self.contract, 4)
        for learner in ("opponent_pool", "both_shared_policy"):
            self.saved["metadata"]["learner"] = learner
            torch.save(self.saved, self.path)
            for deterministic in (False, True):
                with self.subTest(learner=learner, deterministic=deterministic):
                    definition = {"type": "checkpoint", "path": str(self.path),
                                  "sha256": checkpoint_sha256(self.path),
                                  "deterministic": deterministic}
                    before = torch.get_rng_state().clone()
                    controller = create_controller(definition, self.contract, 4, seed=17)
                    actions = controller.act(obs, indices=[3, 1, 2, 0])
                    self.assertTrue(torch.equal(before, torch.get_rng_state()))
                    np.testing.assert_array_equal(decoded(self.contract, actions, "movement"),
                                                  ["forward"] * 4)
                    np.testing.assert_array_equal(decoded(self.contract, actions, "yaw"), [-5] * 4)
                    np.testing.assert_array_equal(decoded(self.contract, actions, "attack"), [False] * 4)
                    if not deterministic:
                        generator = torch.Generator().manual_seed(17)
                        with torch.inference_mode():
                            logits, _ = controller.model(torch.as_tensor(obs))
                            expected_pitch = torch.multinomial(
                                logits["pitch"].softmax(-1), 1, generator=generator).squeeze(-1)
                        np.testing.assert_array_equal(actions[:, 3], expected_pitch.numpy())
                        self.assertTrue(torch.equal(controller.state_dict()["rng_state"],
                                                    generator.get_state()))
                    archive = io.BytesIO()
                    torch.save(controller.state_dict(), archive)
                    archive.seek(0)
                    clone = create_controller(definition, self.contract, 4, seed=17)
                    clone.load_state_dict(torch.load(archive, weights_only=True))
                    for _ in range(3):
                        np.testing.assert_array_equal(controller.act(obs), clone.act(obs))

    def test_checkpoint_rejects_invalid_saved_controls(self):
        for controls in (None, [], {"extra": {}}, {"fixed": []},
                         {"fixed": {"missing_head": True}}, {"fixed": {"yaw": 17}},
                         {"fixed": {"attack": 0}}):
            with self.subTest(controls=controls):
                self.saved["metadata"]["controls"] = controls
                torch.save(self.saved, self.path)
                with self.assertRaises(ValueError):
                    CheckpointController(self.path, self.contract, 1)

    def test_normalized_contract_and_exact_legacy_conversion(self):
        raw = deepcopy(self.contract)
        for key in ("observation_dim", "action_names", "action_sizes", "requires"):
            raw.pop(key)
        for entry in raw["observations"]:
            entry.pop("scale")
        self.saved["metadata"]["contract"] = raw
        torch.save(self.saved, self.path)
        self.assertEqual(CheckpointController(self.path, self.contract, 1).contract_hash,
                         fingerprint(raw))
        self.saved["metadata"]["schema"] = SCHEMA
        self.saved["metadata"].pop("contract")
        with torch.random.fork_rng(devices=[]):
            self.saved["model"] = ActorCritic(22, 8, movement_bins=9).state_dict()
        torch.save(self.saved, self.path)
        legacy = from_legacy(SCHEMA)
        controller = CheckpointController(self.path, legacy, 3)
        self.assertEqual(controller.act(np.zeros((3, 22))).shape, (3, 6))
        self.saved["metadata"]["schema"] = {**SCHEMA, "movement": SCHEMA["movement"][:-1]}
        torch.save(self.saved, self.path)
        with self.assertRaisesRegex(ValueError, "legacy"):
            CheckpointController(self.path, legacy, 1)

    def test_contract_mismatches_even_when_dimensions_match(self):
        for change in ("scale", "order", "choice", "requires"):
            altered = deepcopy(self.contract)
            if change == "scale":
                altered["observations"][0]["scale"] = 2
            elif change == "order":
                altered["observations"].reverse()
            elif change == "choice":
                altered["actions"][2]["values"].reverse()
            else:
                altered["requires"] = ["melee"]
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "fingerprint"):
                CheckpointController(self.path, altered, 1)
        self.saved["metadata"]["contract"]["observations"][0]["scale"] = 2
        torch.save(self.saved, self.path)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            CheckpointController(self.path, self.contract, 1)

    def test_bad_checkpoint_metadata_and_strict_parameter_validation(self):
        original = deepcopy(self.saved)
        for mutate in (
            lambda value: value.update(format="legacy"),
            lambda value: value.update(version=True),
            lambda value: value["metadata"].update(players_per_environment=1),
            lambda value: value["metadata"].update(learner="other"),
            lambda value: value["metadata"].update(hidden_dim=True),
            lambda value: value["metadata"].update(hidden_dim=9),
            lambda value: value["metadata"].update(schema={"id": "wrong"}),
            lambda value: value["model"].pop("value_head.bias"),
            lambda value: value["model"].update(extra=torch.zeros(1)),
            lambda value: value["model"]["yaw_head.bias"].fill_(float("inf")),
        ):
            value = deepcopy(original)
            mutate(value)
            torch.save(value, self.path)
            with self.assertRaises(ValueError):
                CheckpointController(self.path, self.contract, 1)

    def test_checkpoint_hash_provenance_and_replacement_guard(self):
        expected = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.assertEqual(checkpoint_sha256(self.path), expected)
        controller = create_controller({"type": "checkpoint", "path": str(self.path),
                                        "sha256": expected}, self.contract, 2)
        self.assertEqual(controller.provenance["sha256"], expected)
        controller.verify_checkpoint(expected)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            CheckpointController(self.path, self.contract, 2, expected_sha256="0" * 64)
        self.saved["update"] = 4
        torch.save(self.saved, self.path)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            controller.verify_checkpoint()
        with self.assertRaisesRegex(ValueError, "SHA256"):
            create_controller({"type": "checkpoint", "path": str(self.path), "sha256": expected},
                              self.contract, 2)

    def test_checkpoint_invalid_state_precision_and_configuration(self):
        controller = CheckpointController(self.path, self.contract, 2)
        state = controller.state_dict()
        for invalid in ({}, {**state, "rng_state": [1]}, {**state, "rng_state": torch.zeros(1)},
                        {**state, "rng_state": torch.zeros(1, dtype=torch.uint8)}):
            with self.assertRaises(ValueError):
                controller.load_state_dict(invalid)
        with self.assertRaisesRegex(ValueError, "precision"):
            controller.act(np.full((2, 3), 1e100))
        with self.assertRaises(ValueError):
            CheckpointController(self.path, self.contract, 2, deterministic="yes")
        with self.assertRaises(ValueError):
            CheckpointController(self.path, self.contract, 2, device="meta")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_local_generator_preserves_cpu_and_device_rng(self):
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        controller = CheckpointController(self.path, self.contract, 3, device="cuda", seed=7)
        obs = observations(self.contract)
        controller.act(obs)
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))
        state = controller.state_dict()
        expected = controller.act(obs)
        controller.load_state_dict(state)
        np.testing.assert_array_equal(expected, controller.act(obs))


if __name__ == "__main__":
    unittest.main()
