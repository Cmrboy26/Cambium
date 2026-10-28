import contextlib
import io
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import torch

from backend.fast_rl_model import ActorCritic
from simulator.training.environment import (
    DEFAULT_REWARD_WEIGHTS, Environment, REWARD_NAMES, REWARD_SEMANTICS, ROOT, SCHEMA, SCHEMA_V2, SHAPING_REWARD_NAMES,
    resolve_reward_weights,
)
from simulator.training.contract import FEATURES, from_legacy, validate_contract, resolve_scenario
from simulator.training.controllers import CONTROLLER_REGISTRY
from simulator.training.serve import DuelPolicy, Policy, make_server
from simulator.training.compatibility_training import ReplaySamples, collect_rollout, compute_gae, main


def neutral(count, heads=6):
    actions = np.zeros((count, 2, heads), dtype=np.int32)
    actions[:, :, 4] = 4
    actions[:, :, 5] = 2
    return actions


def unshaped_weights():
    weights = DEFAULT_REWARD_WEIGHTS
    return {name: 0.0 if name in SHAPING_REWARD_NAMES else value for name, value in weights.items()}


class InferenceChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "policy.pt"
        self.saved = {"format": "minecraft_sim_ppo", "version": 1,
                      "metadata": {"schema": SCHEMA, "hidden_dim": 16, "max_ticks": 300,
                                   "players_per_environment": 2, "learner": "both_shared_policy",
                                   "environment": json.loads((ROOT / "simulator/environments/duel.json").read_text())},
                      "model": ActorCritic(22, 16, 9).state_dict()}
        torch.save(self.saved, self.path)

    def test_single_player_inference_keeps_duel_protocol_and_fixed_controls(self):
        self.saved["metadata"]["controls"] = {"fixed": {"attack": False}}
        torch.save(self.saved, self.path)
        policy = Policy(self.path, deterministic=True)
        request = {"protocol": 1, "session": "single", "tick": 1,
                   "contract_hash": policy.info["contract_hash"], "observations": [[0.0] * 22]}
        with make_server(policy, 0) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                def post(path, body):
                    return urlopen(Request(base + path, json.dumps(body).encode(),
                                           {"Content-Type": "application/json"}), timeout=3)
                with post("/act-one", request) as response:
                    single = json.load(response)["actions"]
                self.assertEqual(len(single), 1)
                self.assertEqual(single[0][3], 0)
                with post("/act", {**request, "observations": request["observations"] * 2}) as response:
                    self.assertEqual(json.load(response)["actions"][0], single[0])
                for invalid in ({**request, "observations": request["observations"] * 2},
                                {**request, "observations": [[float("nan")] * 22]},
                                {**request, "contract_hash": "0" * 64},
                                {key: value for key, value in request.items() if key != "contract_hash"}):
                    with self.assertRaises(HTTPError) as failure:
                        post("/act-one", invalid)
                    self.assertEqual(failure.exception.code, 400)
                    failure.exception.close()
                with self.assertRaises(HTTPError) as failure:
                    post("/act", request)
                failure.exception.close()
            finally:
                server.shutdown()
                worker.join(timeout=3)

    def test_offline_single_checkpoint_launcher(self):
        game = self.path.parent / "Minecraft with spaces"
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen([sys.executable, "-u", "-B", "-m", "simulator.training.rosie",
                "local-policy", "--minecraft-directory", str(game), "--checkpoint", str(self.path), "--offline", "--json"],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=errors)
            events = queue.Queue()
            reader = threading.Thread(target=lambda: events.put(next((line for line in process.stdout
                if json.loads(line)["event"] in ("policy", "error")), b"")), daemon=True)
            try:
                reader.start()
                event = json.loads(events.get(timeout=30))
                self.assertEqual(event["event"], "policy", event)
                base = f"http://127.0.0.1:{event['port']}"
                with urlopen(base + "/info", timeout=3) as response:
                    info = json.load(response)
                body = {"protocol": 1, "session": "offline-player", "tick": 0,
                        "contract_hash": info["contract_hash"], "observations": [[0.0] * 22]}
                with urlopen(Request(base + "/act-one", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"}), timeout=3) as response:
                    self.assertEqual(len(json.load(response)["actions"]), 1)
                self.assertFalse((game / "config/cambium-rosie.json").exists())
            finally:
                process.terminate()
                process.wait(timeout=10)
                process.stdout.close()
                if reader.ident is not None:
                    reader.join(timeout=3)

    @unittest.skipUnless(os.environ.get("CAMBIUM_BUNDLED_HELPER"), "No bundled helper selected")
    def test_bundled_helper_import_and_inference(self):
        import signal

        executable = os.environ["CAMBIUM_BUNDLED_HELPER"]
        game = self.path.parent / "Fresh Minecraft"
        imported = subprocess.run([executable, "import", "--checkpoint", str(self.path),
                                   "--minecraft-directory", str(game), "--json"],
                                  capture_output=True, text=True, timeout=120)
        self.assertEqual(imported.returncode, 0, imported.stdout + imported.stderr)
        self.assertEqual(json.loads(imported.stdout.splitlines()[-1])["event"], "imported")
        library = json.loads((game / "sim-rosie-cache/library.json").read_text())
        checkpoint = library["policies"][0]["path"]
        duel = {"format": "minecraft_sim_duel", "version": 1, "reference": checkpoint,
                "fighters": [{"type": "checkpoint", "path": checkpoint},
                             {"type": "scripted", "controller": "stationary_target"}]}
        path = game / "sim-rosie-cache/duel.json"
        path.write_text(json.dumps(duel))
        options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen([executable, "local", "--minecraft-directory", str(game),
                                        "--duel", str(path), "--offline", "--json"],
                                       stdout=subprocess.PIPE, stderr=errors, **options)
            events = queue.Queue()
            reader = threading.Thread(target=lambda: events.put(next((line for line in process.stdout
                if json.loads(line)["event"] in ("policy", "error")), b"")), daemon=True)
            try:
                reader.start()
                event = json.loads(events.get(timeout=120))
                self.assertEqual(event["event"], "policy", event)
                base = f"http://127.0.0.1:{event['port']}"
                with urlopen(base + "/info", timeout=3) as response:
                    info = json.load(response)
                body = {"protocol": 1, "session": "bundled-check", "tick": 0,
                        "contract_hash": info["contract_hash"], "observations": np.zeros((2, 22)).tolist()}
                with urlopen(Request(base + "/act", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3) as response:
                    self.assertEqual(len(json.load(response)["actions"]), 2)
            finally:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
                process.stdout.close()
                reader.join(timeout=5)

        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen([executable, "local-policy", "--minecraft-directory", str(game),
                                        "--checkpoint", checkpoint, "--offline", "--json"],
                                       stdout=subprocess.PIPE, stderr=errors, **options)
            events = queue.Queue()
            reader = threading.Thread(target=lambda: events.put(next((line for line in process.stdout
                if json.loads(line)["event"] in ("policy", "error")), b"")), daemon=True)
            try:
                reader.start()
                event = json.loads(events.get(timeout=120))
                self.assertEqual(event["event"], "policy", event)
                base = f"http://127.0.0.1:{event['port']}"
                with urlopen(base + "/info", timeout=3) as response:
                    info = json.load(response)
                body = {"protocol": 1, "session": "bundled-player", "tick": 0,
                        "contract_hash": info["contract_hash"], "observations": np.zeros((1, 22)).tolist()}
                with urlopen(Request(base + "/act-one", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"}), timeout=3) as response:
                    self.assertEqual(len(json.load(response)["actions"]), 1)
            finally:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
                process.stdout.close()
                reader.join(timeout=5)

    def test_live_duels_accept_only_builtin_scripted_controllers(self):
        contract = from_legacy(SCHEMA_V2)
        contract.pop("observation_dim")
        contract.pop("action_names")
        contract.pop("action_sizes")
        contract["observations"].extend({"feature": feature} for feature in (
            "self.arrow_count_fraction", "self.bow_using", "self.bow_power_fraction",
            "self.bow_equipped", "opponent.bow_equipped", "opponent.bow_using", "arrow.present"))
        contract["requires"].append("bow")
        contract = validate_contract(contract)
        environment = json.loads((ROOT / "simulator/environments/duel.json").read_text())
        environment["profile"] = "configured"
        for player in environment["players"]:
            player["loadout"] = {"weapons": ["minecraft:bow"] + ["minecraft:air"] * 8,
                                 "selected": 0, "arrows": 1}
        path = self.path.parent / "duel.json"
        for controller in ("stationary_target", "basic_melee", "stationary_melee", "tactical_melee",
                           "random_walk", "archer"):
            definition = {"format": "minecraft_sim_duel", "version": 1, "policy": contract,
                          "environment": environment,
                          "fighters": [{"type": "scripted", "controller": controller},
                                       {"type": "scripted", "controller": "stationary_target"}]}
            path.write_text(json.dumps(definition))
            duel = DuelPolicy(path)
            self.assertEqual(duel.info["fighters"][0]["controller"], controller)

        definition["fighters"][0]["controller"] = "custom"
        path.write_text(json.dumps(definition))
        custom_factory = mock.Mock(side_effect=AssertionError("custom factory must not run"))
        with mock.patch.dict(CONTROLLER_REGISTRY, {"custom": custom_factory}):
            with self.assertRaisesRegex(ValueError, "built-in scripted controllers"):
                DuelPolicy(path)
        custom_factory.assert_not_called()

    def test_independent_fighters_and_scripted_round_reset(self):
        definition = {"format": "minecraft_sim_duel", "version": 1, "reference": "policy.pt",
                      "fighters": [{"type": "checkpoint", "path": "policy.pt", "name": "Saved fighter"},
                                   {"type": "scripted", "controller": "basic_melee",
                                    "options": {"attack_interval": 2}, "name": "Practice fighter"}]}
        path = self.path.parent / "duel.json"
        path.write_text(json.dumps(definition))
        duel = DuelPolicy(path, deterministic=True)
        request = {"protocol": 1, "session": "round-one", "tick": 0, "observations": np.zeros((2, 22)).tolist()}
        first = duel.act(request)
        self.assertEqual(first["actions"][0], Policy(self.path, deterministic=True).act(request)["actions"][0])
        self.assertEqual(first["actions"][1][3], 1)
        self.assertEqual(duel.act({**request, "tick": 1})["actions"][1][3], 0)
        self.assertEqual(duel.act({**request, "session": "round-two"})["actions"][1][3], 1)
        self.assertEqual(duel.info["fighters"][0]["name"], "Saved fighter")
        separate = np.zeros((2, 22), dtype=np.float32)
        distance_column = next(index for index, observation in enumerate(duel.contract["observations"])
                       if observation["feature"] == "target.distance_fraction")
        separate[1, distance_column] = 0.8
        separated = duel.act({**request, "session": "separate-observations", "observations": separate.tolist()})
        self.assertEqual(separated["actions"][0], first["actions"][0])
        self.assertEqual(separated["actions"][1][3], 0)
        other = self.path.parent / "other.pt"
        changed = {**self.saved, "metadata": {**self.saved["metadata"], "controls": {"fixed": {"attack": False}}}}
        torch.save(changed, other)
        definition["fighters"][1] = {"type": "checkpoint", "path": "other.pt"}
        definition.pop("reference")
        definition.update(policy=duel.contract, environment=duel.info["environment"])
        definition["max_ticks"] = 1200
        path.write_text(json.dumps(definition))
        paired = DuelPolicy(path, deterministic=True)
        self.assertEqual(paired.info["max_ticks"], 1200)
        self.assertEqual(paired.act({**request, "contract_hash": paired.info["contract_hash"]})["actions"][1][3], 0)
        for invalid in (True, 0, 1_000_001, 20.5):
            path.write_text(json.dumps({**definition, "max_ticks": invalid}))
            with self.assertRaisesRegex(ValueError, "max_ticks"):
                DuelPolicy(path)
        definition["fighters"][1] = {"type": "self_play"}
        path.write_text(json.dumps(definition))
        with self.assertRaisesRegex(ValueError, "Live fighters"):
            DuelPolicy(path)

    def test_offline_independent_launcher_without_rosie_configuration(self):
        source = Policy(self.path)
        game = self.path.parent / "Offline Minecraft with spaces"
        path = game / "sim-rosie-cache" / "duel.json"
        path.parent.mkdir(parents=True)
        definition = {"format": "minecraft_sim_duel", "version": 1,
                      "policy": source.contract, "environment": source.info["environment"],
                      "fighters": [{"type": "scripted", "controller": "stationary_target"},
                                   {"type": "scripted", "controller": "basic_melee"}]}
        path.write_text(json.dumps(definition))
        self.path.unlink()
        self.assertFalse((game / "config/cambium-rosie.json").exists())
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen([sys.executable, "-u", "-B", "-m", "simulator.training.rosie",
                "local", "--minecraft-directory", str(game), "--duel", str(path), "--offline", "--json"],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=errors)
            events = queue.Queue()
            reader = threading.Thread(target=lambda: events.put(next((line for line in process.stdout
                if json.loads(line)["event"] in ("policy", "error")), b"")), daemon=True)
            try:
                reader.start()
                event = json.loads(events.get(timeout=30))
                self.assertEqual(event["event"], "policy", event)
                base = f"http://127.0.0.1:{event['port']}"
                with urlopen(base + "/info", timeout=3) as response:
                    info = json.load(response)
                self.assertEqual(info["device"], "cpu")
                self.assertEqual([fighter["controller"] for fighter in info["fighters"]], ["stationary_target", "basic_melee"])
                body = {"protocol": 1, "session": "offline-menu-duel", "tick": 0,
                        "contract_hash": info["contract_hash"], "observations": np.zeros((2, 22)).tolist()}
                with urlopen(Request(base + "/act", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3) as response:
                    actions = json.load(response)["actions"]
                self.assertEqual(actions[0][3], 0)
                self.assertEqual(actions[1][3], 1)
                with self.assertRaises(HTTPError) as single_denied:
                    urlopen(Request(base + "/act-one", json.dumps({**body, "observations": [[0.0] * 22]}).encode(),
                                    {"Content-Type": "application/json"}), timeout=3)
                self.assertEqual(single_denied.exception.code, 400)
                single_denied.exception.close()
                with self.assertRaises(HTTPError) as denied:
                    urlopen(Request(base + "/reload", b"{}", {"Content-Type": "application/json"}), timeout=3)
                self.assertEqual(denied.exception.code, 403)
                denied.exception.close()
                definition["max_ticks"] = 1600
                definition["fighters"][1]["controller"] = "stationary_target"
                path.write_text(json.dumps(definition))
                headers = {"Content-Type": "application/json", "Authorization": "Bearer " + event["reload_token"]}
                with urlopen(Request(base + "/reload", b"{}", headers), timeout=10) as response:
                    reloaded = json.load(response)
                self.assertTrue(reloaded["reloaded"])
                self.assertGreaterEqual(reloaded["seconds"], 0)
                self.assertIsNone(process.poll())
                with urlopen(base + "/info", timeout=3) as response:
                    self.assertEqual(json.load(response)["max_ticks"], 1600)
                with urlopen(Request(base + "/act", json.dumps({**body, "session": "warm-round"}).encode(), {"Content-Type": "application/json"}), timeout=3) as response:
                    self.assertEqual(json.load(response)["actions"][1][3], 0)
                path.write_text("invalid configuration")
                with self.assertRaises(HTTPError) as invalid:
                    urlopen(Request(base + "/reload", b"{}", headers), timeout=3)
                self.assertEqual(invalid.exception.code, 400)
                invalid.exception.close()
                with urlopen(base + "/info", timeout=3) as response:
                    self.assertEqual(json.load(response)["max_ticks"], 1600)
            finally:
                process.terminate()
                process.wait(timeout=10)
                process.stdout.close()
                if reader.ident is not None:
                    reader.join(timeout=3)

    def test_policy_and_validation(self):
        policy = Policy(self.path, deterministic=True)
        self.assertIsNone(policy.info["training_build"])
        self.assertEqual(len(policy.info["catalogue_hash"]), 64)
        request = {"protocol": 1, "session": "test", "tick": 12, "observations": np.zeros((2, 22)).tolist()}
        response = policy.act(request)
        self.assertEqual(response["session"], "test")
        self.assertEqual(response["tick"], 12)
        self.assertEqual(np.asarray(response["actions"]).shape, (2, 6))
        self.assertTrue((np.asarray(response["actions"]) < np.asarray(SCHEMA["action_sizes"])).all())
        self.assertEqual(response, policy.act(request))
        for invalid in (None, {**request, "tick": True}, {**request, "observations": [[0]]},
                        {**request, "observations": np.full((2, 22), np.nan).tolist()}):
            with self.assertRaises((ValueError, TypeError)):
                policy.act(invalid)
        self.saved["metadata"]["schema"] = {"id": "wrong"}
        torch.save(self.saved, self.path)
        with self.assertRaisesRegex(ValueError, "compatible"):
            Policy(self.path)

    def test_equipment_inference_contract(self):
        self.saved["metadata"]["schema"] = SCHEMA_V2
        self.saved["metadata"]["environment"]["profile"] = "unarmored_sword_axe_shield"
        self.saved["model"] = ActorCritic(31, 16, 9, equipment_heads=True).state_dict()
        torch.save(self.saved, self.path)
        policy = Policy(self.path, deterministic=True)
        self.assertEqual(policy.info["schema"], SCHEMA_V2)
        self.assertTrue(policy.info["minecraft_bridge_supported"])
        request = {"protocol": 1, "session": "v2-round", "tick": 1, "observations": np.zeros((2, 31)).tolist()}
        with make_server(policy, 0) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                with urlopen(base + "/info", timeout=3) as response:
                    self.assertEqual(json.load(response)["schema"], SCHEMA_V2)
                with urlopen(Request(base + "/act", json.dumps(request).encode(),
                                     {"Content-Type": "application/json"}), timeout=3) as response:
                    actions = np.asarray(json.load(response)["actions"])
                self.assertEqual(actions.shape, (2, 8))
                self.assertTrue((actions < np.asarray(SCHEMA_V2["action_sizes"])).all())
            finally:
                server.shutdown()
                worker.join(timeout=3)
        with self.assertRaisesRegex(ValueError, "2,31"):
            policy.act({**request, "observations": np.zeros((2, 22)).tolist()})

    def test_use_item_checkpoint_inference_and_old_weights_rejection(self):
        contract = validate_contract({
            "format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
            "ticks_per_action": 1,
            "observations": [{"feature": "self.arrow_count_fraction"}, {"feature": "self.bow_equipped"}],
            "actions": [{"name": "use_item", "values": [False, True]}],
            "requires": ["bow"],
        })
        environment = self.saved["metadata"]["environment"]
        environment["profile"] = "configured"
        for player in environment["players"]:
            player["loadout"] = {"weapons": ["minecraft:bow"] + ["minecraft:air"] * 8,
                                 "selected": 0, "arrows": 1}
        old_model = self.saved["model"]
        self.saved["metadata"]["schema"] = contract
        self.saved["model"] = ActorCritic(2, 16, action_sizes={"use_item": 2}).state_dict()
        torch.save(self.saved, self.path)
        policy = Policy(self.path, deterministic=True)
        self.assertEqual(policy.contract["action_names"], ["use_item"])
        self.assertEqual(policy.info["environment"]["players"][0]["loadout"]["arrows"], 1)
        request = {"protocol": 1, "session": "bow-round", "tick": 0,
                   "contract_hash": policy.info["contract_hash"],
                   "observations": np.zeros((2, 2)).tolist()}
        self.assertEqual(np.asarray(policy.act(request)["actions"]).shape, (2, 1))

        self.saved["model"] = old_model
        torch.save(self.saved, self.path)
        with self.assertRaisesRegex(ValueError, "model parameters"):
            Policy(self.path)

    def test_loopback_protocol(self):
        with make_server(Policy(self.path), 0) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                with urlopen(base + "/info", timeout=3) as response:
                    self.assertEqual(json.load(response)["schema"], SCHEMA)
                body = {"protocol": 1, "session": "round-1", "tick": 0, "observations": np.zeros((2, 22)).tolist()}
                request = Request(base + "/act", json.dumps(body).encode(), {"Content-Type": "application/json"})
                with urlopen(request, timeout=3) as response:
                    self.assertEqual(json.load(response)["tick"], 0)
                with self.assertRaises(HTTPError) as failure:
                    urlopen(Request(base + "/act", b"{}", {"Content-Type": "application/json"}), timeout=3)
                self.assertEqual(failure.exception.code, 400)
                failure.exception.close()
            finally:
                server.shutdown()
                worker.join(timeout=3)

    def test_rosie_cached_policy_launcher(self):
        from simulator.training.rosie import Rosie, configure

        game = Path(self.directory.name) / "Minecraft with spaces"
        configure(game, "offline-host", "student", "/unreachable/remote/runs")
        folder = Rosie(game).cache / "cached-run"
        folder.mkdir(parents=True)
        torch.save(self.saved, folder / "latest.pt")
        (folder / "manifest.json").write_text(json.dumps({"files": {"latest.pt": {}}, "replays": []}), encoding="utf-8")
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen([sys.executable, "-u", "-B", "-m", "simulator.training.rosie",
                "duel", "--minecraft-directory", str(game), "--offline", "--json"],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=errors)
            try:
                events = queue.Queue()
                reader = threading.Thread(target=lambda: events.put(next((line for line in process.stdout
                    if json.loads(line)["event"] in ("policy", "error")), b"")), daemon=True)
                reader.start()
                event = json.loads(events.get(timeout=30))
                self.assertEqual(event["event"], "policy", event)
                self.assertEqual(event["run"], "cached-run")
                base = f"http://127.0.0.1:{event['port']}"
                with urlopen(base + "/info", timeout=3) as response:
                    info = json.load(response)
                self.assertEqual(info["device"], "cpu")
                self.assertEqual(Path(info["checkpoint"]), folder / "latest.pt")
                body = {"protocol": 1, "session": "rosie-launcher", "tick": 0, "observations": np.zeros((2, 22)).tolist()}
                with urlopen(Request(base + "/act", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3) as response:
                    self.assertEqual(np.asarray(json.load(response)["actions"]).shape, (2, 6))
            finally:
                process.terminate()
                process.wait(timeout=10)
                process.stdout.close()
                reader.join(timeout=3)

    @unittest.skipUnless(os.environ.get("MCSIM_LIVE_JAVA_CHECK") == "1", "Explicit Java/Python integration check")
    def test_java_bridge(self):
        for configured in (False, True, "encoded", "bow"):
            with self.subTest(configured=configured):
                folder = Path(self.directory.name) / str(configured)
                if configured:
                    environment_path = ROOT / "simulator/environments/research-duel.json"
                    if configured in ("encoded", "bow"):
                        definition = json.loads(environment_path.read_text())
                        definition["policy"]["observations"] = [
                            {"feature": "self.weapon_slot", "encoding": "one_hot", "categories": list(range(9))},
                            {"feature": "self.health_fraction", "clip": [0, 0.75]},
                            {"feature": "self.on_ground"}, {"feature": "opponent.on_ground"},
                            {"feature": "self.sprinting", "encoding": "one_hot", "categories": [0, 1]},
                            {"feature": "opponent.sprinting", "encoding": "one_hot", "categories": [1, 0]}]
                        definition["policy"].pop("observation_dim", None)
                        if configured == "bow":
                            definition["policy"] = {
                                "format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
                                "ticks_per_action": 1, "requires": ["bow"],
                                "observations": [{"feature": feature} for feature in FEATURES[89:]],
                                "actions": [{"name": "use_item", "values": [False, True]}],
                            }
                            for player in definition["players"]:
                                player["loadout"] = {"weapons": ["minecraft:bow"] + ["minecraft:air"] * 8,
                                                     "selected": 0, "arrows": 64, "offhand": "minecraft:air"}
                        environment_path = Path(self.directory.name) / "encoded.json"
                        environment_path.write_text(json.dumps(definition))
                    main(["--environment", str(environment_path),
                          "--envs", "1", "--updates", "1", "--rollout-steps", "2", "--max-ticks", "2",
                          "--hidden-dim", "16", "--epochs", "1", "--threads", "1",
                          "--output", str(folder), "--replay-every", "1"])
                    policy = Policy(folder / "latest.pt")
                else:
                    policy = Policy(self.path)
                    with Environment(1, max_ticks=2) as environment:
                        ReplaySamples(environment, 1, folder / "replays", 1).export(policy.model, 1, 0)
                replay = next((folder / "replays").glob("*.jsonl"))
                with make_server(policy, 0) as server:
                    worker = threading.Thread(target=server.serve_forever, daemon=True)
                    worker.start()
                    try:
                        wrapper = ROOT / "cambiummod" / ("gradlew.bat" if os.name == "nt" else "gradlew")
                        command = [str(wrapper)] if os.name == "nt" else ["bash", str(wrapper)]
                        subprocess.run([*command, "-p", str(ROOT / "cambiummod"), "replayAnalysisChecks",
                                        f"-PlivePolicyPort={server.server_port}", f"-PreplayFile={replay}",
                                        "--console=plain", "--no-daemon"], cwd=ROOT, check=True)
                    finally:
                        server.shutdown()
                        worker.join(timeout=3)


class TrainingChecks(unittest.TestCase):
    def test_native_capture_records_selected_pre_step_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.bin"
            with mock.patch.dict(os.environ, {"MCSIM_CAPTURE_STEP": "", "MCSIM_CAPTURE_PATH": str(path)}):
                with Environment(2, max_ticks=10) as environment:
                    environment.step(neutral(2))
                self.assertFalse(path.exists())
            with mock.patch.dict(os.environ, {"MCSIM_CAPTURE_STEP": "2", "MCSIM_CAPTURE_PATH": str(path)}):
                with Environment(2, max_ticks=10) as environment:
                    environment.step(neutral(2))
                    self.assertFalse(path.exists())
                    environment.step(neutral(2))
                    environment.step(neutral(2))
            with path.open("rb") as capture:
                header_size = int.from_bytes(capture.read(8), byteorder=sys.byteorder)
                metadata = json.loads(capture.read(header_size))
                payload = capture.read()
            self.assertEqual((metadata["format"], metadata["version"], metadata["step"]),
                             ("mcsim_native_capture", 1, 2))
            self.assertEqual(metadata["worlds"], 2)
            per_world = (2 * metadata["player_size"] + 4 * metadata["controls_size"]
                         + metadata["world_size"] + metadata["projectile_world_size"]
                         + metadata["arrow_capacity"] * metadata["arrow_size"]
                         + metadata["wind_capacity"] * metadata["wind_size"]
                         + metadata["thrown_capacity"] * metadata["thrown_size"])
            self.assertEqual(len(payload), 2 * per_world + metadata["cell_count"] * metadata["block_size"]
                             + metadata["sine_count"] * 4)

    def test_self_launch_reward_matches_owner_impulse_and_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            definition = json.loads((ROOT / "simulator/environments/duel.json").read_text())
            definition["players"][0]["pitch"] = 90
            definition["players"][0]["loadout"] = {"weapons": ["minecraft:wind_charge"], "selected": 0,
                                                  "wind_charges": 2, "offhand": "minecraft:air"}
            path = Path(directory) / "wind-reward.json"
            path.write_text(json.dumps(definition), encoding="utf-8")
            contract = {"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
                        "ticks_per_action": 1, "observations": [{"feature": "self.velocity.y"}],
                        "actions": [{"name": "use_item", "values": [False, True]}]}
            weights = {name: 0.0 for name in REWARD_NAMES}
            weights["wind_self_launch"] = 2.5
            with Environment(2, environment=path, weights=weights, contract=contract) as environment:
                actions = np.zeros((2, 2, 1), dtype=np.int32)
                actions[0, 0, 0] = 1
                accumulated = 0.0
                for tick in range(12):
                    _, rewards, _, _ = environment.step(actions)
                    actions.fill(0)
                    events = environment.step_behavior_events()
                    np.testing.assert_allclose(rewards, events[:, :, 0] * 2.5, atol=1e-6)
                    self.assertEqual(events[1, :, 0].sum(), 0)
                    self.assertEqual(events[0, 1, 0], 0)
                    accumulated += float(events[0, 0, 0])
                self.assertGreater(accumulated, 0.2)
                self.assertAlmostEqual(environment.reward_totals()["wind_self_launch"], 2.5 * accumulated, places=5)
                environment.reset(np.array([True, False]))
                self.assertEqual(environment.step_behavior_events()[0, 0, 0], 0)

    def test_missed_attack_ready_time_uses_selected_weapon(self):
        with tempfile.TemporaryDirectory() as directory:
            definition = json.loads((ROOT / "simulator/environments/duel.json").read_text())
            for player, weapon in zip(definition["players"], ("minecraft:mace", "minecraft:diamond_sword")):
                player["loadout"] = {"weapons": [weapon] + ["minecraft:air"] * 8,
                                     "selected": 0, "offhand": "minecraft:air"}
            path = Path(directory) / "ready-time.json"
            path.write_text(json.dumps(definition), encoding="utf-8")
            weights = {name: 0.0 for name in REWARD_NAMES}
            weights.update(missed_attack=-0.25, missed_attack_ready_time=-0.6)
            with Environment(1, environment=path, weights=weights) as environment:
                actions = neutral(1)
                actions[0, :, 3] = 1
                _, rewards, _, _ = environment.step(actions)
                self.assertAlmostEqual(float(rewards[0, 0]), -0.25 - 0.6 / (4.0 + float(np.float32(-3.4))), places=5)
                self.assertAlmostEqual(float(rewards[0, 1]), -0.25 - 0.6 / (4.0 + float(np.float32(-2.4))), places=5)
                self.assertAlmostEqual(environment.reward_totals()["missed_attack"], -0.5)
                self.assertAlmostEqual(environment.reward_totals()["missed_attack_ready_time"],
                                       float(rewards.sum()) + 0.5, places=5)
                environment.reset()
                self.assertEqual(environment.step_reward_totals()["missed_attack_ready_time"], 0)

    def test_optional_combat_and_win_speed_weights(self):
        additions = {"missed_attack": -0.1, "shield_disabled": 2.0, "win_speed": 20.0,
                 "arrow_progress": 1.0}
        resolved = resolve_reward_weights({}, additions)
        for name, value in additions.items():
            self.assertEqual(resolved[name], value)
            self.assertEqual(resolve_reward_weights({})[name], 0.0)

    def test_missed_attack_reward_tracks_swings_not_damage(self):
        for hit_block in (False, True):
            with self.subTest(hit_block=hit_block), tempfile.TemporaryDirectory() as directory:
                definition = json.loads((ROOT / "simulator/environments/duel.json").read_text())
                if hit_block:
                    for player in definition["players"]:
                        player["yaw"] = -90
                path = Path(directory) / "missed-swings.json"
                path.write_text(json.dumps(definition), encoding="utf-8")
                weights = {name: 0.0 for name in REWARD_NAMES}
                weights["missed_attack"] = -0.25
                with Environment(2, environment=path, weights=weights) as environment:
                    _, rewards, _, _ = environment.step(neutral(2))
                    np.testing.assert_array_equal(rewards, 0)
                    actions = neutral(2)
                    actions[0, 0, 3] = actions[1, 1, 3] = 1
                    _, rewards, _, _ = environment.step(actions)
                    np.testing.assert_array_equal(rewards, [[-0.25, 0], [0, -0.25]])
                    self.assertEqual(environment.reward_totals()["missed_attack"], -0.5)
                    selected = np.array([[True, False], [False, False]])
                    self.assertEqual(environment.step_reward_totals(selected)["missed_attack"], -0.25)
                    environment.reset(np.array([True, False]))
                    self.assertEqual(environment.step_reward_totals()["missed_attack"], -0.25)
                    _, rewards, _, _ = environment.step(neutral(2))
                    np.testing.assert_array_equal(rewards, 0)
                    environment.reset()
                    environment.step(actions)
                    if not hit_block:
                        _, rewards, _, _ = environment.step(actions)
                        np.testing.assert_array_equal(rewards, 0)

    def test_missed_attack_reward_excludes_absorption_and_immunity(self):
        for absorption in (0, 20):
            with self.subTest(absorption=absorption), tempfile.TemporaryDirectory() as directory:
                definition = json.loads((ROOT / "simulator/environments/duel.json").read_text())
                definition["players"][0]["position"][2] = -0.5
                definition["players"][1]["position"][2] = 1.5
                definition["players"][1]["absorption"] = absorption
                path = Path(directory) / "contact-swings.json"
                path.write_text(json.dumps(definition), encoding="utf-8")
                weights = {name: 0.0 for name in REWARD_NAMES}
                weights["missed_attack"] = -0.25
                with Environment(1, environment=path, weights=weights) as environment:
                    actions = neutral(1)
                    actions[0, 0, 3] = 1
                    _, rewards, _, _ = environment.step(actions)
                    np.testing.assert_array_equal(rewards, 0)
                    health = environment.replay_frame()["actors"][1]["health"]
                    _, rewards, _, _ = environment.step(actions)
                    np.testing.assert_array_equal(rewards, 0)
                    self.assertEqual(environment.replay_frame()["actors"][1]["health"], health)
                    self.assertEqual(environment.reward_totals()["missed_attack"], 0)

    def test_win_speed_reward_scales_with_completion_time(self):
        for winner in (0, 1):
            for limit, wait in ((8, 0), (8, 3), (4, 3)):
                with self.subTest(winner=winner, limit=limit, wait=wait), tempfile.TemporaryDirectory() as directory:
                    definition = json.loads((ROOT / "simulator/environments/duel.json").read_text())
                    definition["players"][0]["position"][2] = -0.5
                    definition["players"][1]["position"][2] = 1.5
                    definition["players"][1 - winner]["health"] = 1
                    path = Path(directory) / "timed-win.json"
                    path.write_text(json.dumps(definition), encoding="utf-8")
                    weights = {name: 0.0 for name in REWARD_NAMES}
                    with Environment(2, environment=path, max_ticks=16, weights=weights) as environment:
                        weights.update(win_speed=20.0, won_duel=50.0)
                        environment.set_rewards(weights)
                        environment.set_max_ticks(limit)
                        for _ in range(wait):
                            _, rewards, _, _ = environment.step(neutral(2))
                            np.testing.assert_array_equal(rewards, 0)
                        actions = neutral(2)
                        actions[:, winner, 3] = 1
                        _, rewards, terminated, truncated = environment.step(actions)
                        self.assertTrue(terminated.all())
                        self.assertFalse(truncated.any())
                        bonus = 20 * (1 - (wait + 1) / limit)
                        expected = np.zeros((2, 2))
                        expected[:, winner] = 50 + bonus
                        np.testing.assert_allclose(rewards, expected)
                        self.assertEqual(environment.reward_totals()["win_speed"], 2 * bonus)
                        self.assertEqual(environment.reward_totals()["won_duel"], 100)
                        self.assertEqual(sum(environment.step_reward_totals().values()), rewards.sum())
                        environment.reset(np.array([True, False]))
                        self.assertEqual(environment.step_reward_totals()["win_speed"], bonus)
                        environment.reset()
                        self.assertEqual(environment.step_reward_totals()["win_speed"], 0)
                        _, rewards, _, _ = environment.step(actions)
                        expected[:, winner] = 50 + 20 * (1 - 1 / limit)
                        np.testing.assert_allclose(rewards, expected)

    def test_win_speed_reward_excludes_timeouts_and_double_losses(self):
        weights = {name: 0.0 for name in REWARD_NAMES}
        weights["win_speed"] = 20
        with Environment(1, max_ticks=2, weights=weights) as environment:
            for _ in range(2):
                _, rewards, terminated, truncated = environment.step(neutral(1))
                np.testing.assert_array_equal(rewards, 0)
            self.assertFalse(terminated.any())
            self.assertTrue(truncated.all())
            self.assertEqual(environment.reward_totals()["win_speed"], 0)
        with tempfile.TemporaryDirectory() as directory:
            definition = json.loads((ROOT / "simulator/environments/duel.json").read_text())
            definition["blocks"] = []
            path = Path(directory) / "double-loss.json"
            path.write_text(json.dumps(definition), encoding="utf-8")
            with Environment(1, environment=path, max_ticks=100, weights=weights) as environment:
                for _ in range(100):
                    _, rewards, terminated, truncated = environment.step(neutral(1))
                    np.testing.assert_array_equal(rewards, 0)
                    if terminated.any() or truncated.any():
                        break
                self.assertTrue(terminated.all())
                self.assertFalse(truncated.any())
                np.testing.assert_array_equal(environment.losses(np.array([True])), True)
                self.assertEqual(environment.reward_totals()["win_speed"], 0)

    def test_native_build_contract_diagnostics(self):
        with Environment(1) as environment:
            self.assertTrue(Path(environment.build_info["library_path"]).is_file())
            self.assertEqual(environment.build_info["abi"], 5)
            self.assertEqual(len(environment.build_info["source_hash"]), 64)
            self.assertEqual(environment.build_info["reward_inputs"], {"base": 10, "events": 2, "shaping": 5})
            self.assertEqual(environment.build_info["reward_totals"], {"base": 13, "shaping": 5})
        with mock.patch("simulator.training.environment.CATALOGUE_HASH", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "catalogue differs"):
                Environment(1)

    def test_shared_policy_contract_fixtures(self):
        fixture = json.loads((ROOT / "contracts/policy_fixtures.json").read_text(encoding="utf-8"))
        descriptor = validate_contract(fixture["policy"])
        actual = [fixture["catalogue"][FEATURES.index(item["feature"])] * item.get("scale", 1)
                  for item in descriptor["observations"]]
        np.testing.assert_allclose(actual, fixture["projected"], atol=fixture["tolerance"], rtol=0)
        self.assertEqual(len(fixture["catalogue"]), len(FEATURES))
        for change in fixture["invalid"]:
            with self.subTest(pointer=change["pointer"]):
                invalid = json.loads(json.dumps(fixture["policy"]))
                path = change["pointer"].strip("/").split("/")
                parent = invalid
                for key in path[:-1]:
                    parent = parent[int(key)] if isinstance(parent, list) else parent[key]
                parent[int(path[-1]) if isinstance(parent, list) else path[-1]] = change["value"]
                with self.assertRaises(ValueError):
                    validate_contract(invalid)

    def test_movement_flags_track_actual_state(self):
        contract = validate_contract({"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3", "ticks_per_action": 1,
            "observations": [{"feature": feature} for feature in
                ("self.on_ground", "opponent.on_ground", "self.sprinting", "opponent.sprinting")],
            "actions": [{"name": "movement", "values": ["neutral", "forward"]},
                        {"name": "sprint", "values": [False, True]}, {"name": "jump", "values": [False, True]}]})
        with Environment(1, contract=contract) as environment:
            np.testing.assert_array_equal(environment.observe()[0], [[1, 1, 0, 0], [1, 1, 0, 0]])
            actions = np.zeros((1, 2, 3), dtype=np.int32)
            actions[:, :, 1] = 1
            observed, *_ = environment.step(actions)
            np.testing.assert_array_equal(observed[0], [[1, 1, 0, 0], [1, 1, 0, 0]])
            actions[0, 0, 0] = 1
            observed, *_ = environment.step(actions)
            np.testing.assert_array_equal(observed[0], [[1, 1, 1, 0], [1, 1, 0, 1]])
            actions[0, 0, 2] = 1
            observed, *_ = environment.step(actions)
            np.testing.assert_array_equal(observed[0], [[0, 1, 1, 0], [1, 0, 0, 1]])
            actions.fill(0)
            observed, *_ = environment.step(actions)
            np.testing.assert_array_equal(observed[0], [[0, 1, 0, 0], [1, 0, 0, 0]])
            for _ in range(30):
                observed, *_ = environment.step(actions)
            np.testing.assert_array_equal(observed[0], [[1, 1, 0, 0], [1, 1, 0, 0]])
            environment.reset()
            np.testing.assert_array_equal(environment.observe()[0], [[1, 1, 0, 0], [1, 1, 0, 0]])

    def test_environment_json_max_ticks_and_cli_override(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            definition = json.loads((ROOT / "simulator/environments/duel.json").read_text())
            definition["max_ticks"] = 2
            environment_path = Path(directory) / "timed-duel.json"
            environment_path.write_text(json.dumps(definition), encoding="utf-8")

            with Environment(1, environment=environment_path) as environment:
                self.assertEqual(environment.max_ticks, 2)
                actions = neutral(1)
                _, _, _, truncated = environment.step(actions)
                self.assertFalse(truncated.any())
                _, _, _, truncated = environment.step(actions)
                self.assertTrue(truncated.all())

            base_options = ["--environment", str(environment_path), "--envs", "1", "--updates", "1",
                            "--rollout-steps", "2", "--hidden-dim", "16", "--epochs", "1",
                            "--threads", "1"]
            for cli_ticks, expected in ((None, 2), (3, 3)):
                output = Path(directory) / f"run-{expected}"
                options = [*base_options, "--output", str(output)]
                if cli_ticks is not None:
                    options.extend(("--max-ticks", str(cli_ticks)))
                main(options)
                saved = json.loads((output / "config.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["options"]["max_ticks"], expected)

    def test_configured_training_resume_replay_and_inference(self):
        self._configured_training_resume_replay_and_inference(False)

    def test_encoded_training_resume_replay_and_inference(self):
        self._configured_training_resume_replay_and_inference(True)

    def _configured_training_resume_replay_and_inference(self, encoded):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory) / "run"
            definition = json.loads((ROOT / "simulator/environments/duel.json").read_text())
            definition["profile"] = "configured"
            for player in definition["players"]:
                player["loadout"] = {"weapons": ["minecraft:iron_sword", "minecraft:netherite_axe", "minecraft:air"],
                    "selected": 1, "offhand": "minecraft:shield", "armor": {"helmet": "minecraft:diamond_helmet"}}
            definition["policy"] = {"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3", "ticks_per_action": 1,
                "observations": [{"feature": "self.weapon_slot"}, {"feature": "self.attack_charge"},
                                 {"feature": "self.health_fraction", "scale": 2}],
                "actions": [{"name": "weapon", "values": [2, 1, 0]}, {"name": "yaw", "values": [0, 15]},
                            {"name": "shield", "values": [False, True]}]}
            if encoded:
                definition["policy"]["observations"][0].update(encoding="one_hot", categories=[2, 1, 0])
                definition["policy"]["observations"][1]["clip"] = [0, 0.5]
            path = Path(directory) / "experiment.json"
            path.write_text(json.dumps(definition))
            if encoded:
                with Environment(1, environment=path) as environment:
                    np.testing.assert_allclose(environment.observe()[0], [[0, 1, 0, 0.5, 2]] * 2)
                with mock.patch("simulator.training.environment.load_library", return_value=mock.Mock(_build_info={})), \
                        self.assertRaisesRegex(RuntimeError, "observation encodings"):
                    Environment(1, environment=path)
            options = ["--environment", str(path), "--envs", "2", "--updates", "1", "--rollout-steps", "4",
                       "--max-ticks", "2", "--hidden-dim", "16", "--epochs", "1", "--threads", "1",
                       "--output", str(output), "--replay-every", "4"]
            main(options)
            checkpoint = output / "latest.pt"
            path.unlink()
            main(options[2:] + ["--resume", str(checkpoint)])
            saved = torch.load(checkpoint, weights_only=True)
            self.assertEqual(saved["update"], 2)
            self.assertEqual(saved["metadata"]["scenario"],
                             {**resolve_scenario(definition), "reward_weights": DEFAULT_REWARD_WEIGHTS})
            self.assertEqual(saved["metadata"]["contract"]["action_names"], ["weapon", "yaw", "shield"])
            files = list((output / "replays").glob("*.jsonl"))
            self.assertEqual(len(files), 2)
            for replay in files:
                header, *frames = [json.loads(line) for line in replay.read_text().splitlines()]
                self.assertEqual(header["environment"], saved["metadata"]["scenario"])
                self.assertEqual(header["policy"], saved["metadata"]["contract"])
                self.assertEqual(frames[0]["actors"][0]["weapon"], "minecraft:netherite_axe")
                self.assertIn(frames[-1]["actors"][0]["controls"]["weapon"], definition["players"][0]["loadout"]["weapons"])
            policy = Policy(checkpoint, deterministic=True)
            dimension = saved["metadata"]["contract"]["observation_dim"]
            self.assertEqual(dimension, 5 if encoded else 3)
            request = {"protocol": 1, "session": "custom", "tick": 0, "observations": np.zeros((2, dimension)).tolist(),
                       "contract_hash": policy.info["contract_hash"]}
            self.assertEqual(np.asarray(policy.act(request)["actions"]).shape, (2, 3))
            with self.assertRaisesRegex(ValueError, "contract"):
                policy.act({**request, "contract_hash": "wrong"})
            if os.name == "nt":
                subprocess.run(["powershell", "-NoProfile", "-File", str(ROOT / "scripts/train.ps1"),
                                "-Resume", str(checkpoint), "-ReplayEvery", "4", "-DryRun"], cwd=ROOT, check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def test_mace_scenario_loadout(self):
        definition = json.loads((ROOT / "simulator/environments/duel.json").read_text())
        definition["players"][0]["loadout"] = {"weapons": ["minecraft:mace"]}
        definition["rules"] = {"simulate_fall_damage": True}
        resolved = resolve_scenario(definition)
        self.assertEqual(resolved["players"][0]["loadout"]["weapons"], ["minecraft:mace"])
        self.assertIs(resolved["rules"]["simulate_fall_damage"], True)

    def test_scenario_rejects_unsupported_conditions(self):
        definition = json.loads((ROOT / "simulator/environments/duel.json").read_text())
        self.assertEqual(resolve_scenario(definition)["rules"], {"simulate_food": False, "natural_regeneration": False})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            for invalid in ({"rules": {"natural_regeneration": True}}, {"requires": ["projectiles"]}):
                changed = {**definition, **invalid}
                with self.assertRaises(ValueError):
                    resolve_scenario(changed)
                path.write_text(json.dumps(changed))
                with self.assertRaises(RuntimeError):
                    Environment(1, environment=path)
        definition["players"][0]["loadout"] = {"weapons": ["minecraft:leather_sword"]}
        with self.assertRaisesRegex(ValueError, "weapon loadout"):
            resolve_scenario(definition)

    def test_configured_native_layout_and_loadouts(self):
        with tempfile.TemporaryDirectory() as directory:
            definition = json.loads((ROOT / "simulator/environments/duel.json").read_text())
            definition["profile"] = "configured"
            for player in definition["players"]:
                player["loadout"] = {"weapons": ["minecraft:iron_sword", "minecraft:netherite_axe", "minecraft:air"],
                    "offhand": "minecraft:shield", "armor": {"helmet": "minecraft:diamond_helmet"}}
            definition["policy"] = {"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3", "ticks_per_action": 1,
                "observations": [{"feature": "self.weapon_slot"}, {"feature": "self.weapon_is_axe"},
                                 {"feature": "self.health_fraction", "scale": 2}],
                "actions": [{"name": "weapon", "values": [2, 1, 0]}, {"name": "yaw", "values": [0, 15]},
                            {"name": "shield", "values": [False, True]}]}
            path = Path(directory) / "configured.json"
            path.write_text(json.dumps(definition))
            with Environment(2, environment=path) as environment:
                self.assertEqual(environment.action_names, ("weapon", "yaw", "shield"))
                initial = environment.observe()
                np.testing.assert_array_equal(initial[:, :, 2], 2)
                actions = np.ones((2, 2, 3), dtype=np.int32)
                result, *_ = environment.step(actions)
                np.testing.assert_array_equal(result[:, :, :2], 1)
                frame = environment.replay_frame()
                self.assertEqual(frame["actors"][0]["weapon"], "minecraft:netherite_axe")
                self.assertEqual(frame["actors"][0]["armor"]["helmet"], "minecraft:diamond_helmet")
                self.assertEqual(environment.scenario["players"][0]["loadout"]["weapons"], definition["players"][0]["loadout"]["weapons"])
                np.testing.assert_array_equal(environment.reset()[0], initial[0])

    def test_bow_native_adapter_draw_cancel_release_observations_and_replay(self):
        contract = validate_contract({
            "format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
            "ticks_per_action": 1,
            "observations": [{"feature": feature} for feature in (
                "self.arrow_count_fraction", "self.bow_using", "self.bow_power_fraction",
                "self.bow_equipped", "opponent.bow_equipped", "arrow.present")],
            "actions": [{"name": "use_item", "values": [False, True]},
                        {"name": "weapon", "values": [0, 1]}],
            "requires": ["bow", "equipment"],
        })
        scene = {
            "format": "minecraft_sim_environment", "version": 1, "minecraft": "26.3",
            "name": "test-bow-adapter", "profile": "configured", "requires": ["movement", "bow"],
            "rules": {"simulate_food": False, "natural_regeneration": False},
            "bounds": {"min": [-6, 0, -4], "size": [13, 4, 9]},
            "blocks": [{"from": [-6, 0, -4], "to": [6, 0, 4], "block": "minecraft:stone"}],
            "players": [
                {"position": [-2, 1, 0], "velocity": [0, -0.0784, 0], "yaw": -90, "pitch": 0,
                 "on_ground": True, "health": 20, "absorption": 0, "attack_strength_ticks": 20,
                 "loadout": {"weapons": ["minecraft:bow", "minecraft:iron_sword"],
                             "selected": 0, "arrows": 1}},
                {"position": [2, 1, 0], "velocity": [0, -0.0784, 0], "yaw": 90, "pitch": 0,
                 "on_ground": True, "health": 20, "absorption": 0, "attack_strength_ticks": 20,
                 "loadout": {"weapons": ["minecraft:iron_sword", "minecraft:air"],
                             "selected": 0, "arrows": 0}},
            ],
            "policy": contract,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bow-environment.json"
            path.write_text(json.dumps(scene), encoding="utf-8")
            with Environment(1, environment=path, max_ticks=100) as environment:
                columns = {entry["feature"]: index for index, entry in enumerate(contract["observations"])}
                use_column = contract["action_names"].index("use_item")
                weapon_column = contract["action_names"].index("weapon")
                actions = np.zeros((1, 2, 2), dtype=np.int32)
                initial = environment.observe()
                self.assertAlmostEqual(initial[0, 0, columns["self.arrow_count_fraction"]], 1 / 64)
                self.assertEqual(initial[0, 0, columns["self.bow_equipped"]], 1)
                self.assertEqual(initial[0, 1, columns["self.bow_equipped"]], 0)
                self.assertEqual(initial[0, 1, columns["arrow.present"]], 0)

                actions[0, 0, use_column] = 1
                for _ in range(20):
                    observed, *_ = environment.step(actions)
                self.assertEqual(observed[0, 0, columns["self.bow_using"]], 1)
                self.assertGreater(observed[0, 0, columns["self.bow_power_fraction"]], 0.9)

                cancel = actions.copy()
                cancel[0, 0, use_column] = 0
                cancel[0, 0, weapon_column] = 1
                cancelled, *_ = environment.step(cancel)
                self.assertEqual(cancelled[0, 0, columns["self.bow_using"]], 0)
                self.assertEqual(cancelled[0, 0, columns["self.arrow_count_fraction"]], 1 / 64)
                cancel_frame = environment.replay_frame()
                self.assertEqual(cancel_frame["actors"][0]["weapon"], "minecraft:iron_sword")
                self.assertEqual(cancel_frame["actors"][0]["controls"]["select_slot"], 1)
                self.assertFalse(cancel_frame["actors"][0]["controls"]["use_item"])

                reselect = actions.copy()
                reselect[0, 0, weapon_column] = 0
                environment.step(reselect)
                for _ in range(20):
                    observed, *_ = environment.step(actions)
                release = actions.copy()
                release[0, 0, use_column] = 0
                observed, *_ = environment.step(release)
                self.assertEqual(observed[0, 0, columns["self.arrow_count_fraction"]], 0)
                self.assertEqual(observed[0, 1, columns["arrow.present"]], 1)
                scenario = environment.episode_scenario(0)
                self.assertEqual(scenario["players"][0]["loadout"]["arrows"], 1)
                frame = environment.replay_frame()
                self.assertEqual(frame["actors"][0]["weapon"], "minecraft:bow")
                self.assertFalse(frame["actors"][0]["controls"]["use_item"])

    # Source: BowItem.releaseUsing and AbstractArrow.tick supply speed/drop; aim rewards are simulator-only.
    def test_bow_aim_rewards_compensate_drop_and_lead(self):
        contract = validate_contract({
            "format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
            "ticks_per_action": 1,
            "observations": [{"feature": "self.health_fraction"}],
            "actions": [{"name": "use_item", "values": [False, True]}],
            "requires": ["movement", "equipment", "bow"],
        })
        weights = {name: 0.0 for name in DEFAULT_REWARD_WEIGHTS}
        weights["good_aim"] = 2.0
        scores = {}
        cases = (("flat", 0, -90, 64, 0), ("arc", -4, -90, 64, 0),
                 ("away", -4, 90, 64, 0), ("empty", -4, -90, 0, 0),
                 ("unled", -4, -90, 64, 1), ("led", -4, -77, 64, 1),
                 ("wall", -4, -90, 64, 0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bow-aim.json"
            for name, pitch, yaw, arrows, target_speed in cases:
                with self.subTest(case=name):
                    definition = json.loads((ROOT / "simulator/environments/duel.json").read_text())
                    definition.update(profile="configured", policy=contract)
                    definition["bounds"] = {"min": [-20, 0, -8], "size": [41, 12, 17]}
                    definition["blocks"] = [{"from": [-20, 0, -8], "to": [20, 0, 8],
                                             "block": "minecraft:stone"}]
                    if name == "wall":
                        definition["blocks"].append({"from": [0, 1, -8], "to": [0, 6, 8],
                                                     "block": "minecraft:stone"})
                    for player, position, facing in zip(definition["players"], (-16, 16), (yaw, 90)):
                        player.update(position=[position, 1, 0], yaw=facing, pitch=pitch)
                        player["loadout"] = {"weapons": ["minecraft:bow"], "selected": 0, "arrows": arrows}
                    definition["players"][1]["velocity"][2] = target_speed
                    path.write_text(json.dumps(definition), encoding="utf-8")
                    with Environment(1, environment=path, max_ticks=100, weights=weights) as environment:
                        _, rewards, terminated, truncated = environment.step(np.zeros((1, 2, 1), dtype=np.int32))
                        self.assertFalse(terminated.any() or truncated.any())
                        self.assertTrue(np.all((rewards >= 0) & (rewards <= 2 / 100)))
                        self.assertAlmostEqual(environment.reward_totals()["good_aim"], float(rewards.sum()), places=6)
                        scores[name] = float(rewards[0, 0])
        self.assertGreater(scores["arc"], scores["flat"])
        self.assertGreater(scores["led"], scores["unled"])
        for name in ("away", "empty", "wall"):
            self.assertEqual(scores[name], 0.0)

    def test_policy_contract_is_descriptive_not_a_schema_id(self):
        contract = from_legacy(SCHEMA_V2)
        contract.pop("action_names")
        contract.pop("action_sizes")
        contract.pop("observation_dim")
        contract["observations"] = [{"feature": "self.attack_charge"}, {"feature": "self.health_fraction", "scale": 2}]
        contract["actions"] = [{"name": "yaw", "values": [-15, 0, 15]}, {"name": "attack", "values": [False, True]}]
        parsed = validate_contract(contract)
        self.assertEqual(parsed["action_names"], ["yaw", "attack"])
        self.assertEqual(parsed["action_sizes"], [3, 2])
        self.assertEqual(parsed["observation_dim"], 2)
        contract["observations"][0]["feature"] = "arbitrary_server_secret"
        with self.assertRaisesRegex(ValueError, "Unsupported observation"):
            validate_contract(contract)

    def test_shield_disable_is_observed_not_lowered(self):
        with tempfile.TemporaryDirectory() as directory:
            with Environment(1) as original:
                definition = original.definition
            definition["players"][0]["position"][2] = -0.5
            definition["players"][1]["position"][2] = 1.5
            path = Path(directory) / "close-duel.json"
            path.write_text(json.dumps(definition), encoding="utf-8")
            with Environment(2, environment=path, schema="v2") as environment:
                actions = neutral(2, 8)
                actions[0, 0, 6] = 1
                actions[:, 1, 7] = 1
                for _ in range(6):
                    raised, *_ = environment.step(actions)
                np.testing.assert_array_equal(raised[:, 1, 30], 1)
                np.testing.assert_array_equal(raised[:, :, 26:28], 0)
                actions[0, 0, 3] = 1
                actions[1, 1, 7] = 0
                disabled, *_ = environment.step(actions)
                self.assertEqual(disabled[0, 1, 26], 1)
                self.assertEqual(disabled[0, 0, 27], 1)
                self.assertAlmostEqual(disabled[0, 1, 28], 0.99)
                self.assertEqual(disabled[0, 1, 24], 0)
                self.assertEqual(disabled[0, 1, 0], raised[0, 1, 0])
                self.assertEqual(disabled[1, 1, 24], 0)
                np.testing.assert_array_equal(disabled[1, :, 26:29], 0)
                actions[0, 0, 3] = 0
                for _ in range(98):
                    pending, *_ = environment.step(actions)
                self.assertEqual(pending[0, 1, 26], 1)
                self.assertEqual(pending[0, 0, 27], 1)
                recovered, *_ = environment.step(actions)
                self.assertEqual(recovered[0, 1, 26], 0)
                self.assertEqual(recovered[0, 0, 27], 0)
                environment.reset([True, False])
                np.testing.assert_array_equal(environment.observe()[0, :, 26:29], 0)

    def test_equipment_training_updates_and_schema_resume(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory) / "run"
            options = ["--schema", "v2", "--envs", "2", "--updates", "1", "--rollout-steps", "4",
                       "--max-ticks", "2", "--hidden-dim", "16", "--epochs", "1",
                       "--minibatch-size", "7", "--threads", "1", "--seed", "7", "--output", str(output)]
            torch.manual_seed(7)
            initial = ActorCritic(31, 16, 9, equipment_heads=True).state_dict()
            main(options)
            path = output / "latest.pt"
            saved = torch.load(path, weights_only=True)
            self.assertEqual(saved["metadata"]["schema"], SCHEMA_V2)
            self.assertEqual(saved["samples"], 16)
            for head in ("weapon_head.weight", "shield_head.weight"):
                self.assertFalse(torch.equal(saved["model"][head], initial[head]))
            main(options + ["--resume", str(path)])
            self.assertEqual(torch.load(path, weights_only=True)["update"], 2)
            with self.assertRaisesRegex(ValueError, "does not match"):
                main(options + ["--schema", "v1", "--resume", str(path)])
            main(options + ["--resume", str(path), "--replay-every", "4"])
            with Environment(1, schema="v2") as environment:
                self.assertEqual(ReplaySamples(environment, 1, output / "replays", 1).interval, 1)
            self.assertTrue(list((output / "replays").glob("*.jsonl")))

    def test_equipment_schema_actions_observations_and_reset(self):
        torch.set_num_threads(1)
        model = ActorCritic(31, 16, 9, equipment_heads=True)
        observations = torch.zeros(3, 31)
        logits, _ = model(observations)
        self.assertEqual(list(logits), SCHEMA_V2["action_names"])
        self.assertEqual([head.shape[-1] for head in logits.values()], SCHEMA_V2["action_sizes"])
        actions = [torch.zeros(3, dtype=torch.long) for _ in logits]
        log_probs, _, entropy = model.evaluate_actions(observations, *actions)
        expected = [torch.distributions.Categorical(logits=head) for head in logits.values()]
        torch.testing.assert_close(log_probs, sum(head.log_prob(action) for head, action in zip(expected, actions)))
        torch.testing.assert_close(entropy, sum(head.entropy() for head in expected))
        with Environment(2, schema="v2") as environment:
            initial = environment.observe()
            self.assertEqual(initial.shape, (2, 2, 31))
            self.assertEqual(environment.definition["profile"], "unarmored_sword_axe_shield")
            np.testing.assert_array_equal(initial[:, :, 22:29], 0)
            actions = neutral(2, 8)
            actions[0, 0, 6:] = [1, 1]
            first, *_ = environment.step(actions)
            self.assertEqual(first[0, 0, 22], 1)
            self.assertEqual(first[0, 1, 23], 1)
            self.assertEqual(first[0, 0, 24], 1)
            self.assertEqual(first[0, 1, 25], 1)
            second, *_ = environment.step(actions)
            self.assertGreater(second[0, 0, 29], first[0, 0, 29])
            self.assertLess(second[0, 0, 29], 1)
            self.assertGreater(second[0, 0, 30], first[0, 0, 30])
            invalid = actions.copy()
            invalid[-1, -1, -1] = 2
            with self.assertRaises(ValueError):
                environment.step(invalid)
            result = environment.native.mcsim_step(environment._handle, invalid.ctypes.data,
                environment._observations.ctypes.data, environment._rewards.ctypes.data,
                environment._terminated.ctypes.data, environment._truncated.ctypes.data)
            self.assertNotEqual(result, 0)
            np.testing.assert_array_equal(environment.observe(), second)
            reset = environment.reset([True, False])
            np.testing.assert_array_equal(reset[0], initial[0])
            np.testing.assert_array_equal(reset[1], second[1])
            frame = environment.replay_frame()
            self.assertEqual(frame["actors"][0]["weapon"], "minecraft:diamond_sword")
            self.assertTrue(frame["actors"][0]["shield_equipped"])
            self.assertEqual(frame["actors"][0]["shield_cooldown"], 0)
            self.assertFalse(frame["actors"][0]["controls"]["use_item"])
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "equipment-duel.json"
                path.write_text(json.dumps(environment.definition), encoding="utf-8")
                with Environment(1, environment=path, schema="v2") as restored:
                    self.assertEqual(restored.definition, environment.definition)
                    np.testing.assert_array_equal(restored.observe()[0], initial[0])
                with self.assertRaisesRegex(RuntimeError, "profile"):
                    Environment(1, environment=path)

    def test_model_keeps_legacy_shape(self):
        torch.set_num_threads(1)
        self.assertEqual(ActorCritic(22, 16)(torch.zeros(3, 22))[0]["movement"].shape, (3, 8))
        self.assertEqual(ActorCritic(22, 16, 9)(torch.zeros(3, 22))[0]["movement"].shape, (3, 9))

    def test_parallel_worlds_and_selective_reset(self):
        with Environment(3, max_ticks=2, weights=unshaped_weights()) as environment:
            initial = environment.observe()
            actions = neutral(3)
            actions[0, 0, 0] = 1
            observations, _, terminated, truncated = environment.step(actions)
            self.assertFalse(terminated.any() or truncated.any())
            self.assertFalse(np.array_equal(observations[0], initial[0]))
            np.testing.assert_array_equal(observations[1:], initial[1:])
            reset = environment.reset([True, False, False])
            np.testing.assert_array_equal(reset[0], initial[0])
            np.testing.assert_array_equal(reset[1:], observations[1:])
            _, rewards, terminated, truncated = environment.step(neutral(3))
            np.testing.assert_array_equal(truncated, [False, True, True])
            np.testing.assert_array_equal(rewards[1:], -20)
            self.assertFalse(terminated.any())
            with self.assertRaisesRegex(RuntimeError, "Reset finished"):
                environment.step(actions)
            environment.reset(truncated)
            _, _, _, truncated = environment.step(neutral(3))
            np.testing.assert_array_equal(truncated, [True, False, False])

    def test_action_validation_is_atomic(self):
        with Environment(2) as environment:
            before = environment.observe()
            invalid = neutral(2)
            invalid[1, 1, 0] = 9
            with self.assertRaises(ValueError):
                environment.step(invalid)
            with self.assertRaises(ValueError):
                environment.step(neutral(2).astype(float))
            with self.assertRaises(ValueError):
                environment.reset([256, 0])
            np.testing.assert_array_equal(environment.observe(), before)

    def test_shaping_rewards_are_episode_budgets(self):
        defaults = DEFAULT_REWARD_WEIGHTS
        weights = {name: 0.0 for name in defaults}
        weights["pitch_control"] = 2.0
        for max_ticks in (2, 8):
            with self.subTest(max_ticks=max_ticks), Environment(1, max_ticks=max_ticks, weights=weights) as environment:
                total = np.zeros((1, 2), dtype=np.float64)
                for _ in range(max_ticks):
                    _, rewards, terminated, truncated = environment.step(neutral(1))
                    total += rewards
                self.assertFalse(terminated.any())
                self.assertTrue(truncated.all())
                np.testing.assert_allclose(total, 2.0)
                self.assertAlmostEqual(environment.reward_totals()["pitch_control"], 4.0)
                environment.reset()
                _, rewards, _, _ = environment.step(neutral(1))
                np.testing.assert_allclose(rewards, 2.0 / max_ticks)

    def test_environment_reward_overrides(self):
        definition = json.loads((ROOT / "simulator/environments/duel.json").read_text())
        definition["reward_weights"] = {"pitch_control": 2.0, "proximity": 0.0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reward-overrides.json"
            path.write_text(json.dumps(definition), encoding="utf-8")
            with Environment(1, environment=path) as environment:
                self.assertEqual(environment.weights["pitch_control"], 2.0)
                self.assertEqual(environment.weights["proximity"], 0.0)
                self.assertEqual(environment.weights["doing_damage"], 10.0)
                self.assertEqual(environment.definition["reward_weights"], environment.weights)

    def test_shaping_reward_weights_reach_native_trainer(self):
        defaults = DEFAULT_REWARD_WEIGHTS
        cases = {"good_aim": 1.0, "proximity": 6 / 7, "pitch_control": 1.0,
                 "aim_hold": 1.0, "extreme_pitch_penalty": -1.0}
        for name, expected in cases.items():
            with self.subTest(reward=name), tempfile.TemporaryDirectory() as directory:
                definition = json.loads((ROOT / "simulator/environments/duel.json").read_text())
                if name == "extreme_pitch_penalty":
                    for player in definition["players"]:
                        player["pitch"] = 65.0
                path = Path(directory) / "shaping-duel.json"
                path.write_text(json.dumps(definition), encoding="utf-8")
                weights = {key: 0.0 for key in defaults}
                weights[name] = -1.0 if name == "extreme_pitch_penalty" else 1.0
                with Environment(1, environment=path, weights=weights) as environment:
                    _, rewards, terminated, truncated = environment.step(neutral(1))
                    self.assertFalse(terminated.any() or truncated.any())
                    np.testing.assert_allclose(rewards, expected / environment.max_ticks, rtol=1e-6)
                    self.assertAlmostEqual(environment.reward_totals()[name], 2 * expected / environment.max_ticks, places=6)

    def test_sweep_hit_reward_uses_health_fraction(self):
        definition = json.loads((ROOT / "simulator/environments/duel.json").read_text())
        definition["players"][0]["position"][2] = -0.5
        definition["players"][1]["position"][2] = 1.5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sweep-duel.json"
            path.write_text(json.dumps(definition), encoding="utf-8")
            with Environment(1, environment=path, weights=unshaped_weights()) as environment:
                actions = neutral(1)
                health_before = environment.replay_frame()["actors"][1]["health"]
                actions[0, 0, 3] = 1
                _, rewards, terminated, truncated = environment.step(actions)
                self.assertFalse(terminated.any() or truncated.any())
                health_lost = health_before - environment.replay_frame()["actors"][1]["health"]
                self.assertGreater(health_lost, 0)
                fraction = health_lost / 20
                np.testing.assert_allclose(rewards, [[10 * fraction, -2 * fraction]], rtol=1e-6)
                totals = environment.reward_totals()
                self.assertAlmostEqual(totals["doing_damage"], 10 * fraction, places=6)
                self.assertAlmostEqual(totals["taking_damage"], -2 * fraction, places=6)
                self.assertAlmostEqual(sum(totals.values()), float(rewards.sum()), places=6)

    def test_critical_and_weak_hits_reward_actual_health_loss(self):
        for critical in (False, True):
            with self.subTest(critical=critical), tempfile.TemporaryDirectory() as directory:
                definition = json.loads((ROOT / "simulator/environments/duel.json").read_text())
                definition["players"][0]["position"][2] = -0.5
                definition["players"][1]["position"][2] = 1.5
                for player in definition["players"]:
                    if critical:
                        player["position"][1] = 2.0
                        player["velocity"][1] = -0.1
                        player["on_ground"] = False
                    else:
                        player["attack_strength_ticks"] = 0
                path = Path(directory) / "hit-duel.json"
                path.write_text(json.dumps(definition), encoding="utf-8")
                with Environment(1, environment=path, weights=unshaped_weights()) as environment:
                    actions = neutral(1)
                    if critical:
                        environment.step(actions)
                    health_before = environment.replay_frame()["actors"][1]["health"]
                    actions[0, 0, 3] = 1
                    _, rewards, terminated, truncated = environment.step(actions)
                    self.assertFalse(terminated.any() or truncated.any())
                    health_lost = health_before - environment.replay_frame()["actors"][1]["health"]
                    self.assertGreater(health_lost, 0)
                    if critical:
                        self.assertGreater(health_lost, 7)
                    else:
                        self.assertLess(health_lost, 7)
                    fraction = health_lost / 20
                    np.testing.assert_allclose(rewards, [[10 * fraction, -2 * fraction]], rtol=1e-6)
                    self.assertAlmostEqual(environment.reward_totals()["doing_damage"], 10 * fraction, places=6)
                    self.assertAlmostEqual(environment.reward_totals()["taking_damage"], -2 * fraction, places=6)

    def test_armor_reduces_damage_reward(self):
        definition = json.loads((ROOT / "simulator/environments/duel.json").read_text())
        definition["players"][0]["position"][2] = -0.5
        definition["players"][1]["position"][2] = 1.5
        definition["players"][1]["loadout"] = {
            "weapons": ["minecraft:diamond_sword"], "armor": {"chestplate": "minecraft:diamond_chestplate"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "armored-duel.json"
            path.write_text(json.dumps(definition), encoding="utf-8")
            with Environment(1, environment=path, weights=unshaped_weights()) as environment:
                actions = neutral(1)
                actions[0, 0, 3] = 1
                _, rewards, _, _ = environment.step(actions)
                health_lost = 20 - environment.replay_frame()["actors"][1]["health"]
                self.assertGreater(health_lost, 0)
                self.assertLess(health_lost, 7)
                np.testing.assert_allclose(rewards, [[10 * health_lost / 20, -2 * health_lost / 20]], rtol=1e-6)

    def test_misses_and_absorption_do_not_count_as_health_damage(self):
        for absorbed in (False, True):
            with self.subTest(absorbed=absorbed), tempfile.TemporaryDirectory() as directory:
                definition = json.loads((ROOT / "simulator/environments/duel.json").read_text())
                definition["players"][0]["position"][2] = -0.5
                definition["players"][1]["position"][2] = 1.5
                if absorbed:
                    definition["players"][1]["absorption"] = 20
                else:
                    definition["players"][0]["yaw"] = 180
                path = Path(directory) / "no-health-damage.json"
                path.write_text(json.dumps(definition), encoding="utf-8")
                with Environment(1, environment=path, max_ticks=1, weights=unshaped_weights()) as environment:
                    actions = neutral(1)
                    actions[0, 0, 3] = 1
                    _, rewards, terminated, truncated = environment.step(actions)
                    self.assertFalse(terminated.any())
                    self.assertTrue(truncated.all())
                    np.testing.assert_array_equal(rewards, -20)
                    totals = environment.reward_totals()
                    self.assertEqual(totals["doing_damage"], 0)
                    self.assertEqual(totals["taking_damage"], 0)
                    self.assertEqual(totals["no_damage_penalty"], -20)
                    self.assertEqual(environment.replay_frame()["actors"][1]["health"], 20)

    def test_old_native_reward_abi_requires_rebuild(self):
        with mock.patch("simulator.training.environment.ctypes.CDLL") as library:
            for version in (1, 2, 3, 4):
                library.return_value.mcsim_api_version.return_value = version
                with self.subTest(version=version), self.assertRaisesRegex(RuntimeError, "rebuild for wind event rewards"):
                    Environment(1)

    def test_shield_block_reward_excludes_axes(self):
        for attacker in (0, 1):
            for weapon in (0, 1):
                with self.subTest(attacker=attacker, weapon=weapon), tempfile.TemporaryDirectory() as directory:
                    definition = json.loads((ROOT / "simulator/environments/duel.json").read_text())
                    definition["players"][0]["position"][2] = -0.5
                    definition["players"][1]["position"][2] = 1.5
                    path = Path(directory) / "shield-duel.json"
                    path.write_text(json.dumps(definition), encoding="utf-8")
                    with Environment(1, environment=path, schema="v2",
                                     weights={**unshaped_weights(), "blocked_with_shield": 7,
                                              "shield_disabled": 3, "missed_attack": -0.25}) as environment:
                        defender = 1 - attacker
                        actions = neutral(1, 8)
                        actions[0, attacker, 6] = weapon
                        actions[0, defender, 7] = 1
                        for _ in range(6):
                            _, rewards, _, _ = environment.step(actions)
                            np.testing.assert_array_equal(rewards, 0)
                        actions[0, attacker, 3] = 1
                        _, rewards, terminated, truncated = environment.step(actions)
                        self.assertFalse(terminated.any() or truncated.any())
                        expected = np.zeros((1, 2), dtype=np.float32)
                        expected[0, defender] = 7 if weapon == 0 else 0
                        expected[0, attacker] = 3 if weapon == 1 else 0
                        np.testing.assert_array_equal(rewards, expected)
                        self.assertEqual(environment.reward_totals()["blocked_with_shield"], 7 if weapon == 0 else 0)
                        self.assertEqual(environment.reward_totals()["shield_disabled"], 3 if weapon == 1 else 0)
                        self.assertEqual(environment.reward_totals()["missed_attack"], 0)
                        state = environment.replay_frame()["actors"][defender]
                        self.assertEqual(state["health"], 20)
                        self.assertEqual(state["shield_cooldown"] > 0, weapon == 1)
                        actions[0, attacker, 3] = 0
                        _, rewards, _, _ = environment.step(actions)
                        np.testing.assert_array_equal(rewards, 0)

    def test_hit_reward_configuration_and_validation(self):
        definition = json.loads((ROOT / "simulator/environments/duel.json").read_text())
        definition["players"][0]["position"][2] = -0.5
        definition["players"][1]["position"][2] = 1.5
        weights = unshaped_weights()
        weights["doing_damage"] = 8
        weights["taking_damage"] = -4
        weights["blocked_with_shield"] = 7
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "configured-rewards.json"
            path.write_text(json.dumps(definition), encoding="utf-8")
            with Environment(1, environment=path, weights=weights) as environment:
                actions = neutral(1)
                actions[0, 0, 3] = 1
                _, rewards, _, _ = environment.step(actions)
                np.testing.assert_allclose(rewards, [[2.8, -1.4]], rtol=1e-6)
            for name in ("sweep_hit_multiplier", "critical_hit_multiplier"):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unknown reward weights"):
                    Environment(1, weights={**weights, name: 1})
            for name in ("doing_damage", "taking_damage", "blocked_with_shield", "missed_attack", "shield_disabled", "win_speed"):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "finite"):
                    Environment(1, weights={**weights, name: float("nan")})

    def test_terminal_reward_and_cached_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            with Environment(1) as original:
                definition = original.definition
            for player in definition["players"]:
                player["health"] = 1
            definition["players"][0]["position"][2] = -0.5
            definition["players"][1]["position"][2] = 1.5
            path = Path(directory) / "duel.json"
            path.write_text(json.dumps(definition), encoding="utf-8")
            with Environment(2, environment=path) as environment:
                path.unlink()
                actions = neutral(2)
                actions[:, :, 3] = 1
                _, rewards, terminated, truncated = environment.step(actions)
                self.assertTrue(terminated.all())
                self.assertFalse(truncated.any())
                np.testing.assert_allclose(rewards, [[90.5, -20.1], [90.5, -20.1]], rtol=1e-6)
                totals = environment.reward_totals()
                self.assertEqual(totals["won_duel"], 180)
                self.assertEqual(totals["death"], -40)
                self.assertEqual(totals["doing_damage"], 1)
                self.assertAlmostEqual(totals["taking_damage"], -0.2, places=6)
                np.testing.assert_allclose(sum(totals.values()), rewards.sum(dtype=np.float64), rtol=1e-6)
                environment.reset()
                self.assertEqual(environment.observe()[0, 1, 0], np.float32(0.05))

    def test_outside_arena_is_task_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            with Environment(1) as original:
                definition = original.definition
            definition["blocks"] = definition["blocks"][:1]
            path = Path(directory) / "open-arena.json"
            path.write_text(json.dumps(definition), encoding="utf-8")
            with Environment(1, environment=path, max_ticks=100) as environment:
                actions = neutral(1)
                actions[0, 0, 0] = 5
                for _ in range(100):
                    _, rewards, terminated, truncated = environment.step(actions)
                    if terminated.any():
                        break
                self.assertTrue(terminated.all())
                self.assertFalse(truncated.any())
                np.testing.assert_array_equal(rewards, [[-20, 90]])

    def test_gae_timeout_bootstraps_but_does_not_cross_reset(self):
        rewards = torch.tensor([[1.0], [2.0]])
        values = torch.tensor([[0.5], [0.7]])
        next_values = torch.tensor([[3.0], [9.0]])
        terminated = torch.tensor([[False], [True]])
        truncated = torch.tensor([[True], [False]])
        advantages, returns = compute_gae(rewards, values, next_values, terminated, truncated, 1, 1)
        torch.testing.assert_close(advantages, torch.tensor([[3.5], [1.3]]))
        torch.testing.assert_close(returns, torch.tensor([[4.0], [2.0]]))

    def test_step_reward_totals_masks_and_resets(self):
        with Environment(2, max_ticks=1) as environment:
            self.assertEqual(set(environment.step_reward_totals()), set(REWARD_NAMES))
            self.assertEqual(sum(environment.step_reward_totals().values()), 0)
            _, rewards, _, _ = environment.step(neutral(2))
            snapshot = environment.reward_totals()
            np.testing.assert_allclose(sum(environment.step_reward_totals().values()), rewards.sum(dtype=np.float64),
                                       rtol=1e-6, atol=2e-5)
            for world in range(2):
                for player in range(2):
                    selected = np.zeros((2, 2), dtype=bool)
                    selected[world, player] = True
                    np.testing.assert_allclose(sum(environment.step_reward_totals(selected).values()),
                                               rewards[world, player], rtol=1e-6, atol=2e-5)
            environment.reset(np.array([True, False]))
            self.assertEqual(environment.reward_totals(), snapshot)
            np.testing.assert_allclose(sum(environment.step_reward_totals().values()),
                                       rewards[1].sum(dtype=np.float64), rtol=1e-6, atol=2e-5)
            environment.reset()
            self.assertEqual(sum(environment.step_reward_totals().values()), 0)
            _, rewards, _, _ = environment.step(neutral(2))
            np.testing.assert_allclose(sum(environment.step_reward_totals().values()), rewards.sum(dtype=np.float64),
                                       rtol=1e-6, atol=2e-5)
            for invalid in (np.zeros(2), np.full((2, 2), 0.5), np.full((2, 2), 2)):
                with self.assertRaisesRegex(ValueError, "reward mask"):
                    environment.step_reward_totals(invalid)
            with mock.patch.object(environment.native, "mcsim_step_reward_totals", None):
                self.assertIsNone(environment.step_reward_totals())

    def test_reward_totals_and_rollout_boundaries(self):
        with Environment(2, max_ticks=1) as environment:
            self.assertEqual(sum(environment.reward_totals().values()), 0)
            environment.step(neutral(2))
            totals = environment.reward_totals()
            self.assertEqual(totals["episode_timeout"], -40)
            self.assertEqual(totals["no_damage_penalty"], -40)
            observations = environment.reset()
            self.assertEqual(environment.reward_totals(), totals)
            model = ActorCritic(22, 16, 9)
            rollout, _, _, _ = collect_rollout(model, environment, observations, 3, torch.device("cpu"))
            self.assertEqual(rollout["reward_types"]["episode_timeout"], -120)
            self.assertEqual(rollout["reward_types"]["no_damage_penalty"], -120)
            np.testing.assert_allclose(sum(rollout["reward_types"].values()),
                                       rollout["rewards"].sum(dtype=torch.float64).item(), rtol=1e-6, atol=2e-5)
            with mock.patch.object(environment.native, "mcsim_reward_totals", None):
                self.assertIsNone(environment.reward_totals())

    def test_training_updates_weights_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = str(Path(directory) / "run")
            options = ["--device", "cpu", "--envs", "2", "--updates", "1", "--rollout-steps", "4",
                       "--max-ticks", "2", "--hidden-dim", "16", "--epochs", "1",
                       "--minibatch-size", "7", "--threads", "1", "--seed", "7", "--output", output]
            torch.manual_seed(7)
            initial = ActorCritic(22, 16, 9).state_dict()
            main(options)
            checkpoint_path = str(Path(output) / "latest.pt")
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            self.assertEqual(checkpoint["samples"], 16)
            self.assertEqual(json.loads((Path(output) / "config.json").read_text())["training"],
                             {"start_update": 0, "target_update": 1})
            self.assertEqual(checkpoint["metadata"]["reward_semantics"], REWARD_SEMANTICS)
            self.assertFalse((Path(output) / "replays").exists())
            self.assertTrue(any(not torch.equal(value, initial[key]) for key, value in checkpoint["model"].items()))
            self.assertEqual(Policy(checkpoint_path).info["training_build"], checkpoint["metadata"]["native_build"])
            checkpoint["metadata"]["native_build"]["library_path"] = "other-machine/artifact"
            checkpoint["metadata"]["native_build"]["source_hash"] = "0" * 64
            torch.save(checkpoint, checkpoint_path)
            main(options + ["--resume", checkpoint_path])
            resumed = torch.load(checkpoint_path, weights_only=True)
            self.assertNotEqual(resumed["metadata"]["native_build"]["library_path"], "other-machine/artifact")
            self.assertEqual(resumed["update"], 2)
            self.assertEqual(resumed["samples"], 32)
            self.assertEqual(json.loads((Path(output) / "config.json").read_text())["training"],
                             {"start_update": 1, "target_update": 2})
            metrics = [json.loads(line) for line in (Path(output) / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(metrics), 2)
            for row in metrics:
                np.testing.assert_allclose(sum(row["reward_types"].values()), row["reward_total"], rtol=1e-6, atol=2e-5)
                np.testing.assert_allclose(row["reward_mean"] * 16, row["reward_total"], rtol=1e-6, atol=2e-5)
            for semantics in (None, "episode_shaping_budgets_v1", "health_fraction_damage_episode_shaping_budgets_v2",
                              "health_fraction_damage_combat_events_win_speed_v3",
                              "opponent_damage_bow_aim_episode_budgets_v4"):
                incompatible = {**checkpoint, "metadata": checkpoint["metadata"].copy()}
                if semantics is None:
                    incompatible["metadata"].pop("reward_semantics")
                else:
                    incompatible["metadata"]["reward_semantics"] = semantics
                torch.save(incompatible, Path(directory) / "old-rewards.pt")
                with self.subTest(semantics=semantics), self.assertRaisesRegex(ValueError, "reward semantics"):
                    main(options + ["--resume", str(Path(directory) / "old-rewards.pt")])
            checkpoint["metadata"]["schema"]["id"] = "incompatible"
            torch.save(checkpoint, Path(directory) / "wrong.pt")
            with self.assertRaisesRegex(ValueError, "does not match"):
                main(options + ["--resume", str(Path(directory) / "wrong.pt")])

    def test_periodic_replays_are_complete_and_resume_cadence(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory) / "run"
            replays = Path(directory) / "minecraft-replays"
            options = ["--envs", "2", "--updates", "1", "--rollout-steps", "4", "--max-ticks", "2",
                       "--epochs", "1", "--hidden-dim", "16", "--threads", "1", "--output", str(output),
                       "--replay-every", "3", "--replay-dir", str(replays)]
            main(options)
            self.assertEqual(len(list(replays.glob("*.jsonl"))), 1)
            checkpoint = output / "latest.pt"
            self.assertEqual(torch.load(checkpoint, weights_only=True)["completed_rounds"], 4)
            main(options + ["--resume", str(checkpoint)])
            files = sorted(replays.glob("*.jsonl"))
            self.assertEqual(len(files), 2)
            for path, round_number in zip(files, (3, 6)):
                header, *frames = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(header["version"], 3)
                self.assertEqual(header["training"]["completed_rounds"], round_number)
                self.assertEqual(header["training"]["sample"], "evaluation")
                self.assertEqual([frame["tick"] for frame in frames], [0, 1, 2])
                self.assertTrue(frames[-1]["truncated"])
                for player in range(2):
                    self.assertEqual(frames[0]["actors"][player]["position"], header["environment"]["players"][player]["position"])
                    self.assertEqual(len(frames[-1]["actors"][player]["position"]), 3)
            self.assertFalse(list(replays.glob("*.part")))

    def test_replay_sampling_preserves_training_rng_and_experience(self):
        torch.set_num_threads(1)
        model = ActorCritic(22, 16, 9)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            with Environment(2, max_ticks=3) as baseline, Environment(2, max_ticks=3) as sampled, Environment(1, max_ticks=3) as sample:
                torch.manual_seed(42)
                expected, _, _, _ = collect_rollout(model, baseline, baseline.observe(), 5, torch.device("cpu"))
                expected_rng = torch.get_rng_state().clone()
                replays = ReplaySamples(sample, 1, directory, 1)
                torch.manual_seed(42)
                actual, _, _, _ = collect_rollout(model, sampled, sampled.observe(), 5,
                                                 torch.device("cpu"), replays)
                self.assertEqual(replays.exported, 2)
                self.assertTrue(model.training)
                torch.testing.assert_close(torch.get_rng_state(), expected_rng)
                for name in expected:
                    torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    @unittest.skipUnless(os.getenv("MCSIM_TEST_CUDA") == "1", "Set MCSIM_TEST_CUDA=1 on a CUDA build")
    def test_cuda_parity_and_selective_reset(self):
        generator = np.random.default_rng(123)
        for schema in ("v1", "v2"):
            with self.subTest(schema=schema), Environment(3, "cpu", max_ticks=5, schema=schema) as cpu, \
                    Environment(3, "cuda", max_ticks=5, schema=schema) as gpu:
                for step in range(30):
                    actions = generator.integers(0, cpu.action_sizes, size=(3, 2, cpu.action_count), dtype=np.int32)
                    cpu_result = cpu.step(actions)
                    gpu_result = gpu.step(actions)
                    np.testing.assert_allclose(cpu_result[0], gpu_result[0], atol=2e-5, rtol=0)
                    for index in range(1, 4):
                        np.testing.assert_array_equal(cpu_result[index], gpu_result[index])
                    reset = cpu_result[2] | cpu_result[3]
                    if step % 3 == 0:
                        reset[0] = True
                    np.testing.assert_allclose(cpu.reset(reset), gpu.reset(reset), atol=2e-5, rtol=0)


if __name__ == "__main__":
    unittest.main()