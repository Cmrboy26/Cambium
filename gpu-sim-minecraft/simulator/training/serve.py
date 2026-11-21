"""Serve a trained duel policy to the Minecraft mod over loopback HTTP."""

import argparse
import hashlib
import hmac
import io
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

import numpy as np
import torch

from backend.fast_rl_model import ActorCritic
from simulator.training.bridge import validate_bridge
from simulator.training.environment import SCHEMA, SCHEMAS
from simulator.training.contract import (CATALOGUE_HASH, fingerprint, from_legacy, reject_structured_bridge_checkpoint,
                                         resolve_scenario, validate_bridge_observation_descriptor, validate_contract)
from simulator.training.policies import fixed_indices
from simulator.training.controllers import CheckpointController, create_controller

LIVE_SCRIPTED_CONTROLLERS = frozenset((
    "stationary_target", "basic_melee", "stationary_melee", "tactical_melee", "random_walk", "archer",
))


def reject_bridge_checkpoint(manifest):
    if any(isinstance(manifest.get(name), dict) and "experimental_bridge" in manifest[name]
           for name in ("scenario", "environment")):
        raise ValueError("Bridge checkpoints are not supported by the live duel adapter")


def inference_observations(request, schema, contract_hash, players=2):
    if (not isinstance(request, dict) or request.get("protocol") != 1
            or not isinstance(request.get("session"), str) or not 1 <= len(request["session"]) <= 64
            or type(request.get("tick")) is not int or not 0 <= request["tick"] < 2**53):
        raise ValueError("Expected protocol 1, session ID and nonnegative integer tick")
    observations = np.asarray(request.get("observations"), dtype=np.float32)
    if (request.get("contract_hash") is not None or schema.get("format") == "minecraft_sim_policy") \
            and request.get("contract_hash") != contract_hash:
        raise ValueError("Inference request policy contract fingerprint does not match")
    features = schema["observation_dim"]
    if observations.shape != (players, features) or not np.isfinite(observations).all():
        raise ValueError(f"Expected finite observations with shape [{players},{features}]")
    return observations


class Policy:
    def __init__(self, checkpoint, device="cpu", deterministic=False, seed=1):
        self.path = Path(checkpoint).resolve()
        self.device = torch.device(device)
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA inference requested but unavailable")
        saved = torch.load(self.path, map_location=self.device, weights_only=True)
        manifest = saved.get("metadata", {})
        reject_structured_bridge_checkpoint(manifest)
        reject_bridge_checkpoint(manifest)
        schema = next((candidate for candidate in SCHEMAS.values() if manifest.get("schema") == candidate), None)
        if schema is None:
            try:
                schema = validate_contract(manifest.get("schema"))
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError("Checkpoint is not a compatible standalone two-player PPO policy") from error
        if (saved.get("format") != "minecraft_sim_ppo" or saved.get("version") != 1
            or schema is None or manifest.get("players_per_environment") != 2
                or manifest.get("learner") not in ("both_shared_policy", "opponent_pool")):
            raise ValueError("Checkpoint is not a compatible standalone two-player PPO policy")
        self.schema = schema
        self.contract = from_legacy(schema) if schema.get("format") != "minecraft_sim_policy" else validate_contract(schema)
        if manifest.get("contract", self.contract) != self.contract:
            raise ValueError("Checkpoint policy descriptors disagree")
        hidden = manifest.get("hidden_dim")
        limit = manifest.get("max_ticks")
        if type(hidden) is not int or not 1 <= hidden <= 4096 or type(limit) is not int or not 1 <= limit <= 1000000:
            raise ValueError("Invalid checkpoint dimensions or round limit")
        definition = resolve_scenario(manifest.get("scenario", manifest.get("environment", {})))
        for action in self.contract["actions"]:
            if action["name"] == "weapon" and any(slot >= len(player["loadout"]["weapons"])
                    for slot in action["values"] for player in definition["players"]):
                raise ValueError("Policy weapon action outside scenario loadout")
        json.dumps(definition, allow_nan=False)
        torch.manual_seed(seed)
        self.model = ActorCritic(self.schema["observation_dim"], hidden, movement_bins=9,
            action_sizes=dict(zip(self.contract["action_names"], self.contract["action_sizes"]))).to(self.device)
        try:
            self.model.load_state_dict(saved["model"], strict=True)
        except (KeyError, TypeError, RuntimeError) as error:
            raise ValueError("Checkpoint model parameters do not match its policy contract") from error
        if not all(torch.isfinite(parameter).all() for parameter in self.model.parameters()):
            raise ValueError("Checkpoint contains non-finite parameters")
        self.model.eval()
        self.deterministic = deterministic
        self.fixed = fixed_indices(self.contract, manifest.get("controls", {"fixed": {}}))
        self.lock = threading.Lock()
        self.info = {"protocol": 1, "schema": self.schema, "environment": definition,
                     "contract": self.contract, "contract_hash": fingerprint(self.contract),
                     "catalogue_hash": CATALOGUE_HASH, "training_build": manifest.get("native_build"),
                     "minecraft_bridge_supported": True,
                     "checkpoint": str(self.path), "update": saved.get("update", 0),
                     "max_ticks": limit, "device": str(self.device), "deterministic": deterministic,
                     "controls": manifest.get("controls", {"fixed": {}})}
        if "experiment" in manifest:
            self.info["experiment"] = manifest["experiment"]

    @torch.inference_mode()
    def act(self, request, players=2):
        if players == 1 and (not isinstance(request, dict) or request.get("contract_hash") != self.info["contract_hash"]):
            raise ValueError("Single-player inference requires the negotiated policy contract hash")
        observations = inference_observations(request, self.schema, self.info["contract_hash"], players)
        with self.lock:
            logits, _ = self.model(torch.as_tensor(observations, device=self.device))
            actions = torch.stack([
                torch.full((players,), self.fixed[name], dtype=torch.long, device=self.device) if name in self.fixed
                else logits[name].argmax(-1) if self.deterministic
                else torch.distributions.Categorical(logits=logits[name]).sample()
                for name in self.schema["action_names"]], dim=-1).cpu().tolist()
        return {"protocol": 1, "session": request["session"], "tick": request["tick"],
            "contract_hash": self.info["contract_hash"], "actions": actions}


class BridgePolicy:
    def __init__(self, checkpoint, device="cpu", deterministic=False, seed=1):
        self.path = Path(checkpoint).resolve()
        data = self.path.read_bytes()
        saved = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        if not isinstance(saved, dict) or not isinstance(saved.get("metadata"), dict):
            raise ValueError("Invalid Bridge checkpoint")
        manifest = saved["metadata"]
        descriptor = validate_bridge_observation_descriptor(manifest.get("bridge_observation"))
        schema = manifest.get("schema")
        contract = (from_legacy(schema) if schema in SCHEMAS.values() else validate_contract(schema))
        scenario = resolve_scenario(manifest.get("scenario", {}))
        validate_bridge(scenario.get("experimental_bridge"), scenario)
        limit = manifest.get("max_ticks")
        if type(limit) is not int or not 1 <= limit <= 1000000:
            raise ValueError("Invalid checkpoint round limit")
        self.controller = CheckpointController(self.path, contract, 2, device, seed, deterministic,
                               expected_sha256=hashlib.sha256(data).hexdigest(),
                                               bridge_observation=descriptor)
        descriptor_hash = hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(",", ":"),
                                                     allow_nan=False).encode()).hexdigest()
        self.info = {"protocol": 2, "schema": contract, "contract": contract, "contract_hash": fingerprint(contract),
                     "bridge_observation": descriptor, "bridge_observation_hash": descriptor_hash,
                     "environment": scenario, "max_ticks": limit, "catalogue_hash": CATALOGUE_HASH,
                     "minecraft_bridge_supported": False, "checkpoint": str(self.path),
                     "update": saved.get("update", 0), "device": str(self.controller.device),
                     "deterministic": deterministic}
        self.lock = threading.Lock()

    def observations(self, request):
        if (not isinstance(request, dict) or request.get("protocol") != 2
                or not isinstance(request.get("session"), str) or not 1 <= len(request["session"]) <= 64
                or type(request.get("tick")) is not int or not 0 <= request["tick"] < 2**53
                or request.get("contract_hash") != self.info["contract_hash"]
                or request.get("bridge_observation_hash") != self.info["bridge_observation_hash"]):
            raise ValueError("Expected protocol 2, session, tick and negotiated Bridge contract hashes")
        players = request.get("players")
        if (not isinstance(players, list) or not 1 <= len(players) <= 2
                or any(type(player) is not int or player not in (0, 1) for player in players)
                or len(set(players)) != len(players)):
            raise ValueError("Bridge inference requires unique player seats")
        source = request.get("observations")
        if not isinstance(source, dict) or set(source) != {"scalar", "match", "voxels"}:
            raise ValueError("Bridge inference requires scalar, match and voxel observations")
        scalar = np.asarray(source["scalar"])
        match = np.asarray(source["match"])
        voxels = np.asarray(source["voxels"])
        count = len(players)
        if (scalar.shape != (count, self.controller.contract["observation_dim"]) or scalar.dtype.kind not in "iuf"
                or match.shape != (count, len(self.info["bridge_observation"]["match_features"])) or match.dtype.kind not in "iuf"
                or voxels.shape != (count, 5, 9, 9) or voxels.dtype.kind not in "iu"
                or not np.isfinite(scalar).all() or not np.isfinite(match).all()
                or (voxels < 0).any() or (voxels >= 5).any()):
            raise ValueError("Invalid structured Bridge observation shapes or values")
        inputs = {"scalar": scalar.astype(np.float32), "match": match.astype(np.float32),
                  "voxels": voxels.astype(np.uint8)}
        if not np.isfinite(inputs["scalar"]).all() or not np.isfinite(inputs["match"]).all():
            raise ValueError("Bridge observations exceed float32 range")
        return players, inputs

    def act(self, request):
        players, inputs = self.observations(request)
        with self.lock:
            actions = self.controller.act(inputs, indices=np.asarray(players, dtype=np.int64)).tolist()
        return {"protocol": 2, "session": request["session"], "tick": request["tick"], "players": players,
                "contract_hash": self.info["contract_hash"],
                "bridge_observation_hash": self.info["bridge_observation_hash"], "actions": actions}


class BridgeDuelPolicy:
    def __init__(self, first, second, device="cpu", deterministic=False, seed=1):
        self.policies = [BridgePolicy(first, device, deterministic, seed),
                         BridgePolicy(second, device, deterministic, seed + 1)]
        self.info = deepcopy(self.policies[0].info)
        if any(self.policies[1].info[key] != self.info[key]
               for key in ("contract_hash", "bridge_observation_hash", "catalogue_hash")):
            raise ValueError("Bridge fighters require matching policy and observation contracts")
        self.info["fighters"] = [{"name": str(policy.path.name), "checkpoint": str(policy.path),
                                  "update": policy.info["update"]} for policy in self.policies]
        self.lock = threading.Lock()

    def act(self, request):
        players, inputs = self.policies[0].observations(request)
        with self.lock:
            actions = [self.policies[seat].controller.act(
                {name: value[index:index + 1] for name, value in inputs.items()},
                indices=np.asarray([seat], dtype=np.int64))[0].tolist()
                for index, seat in enumerate(players)]
        return {"protocol": 2, "session": request["session"], "tick": request["tick"], "players": players,
                "contract_hash": self.info["contract_hash"],
                "bridge_observation_hash": self.info["bridge_observation_hash"], "actions": actions}


class DuelPolicy:
    def __init__(self, configuration, device="cpu", deterministic=False, seed=1):
        self.path = Path(configuration).resolve()
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if (not isinstance(value, dict) or set(value) - {"format", "version", "reference", "policy", "environment", "fighters", "max_ticks"}
                or value.get("format") != "minecraft_sim_duel" or type(value.get("version")) is not int
                or value["version"] != 1):
            raise ValueError("Expected minecraft_sim_duel version 1")
        fighters = value.get("fighters")
        if not isinstance(fighters, list) or len(fighters) != 2:
            raise ValueError("A duel requires exactly two fighters")
        if "reference" in value:
            reference = Path(value["reference"])
            reference = reference if reference.is_absolute() else self.path.parent / reference
            source = Policy(reference, device, deterministic, seed)
            self.info = deepcopy(source.info)
            self.schema, self.contract = source.schema, source.contract
        else:
            self.contract = validate_contract(value.get("policy"))
            self.schema = self.contract
            self.info = {"protocol": 1, "schema": self.schema, "contract": self.contract,
                         "contract_hash": fingerprint(self.contract), "catalogue_hash": CATALOGUE_HASH,
                         "minecraft_bridge_supported": True, "device": device, "controls": {"fixed": {}},
                         "checkpoint": str(self.path), "update": 0, "deterministic": deterministic}
        if "environment" in value:
            self.info["environment"] = resolve_scenario(value["environment"])
        if "environment" not in self.info:
            raise ValueError("A duel requires an environment or reference checkpoint")
        scenario = self.info["environment"]
        limit = value.get("max_ticks", scenario.get("max_ticks", self.info.get("max_ticks", 300)))
        if type(limit) is not int or not 1 <= limit <= 1_000_000:
            raise ValueError("Duel max_ticks must be an integer from 1 to 1000000")
        self.info["max_ticks"] = limit
        for action in self.contract["actions"]:
            if action["name"] == "weapon" and any(slot >= len(player["loadout"]["weapons"])
                    for slot in action["values"] for player in scenario["players"]):
                raise ValueError("Policy weapon action outside duel environment loadout")
        self.controllers = []
        self.info["fighters"] = []
        for index, fighter in enumerate(fighters):
            if not isinstance(fighter, dict):
                raise ValueError("Fighter must be an object")
            definition = {key: deepcopy(item) for key, item in fighter.items() if key != "name"}
            if definition.get("type") not in ("checkpoint", "scripted"):
                raise ValueError("Live fighters must be checkpoint policies or scripted controllers")
            if definition["type"] == "scripted" and definition.get("controller") not in LIVE_SCRIPTED_CONTROLLERS:
                raise ValueError("Live duels accept only built-in scripted controllers")
            if definition["type"] == "checkpoint":
                path = Path(definition.get("path", ""))
                definition["path"] = str(path if path.is_absolute() else self.path.parent / path)
                definition.setdefault("deterministic", deterministic)
                data = Path(definition["path"]).read_bytes()
                saved = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
                if isinstance(saved, dict) and isinstance(saved.get("metadata"), dict):
                    reject_bridge_checkpoint(saved["metadata"])
                digest = hashlib.sha256(data).hexdigest()
                if definition.get("sha256") is not None and definition["sha256"] != digest:
                    raise ValueError("Checkpoint SHA256 differs from the resolved plan")
                definition["sha256"] = digest
            controller = create_controller(definition, self.contract, 1, device, seed + index)
            self.controllers.append(controller)
            self.info["fighters"].append({"name": str(fighter.get("name", f"Fighter {index + 1}")),
                                          **controller.describe})
        self.lock = threading.Lock()
        self.session = None

    @torch.inference_mode()
    def act(self, request):
        observations = inference_observations(request, self.schema, self.info["contract_hash"])
        with self.lock:
            if request["session"] != self.session:
                for controller in self.controllers:
                    controller.reset(np.ones(1, dtype=bool))
                self.session = request["session"]
            actions = [controller.act(observations[index:index + 1])[0].tolist()
                       for index, controller in enumerate(self.controllers)]
        return {"protocol": 1, "session": request["session"], "tick": request["tick"],
                "contract_hash": self.info["contract_hash"], "actions": actions}


def make_server(policy, port, reload_policy=None, reload_token=None):
    active = [policy]
    reload_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def reply(self, status, value):
            body = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(200, active[0].info) if self.path == "/info" else self.reply(404, {"error": "Not found"})

        def do_POST(self):
            if self.path not in ("/act", "/act-one") and not (self.path == "/reload" and reload_policy is not None):
                self.reply(404, {"error": "Not found"})
                return
            if self.path == "/reload" and (not reload_token or not hmac.compare_digest(
                    self.headers.get("Authorization", ""), "Bearer " + reload_token)):
                self.reply(403, {"error": "Unauthorized reload"})
                return
            try:
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Content-Type must be application/json")
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 16384:
                    raise ValueError("Request must contain 1..16384 bytes")
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                if self.path == "/reload":
                    if request != {}:
                        raise ValueError("Reload uses only the configured local duel file")
                    started = time.monotonic()
                    with reload_lock:
                        replacement = reload_policy()
                        active[0] = replacement
                    self.reply(200, {"reloaded": True, "seconds": time.monotonic() - started})
                elif self.path == "/act-one":
                    if not isinstance(active[0], Policy):
                        raise ValueError("Single-player inference requires a standalone checkpoint")
                    self.reply(200, active[0].act(request, players=1))
                else:
                    self.reply(200, active[0].act(request))
            except (ValueError, TypeError, OverflowError) as error:
                self.reply(400, {"error": str(error)})
            except (RuntimeError, OSError) as error:
                self.reply(500, {"error": str(error)})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?", help="Standalone trainer checkpoint (trusted files only)")
    parser.add_argument("--duel", type=Path, help="Local two-fighter duel configuration")
    parser.add_argument("--bridge-protocol", action="store_true", help="Serve experimental structured Bridge protocol 2")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--deterministic", action="store_true", help="Use argmax instead of training-style stochastic actions")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    options = parser.parse_args(argv)
    if not 1 <= options.port <= 65535 or options.threads < 1:
        parser.error("Invalid port or thread count")
    if (options.checkpoint is None) == (options.duel is None):
        parser.error("Choose a checkpoint or --duel, not both")
    if options.bridge_protocol and options.checkpoint is None:
        parser.error("Bridge protocol requires a standalone checkpoint")
    torch.set_num_threads(options.threads)
    policy = (DuelPolicy(options.duel, options.device, options.deterministic, options.seed) if options.duel else
              BridgePolicy(options.checkpoint, options.device, options.deterministic, options.seed) if options.bridge_protocol else
              Policy(options.checkpoint, options.device, options.deterministic, options.seed))
    with make_server(policy, options.port) as server:
        print(f"Policy: {policy.path}\nLoopback inference: http://127.0.0.1:{options.port}", flush=True)
        if policy.info["minecraft_bridge_supported"]:
            print("Minecraft: /simduel arena OR /simduel pos1, /simduel pos2, /simduel world", flush=True)
        else:
            print("Experimental Bridge protocol ready; the Minecraft live Bridge adapter is not yet supported.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()