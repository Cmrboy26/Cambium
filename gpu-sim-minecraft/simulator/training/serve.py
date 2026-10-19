"""Serve a trained duel policy to the Minecraft mod over loopback HTTP."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

import numpy as np
import torch

from backend.fast_rl_model import ActorCritic
from simulator.training.environment import SCHEMA, SCHEMAS
from simulator.training.contract import CATALOGUE_HASH, fingerprint, from_legacy, resolve_scenario, validate_contract


class Policy:
    def __init__(self, checkpoint, device="cpu", deterministic=False, seed=1):
        self.path = Path(checkpoint).resolve()
        self.device = torch.device(device)
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA inference requested but unavailable")
        saved = torch.load(self.path, map_location=self.device, weights_only=True)
        manifest = saved.get("metadata", {})
        schema = next((candidate for candidate in SCHEMAS.values() if manifest.get("schema") == candidate), None)
        if schema is None:
            try:
                schema = validate_contract(manifest.get("schema"))
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError("Checkpoint is not a compatible standalone two-player PPO policy") from error
        if (saved.get("format") != "minecraft_sim_ppo" or saved.get("version") != 1
            or schema is None or manifest.get("players_per_environment") != 2
                or manifest.get("learner") != "both_shared_policy"):
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
        self.model.load_state_dict(saved["model"], strict=True)
        if not all(torch.isfinite(parameter).all() for parameter in self.model.parameters()):
            raise ValueError("Checkpoint contains non-finite parameters")
        self.model.eval()
        self.deterministic = deterministic
        self.lock = threading.Lock()
        self.info = {"protocol": 1, "schema": self.schema, "environment": definition,
                     "contract": self.contract, "contract_hash": fingerprint(self.contract),
                     "catalogue_hash": CATALOGUE_HASH, "training_build": manifest.get("native_build"),
                     "minecraft_bridge_supported": True,
                     "checkpoint": str(self.path), "update": saved.get("update", 0),
                     "max_ticks": limit, "device": str(self.device), "deterministic": deterministic}

    @torch.inference_mode()
    def act(self, request):
        if (not isinstance(request, dict) or request.get("protocol") != 1
                or not isinstance(request.get("session"), str) or not 1 <= len(request["session"]) <= 64
                or type(request.get("tick")) is not int or not 0 <= request["tick"] < 2**53):
            raise ValueError("Expected protocol 1, session ID and nonnegative integer tick")
        observations = np.asarray(request.get("observations"), dtype=np.float32)
        if (request.get("contract_hash") is not None or self.schema.get("format") == "minecraft_sim_policy") \
                and request.get("contract_hash") != self.info["contract_hash"]:
            raise ValueError("Inference request policy contract fingerprint does not match")
        features = self.schema["observation_dim"]
        if observations.shape != (2, features) or not np.isfinite(observations).all():
            raise ValueError(f"Expected finite observations with shape [2,{features}]")
        with self.lock:
            logits, _ = self.model(torch.as_tensor(observations, device=self.device))
            actions = torch.stack([
                logits[name].argmax(-1) if self.deterministic else torch.distributions.Categorical(logits=logits[name]).sample()
                for name in self.schema["action_names"]], dim=-1).cpu().tolist()
        return {"protocol": 1, "session": request["session"], "tick": request["tick"],
            "contract_hash": self.info["contract_hash"], "actions": actions}


def make_server(policy, port):
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
            self.reply(200, policy.info) if self.path == "/info" else self.reply(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/act":
                self.reply(404, {"error": "Not found"})
                return
            try:
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Content-Type must be application/json")
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 16384:
                    raise ValueError("Request must contain 1..16384 bytes")
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                self.reply(200, policy.act(request))
            except (ValueError, TypeError, OverflowError) as error:
                self.reply(400, {"error": str(error)})
            except RuntimeError as error:
                self.reply(500, {"error": str(error)})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Standalone trainer checkpoint (trusted files only)")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--deterministic", action="store_true", help="Use argmax instead of training-style stochastic actions")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    options = parser.parse_args(argv)
    if not 1 <= options.port <= 65535 or options.threads < 1:
        parser.error("Invalid port or thread count")
    torch.set_num_threads(options.threads)
    policy = Policy(options.checkpoint, options.device, options.deterministic, options.seed)
    with make_server(policy, options.port) as server:
        print(f"Policy: {policy.path}\nLoopback inference: http://127.0.0.1:{options.port}", flush=True)
        if policy.info["minecraft_bridge_supported"]:
            print("Minecraft: /simduel arena OR /simduel pos1, /simduel pos2, /simduel world", flush=True)
        else:
            print("V2 inference API ready; the current Minecraft mod only supports v1 sword policies.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()