"""Host-staged two-player training adapter over the shared C++/CUDA simulator."""

import ctypes
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

import numpy as np
from simulator.training.contract import CATALOGUE_HASH, from_legacy, validate_contract
from simulator.training.legacy import ACTION_NAMES, ACTION_SIZES, SCHEMA, SCHEMA_V2, SCHEMAS
from simulator.training.timing import CollectionTimer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAX_TICKS = 300
MAX_MAX_TICKS = 1_000_000
BASE_REWARD_NAMES = ("doing_damage", "taking_damage", "won_duel", "death",
                     "episode_timeout", "no_damage_penalty", "missed_attack",
                     "shield_disabled", "win_speed")
SHAPING_REWARD_NAMES = ("good_aim", "proximity", "pitch_control", "aim_hold", "extreme_pitch_penalty")
REWARD_NAMES = (*BASE_REWARD_NAMES, "blocked_with_shield", *SHAPING_REWARD_NAMES)
REWARD_TYPE_NAMES = (*BASE_REWARD_NAMES, "blocked_with_shield")
DEFAULT_REWARD_WEIGHTS = json.loads(Path(__file__).with_name("reward_weights.json").read_text(encoding="utf-8"))
REWARD_SEMANTICS = "health_fraction_damage_combat_events_win_speed_v3"


def resolve_max_ticks(environment=None, override=None):
    if override is not None:
        max_ticks = override
    else:
        path = Path(environment) if environment else ROOT / "simulator" / "environments" / "duel.json"
        definition = json.loads(path.read_text(encoding="utf-8"))
        max_ticks = definition.get("max_ticks", DEFAULT_MAX_TICKS)
    if type(max_ticks) is not int or not 1 <= max_ticks <= MAX_MAX_TICKS:
        raise ValueError(f"max_ticks must be an integer from 1 to {MAX_MAX_TICKS}")
    return max_ticks


def resolve_reward_weights(definition, overrides=None):
    resolved = DEFAULT_REWARD_WEIGHTS.copy()
    for source in (definition.get("reward_weights", {}), {} if overrides is None else overrides):
        if not isinstance(source, dict):
            raise ValueError("reward_weights must be an object")
        unknown = set(source) - set(REWARD_NAMES)
        if unknown:
            raise ValueError(f"Unknown reward weights: {', '.join(sorted(unknown))}")
        for name, value in source.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Reward weight {name} must be a finite number")
            resolved[name] = float(value)
    return resolved


def load_library(library=None, device="cpu"):
    suffix = ".dll" if sys.platform == "win32" else (".dylib" if sys.platform == "darwin" else ".so")
    if library is not None:
        candidates = [Path(library).resolve()]
    else:
        builds = sorted((ROOT / "simulator").glob("build*"),
                        key=lambda path: ("cuda" not in path.name if device == "cuda" else path.name != "build", str(path)))
        candidates = [build / configuration / ("minecraft_sim_training" + suffix)
                      for build in builds for configuration in ("Release", "", "Debug")]
    errors = []
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    for path in candidates:
        if not path.is_file():
            continue
        directory = os.add_dll_directory(str(path.parent)) if sys.platform == "win32" else None
        try:
            native = ctypes.CDLL(str(path))
            native.mcsim_api_version.restype = ctypes.c_int
            if native.mcsim_api_version() != 3:
                raise RuntimeError("Unsupported training library ABI; rebuild for combat-event and win-speed rewards")
            native.mcsim_build_info.restype = ctypes.c_char_p
            build_info = json.loads(native.mcsim_build_info())
            if build_info.get("catalogue_hash") != CATALOGUE_HASH:
                raise RuntimeError("Native policy catalogue differs from Python; rebuild the selected library")
            expected_layout = {"scalar_bytes": {"observation": 4, "action": 4, "reward_total": 8, "done": 1},
                               "reward_inputs": {"base": len(BASE_REWARD_NAMES), "shaping": len(SHAPING_REWARD_NAMES)},
                               "reward_totals": {"base": len(REWARD_TYPE_NAMES), "shaping": len(SHAPING_REWARD_NAMES)},
                               "players_per_environment": 2}
            if any(build_info.get(name) != value for name, value in expected_layout.items()):
                raise RuntimeError("Native buffer layout differs from Python; rebuild the selected library")
            if device == "cuda" and not native.mcsim_cuda_available():
                raise RuntimeError("Library has no available CUDA backend")
            native._dll_directory = directory
            native._build_info = {**build_info, "library_path": str(path), "device": device}
            return native
        except (OSError, AttributeError, RuntimeError, ValueError, TypeError) as error:
            errors.append(f"{path}: {error}")
            if directory is not None:
                directory.close()
    detail = "\n".join(errors) or "No built training library found."
    raise RuntimeError(f"{detail}\nBuild target minecraft_sim_training; use --library to select it explicitly.")


def actor_lost(actor, scenario):
    position = actor["position"]
    lower, size = scenario["bounds"]["min"], scenario["bounds"]["size"]
    return (actor["health"] <= 0 or position[0] < lower[0] or position[0] >= lower[0] + size[0]
            or position[2] < lower[2] or position[2] >= lower[2] + size[2] or position[1] < lower[1] - 4)


class Environment:
    """Explicit-reset batch; step returns final observations, never auto-reset states."""

    def __init__(self, environments=16, device="cpu", environment=None, library=None,
                 max_ticks=None, seed=1, weights=None, schema="v1", contract=None):
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if not 1 <= environments <= 65536 or not 0 <= seed < 2**64:
            raise ValueError("Invalid environment count, episode length or seed")
        max_ticks = resolve_max_ticks(environment, max_ticks)
        path = Path(environment) if environment else ROOT / "simulator" / "environments" / "duel.json"
        definition = json.loads(path.read_text(encoding="utf-8"))
        if schema not in SCHEMAS:
            raise ValueError("schema must be v1 or v2")
        if contract is None and environment:
            contract = definition.get("policy")
        self.schema = validate_contract(contract) if contract is not None else SCHEMAS[schema]
        self.contract = validate_contract(contract) if contract is not None else from_legacy(self.schema)
        self.observation_dim = self.schema["observation_dim"]
        self.action_names = tuple(self.schema["action_names"])
        self.action_sizes = tuple(self.schema["action_sizes"])
        self.action_count = len(self.action_names)
        self.count = environments
        self.device = device
        self.max_ticks = max_ticks
        self.weights = resolve_reward_weights(definition, weights)
        self.effective_weights = {name: value / max_ticks if name in SHAPING_REWARD_NAMES else value
                      for name, value in self.weights.items()}
        reward_weights = np.array([self.effective_weights[name] for name in REWARD_NAMES], dtype=np.float32)
        if not np.isfinite(reward_weights).all():
            raise ValueError("Reward weights must be finite")
        self.native = load_library(library, device)
        self.build_info = self.native._build_info.copy()
        pointer = ctypes.c_void_p
        self.native.mcsim_last_error.restype = ctypes.c_char_p
        set_block_reward = self.native.mcsim_set_block_reward
        set_block_reward.argtypes = [pointer, ctypes.c_float]
        set_block_reward.restype = ctypes.c_int
        set_shaping_rewards = getattr(self.native, "mcsim_set_shaping_rewards", None)
        if set_shaping_rewards is None:
            raise RuntimeError("Rebuild minecraft_sim_training to enable shaping rewards")
        set_shaping_rewards.argtypes = [pointer, pointer]
        set_shaping_rewards.restype = ctypes.c_int
        self.native.mcsim_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
                                            ctypes.c_int, ctypes.c_uint64, pointer]
        self.native.mcsim_create.restype = pointer
        self.native.mcsim_destroy.argtypes = [pointer]
        self.native.mcsim_destroy.restype = None
        self.native.mcsim_environment.argtypes = [pointer]
        self.native.mcsim_environment.restype = ctypes.c_char_p
        for name, arguments in (("mcsim_observe", 2), ("mcsim_reset", 3), ("mcsim_step", 6)):
            function = getattr(self.native, name)
            function.argtypes = [pointer] * arguments
            function.restype = ctypes.c_int
        create = self.native.mcsim_create
        arguments = (environments, int(device == "cuda"), os.fsencode(environment) if environment else None,
                     max_ticks, seed, reward_weights.ctypes.data)
        if contract is not None:
            create = getattr(self.native, "mcsim_create_contract", None)
            if create is None:
                raise RuntimeError("Rebuild minecraft_sim_training to enable configured policy contracts")
            create.argtypes = [*self.native.mcsim_create.argtypes, ctypes.c_char_p]
            create.restype = pointer
            arguments += (json.dumps(self.contract, allow_nan=False).encode("utf-8"),)
        elif schema == "v2":
            create = getattr(self.native, "mcsim_create_schema", None)
            if create is None:
                directory = getattr(self.native, "_dll_directory", None)
                if directory is not None:
                    directory.close()
                raise RuntimeError("Rebuild minecraft_sim_training to enable the v2 sword/axe/shield schema")
            create.argtypes = [*self.native.mcsim_create.argtypes, ctypes.c_int]
            create.restype = pointer
            arguments += (2,)
        self._handle = create(*arguments)
        if not self._handle:
            raise RuntimeError(self.native.mcsim_last_error().decode("utf-8", errors="replace"))
        self.native.mcsim_batch_info.argtypes = [pointer]
        self.native.mcsim_batch_info.restype = ctypes.c_char_p
        native_layout = json.loads(self.native.mcsim_batch_info(self._handle))
        expected_layout = {"environments": environments, "players_per_environment": 2,
                           "observation_dim": self.observation_dim, "action_sizes": list(self.action_sizes)}
        if native_layout != expected_layout:
            self.close()
            raise RuntimeError(f"Native batch layout differs from policy: {native_layout} != {expected_layout}")
        shaping_weights = reward_weights[-len(SHAPING_REWARD_NAMES):]
        for setter, value in ((set_block_reward, self.weights["blocked_with_shield"]),
                              (set_shaping_rewards, shaping_weights.ctypes.data)):
            if setter(self._handle, value) != 0:
                error = self.native.mcsim_last_error().decode("utf-8", errors="replace")
                self.close()
                raise RuntimeError(error)
        self.definition = json.loads(self.native.mcsim_environment(self._handle))
        self.definition["reward_weights"] = self.weights.copy()
        describe = getattr(self.native, "mcsim_scenario", None)
        if describe is None:
            self.close()
            raise RuntimeError("Rebuild minecraft_sim_training to enable self-contained scenarios")
        describe.argtypes = [pointer]
        describe.restype = ctypes.c_char_p
        self.scenario = json.loads(describe(self._handle))
        self.scenario["reward_weights"] = self.weights.copy()
        self._observations = np.empty((environments, 2, self.observation_dim), dtype=np.float32)
        self._rewards = np.empty((environments, 2), dtype=np.float32)
        self._terminated = np.empty(environments, dtype=np.uint8)
        self._truncated = np.empty(environments, dtype=np.uint8)

    def _check(self, result):
        if result != 0:
            raise RuntimeError(self.native.mcsim_last_error().decode("utf-8", errors="replace"))

    def _require_open(self):
        if not self._handle:
            raise RuntimeError("Environment is closed")

    @contextmanager
    def profile_collection(self, timer):
        if not timer.enabled:
            yield
            return
        self._require_open()
        setter = getattr(self.native, "mcsim_set_profiling", None)
        getter = getattr(self.native, "mcsim_profile", None)
        if setter is None or getter is None:
            raise RuntimeError("Rebuild minecraft_sim_training to enable collection profiling")
        setter.argtypes = [ctypes.c_void_p, ctypes.c_int]
        setter.restype = ctypes.c_int
        getter.argtypes = [ctypes.c_void_p]
        getter.restype = ctypes.c_char_p
        self._check(setter(self._handle, 1))
        try:
            yield
        finally:
            try:
                encoded = getter(self._handle)
                if encoded is None:
                    raise RuntimeError(self.native.mcsim_last_error().decode("utf-8", errors="replace"))
                report = json.loads(encoded)
                if report.get("version") != 1:
                    raise RuntimeError("Unsupported native collection profile version")
                for name, seconds in report["seconds"].items():
                    timer.native_seconds[name] = timer.native_seconds.get(name, 0.0) + seconds
                timer.count("native_batches")
            finally:
                self._check(setter(self._handle, 0))

    def observe(self):
        self._require_open()
        self._check(self.native.mcsim_observe(self._handle, self._observations.ctypes.data))
        return self._observations.copy()

    def reward_totals(self):
        self._require_open()
        function = getattr(self.native, "mcsim_reward_totals", None)
        if function is None:
            return None
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        function.restype = ctypes.c_int
        totals = np.empty(len(REWARD_TYPE_NAMES), dtype=np.float64)
        self._check(function(self._handle, totals.ctypes.data))
        result = dict(zip(REWARD_TYPE_NAMES, totals.tolist()))
        shaping = self.native.mcsim_shaping_reward_totals
        shaping.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        shaping.restype = ctypes.c_int
        shaping_totals = np.empty(len(SHAPING_REWARD_NAMES), dtype=np.float64)
        self._check(shaping(self._handle, shaping_totals.ctypes.data))
        result.update(zip(SHAPING_REWARD_NAMES, shaping_totals.tolist()))
        return result

    def step_reward_totals(self, mask=None):
        self._require_open()
        selected = None
        if mask is not None:
            raw = np.asarray(mask)
            if raw.shape != (self.count, 2) or raw.dtype.kind not in "bui" or not np.isin(raw, [0, 1]).all():
                raise ValueError("Step reward mask must contain one boolean per player")
            selected = np.ascontiguousarray(raw, dtype=np.uint8)
        function = getattr(self.native, "mcsim_step_reward_totals", None)
        if function is None:
            return None
        function.argtypes = [ctypes.c_void_p] * 3
        function.restype = ctypes.c_int
        totals = np.empty(len(REWARD_NAMES), dtype=np.float64)
        self._check(function(self._handle, None if selected is None else selected.ctypes.data, totals.ctypes.data))
        return dict(zip(REWARD_NAMES, totals.tolist()))

    def losses(self, mask):
        self._require_open()
        selected = np.asarray(mask)
        if selected.shape != (self.count,) or selected.dtype.kind != "b":
            raise ValueError("Loss mask must contain one boolean per environment")
        result = np.zeros((self.count, 2), dtype=np.uint8)
        function = getattr(self.native, "mcsim_losses", None)
        if function is not None:
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            function.restype = ctypes.c_int
            self._check(function(self._handle, result.ctypes.data))
        else:
            for world in np.flatnonzero(selected):
                scenario = self.episode_scenario(int(world))
                frame = self.replay_frame(int(world))
                result[world] = [actor_lost(actor, scenario) for actor in frame["actors"]]
        return result.astype(bool)

    def replay_frame(self, environment=0):
        self._require_open()
        if not isinstance(environment, int) or not 0 <= environment < self.count:
            raise ValueError("Replay environment index outside batch")
        function = getattr(self.native, "mcsim_replay_frame", None)
        if function is None:
            raise RuntimeError("Rebuild minecraft_sim_training to enable replay snapshots")
        function.argtypes = [ctypes.c_void_p, ctypes.c_int]
        function.restype = ctypes.c_char_p
        frame = function(self._handle, environment)
        if frame is None:
            raise RuntimeError(self.native.mcsim_last_error().decode("utf-8", errors="replace"))
        return json.loads(frame)

    def reset(self, mask=None):
        self._require_open()
        raw = np.ones(self.count, dtype=np.uint8) if mask is None else np.asarray(mask)
        if raw.shape != (self.count,) or raw.dtype.kind not in "bui" or not np.isin(raw, [0, 1]).all():
            raise ValueError("Reset mask must contain one boolean per environment")
        selected = np.ascontiguousarray(raw, dtype=np.uint8)
        self._check(self.native.mcsim_reset(self._handle, selected.ctypes.data, self._observations.ctypes.data))
        return self._observations.copy()

    def reset_players(self, players, mask, timer=None):
        """Reset selected episodes without rebuilding their immutable terrain."""
        timer = timer or CollectionTimer("cpu")
        self._require_open()
        raw = np.asarray(mask)
        if raw.shape != (self.count,) or raw.dtype.kind not in "bui" or not np.isin(raw, [0, 1]).all():
            raise ValueError("Reset mask must contain one boolean per environment")
        selected = np.ascontiguousarray(raw, dtype=np.uint8)
        if not isinstance(players, list) or len(players) != self.count:
            raise ValueError("Player definitions must contain one entry per environment")
        function = getattr(self.native, "mcsim_reset_players", None)
        if function is None:
            raise RuntimeError("Rebuild minecraft_sim_training for sampled episode resets")
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        function.restype = ctypes.c_int
        with timer.phase("environment_reset.json_encode"):
            payload = json.dumps(players, allow_nan=False).encode("utf-8")
        with timer.phase("environment_reset.native"):
            self._check(function(self._handle, selected.ctypes.data, payload, self._observations.ctypes.data))
        with timer.phase("environment_reset.copy_outputs"):
            return self._observations.copy()

    def episode_scenario(self, environment=0):
        self._require_open()
        if type(environment) is not int or not 0 <= environment < self.count:
            raise ValueError("Scenario environment index outside batch")
        function = getattr(self.native, "mcsim_scenario_for", None)
        if function is None:
            raise RuntimeError("Rebuild minecraft_sim_training for sampled episode scenarios")
        function.argtypes = [ctypes.c_void_p, ctypes.c_int]
        function.restype = ctypes.c_char_p
        result = function(self._handle, environment)
        if result is None:
            raise RuntimeError(self.native.mcsim_last_error().decode("utf-8", errors="replace"))
        return {**json.loads(result), "reward_weights": self.weights.copy(), "max_ticks": self.max_ticks}

    def set_max_ticks(self, max_ticks):
        if type(max_ticks) is not int or not 1 <= max_ticks <= MAX_MAX_TICKS:
            raise ValueError("Episode limit must be an integer in [1,1000000]")
        function = getattr(self.native, "mcsim_set_max_ticks", None)
        if function is None:
            raise RuntimeError("Rebuild minecraft_sim_training for stage-specific episode limits")
        function.argtypes = [ctypes.c_void_p, ctypes.c_int]
        function.restype = ctypes.c_int
        self._check(function(self._handle, max_ticks))
        self.max_ticks = max_ticks
        self.set_rewards(self.weights)

    def set_rewards(self, weights):
        resolved = resolve_reward_weights({}, weights)
        if not all(np.isfinite(value) for value in resolved.values()):
            raise ValueError("Reward weights must be finite")
        effective = {name: value / self.max_ticks if name in SHAPING_REWARD_NAMES else value
                     for name, value in resolved.items()}
        base = getattr(self.native, "mcsim_set_reward_weights", None)
        if base is None:
            raise RuntimeError("Rebuild minecraft_sim_training for curriculum rewards")
        values = np.array([effective[name] for name in BASE_REWARD_NAMES], dtype=np.float32)
        base.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        base.restype = ctypes.c_int
        self._check(base(self._handle, values.ctypes.data))
        self._check(self.native.mcsim_set_block_reward(self._handle, resolved["blocked_with_shield"]))
        shaping = np.array([effective[name] for name in SHAPING_REWARD_NAMES], dtype=np.float32)
        self._check(self.native.mcsim_set_shaping_rewards(self._handle, shaping.ctypes.data))
        self.weights, self.effective_weights = resolved, effective

    def step(self, actions):
        self._require_open()
        raw = np.asarray(actions)
        if raw.shape != (self.count, 2, self.action_count) or raw.dtype.kind not in "iu":
            raise ValueError(f"Actions must be an integer array [environments, 2, {self.action_count}]")
        if np.any(raw < 0) or np.any(raw >= np.array(self.action_sizes)):
            raise ValueError("Action index outside declared policy head")
        packed = np.ascontiguousarray(raw, dtype=np.int32)
        self._check(self.native.mcsim_step(self._handle, packed.ctypes.data,
            self._observations.ctypes.data, self._rewards.ctypes.data,
            self._terminated.ctypes.data, self._truncated.ctypes.data))
        return (self._observations.copy(), self._rewards.copy(),
                self._terminated.astype(bool), self._truncated.astype(bool))

    def close(self):
        if self._handle:
            self.native.mcsim_destroy(self._handle)
            self._handle = None
        directory = getattr(self.native, "_dll_directory", None)
        if directory is not None:
            directory.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()