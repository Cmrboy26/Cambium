"""Host-staged two-player training adapter over the shared C++/CUDA simulator."""

import ctypes
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import threading

import numpy as np
from simulator.training.contract import CATALOGUE_HASH, from_legacy, validate_contract
from simulator.training.legacy import ACTION_NAMES, ACTION_SIZES, SCHEMA, SCHEMA_V2, SCHEMAS
from simulator.training.timing import CollectionTimer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAX_TICKS = 300
MAX_MAX_TICKS = 1_000_000
BASE_REWARD_NAMES = ("doing_damage", "taking_damage", "won_duel", "death",
                     "episode_timeout", "no_damage_penalty", "missed_attack",
                     "shield_disabled", "win_speed", "arrow_progress")
SHAPING_REWARD_NAMES = ("good_aim", "proximity", "pitch_control", "aim_hold", "extreme_pitch_penalty")
EVENT_REWARD_NAMES = ("wind_self_launch", "missed_attack_ready_time")
REWARD_TYPE_NAMES = (*BASE_REWARD_NAMES, "blocked_with_shield", *EVENT_REWARD_NAMES)
REWARD_NAMES = (*REWARD_TYPE_NAMES, *SHAPING_REWARD_NAMES)
DEFAULT_REWARD_WEIGHTS = json.loads(Path(__file__).with_name("reward_weights.json").read_text(encoding="utf-8"))
REWARD_SEMANTICS = "wind_launch_ready_time_episode_budgets_v6"


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
            if native.mcsim_api_version() != 5:
                raise RuntimeError("Unsupported training library ABI; rebuild for wind event rewards")
            native.mcsim_build_info.restype = ctypes.c_char_p
            build_info = json.loads(native.mcsim_build_info())
            if build_info.get("catalogue_hash") != CATALOGUE_HASH:
                raise RuntimeError("Native policy catalogue differs from Python; rebuild the selected library")
            expected_layout = {"scalar_bytes": {"observation": 4, "action": 4, "reward_total": 8, "done": 1},
                               "reward_inputs": {"base": len(BASE_REWARD_NAMES), "events": len(EVENT_REWARD_NAMES), "shaping": len(SHAPING_REWARD_NAMES)},
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
        if (any(observation.get("encoding") == "one_hot" or "clip" in observation
                for observation in self.contract["observations"])
                and self.build_info.get("observation_projection_version", 0) < 1):
            raise RuntimeError("Rebuild minecraft_sim_training to enable observation encodings and clipping")
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
        set_event_rewards = self.native.mcsim_set_event_rewards
        set_event_rewards.argtypes = [pointer, pointer]
        set_event_rewards.restype = ctypes.c_int
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
        event_weights = np.array([self.effective_weights[name] for name in EVENT_REWARD_NAMES], dtype=np.float32)
        for setter, value in ((set_block_reward, self.weights["blocked_with_shield"]),
                      (set_event_rewards, event_weights.ctypes.data),
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
        self._cuda_observations = None
        self._cuda_observation_pointer = None
        self._cuda_observation_device = None
        self._cuda_observation_thread = None
        self._cuda_actions = None
        self._cuda_action_pointer = None
        self._cuda_action_device = None
        self._cuda_resident_buffers = None
        self._cuda_resident_pointers = None
        self._cuda_resident_ready = False

    def _check(self, result):
        if result != 0:
            raise RuntimeError(self.native.mcsim_last_error().decode("utf-8", errors="replace"))

    def _require_open(self):
        if not self._handle:
            raise RuntimeError("Environment is closed")
        owner = getattr(self, "_cuda_observation_thread", None)
        if owner is not None and owner != threading.get_ident():
            raise RuntimeError("CUDA observation environments must be used from their registration thread")

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

    def observe_base_cuda(self):
        """Return synchronized raw 22-feature CUDA observations, without policy projection."""
        self._require_open()
        if self.device != "cuda":
            raise ValueError("Base CUDA observations require a CUDA environment")
        function = getattr(self.native, "mcsim_observe_base_cuda_sync", None)
        if function is None or self.build_info.get("cuda_base_observation_version") != 1:
            raise RuntimeError("Rebuild minecraft_sim_training for base CUDA observations")
        import torch

        observations = torch.empty((self.count, 2, 22), dtype=torch.float32, device="cuda")
        stream = torch.cuda.current_stream(observations.device)
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        function.restype = ctypes.c_int
        self._check(function(self._handle, observations.data_ptr(), observations.numel(), stream.cuda_stream))
        return observations

    def register_cuda_observations(self, output=None):
        """Register a reusable CUDA tensor for configured asynchronous observations."""
        self._require_open()
        if self.device != "cuda":
            raise ValueError("CUDA observations require a CUDA environment")
        function = getattr(self.native, "mcsim_register_cuda_observations", None)
        if function is None or self.build_info.get("cuda_async_observation_version") != 1:
            raise RuntimeError("Rebuild minecraft_sim_training for asynchronous CUDA observations")
        import torch

        if output is None:
            output = torch.empty((self.count, 2, self.observation_dim), dtype=torch.float32, device="cuda")
        if (not isinstance(output, torch.Tensor) or output.device.type != "cuda" or output.dtype != torch.float32
                or output.shape != (self.count, 2, self.observation_dim) or not output.is_contiguous()
                or output.requires_grad):
            raise ValueError("CUDA observation buffer must be a contiguous float32 tensor with the configured shape")
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        function.restype = ctypes.c_int
        previous_thread = self._cuda_observation_thread
        self._cuda_observation_thread = threading.get_ident()
        try:
            with torch.cuda.device(output.device):
                self._check(function(self._handle, output.data_ptr(), output.numel()))
        except BaseException:
            self._cuda_observation_thread = previous_thread
            raise
        self._cuda_observations = output
        self._cuda_observation_pointer = output.data_ptr()
        self._cuda_observation_device = output.device
        return output

    def observe_cuda_async(self):
        """Enqueue configured observations on the current stream into the registered tensor."""
        self._require_open()
        output = self._cuda_observations
        if output is None:
            raise RuntimeError("Register a CUDA observation buffer before use")
        import torch

        if (output.data_ptr() != self._cuda_observation_pointer or output.device != self._cuda_observation_device
                or output.dtype != torch.float32 or output.shape != (self.count, 2, self.observation_dim)
                or not output.is_contiguous() or output.requires_grad):
            raise RuntimeError("Registered CUDA observation buffer changed; register it again")
        function = self.native.mcsim_observe_cuda_async
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        function.restype = ctypes.c_int
        with torch.cuda.device(output.device):
            stream = torch.cuda.current_stream(output.device)
            output.record_stream(stream)
            self._check(function(self._handle, stream.cuda_stream))
        return output

    def wait_cuda_observations(self):
        """Wait for the latest registered CUDA observation and report errors."""
        self._require_open()
        if self._cuda_observations is None:
            raise RuntimeError("Register a CUDA observation buffer before use")
        import torch

        function = self.native.mcsim_wait_cuda_observations
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_int
        with torch.cuda.device(self._cuda_observation_device):
            self._check(function(self._handle))

    def register_cuda_actions(self, actions=None):
        """Register a reusable int32 CUDA tensor for asynchronous action decoding."""
        self._require_open()
        if self.device != "cuda":
            raise ValueError("CUDA actions require a CUDA environment")
        if self._cuda_observations is None:
            raise RuntimeError("Register a CUDA observation buffer before CUDA actions")
        function = getattr(self.native, "mcsim_register_cuda_actions", None)
        if function is None or self.build_info.get("cuda_action_decode_version") != 1:
            raise RuntimeError("Rebuild minecraft_sim_training for CUDA action decoding")
        import torch

        if actions is None:
            actions = torch.empty((self.count, 2, self.action_count), dtype=torch.int32,
                                  device=self._cuda_observation_device)
        if (not isinstance(actions, torch.Tensor) or actions.device != self._cuda_observation_device
                or actions.dtype != torch.int32 or actions.shape != (self.count, 2, self.action_count)
                or not actions.is_contiguous()):
            raise ValueError("CUDA action buffer must be a contiguous int32 tensor with the configured shape and device")
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        function.restype = ctypes.c_int
        with torch.cuda.device(actions.device):
            self._check(function(self._handle, actions.data_ptr(), actions.numel()))
        self._cuda_actions = actions
        self._cuda_action_pointer = actions.data_ptr()
        self._cuda_action_device = actions.device
        return actions

    def decode_cuda_actions_async(self):
        """Enqueue batch-atomic action validation and decoding on the current stream."""
        self._require_open()
        actions = self._cuda_actions
        if actions is None:
            raise RuntimeError("Register a CUDA action buffer before use")
        import torch

        if (actions.data_ptr() != self._cuda_action_pointer or actions.device != self._cuda_action_device
                or actions.dtype != torch.int32 or actions.shape != (self.count, 2, self.action_count)
                or not actions.is_contiguous()):
            raise RuntimeError("Registered CUDA action buffer changed; register it again")
        function = self.native.mcsim_decode_cuda_actions_async
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        function.restype = ctypes.c_int
        with torch.cuda.device(actions.device):
            stream = torch.cuda.current_stream(actions.device)
            actions.record_stream(stream)
            self._check(function(self._handle, stream.cuda_stream))

    def wait_cuda_actions(self):
        """Wait for CUDA action decoding and raise if the sticky status is invalid."""
        self._require_open()
        if self._cuda_actions is None:
            raise RuntimeError("Register a CUDA action buffer before use")
        import torch

        function = self.native.mcsim_cuda_action_status
        function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        function.restype = ctypes.c_int
        status = ctypes.c_int()
        with torch.cuda.device(self._cuda_action_device):
            self._check(function(self._handle, ctypes.byref(status)))
        if status.value & 1:
            raise ValueError("Invalid CUDA action index or selected slot; clear the action status before reuse")
        if status.value & 2:
            raise ValueError("CUDA batch contains a finished world; clear the action status and reset it before reuse")
        if status.value & 4:
            raise ValueError("Invalid CUDA reset mask; clear the action status before reuse")

    def clear_cuda_action_status(self):
        """Clear the sticky CUDA action error after it has been inspected."""
        self._require_open()
        if self._cuda_actions is None:
            raise RuntimeError("Register a CUDA action buffer before use")
        import torch

        function = self.native.mcsim_clear_cuda_action_status
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_int
        with torch.cuda.device(self._cuda_action_device):
            self._check(function(self._handle))

    def register_cuda_resident(self):
        """Register fixed-reset CUDA step outputs; host-state operations become unavailable."""
        self._require_open()
        if self.device != "cuda":
            raise ValueError("CUDA resident stepping requires a CUDA environment")
        if self._cuda_observations is None or self._cuda_actions is None:
            raise RuntimeError("Register CUDA observation and action buffers before resident stepping")
        if self.build_info.get("cuda_resident_step_version") != 1:
            raise RuntimeError("Rebuild minecraft_sim_training for CUDA resident stepping")
        import torch

        device = self._cuda_observation_device
        buffers = {
            "rewards": torch.empty((self.count, 2), dtype=torch.float32, device=device),
            "terminated": torch.empty((self.count,), dtype=torch.uint8, device=device),
            "truncated": torch.empty((self.count,), dtype=torch.uint8, device=device),
            "losses": torch.empty((self.count, 2), dtype=torch.uint8, device=device),
            "contributions": torch.empty((self.count, 2, len(REWARD_NAMES)), dtype=torch.float64, device=device),
            "reset_mask": torch.empty((self.count,), dtype=torch.uint8, device=device),
        }
        self._cuda_resident_buffers = buffers
        self._cuda_resident_pointers = {name: output.data_ptr() for name, output in buffers.items()}
        self._cuda_resident_ready = False
        register = self.native.mcsim_register_cuda_step_outputs
        register.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_size_t]
        register.restype = ctypes.c_int
        reset = self.native.mcsim_register_cuda_reset_mask
        reset.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        reset.restype = ctypes.c_int
        with torch.cuda.device(device):
            self._check(register(self._handle, buffers["rewards"].data_ptr(), buffers["terminated"].data_ptr(),
                                 buffers["truncated"].data_ptr(), buffers["losses"].data_ptr(),
                                 buffers["contributions"].data_ptr(), self.count))
            self._check(reset(self._handle, buffers["reset_mask"].data_ptr(), self.count))
        self._cuda_resident_ready = True
        return buffers

    def _resident_stream(self):
        self._require_open()
        if not self._cuda_resident_ready:
            raise RuntimeError("Register CUDA resident buffers before use")
        import torch

        buffers = self._cuda_resident_buffers
        shapes = {"rewards": (self.count, 2), "terminated": (self.count,), "truncated": (self.count,),
                  "losses": (self.count, 2), "contributions": (self.count, 2, len(REWARD_NAMES)),
                  "reset_mask": (self.count,)}
        dtypes = {"rewards": torch.float32, "terminated": torch.uint8, "truncated": torch.uint8,
                  "losses": torch.uint8, "contributions": torch.float64, "reset_mask": torch.uint8}
        if set(buffers) != set(shapes):
            raise RuntimeError("Registered CUDA resident buffers changed; recreate the environment")
        for name, output in buffers.items():
            if (not isinstance(output, torch.Tensor) or output.data_ptr() != self._cuda_resident_pointers[name]
                    or output.device != self._cuda_observation_device or output.dtype != dtypes[name]
                    or output.shape != shapes[name] or not output.is_contiguous()):
                raise RuntimeError("Registered CUDA resident buffer changed; recreate the environment")
        if (self._cuda_actions.data_ptr() != self._cuda_action_pointer
                or self._cuda_actions.device != self._cuda_action_device or self._cuda_actions.dtype != torch.int32
                or self._cuda_actions.shape != (self.count, 2, self.action_count) or not self._cuda_actions.is_contiguous()
                or self._cuda_observations.data_ptr() != self._cuda_observation_pointer
                or self._cuda_observations.device != self._cuda_observation_device
                or self._cuda_observations.dtype != torch.float32
                or self._cuda_observations.shape != (self.count, 2, self.observation_dim)
                or not self._cuda_observations.is_contiguous() or self._cuda_observations.requires_grad):
            raise RuntimeError("Registered CUDA action or observation buffer changed; register it again")
        stream = torch.cuda.current_stream(self._cuda_observation_device)
        for output in buffers.values():
            output.record_stream(stream)
        self._cuda_actions.record_stream(stream)
        self._cuda_observations.record_stream(stream)
        return stream

    def step_cuda_resident_async(self):
        """Enqueue one fixed-reset resident step into the registered CUDA buffers."""
        self._require_open()
        if not self._cuda_resident_ready:
            raise RuntimeError("Register CUDA resident buffers before use")
        import torch

        with torch.cuda.device(self._cuda_observation_device):
            stream = self._resident_stream()
            function = self.native.mcsim_step_cuda_resident_async
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            function.restype = ctypes.c_int
            self._check(function(self._handle, stream.cuda_stream))
        return self._cuda_observations, self._cuda_resident_buffers

    def reset_cuda_resident_async(self):
        """Reset worlds selected by the registered CUDA reset mask."""
        self._require_open()
        if not self._cuda_resident_ready:
            raise RuntimeError("Register CUDA resident buffers before use")
        import torch

        with torch.cuda.device(self._cuda_observation_device):
            stream = self._resident_stream()
            function = self.native.mcsim_reset_cuda_resident_async
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            function.restype = ctypes.c_int
            self._check(function(self._handle, stream.cuda_stream))
        return self._cuda_observations

    def cuda_reset_episodes(self):
        """Read fixed-reset episode counters at a checkpoint boundary."""
        self._require_open()
        if not self._cuda_resident_ready or self.build_info.get("cuda_resident_episode_version") != 1:
            raise RuntimeError("CUDA resident episode counters are unavailable")
        import torch

        episodes = np.empty(self.count, dtype=np.uint64)
        function = self.native.mcsim_cuda_reset_episodes
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        function.restype = ctypes.c_int
        with torch.cuda.device(self._cuda_observation_device):
            self._check(function(self._handle, episodes.ctypes.data, self.count))
        return torch.from_numpy(episodes.view(np.int64))

    def set_cuda_reset_episodes(self, episodes):
        """Restore fixed-reset episode counters at a checkpoint boundary."""
        self._require_open()
        if not self._cuda_resident_ready or self.build_info.get("cuda_resident_episode_version") != 1:
            raise RuntimeError("CUDA resident episode counters are unavailable")
        import torch

        if not isinstance(episodes, torch.Tensor) or episodes.dtype != torch.int64 or episodes.shape != (self.count,):
            raise ValueError("CUDA reset episodes must be an int64 tensor with one entry per world")
        values = episodes.cpu().contiguous().numpy().view(np.uint64)
        function = self.native.mcsim_set_cuda_reset_episodes
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        function.restype = ctypes.c_int
        with torch.cuda.device(self._cuda_observation_device):
            self._check(function(self._handle, values.ctypes.data, self.count))

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

    def step_behavior_events(self):
        self._require_open()
        if self.build_info.get("behavior_events_version") != 1:
            raise RuntimeError("Rebuild minecraft_sim_training to enable behavior evaluation")
        function = self.native.mcsim_step_behavior_events
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        function.restype = ctypes.c_int
        events = np.empty((self.count, 2, 3), dtype=np.float32)
        self._check(function(self._handle, events.ctypes.data))
        return events

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
        self._require_open()
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
        self._require_open()
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
        events = np.array([effective[name] for name in EVENT_REWARD_NAMES], dtype=np.float32)
        self._check(self.native.mcsim_set_event_rewards(self._handle, events.ctypes.data))
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
            self._require_open()
        try:
            if self._handle:
                device = getattr(self, "_cuda_observation_device", None)
                if device is None:
                    self.native.mcsim_destroy(self._handle)
                else:
                    import torch

                    with torch.cuda.device(device):
                        try:
                            self.wait_cuda_observations()
                            if self._cuda_actions is not None:
                                self.wait_cuda_actions()
                        finally:
                            self.native.mcsim_destroy(self._handle)
        finally:
            self._handle = None
            self._cuda_observations = None
            self._cuda_observation_pointer = None
            self._cuda_observation_device = None
            self._cuda_observation_thread = None
            self._cuda_actions = None
            self._cuda_action_pointer = None
            self._cuda_action_device = None
            self._cuda_resident_buffers = None
            self._cuda_resident_pointers = None
            self._cuda_resident_ready = False
            directory = getattr(self.native, "_dll_directory", None)
            if directory is not None:
                directory.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()