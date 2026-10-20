"""Batched, checkpointable opponents over semantic policy descriptors.

``act(obs, indices)`` addresses actor slots in a fixed-size controller pool.
Without indices, rows address slots 0..N-1; ``reset`` always takes a full-pool
boolean mask. Custom ``module:factory`` imports execute trusted Python code.

Builtins: stationary_target accepts no options. basic_melee accepts
attack_interval (13), melee_distance (3 blocks), strafe (False),
strafe_interval (20 ticks), sprint (False), shield (False), and optional
axe_slot/sword_slot (explicit loadout slots, supplied together). Equipment
behaviors require their relevant observations and controls, never guessed slots.
Checkpoint files are trusted standalone PPO archives, loaded with weights_only.
Both shared-policy and opponent-pool checkpoints retain saved fixed controls.
Self-play uses the bound learner's rollout-frozen weights and its own sampling RNG.
"""

from copy import deepcopy
import hashlib
import importlib
import io
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

import numpy as np

from simulator.training.contract import fingerprint, from_legacy, validate_contract

if TYPE_CHECKING:
    import torch


@runtime_checkable
class Controller(Protocol):
    def act(self, observations: np.ndarray, indices: np.ndarray | None = None) -> np.ndarray: ...

    def reset(self, mask: np.ndarray) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...

    @property
    def describe(self) -> dict[str, Any]: ...

    @property
    def provenance(self) -> dict[str, Any]: ...


ControllerFactory = Callable[..., Controller]
CONTROLLER_REGISTRY: dict[str, ControllerFactory] = {}


def register_controller(name: str, factory: ControllerFactory) -> None:
    """Register a trusted factory(contract=..., count=..., seed=..., options=...)."""
    if not isinstance(name, str) or not name or ":" in name:
        raise ValueError("Controller name must be nonempty and cannot contain ':'")
    if not callable(factory):
        raise ValueError("Controller factory must be callable")
    if name in CONTROLLER_REGISTRY:
        raise ValueError(f"Controller already registered: {name}")
    CONTROLLER_REGISTRY[name] = factory


def _contract(value):
    if isinstance(value, dict) and value.get("format") == "minecraft_sim_policy":
        return validate_contract(value)
    # Legacy preset identities must be exact; an arbitrary old layout is unsafe.
    from simulator.training.environment import SCHEMAS

    if not any(value == candidate for candidate in SCHEMAS.values()):
        raise ValueError("Unsupported legacy policy schema")
    return from_legacy(value)


def _integer(value, name, minimum=1, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer >= {minimum}"
                         + (f" and <= {maximum}" if maximum is not None else ""))
    return value


def _options(options, allowed):
    if not isinstance(options, dict) or set(options) - set(allowed):
        raise ValueError(f"Unknown or invalid controller options; allowed: {sorted(allowed)}")
    return deepcopy(options)


def _boolean(options, name, default=False):
    value = options.get(name, default)
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


def checkpoint_sha256(path) -> str:
    """Hash the bytes a resolved opponent plan intends to load."""
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest() if hasattr(hashlib, "file_digest") \
            else hashlib.sha256(source.read()).hexdigest()


class _BatchedController:
    def __init__(self, contract, count, seed):
        self.contract = _contract(contract)
        self.count = _integer(count, "count")
        self.seed = _integer(seed, "seed", 0, 2**63 - 1)
        self.contract_hash = fingerprint(self.contract)
        self._description = {"contract_hash": self.contract_hash, "count": count, "seed": seed}

    @property
    def describe(self):
        return deepcopy(self._description)

    @property
    def provenance(self):
        return self.describe

    def _batch(self, observations, indices):
        if (not isinstance(observations, np.ndarray) or observations.ndim != 2
                or observations.shape[1] != self.contract["observation_dim"]
                or len(observations) > self.count or observations.dtype.kind not in "fiu"
                or not np.isfinite(observations).all()):
            raise ValueError(f"Expected finite numeric observations [N,{self.contract['observation_dim']}]"
                             f" with N <= {self.count}")
        size = len(observations)
        if indices is None:
            indices = np.arange(size)
        else:
            indices = np.asarray(indices)
            if (indices.shape != (size,) or indices.dtype.kind not in "iu"
                    or (indices < 0).any() or (indices >= self.count).any()
                    or len(np.unique(indices)) != size):
                raise ValueError("indices must be unique integer actor slots matching the batch")
        return observations, indices

    def _mask(self, mask):
        mask = np.asarray(mask)
        if mask.shape != (self.count,) or mask.dtype != np.bool_:
            raise ValueError(f"reset mask must be boolean with shape [{self.count}]")
        return mask

    def reset(self, mask):
        self._mask(mask)

    def state_dict(self):
        return {"version": 1, "provenance": self.provenance}

    def _state(self, state, keys):
        if (not isinstance(state, dict) or set(state) != {"version", "provenance", *keys}
                or type(state["version"]) is not int or state["version"] != 1
                or state["provenance"] != self.provenance):
            raise ValueError("Controller state is incompatible with this controller")

    def load_state_dict(self, state):
        self._state(state, ())


class StationaryTargetController(_BatchedController):
    def __init__(self, contract, count, seed=1, options=None):
        super().__init__(contract, count, seed)
        self.options = _options({} if options is None else options, ())
        self._description.update(type="scripted", controller="stationary_target", options=self.options)
        self._heads = {head["name"]: (index, head["values"])
                       for index, head in enumerate(self.contract["actions"])}
        neutral = []
        for name, (_, values) in self._heads.items():
            if name == "movement":
                if "neutral" not in values:
                    raise ValueError("Stationary controls require a neutral movement choice")
                neutral.append(values.index("neutral"))
            elif name in ("yaw", "pitch", "weapon"):
                neutral.append(min(range(len(values)), key=lambda index: abs(values[index])))
            else:
                neutral.append(values.index(False) if False in values else 0)
        self._neutral = np.asarray(neutral, dtype=np.int64)

    def act(self, observations, indices=None):
        observations, _ = self._batch(observations, indices)
        return np.tile(self._neutral, (len(observations), 1))


class BasicMeleeController(StationaryTargetController):
    def __init__(self, contract, count, seed=1, options=None):
        options = _options({} if options is None else options,
                           ("attack_interval", "melee_distance", "strafe", "strafe_interval",
                            "sprint", "shield", "axe_slot", "sword_slot"))
        super().__init__(contract, count, seed)
        self.options = options
        self._description.update(controller="basic_melee", options=options)
        self.attack_interval = _integer(options.get("attack_interval", 13), "attack_interval", 1, 1_000_000)
        self.strafe_interval = _integer(options.get("strafe_interval", 20), "strafe_interval", 1, 1_000_000)
        self.melee_distance = options.get("melee_distance", 3.0)
        if (type(self.melee_distance) not in (int, float) or not math.isfinite(self.melee_distance)
                or not 0 < self.melee_distance < 10):
            raise ValueError("melee_distance must be finite and strictly between 0 and 10 blocks")
        self.strafe = _boolean(options, "strafe")
        self.sprint = _boolean(options, "sprint")
        self.shield = _boolean(options, "shield")
        self._features = {}
        for index, observation in enumerate(self.contract["observations"]):
            if observation["scale"] != 0:
                self._features.setdefault(observation["feature"], (index, observation["scale"]))
        for feature in ("target.distance_fraction", "target.yaw_error", "target.pitch_error"):
            self._require_feature(feature)
        if "movement" in self._heads:
            self._require_choice("movement", "forward")
        if self.strafe:
            for choice in ("left", "right", "forward_left", "forward_right"):
                self._require_choice("movement", choice)
        if self.sprint:
            self._require_choice("sprint", True)
        if self.shield:
            self._require_choice("shield", True)
            self._require_choice("shield", False)
            self._require_feature("self.shield_using")
            self._require_feature("self.shield_disabled")
            self._require_feature("self.attack_charge")
        self.axe_slot = options.get("axe_slot")
        self.sword_slot = options.get("sword_slot")
        if "axe_slot" in options or "sword_slot" in options:
            for name, slot in (("axe_slot", self.axe_slot), ("sword_slot", self.sword_slot)):
                _integer(slot, name, 0, 8)
                self._require_choice("weapon", slot)
            if self.axe_slot == self.sword_slot:
                raise ValueError("axe_slot and sword_slot must differ")
            self._require_feature("opponent.shield_using")
            self._require_feature("opponent.shield_disabled_observed")
        self._ticks = np.zeros(self.count, dtype=np.int64)

    def _require_feature(self, name):
        if name not in self._features:
            raise ValueError(f"basic_melee requires a nonzero-scale feature: {name}")

    def _require_choice(self, name, value):
        if name not in self._heads or value not in self._heads[name][1]:
            raise ValueError(f"basic_melee requires action {name} with choice {value!r}")

    def _feature(self, observations, name):
        index, scale = self._features[name]
        return observations[:, index].astype(np.float64) / scale

    def _set(self, actions, name, values):
        if name in self._heads:
            column, choices = self._heads[name]
            for index, choice in enumerate(choices):
                actions[np.asarray(values) == choice, column] = index

    def act(self, observations, indices=None):
        observations, indices = self._batch(observations, indices)
        actions = np.tile(self._neutral, (len(observations), 1))
        distance = self._feature(observations, "target.distance_fraction") * 10.0
        near = (distance <= self.melee_distance) | np.isclose(
            distance, self.melee_distance, rtol=1e-7, atol=0)
        ticks = self._ticks[indices]
        attack = near & (ticks % self.attack_interval == 0)
        if self.shield:
            attack &= self._feature(observations, "self.shield_using") < 0.5
        for name, degrees in (("yaw", 180.0), ("pitch", 90.0)):
            if name in self._heads:
                column, choices = self._heads[name]
                error = self._feature(observations, f"target.{name}_error") * degrees
                actions[:, column] = np.abs(error[:, None] - np.asarray(choices)).argmin(axis=1)
        movement = np.where(near, "neutral", "forward")
        if self.strafe:
            right = (ticks // self.strafe_interval) % 2 == 0
            movement = np.where(near, np.where(right, "right", "left"),
                                np.where(right, "forward_right", "forward_left"))
        self._set(actions, "movement", movement)
        self._set(actions, "attack", attack)
        if self.sprint:
            self._set(actions, "sprint", ~near)
        if self.shield:
            # Releasing an active shield consumes that tick's attack in the simulator.
            release_ahead = (ticks + 1) % self.attack_interval == 0
            use = (near & ~attack & ~release_ahead
                   & (self._feature(observations, "self.attack_charge") < 1)
                   & (self._feature(observations, "self.shield_disabled") < 0.5))
            if "self.has_shield" in self._features:
                use &= self._feature(observations, "self.has_shield") >= 0.5
            self._set(actions, "shield", use)
        if self.axe_slot is not None:
            blocked = ((self._feature(observations, "opponent.shield_using") >= 0.5)
                       & (self._feature(observations, "opponent.shield_disabled_observed") < 0.5))
            slot_present = f"self.weapon_slot_{self.axe_slot}.present"
            if slot_present in self._features:
                blocked &= self._feature(observations, slot_present) >= 0.5
            self._set(actions, "weapon", np.where(blocked, self.axe_slot, self.sword_slot))
        self._ticks[indices] = (ticks + 1) % math.lcm(self.attack_interval, 2 * self.strafe_interval)
        return actions

    def reset(self, mask):
        self._ticks[self._mask(mask)] = 0

    def state_dict(self):
        return {**super().state_dict(), "ticks": self._ticks.tolist()}

    def load_state_dict(self, state):
        self._state(state, ("ticks",))
        ticks = state["ticks"]
        period = math.lcm(self.attack_interval, 2 * self.strafe_interval)
        if (not isinstance(ticks, list) or len(ticks) != self.count
                or any(type(tick) is not int or not 0 <= tick < period for tick in ticks)):
            raise ValueError("Invalid per-actor controller ticks")
        self._ticks[:] = ticks


class _PolicyController(_BatchedController):
    model: "torch.nn.Module"
    device: "torch.device"
    deterministic: bool
    _fixed: dict[str, int]
    _generator: "torch.Generator"

    def act(self, observations, indices=None):
        import torch

        observations, _ = self._batch(observations, indices)
        if not len(observations):
            return np.empty((0, len(self.contract["actions"])), dtype=np.int64)
        with torch.inference_mode():
            inputs = torch.as_tensor(np.array(observations, dtype=np.float32, copy=True), device=self.device)
            if not torch.isfinite(inputs).all():
                raise ValueError("Observations are not finite at model precision")
            logits, _ = self.model(inputs)
            if not all(torch.isfinite(value).all() for value in logits.values()):
                raise ValueError("Policy inference produced non-finite logits")
            actions = [
                torch.full((len(observations),), self._fixed[name], dtype=torch.long, device=self.device)
                if name in self._fixed else
                logits[name].argmax(-1) if self.deterministic else
                torch.multinomial(logits[name].softmax(-1), 1, generator=self._generator).squeeze(-1)
                for name in self.contract["action_names"]
            ]
            return torch.stack(actions, dim=-1).cpu().numpy()

    def state_dict(self):
        return {**super().state_dict(), "rng_state": self._generator.get_state().clone()}

    def load_state_dict(self, state):
        import torch

        self._state(state, ("rng_state",))
        value = state["rng_state"]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
            raise ValueError("Invalid controller RNG state")
        try:
            self._generator.set_state(value.cpu())
        except RuntimeError as error:
            raise ValueError("Invalid controller RNG state") from error


class SelfPlayController(_PolicyController):
    def __init__(self, learner, contract, count, device="cpu", seed=1, deterministic=False):
        import torch

        super().__init__(contract, count, seed)
        if learner is None:
            raise ValueError("Self-play requires the learner policy model")
        if type(deterministic) is not bool:
            raise ValueError("deterministic must be boolean")
        self.model = learner
        self.device = next(learner.parameters()).device
        requested = torch.device(device)
        if (requested.type != self.device.type
                or requested.index is not None and requested.index != self.device.index):
            raise ValueError("Self-play and learner devices must match")
        self.deterministic = deterministic
        self._fixed = {}
        self._generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self._description.update(type="self_play", device=str(self.device), deterministic=deterministic)

    def set_controls(self, controls):
        from simulator.training.policies import fixed_indices

        self._fixed = fixed_indices(self.contract, controls)


class CheckpointController(_PolicyController):
    def __init__(self, path, contract, count, device="cpu", seed=1, deterministic=False,
                 expected_sha256=None):
        import torch
        from backend.fast_rl_model import ActorCritic
        from simulator.training.policies import fixed_indices

        super().__init__(contract, count, seed)
        if type(deterministic) is not bool:
            raise ValueError("deterministic must be boolean")
        self.path = Path(path).resolve()
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Checkpoint inference supports cpu or cuda")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA inference requested but unavailable")
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
        data = self.path.read_bytes()
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.checkpoint_hash = self.sha256
        if expected_sha256 is not None and expected_sha256 != self.sha256:
            raise ValueError("Checkpoint SHA256 differs from the resolved plan")
        saved = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        if (not isinstance(saved, dict) or saved.get("format") != "minecraft_sim_ppo"
                or type(saved.get("version")) is not int or saved["version"] != 1):
            raise ValueError("Checkpoint is not a compatible standalone PPO policy")
        metadata = saved.get("metadata")
        if (not isinstance(metadata, dict) or metadata.get("players_per_environment") != 2
                or metadata.get("learner") not in ("both_shared_policy", "opponent_pool")):
            raise ValueError("Checkpoint is not a compatible standalone PPO policy")
        descriptors = [metadata[key] for key in ("schema", "contract") if key in metadata]
        if not descriptors or any(fingerprint(_contract(value)) != self.contract_hash
                                  for value in descriptors):
            raise ValueError("Checkpoint policy contract fingerprint does not match")
        controls = metadata.get("controls", {})
        if not isinstance(controls, dict):
            raise ValueError("Checkpoint controls must contain a fixed-action object")
        self._fixed = fixed_indices(self.contract, controls)
        hidden = _integer(metadata.get("hidden_dim"), "checkpoint hidden_dim", 1, 4096)
        # Initialization consumes global CPU RNG even though all weights are replaced.
        with torch.random.fork_rng(devices=[]):
            self.model = ActorCritic(self.contract["observation_dim"], hidden,
                                    action_sizes=dict(zip(self.contract["action_names"],
                                                          self.contract["action_sizes"])))
        try:
            self.model.load_state_dict(saved["model"], strict=True)
        except (KeyError, TypeError, RuntimeError) as error:
            raise ValueError("Checkpoint model parameters do not match its contract") from error
        if not all(torch.isfinite(parameter).all() for parameter in self.model.parameters()):
            raise ValueError("Checkpoint contains non-finite parameters")
        self.model.to(self.device).eval().requires_grad_(False)
        self.deterministic = deterministic
        self._generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self._description.update(type="checkpoint", path=str(self.path), sha256=self.sha256,
                                 device=str(self.device), deterministic=deterministic,
                                 controls=deepcopy(controls))

    def verify_checkpoint(self, expected_sha256=None):
        """Reject replacement after resolution/loading, without reloading weights."""
        expected = self.sha256 if expected_sha256 is None else expected_sha256
        if expected != self.sha256 or checkpoint_sha256(self.path) != self.sha256:
            raise ValueError("Checkpoint SHA256 differs from the resolved plan")


def create_controller(definition, contract, count, device="cpu", seed=1, *, learner=None) -> Controller:
    """Instantiate a scripted/checkpoint opponent or learner-bound self-play."""
    if not isinstance(definition, dict):
        raise ValueError("Controller definition must be an object")
    kind = definition.get("type")
    if kind == "scripted":
        _options(definition, ("type", "controller", "options"))
        name = definition.get("controller")
        if not isinstance(name, str) or not name:
            raise ValueError("Scripted controller requires a controller name")
        factory = CONTROLLER_REGISTRY.get(name)
        if factory is None:
            if name.count(":") != 1:
                raise ValueError(f"Unknown controller: {name}")
            module_name, factory_name = name.split(":")
            if (not module_name or not all(part.isidentifier() for part in module_name.split("."))
                    or not factory_name.isidentifier()):
                raise ValueError("Custom controller must use an explicit module:factory")
            factory = getattr(importlib.import_module(module_name), factory_name, None)
            if not callable(factory):
                raise ValueError(f"Custom controller factory is not callable: {name}")
        options = definition.get("options", {})
        if not isinstance(options, dict):
            raise ValueError("Controller options must be an object")
        controller = factory(contract=_contract(contract), count=_integer(count, "count"),
                             seed=_integer(seed, "seed", 0, 2**63 - 1), options=deepcopy(options))
        if not isinstance(controller, Controller):
            raise ValueError("Custom controller does not implement the Controller protocol")
        return controller
    if kind == "self_play":
        _options(definition, ("type", "deterministic"))
        return SelfPlayController(learner, contract, count, device, seed,
                                  definition.get("deterministic", False))
    if kind == "checkpoint":
        _options(definition, ("type", "path", "deterministic", "sha256"))
        path = definition.get("path")
        if not isinstance(path, (str, Path)) or not str(path):
            raise ValueError("Checkpoint controller requires a path")
        return CheckpointController(path, contract, count, device, seed,
                                    definition.get("deterministic", False), definition.get("sha256"))
    raise ValueError(f"Unknown controller type: {kind!r}")


register_controller("stationary_target", StationaryTargetController)
register_controller("basic_melee", BasicMeleeController)
