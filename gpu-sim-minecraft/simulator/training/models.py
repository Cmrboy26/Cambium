"""Trusted, versioned factories for standalone actor-critic architectures."""

from copy import deepcopy
import importlib
import json

import torch
import torch.nn as nn

from backend.fast_rl_model import ActorCritic, BridgeActorCritic
from simulator.training.contract import validate_bridge_observation_descriptor


MODEL_REGISTRY = {}


def load_trusted_models(modules):
    if not isinstance(modules, (list, tuple)):
        raise ValueError("Trusted model modules must be a list")
    for name in modules:
        if not isinstance(name, str) or not name or not all(part.isidentifier() for part in name.split(".")):
            raise ValueError("Trusted model module must be a dotted Python module name")
        importlib.import_module(name)


def register_model(name, version, factory):
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError("Model name must be an identifier")
    if type(version) is not int or version < 1:
        raise ValueError("Model version must be a positive integer")
    if not callable(factory):
        raise ValueError("Model factory must be callable")
    key = (name, version)
    if key in MODEL_REGISTRY:
        raise ValueError(f"Model is already registered: {name} version {version}")
    MODEL_REGISTRY[key] = factory


def validate_model_descriptor(descriptor):
    if (not isinstance(descriptor, dict)
            or set(descriptor) != {"format", "version", "name", "model_version", "options"}
            or descriptor["format"] != "minecraft_sim_model"
            or type(descriptor["version"]) is not int or descriptor["version"] != 1
            or not isinstance(descriptor["name"], str) or not descriptor["name"].isidentifier()
            or type(descriptor["model_version"]) is not int or descriptor["model_version"] < 1
            or not isinstance(descriptor["options"], dict)):
        raise ValueError("Invalid model architecture descriptor")
    try:
        encoded = json.dumps(descriptor["options"], allow_nan=False, sort_keys=True)
        if json.loads(encoded) != descriptor["options"]:
            raise ValueError("Model options must contain only JSON values")
    except (TypeError, ValueError) as error:
        raise ValueError("Model options must contain only finite JSON values") from error
    return deepcopy(descriptor)


def legacy_model_descriptor(bridge_observation):
    if bridge_observation is not None:
        validate_bridge_observation_descriptor(bridge_observation)
    return {"format": "minecraft_sim_model", "version": 1,
            "name": "bridge_voxel" if bridge_observation is not None else "scalar_mlp",
            "model_version": 1, "options": {}}


def checkpoint_model_descriptor(metadata, bridge_observation):
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint is missing model metadata")
    return (validate_model_descriptor(metadata["model"]) if "model" in metadata
            else legacy_model_descriptor(bridge_observation))


def model_factory(descriptor):
    descriptor = validate_model_descriptor(descriptor)
    factory = MODEL_REGISTRY.get((descriptor["name"], descriptor["model_version"]))
    if factory is None:
        raise ValueError(f"Model is not registered: {descriptor['name']} version {descriptor['model_version']}")
    return factory


def bridge_model_descriptor(model):
    if isinstance(model, BridgeActorCritic):
        return model.descriptor
    return getattr(model, "_mcsim_bridge_observation", None)


def model_identity(model):
    descriptor = getattr(model, "_mcsim_model_descriptor", None)
    if descriptor is not None:
        return descriptor
    if type(model) is ActorCritic:
        return legacy_model_descriptor(None)
    if type(model) is BridgeActorCritic:
        return legacy_model_descriptor(model.descriptor)
    return type(model)


def validate_structured_model(model, policy):
    descriptor = validate_bridge_observation_descriptor(bridge_model_descriptor(model))
    if isinstance(model, BridgeActorCritic):
        sizes = [getattr(model.policy, f"{name}_head").out_features for name in model.action_names]
    else:
        sizes = getattr(model, "_mcsim_action_sizes", None)
    if (model.observation_dim != policy["observation_dim"]
            or model.match_dim != len(descriptor["match_features"])
            or model.action_names != tuple(policy["action_names"])
            or sizes != policy["action_sizes"]):
        raise ValueError("Structured Bridge model and scalar policy are incompatible")
    return descriptor


def create_model(descriptor, observation_dim, bridge_observation, hidden_dim, action_sizes):
    descriptor = validate_model_descriptor(descriptor)
    factory = model_factory(descriptor)
    model = factory(observation_dim=observation_dim, bridge_observation=deepcopy(bridge_observation),
                    hidden_dim=hidden_dim, action_sizes=dict(action_sizes), options=deepcopy(descriptor["options"]))
    if not isinstance(model, nn.Module) or getattr(model, "action_names", None) != tuple(action_sizes):
        raise ValueError("Model factory must return a torch module with the configured action names")
    if bridge_observation is not None:
        observation = validate_bridge_observation_descriptor(bridge_observation)
        if (getattr(model, "descriptor", None) != observation
                or getattr(model, "observation_dim", None) != observation_dim
                or getattr(model, "match_dim", None) != len(observation["match_features"])):
            raise ValueError("Structured model attributes differ from its observation contract")
        model._mcsim_bridge_observation = deepcopy(observation)
    model._mcsim_action_sizes = list(action_sizes.values())
    model._mcsim_model_descriptor = deepcopy(descriptor)
    sample = torch.zeros(2, observation_dim, dtype=torch.float32)
    inputs = sample if bridge_observation is None else {
        "scalar": sample, "match": torch.zeros(2, model.match_dim, dtype=torch.float32),
        "voxels": torch.zeros(2, 5, 9, 9, dtype=torch.uint8)}
    was_training = model.training
    try:
        model.eval()
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            logits, values = model(inputs)
        if (not isinstance(logits, dict) or tuple(logits) != tuple(action_sizes)
                or any(not isinstance(logits[name], torch.Tensor)
                       or logits[name].shape != (2, size) or logits[name].device.type != "cpu"
                       or not torch.is_floating_point(logits[name]) or not torch.isfinite(logits[name]).all()
                       for name, size in action_sizes.items())
                or not isinstance(values, torch.Tensor) or values.shape != (2,) or values.device.type != "cpu"
                or not torch.is_floating_point(values) or not torch.isfinite(values).all()):
            raise ValueError("Model outputs differ from the finite logits/value contract")
    finally:
        model.train(was_training)
    return model


def _scalar_model(*, observation_dim, bridge_observation, hidden_dim, action_sizes, options):
    if bridge_observation is not None or options:
        raise ValueError("Scalar MLP requires scalar observations and no model options")
    return ActorCritic(observation_dim, hidden_dim, movement_bins=9, action_sizes=action_sizes)


def _bridge_model(*, observation_dim, bridge_observation, hidden_dim, action_sizes, options):
    if bridge_observation is None or options:
        raise ValueError("Bridge voxel model requires Bridge observations and no model options")
    return BridgeActorCritic(observation_dim, bridge_observation, hidden_dim, action_sizes)


register_model("scalar_mlp", 1, _scalar_model)
register_model("bridge_voxel", 1, _bridge_model)
