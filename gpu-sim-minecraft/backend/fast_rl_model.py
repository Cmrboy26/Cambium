"""Actor-critic network for the standalone simulator's multi-discrete policy."""

import torch
import torch.nn as nn
from typing import Dict, Tuple

YAW_BINS = [-20.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 20.0]
PITCH_BINS = [-5.0, -2.0, 0.0, 2.0, 5.0]


class ActorCritic(nn.Module):
    """
    Actor-critic network with shared feature extractor.
    Actor: multi-discrete policy heads (movement, jump, attack, yaw, pitch)
    Critic: single scalar value estimate V(s)
    """
    def __init__(self, observation_dim: int, hidden_dim: int = 256, movement_bins: int = 8,
                 equipment_heads: bool = False, action_sizes: dict[str, int] | None = None):
        super(ActorCritic, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Actor heads
        sizes = {"movement": movement_bins, "jump": 2, "sprint": 2, "attack": 2,
                 "yaw": len(YAW_BINS), "pitch": len(PITCH_BINS)}
        if equipment_heads:
            sizes.update(weapon=2, shield=2)
        if action_sizes is not None:
            sizes = dict(action_sizes)
        from simulator.training.contract import HEADS
        allowed = set(HEADS)
        if not sizes or any(name not in allowed or type(size) is not int or not 1 <= size <= 64 for name, size in sizes.items()):
            raise ValueError("Invalid categorical policy heads")
        self.action_names = tuple(sizes)
        for name, size in sizes.items():
            setattr(self, f"{name}_head", nn.Linear(hidden_dim, size))

        # Critic head
        self.value_head = nn.Linear(hidden_dim, 1)

        # Orthogonal init for better PPO convergence
        actor_heads = [getattr(self, f"{name}_head") for name in self.action_names]
        for module in actor_heads:
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)
        # Re-apply pitch center bias after orthogonal init
        with torch.no_grad():
            if "pitch" in sizes:
                self.pitch_head.bias[sizes["pitch"] // 2] = 2.0
            if "jump" in sizes:
                self.jump_head.bias[0] = 2.0

    def forward(self, obs: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Returns (logits_dict, value)"""
        features = self.shared(obs)
        logits = {name: getattr(self, f"{name}_head")(features) for name in self.action_names}
        value = self.value_head(features).squeeze(-1)
        return logits, value


class BridgeActorCritic(nn.Module):
    """Experimental standalone model; not admitted by the scalar training contract."""

    def __init__(self, observation_dim: int, descriptor: dict, hidden_dim: int = 256,
                 action_sizes: dict[str, int] | None = None):
        super().__init__()
        from simulator.training.contract import validate_bridge_observation_descriptor
        self.descriptor = validate_bridge_observation_descriptor(descriptor)
        self.observation_dim = observation_dim
        self.voxel_encoder = nn.Sequential(nn.Conv3d(5, 8, 3, padding=1), nn.ReLU(),
                                           nn.AdaptiveAvgPool3d(1), nn.Flatten())
        self.policy = ActorCritic(observation_dim + 10 + 8, hidden_dim, action_sizes=action_sizes)
        self.action_names = self.policy.action_names

    def forward(self, inputs):
        scalar, match, voxels = inputs["scalar"], inputs["match"], inputs["voxels"]
        if (scalar.ndim != 2 or scalar.shape[1] != self.observation_dim
                or match.shape != (scalar.shape[0], 10)
                or voxels.shape != (scalar.shape[0], 5, 9, 9)
                or scalar.dtype != torch.float32 or match.dtype != torch.float32 or voxels.dtype != torch.uint8):
            raise ValueError("Invalid Bridge observation tensors")
        encoded = torch.nn.functional.one_hot(voxels.to(torch.long), num_classes=5).movedim(-1, 1)
        terrain = self.voxel_encoder(encoded.to(dtype=scalar.dtype))
        return self.policy(torch.cat((scalar, match, terrain), dim=1))
