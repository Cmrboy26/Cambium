# Shared Training Dependencies: Development

Follow the repository [development guide](../AGENTS.md).

## Ownership

- [fast_rl_model.py](fast_rl_model.py) provides the `ActorCritic` architecture for
  standalone simulator training and policy inference. `FastRLAgent` also provides
  PPO constants reused by the standalone trainer. The simulator does not run its
  legacy agent loop.
- Default [reward weights](../simulator/training/reward_weights.json) belong to
  the training component, not this folder. Environment files can override
  individual weights; changed effective rewards require a compatible new run.
- The legacy server, controllers, frontend, RCON integration, reward engine,
  logs and model checkpoints have been removed. Do not describe them as active
  services or reintroduce their runtime as a training dependency.

## Downstream Contracts

Read [training internals](../simulator/training/AGENTS.md) before changing network
dimensions or PPO defaults. Training and inference must agree
on checkpoint architecture and observation/action descriptors. Legacy backend
callers retain their default eight-movement-head behavior; standalone training
uses its own versioned layout.

Dependencies live in
[requirements.txt](../simulator/training/requirements.txt). Validate relevant
changes with `python -B -m unittest simulator.training.test_training -v` after
building the native training library. Reward settings are recorded in checkpoint
metadata; preserve compatibility checks rather than silently changing resumed runs.