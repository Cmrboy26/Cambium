# Shared Policy Model

The shared actor-critic network and PPO constants used by simulator training
and live policy inference are in [fast_rl_model.py](fast_rl_model.py).
The legacy Minecraft-connected server and tooling have been removed.

Use the [training guide](../simulator/training/README.md) to set up Python,
train or resume a policy. [Workflow scripts](../scripts/README.md) launch
training and inference; this folder is not a standalone service.

Default rewards live in
[simulator/training/reward_weights.json](../simulator/training/reward_weights.json).
Environment files can override individual weights. Python dependencies are in
[requirements.txt](../simulator/training/requirements.txt).

[AGENTS.md](AGENTS.md) covers model ownership and downstream compatibility.