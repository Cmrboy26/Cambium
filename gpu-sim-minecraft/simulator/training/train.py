"""Train a resolved experiment compiled from arena, environment, policy and plan."""

import argparse
from pathlib import Path

from simulator.training.environment import ROOT
from simulator.training.ppo import compute_gae, metadata, optimize


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return result


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", help="Version 2 training plan; defaults to the shield/axe curriculum")
    parser.add_argument("--resume", help="Resume a resolved-experiment checkpoint with fresh episodes")
    parser.add_argument("--initialize", help="Initialize compatible weights for a new experiment")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--envs", type=positive, help="Worlds: default 16 CPU / 8192 CUDA; restored on resume")
    parser.add_argument("--updates", type=positive, default=100, help="Maximum additional PPO updates")
    parser.add_argument("--rollout-steps", type=positive, help="Ticks per rollout: default 128; restored on resume")
    parser.add_argument("--threads", type=positive, default=4)
    parser.add_argument("--library", help="Explicit native training DLL/SO")
    parser.add_argument("--output", help="Run directory; default timestamp or resumed checkpoint directory")
    parser.add_argument("--save-every", type=positive, default=10)
    parser.add_argument("--replay-every", type=int, default=0, help="Completed training episodes between replay samples")
    parser.add_argument("--replay-dir", help="Replay directory; default OUTPUT/replays")
    parser.add_argument("--hidden-dim", type=positive, help="Override plan learner width for a new experiment")
    parser.add_argument("--epochs", type=positive, help="Override plan learner PPO epochs")
    parser.add_argument("--minibatch-size", type=positive, help="Override plan learner minibatch size")
    parser.add_argument("--lr", type=float, help="Override plan learner learning rate")
    parser.add_argument("--max-ticks", type=positive, help="Explicit override of all episode limits for smoke checks")
    options = parser.parse_args(argv)
    if options.envs is not None and options.envs > 65536:
        parser.error("envs must be <=65536")
    if options.hidden_dim is not None and options.hidden_dim > 4096:
        parser.error("hidden-dim must be <=4096")
    if options.max_ticks is not None and options.max_ticks > 1000000:
        parser.error("max-ticks must be <=1000000")
    if options.lr is not None:
        import math
        if not math.isfinite(options.lr) or options.lr <= 0:
            parser.error("lr must be finite and positive")
    if options.initialize and options.resume:
        parser.error("Choose --initialize or --resume, not both")
    if options.replay_every < 0 or options.replay_dir and not options.replay_every:
        parser.error("replay-every must be nonnegative; replay-dir requires replay-every")
    if not options.plan and not options.resume:
        options.plan = str(ROOT / "simulator/training/presets/shield-axe-curriculum.json")
    if options.plan:
        options.plan = str(Path(options.plan).resolve())
    return options


def main(argv=None):
    from simulator.training.research import run
    return run(arguments(argv))


if __name__ == "__main__":
    raise SystemExit(main())
