import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from backend.fast_rl_model import ActorCritic
from simulator.training.contract import from_legacy
from simulator.training.environment import SCHEMA
from simulator.training.ppo import optimize
from simulator.training.timing import OptimizationTimer
from simulator.training.benchmark_ppo import (
    BENCHMARK_EPOCHS,
    build_parser,
    clone_model,
    clone_rollout,
    main,
    optimizer_steps,
    parse_minibatch_sizes,
    resize_pool_state,
    run_trial,
    safe_controller_metadata,
    safe_metadata,
    subset_rollout,
    synchronized_call,
    synthetic_rollout,
    validate_benchmark_sizes,
    validate_pool_device,
)


class BenchmarkChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.contract = from_legacy(SCHEMA)
        self.model = ActorCritic(self.contract["observation_dim"], 8,
                                 action_sizes=dict(zip(self.contract["action_names"],
                                                       self.contract["action_sizes"])))
        self.rollout = synthetic_rollout(self.model, 3, 5, self.contract, torch.device("cpu"), 47)

    def test_model_and_rollout_clones_are_equal_and_independent(self):
        model_copy = clone_model(self.model, "cpu")
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, model_copy.state_dict()[name]), name)
        rollout_copy = clone_rollout(self.rollout)
        for name, value in self.rollout.items():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(value, rollout_copy[name]), name)
        rollout_copy["observations"][0, 0, 0] += 1
        self.assertFalse(torch.equal(self.rollout["observations"], rollout_copy["observations"]))

    def test_smaller_capture_resizes_per_actor_state_without_mutating_source(self):
        state = {"controllers": {
            "walker": {"provenance": {"count": 4}, "directions": [0, 1, 2, 3],
                       "remaining": [4, 3, 2, 1], "generators": [{"state": index} for index in range(4)]},
            "fighter": {"provenance": {"count": 4}, "ticks": [1, 2, 3, 4]},
            "self": {"provenance": {"count": 4}, "snapshot_updates": [50, 50, 100, 100],
                     "snapshots": {50: {"model": "retained"}}},
        }}
        resized = resize_pool_state(state, 2)
        self.assertEqual(resized["controllers"]["walker"]["directions"], [0, 1])
        self.assertEqual(resized["controllers"]["walker"]["generators"], [{"state": 0}, {"state": 1}])
        self.assertEqual(resized["controllers"]["fighter"]["ticks"], [1, 2])
        self.assertEqual(resized["controllers"]["self"]["snapshot_updates"], [50, 50])
        self.assertEqual(resized["controllers"]["self"]["snapshots"], state["controllers"]["self"]["snapshots"])
        self.assertEqual(state["controllers"]["walker"]["provenance"]["count"], 4)
        self.assertEqual(len(state["controllers"]["walker"]["directions"]), 4)

    def test_real_capture_rejects_cross_device_controller_rng(self):
        state = {"controllers": {"self": {"provenance": {"device": "cuda"}}}}
        validate_pool_device(state, "cuda")
        with self.assertRaisesRegex(ValueError, "checkpoint device type"):
            validate_pool_device(state, "cpu")

    def test_nondivisible_batches_count_real_adam_steps(self):
        learner = {"lr": 3e-4, "clip_eps": 0.2, "vf_coef": 0.5, "ent_coef": 0.01,
                   "max_grad_norm": 0.5}
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learner["lr"], eps=1e-5)
        result = run_trial(self.model, optimizer.state_dict(), self.rollout, 4, learner,
                           torch.device("cpu"), 99)
        self.assertEqual(result["optimizer_steps"], optimizer_steps(15, 4, BENCHMARK_EPOCHS))
        self.assertGreaterEqual(result["seconds"], 0)
        self.assertEqual(set(result["statistics"]), {"loss", "policy_loss", "value_loss", "entropy"})

    def test_optimization_profile_preserves_results_and_rng(self):
        models = [clone_model(self.model, "cpu"), clone_model(self.model, "cpu")]
        results = []
        states = []
        optimizer_states = []
        timer = OptimizationTimer("cpu", enabled=True)
        for model, active_timer in zip(models, (None, timer)):
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
            torch.manual_seed(99)
            results.append(optimize(model, optimizer, self.rollout, 2, 4, timer=active_timer))
            states.append(torch.get_rng_state())
            optimizer_states.append(optimizer.state_dict())
        self.assertEqual(results[0], results[1])
        self.assertTrue(torch.equal(states[0], states[1]))
        for name, value in models[0].state_dict().items():
            self.assertTrue(torch.equal(value, models[1].state_dict()[name]), name)
        torch.testing.assert_close(optimizer_states[0], optimizer_states[1], rtol=0, atol=0)
        profile = timer.finish()
        self.assertEqual(profile["counts"]["samples"], 15)
        self.assertEqual(profile["counts"]["epochs"], 2)
        self.assertEqual(profile["counts"]["minibatches"], 8)
        self.assertEqual(set(profile["seconds"]), {"setup", "shuffle", "policy_evaluation", "loss",
            "finite_check", "backward", "gradient_clip", "optimizer_step", "statistics", "results"})

    def test_optimization_cuda_event_sampling_is_bounded(self):
        timer = OptimizationTimer("cpu", enabled=True)
        timer.device = torch.device("cuda")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4, eps=1e-5)
        with patch("torch.cuda.Event") as event, patch("torch.cuda.current_stream"), \
                patch("torch.cuda.synchronize"):
            event.return_value.elapsed_time.return_value = 1.0
            optimize(self.model, optimizer, self.rollout, 2, 1, timer=timer)
            profile = timer.finish()
        self.assertEqual(profile["counts"]["minibatches"], 30)
        self.assertEqual(profile["counts"]["cuda_profiled_minibatches"], 16)
        self.assertEqual(event.call_count, 16 * 7 * 2)
        self.assertEqual(set(profile["cuda_stream_seconds"]), {"policy_evaluation", "loss", "finite_check",
            "backward", "gradient_clip", "optimizer_step", "statistics"})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for GPU profiling checks")
    def test_optimization_profile_cuda_preserves_training_state(self):
        models = [clone_model(self.model, "cuda"), clone_model(self.model, "cuda")]
        rollout = {name: value.to("cuda") if isinstance(value, torch.Tensor) else value
                   for name, value in self.rollout.items()}
        timer = OptimizationTimer("cuda", enabled=True)
        results, states, optimizer_states = [], [], []
        for model, active_timer in zip(models, (None, timer)):
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
            torch.manual_seed(99)
            results.append(optimize(model, optimizer, rollout, 2, 4, timer=active_timer))
            states.append(torch.cuda.get_rng_state())
            optimizer_states.append(optimizer.state_dict())
        profile = timer.finish()
        torch.testing.assert_close(results[0], results[1])
        self.assertTrue(torch.equal(states[0], states[1]))
        torch.testing.assert_close(models[0].state_dict(), models[1].state_dict())
        torch.testing.assert_close(optimizer_states[0], optimizer_states[1])
        self.assertEqual(profile["counts"]["cuda_profiled_minibatches"], 8)
        self.assertIn("optimizer_step", profile["cuda_stream_seconds"])

    def test_synchronized_timer_orders_cuda_waits_around_measurement(self):
        events = []

        def sync(device):
            events.append("sync")

        def clock():
            events.append("clock")
            return 10.0 if events.count("clock") == 1 else 12.0

        def operation():
            events.append("operation")
            return "done"

        with patch("simulator.training.benchmark_ppo.synchronize_device", side_effect=sync), \
                patch("simulator.training.benchmark_ppo.time.perf_counter", side_effect=clock):
            result, seconds = synchronized_call(operation, torch.device("cuda"))
        self.assertEqual(events, ["sync", "clock", "operation", "sync", "clock"])
        self.assertEqual(result, "done")
        self.assertEqual(seconds, 2.0)

    def test_invalid_options_and_nondivisible_size_validation(self):
        self.assertEqual(parse_minibatch_sizes("1024, 4096"), [1024, 4096])
        for invalid in ("", "0", "1,,2", "2,2", "-1", str(1_048_577)):
            with self.subTest(invalid=invalid), self.assertRaises(Exception):
                parse_minibatch_sizes(invalid)
        self.assertEqual(validate_benchmark_sizes(3, 5, [4], max_samples=15), 15)
        self.assertEqual(optimizer_steps(15, 4), 8)
        self.assertEqual(subset_rollout(self.rollout, 7)["observations"].shape[:2], (1, 7))
        for invalid_samples in (0, 16, True):
            with self.subTest(num_samples=invalid_samples), self.assertRaises(ValueError):
                subset_rollout(self.rollout, invalid_samples)
        for arguments in ((0, 2, [1]), (2, 0, [1]), (2, 2, []), (2, 2, [0]),
                          (2, 2, [1, 1]), (2, 2, [1], 3)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                validate_benchmark_sizes(*arguments)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--output", "/tmp/out", "--minibatch-sizes", "0"])

    def test_metadata_scrubs_paths_and_secrets_but_keeps_opponent_identity(self):
        value = safe_metadata({"checkpoint_path": "/private/run/latest.pt", "api_token": "hidden",
                               "opponent": {"id": "practice", "count": 3}})
        self.assertEqual(value, {"opponent": {"id": "practice", "count": 3}})
        controller = safe_controller_metadata({"type": "checkpoint", "path": "/private/baseline.pt",
                                               "sha256": "a" * 64, "seed": 1, "token": "hidden"})
        self.assertEqual(controller, {"type": "checkpoint", "sha256": "a" * 64})

    def test_smoke_cli_labels_synthetic_artifacts_and_is_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "benchmark"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--smoke", "--device", "cpu", "--envs", "2",
                                       "--rollout-steps", "4", "--minibatch-sizes", "3,5",
                                       "--repeats", "1", "--warmup", "0", "--output", str(output)]), 0)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["classification"],
                             "synthetic_smoke_not_learning_or_performance_evidence")
            self.assertEqual(summary["capture"]["source"], "synthetic_smoke")
            self.assertEqual(summary["capture"]["num_samples"], 8)
            self.assertEqual(len(summary["minibatches"]), 2)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--smoke", "--output", str(output)])


if __name__ == "__main__":
    unittest.main()
