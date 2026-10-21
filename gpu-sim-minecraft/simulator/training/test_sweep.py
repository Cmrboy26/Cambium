"""Sweep supervision and comparison checks independent of installed curricula."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from simulator.training.contract import from_legacy
from simulator.training.environment import ROOT, SCHEMA
from simulator.training.sweep import prepare, run
from simulator.training.plot_sweep import plot, series


class SweepChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "sweeps/test"
        self.runs = self.root / "runs"
        arena = {"format": "minecraft_sim_arena", "version": 1, "minecraft": "26.3", "name": "test",
                 "requires": ["movement", "melee"], "rules": {"simulate_food": False, "natural_regeneration": False},
                 "bounds": {"min": [-5, 0, -5], "size": [11, 4, 11]},
                 "blocks": [{"from": [-5, 0, -5], "to": [5, 0, 5], "block": "minecraft:stone"}]}
        environment = {"format": "minecraft_sim_environment", "version": 2, "minecraft": "26.3", "name": "test",
                       "arena": "arena.json", "players": {"shared": {"loadout": {"weapons": ["minecraft:iron_sword"]}},
                       "overrides": [{"position": [-2, 1, 0]}, {"position": [2, 1, 0]}]},
                       "task": {"max_ticks": 2, "rewards": {"episode_timeout": -1}}}
        plan = {"format": "minecraft_sim_training_plan", "version": 2, "name": "test", "seed": 42,
                "policy": "policy.json", "environment": "environment.json",
                "learner": {"hidden_dim": 8, "epochs": 1, "minibatch_size": 4},
                "opponents": {"target": {"type": "scripted", "controller": "stationary_target"}},
                "mixtures": {"targets": [{"id": "target", "weight": 1}]},
                "stages": [{"name": "combat", "opponents": "targets", "advance": {"updates": 2}}]}
        for name, value in (("arena", arena), ("environment", environment), ("policy", from_legacy(SCHEMA)), ("plan", plan)):
            (self.root / f"{name}.json").write_text(json.dumps(value))
        self.plan = self.root / "plan.json"
        self.options = {"device": "cpu", "envs": 2, "rollout_steps": 2, "updates": 2, "threads": 1}

    def prepare(self, **changes):
        settings = {"plan": self.plan, "directory": self.directory, "runs": self.runs,
                    "name": "test", "parameter": "lr", "values": [".0001", ".0003"], "options": self.options}
        return prepare(**{**settings, **changes})

    def test_prepare_freezes_independent_experiments_and_all_trials_default(self):
        manifest = self.prepare()
        self.assertEqual(manifest["max_parallel"], 0)
        self.assertEqual(manifest["seed"], 42)
        self.assertEqual(len(manifest["trials"]), 2)
        self.assertFalse(self.runs.exists())
        self.plan.unlink()
        for trial, rate in zip(manifest["trials"], (.0001, .0003)):
            frozen = json.loads(Path(trial["plan"]).read_text())
            self.assertEqual(frozen["format"], "minecraft_sim_experiment")
            self.assertEqual(frozen["seed"], 42)
            self.assertEqual(frozen["learner"]["lr"], rate)
            self.assertEqual(Path(trial["output"]).parent, self.runs)
        self.assertEqual(json.loads((self.directory / "sweep.json").read_text()), manifest)

    def test_rejects_duplicates_existing_outputs_and_resource_sweeps(self):
        for changes in ({"values": [".0001", "1e-4"]}, {"parameter": "memory"}, {"max_parallel": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.prepare(**changes)
        (self.runs / "test-0.0001").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.prepare()
        self.assertFalse(self.directory.exists())

    def test_cpu_budget_is_checked_before_any_trial_starts(self):
        self.prepare()
        library = self.root / "library.so"
        library.touch()
        with self.assertRaisesRegex(ValueError, "need 2 CPU threads"):
            run(self.directory / "sweep.json", library, allocated_cpus=1)
        self.assertFalse((self.directory / "status.json").exists())

    def test_default_concurrency_and_trial_failure_isolation(self):
        self.prepare()
        library = self.root / "library.so"
        library.touch()
        worker = self.root / "worker.py"
        worker.write_text('''import json, os, pathlib, sys, time
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
output = pathlib.Path(args["--output"])
output.mkdir(parents=True)
(output / "started").write_text(json.dumps({"time": time.time(), "threads": os.environ["OMP_NUM_THREADS"]}))
sibling = output.parent / ("test-0.0003" if output.name.endswith("0.0001") else "test-0.0001")
deadline = time.monotonic() + 5
while not (sibling / "started").exists():
    if time.monotonic() > deadline: raise SystemExit(9)
    __import__("threading").Event().wait(.01)
print("worker output")
print("worker error", file=sys.stderr)
raise SystemExit(3 if output.name.endswith("0.0001") else 0)
''')
        code = run(self.directory / "sweep.json", library, allocated_cpus=2, trainer=[sys.executable, str(worker)])
        self.assertEqual(code, 1)
        status = json.loads((self.directory / "status.json").read_text())
        self.assertEqual(status["trials"]["test-0.0001"]["state"], "failed")
        self.assertEqual(status["trials"]["test-0.0003"]["state"], "succeeded")
        for name in status["trials"]:
            self.assertEqual(json.loads((self.runs / name / "started").read_text())["threads"], "1")
            self.assertIn("worker output", (self.directory / "logs" / f"{name}.out").read_text())
            self.assertIn("worker error", (self.directory / "logs" / f"{name}.err").read_text())

    def test_optional_limit_runs_trials_sequentially(self):
        self.prepare(max_parallel=1)
        library = self.root / "library.so"
        library.touch()
        code = run(self.directory / "sweep.json", library, allocated_cpus=1,
                   trainer=[sys.executable, "-c", "import sys; print('complete')"])
        self.assertEqual(code, 0)
        trials = list(json.loads((self.directory / "status.json").read_text())["trials"].values())
        self.assertLessEqual(trials[0]["finished_at"], trials[1]["started_at"])

    def test_plot_overlays_trials_and_deduplicates_cached_evaluations(self):
        manifest = self.prepare()
        for trial, score in zip(manifest["trials"], (.25, .75)):
            output = Path(trial["output"])
            output.mkdir(parents=True)
            rows = [{"update": update, "samples": update * 4, "elapsed_seconds": update * 10,
                     "evaluations": {"held": {"snapshot_update": 1, "win_rate": score, "reward_mean": 0, "episodes": 4}}}
                    for update in (1, 2)]
            (output / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows) + '{"update":')
        path = self.directory / "sweep.json"
        report = series(path)
        self.assertEqual([curve["points"] for curve in report["curves"]], [[(10, 25)], [(10, 75)]])
        by_samples = series(path, axis="samples")
        self.assertEqual(by_samples["curves"][0]["points"], [(4, 25)])
        image = plot(path)
        self.assertTrue(image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_plot_handles_pending_trials_and_explicit_run_selection(self):
        manifest = self.prepare()
        self.assertTrue(plot(self.directory / "sweep.json").is_file())
        for trial, loss in zip(manifest["trials"], (1, 2)):
            output = Path(trial["output"])
            output.mkdir(parents=True)
            (output / "metrics.jsonl").write_text(json.dumps({"update": 1, "loss": loss, "iteration_seconds": 3}) + "\n")
        report = series(self.directory / "sweep.json", metric="loss")
        self.assertEqual([curve["points"] for curve in report["curves"]], [[(3, 1)], [(3, 2)]])
        self.assertIn("Legacy elapsed", report["warnings"][0])

    @unittest.skipUnless(os.name == "posix", "Process-group cancellation requires POSIX")
    def test_cancel_stops_running_worker_and_does_not_start_pending_trial(self):
        self.prepare(max_parallel=1)
        library = self.root / "library.so"
        library.touch()
        worker = self.root / "waiting.py"
        worker.write_text('''import json, os, pathlib, sys, threading
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
output = pathlib.Path(args["--output"])
output.mkdir(parents=True)
(output / "pid").write_text(str(os.getpid()))
threading.Event().wait(60)
''')
        command = [sys.executable, "-B", "-c", "from simulator.training.sweep import run; import sys; "
                   "raise SystemExit(run(sys.argv[1], sys.argv[2], allocated_cpus=1, trainer=[sys.executable, sys.argv[3]]))",
                   str(self.directory / "sweep.json"), str(library), str(worker)]
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
            try:
                deadline = time.monotonic() + 15
                marker = self.runs / "test-0.0001/pid"
                while not marker.exists():
                    if time.monotonic() > deadline or process.poll() is not None:
                        self.fail("Sweep did not start its first worker")
                    threading.Event().wait(.02)
                worker_pid = int(marker.read_text())
                process.send_signal(signal.SIGTERM)
                _, errors = process.communicate(timeout=15)
                self.assertEqual(process.returncode, 143, errors)
                with self.assertRaises(ProcessLookupError):
                    os.kill(worker_pid, 0)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
        status = json.loads((self.directory / "status.json").read_text())
        self.assertEqual(status["state"], "interrupted")
        self.assertTrue(all(trial["state"] == "interrupted" for trial in status["trials"].values()))
        self.assertFalse((self.runs / "test-0.0003").exists())

    def test_actual_cpu_trainers_write_independent_checkpoints_metrics_and_chart(self):
        library = ROOT / "simulator/build-research/minecraft_sim_training.so"
        if not library.is_file():
            self.skipTest("Build the CPU training library for end-to-end sweep checks")
        authored = json.loads(self.plan.read_text())
        authored["evaluations"] = {"held": {"stage": "combat", "every_updates": 1, "episodes": 2, "worlds": 2, "seed": 17}}
        self.plan.write_text(json.dumps(authored))
        manifest = self.prepare()
        path = self.directory / "sweep.json"
        self.assertEqual(run(path, library, allocated_cpus=2), 0, (self.directory / "status.json").read_text())
        for trial in manifest["trials"]:
            output = Path(trial["output"])
            self.assertTrue((output / "latest.pt").is_file())
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["experiment"]["learner"]["lr"], trial["value"])
            rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["update"] for row in rows], [1, 2])
            self.assertLess(rows[0]["elapsed_seconds"], rows[1]["elapsed_seconds"])
        self.assertEqual([len(curve["points"]) for curve in series(path)["curves"]], [2, 2])
        self.assertTrue(plot(path).is_file())