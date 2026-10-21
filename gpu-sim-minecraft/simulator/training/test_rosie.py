"""SSH artifact protocol tests using local subprocesses instead of a cluster."""

import base64
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from simulator.training.rosie import (
    REMOTE_SCRIPT, SESSION_SCRIPT, Rosie, configure, connect_terminal, connection_id,
    main, session_path, session_response, valid_name,
)


@unittest.skipUnless(os.name != "nt" and shutil.which("bash"), "Requires a POSIX shell")
class SetupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "project with spaces"
        rosie = self.root / "rosie"
        rosie.mkdir(parents=True)
        source = Path(__file__).resolve().parents[2] / "rosie/setup.sh"
        self.script = rosie / "setup.sh"
        shutil.copyfile(source, self.script)
        self.bin = self.root / "tools"
        self.bin.mkdir()
        self.log = self.root / "calls"
        self.environment = self.root / ".venv-rosie"
        self.python = self.bin / "python3"
        self.python.write_text(
            '#!/usr/bin/env bash\n'
            'printf "%s\\n" "$*" >> "$SETUP_TEST_LOG"\n'
            'if [[ "$1" == "-m" && "$2" == "venv" ]]; then\n'
            '  mkdir -p "$3/bin"\n'
            '  printf "test\\n" > "$3/pyvenv.cfg"\n'
            '  cp "$0" "$3/bin/python"\n'
            '  chmod u+x "$3/bin/python"\n'
            'fi\n'
            'if [[ "$1" == "-c" && "${SETUP_TEST_BROKEN:-0}" == "1" ]]; then exit 7; fi\n',
            encoding="utf-8",
        )
        self.python.chmod(0o700)

    def run_setup(self, broken=False):
        environment = {**os.environ, "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
                       "SETUP_TEST_LOG": str(self.log), "SETUP_TEST_BROKEN": "1" if broken else "0"}
        return subprocess.run(["bash", str(self.script)], env=environment, capture_output=True,
                              text=True, timeout=10)

    def existing(self):
        (self.environment / "bin").mkdir(parents=True)
        (self.environment / "pyvenv.cfg").write_text("test\n")
        shutil.copyfile(self.python, self.environment / "bin/python")
        (self.environment / "bin/python").chmod(0o700)
        activation = self.environment / "bin/activate"
        activation.write_text("preserve this activation script\n")
        activation.chmod(0o444)
        return activation

    def test_reuses_environment_without_rewriting_readonly_activation(self):
        activation = self.existing()
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Reusing Python environment", result.stdout)
        self.assertEqual(activation.read_text(), "preserve this activation script\n")
        self.assertEqual(activation.stat().st_mode & 0o777, 0o444)
        calls = self.log.read_text().splitlines()
        self.assertFalse(any(line.startswith("-m venv ") for line in calls))
        self.assertEqual(sum(line.startswith("-m pip install") for line in calls), 3)

    def test_creates_new_environment(self):
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.environment / "pyvenv.cfg").is_file())
        self.assertTrue(self.log.read_text().startswith("-m venv "))

    def test_incomplete_environment_is_preserved(self):
        self.environment.mkdir()
        marker = self.environment / "keep"
        marker.write_text("untouched")
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("incomplete", result.stderr)
        self.assertEqual(marker.read_text(), "untouched")
        self.assertFalse(self.log.exists())

    def test_broken_environment_fails_before_package_changes(self):
        activation = self.existing()
        result = self.run_setup(broken=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("failed validation", result.stderr)
        self.assertNotIn("-m pip", self.log.read_text())
        self.assertEqual(activation.stat().st_mode & 0o777, 0o444)


@unittest.skipUnless(os.name != "nt" and shutil.which("bash"), "Requires a POSIX shell")
class TrainingJobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "project with spaces"
        self.tools = self.root / "tools"
        self.tools.mkdir(parents=True)
        (self.root / "simulator/training").mkdir(parents=True)
        (self.root / "simulator/setups/curricula").mkdir(parents=True)
        (self.root / "simulator/setups/curricula/shield-axe-curriculum.json").write_text("{}")
        (self.root / "simulator/build-rosie-cuda-75").mkdir()
        (self.root / "simulator/build-rosie-cuda-75/minecraft_sim_training.so").write_bytes(b"compiled simulator")
        (self.root / ".venv-rosie/bin").mkdir(parents=True)
        self.log = self.root / "arguments"
        self.snapshot = self.root / "library-snapshot"
        self.scratch = self.root / "scratch"
        self.scratch.mkdir()
        for name in ("module", "nvcc", "nvidia-smi"):
            self.executable(self.tools / name, "#!/usr/bin/env bash\nexit 0\n")
        self.executable(self.tools / "cmake",
                        '#!/usr/bin/env bash\n'
                        'if [[ "$1" == "--version" ]]; then echo "cmake version 3.25.0"; exit 0; fi\n'
                        'if [[ "${JOB_TEST_BUILD_FAIL:-0}" == "1" ]]; then exit 9; fi\n'
                        'if flock --nonblock "$JOB_TEST_BUILD/.build.lock" true; then\n'
                        '  echo "Build lock was not held" >&2; exit 10\n'
                        'fi\n'
                        'if [[ "${JOB_TEST_VERSIONED_BUILD:-0}" == "1" && "$1" == "--build" ]]; then\n'
                        '  printf "%s" "$SLURM_JOB_ID" > "$JOB_TEST_BUILD/minecraft_sim_training.so"\n'
                        'fi\n')
        self.executable(self.root / ".venv-rosie/bin/python",
                        '#!/usr/bin/env bash\n'
                        'if [[ "$1" == "-B" ]]; then\n'
                        '  printf "%s\\n" "$@" > "$JOB_TEST_LOG"\n'
                        '  while (( $# )); do\n'
                        '    if [[ "$1" == "--library" ]]; then cp -- "$2" "$JOB_TEST_SNAPSHOT"; break; fi\n'
                        '    shift\n'
                        '  done\n'
                        '  if ! flock --nonblock "$JOB_TEST_BUILD/.build.lock" true; then\n'
                        '    echo "Build lock was held during training" >&2; exit 11\n'
                        '  fi\n'
                        '  if [[ "${JOB_TEST_WAIT_FOR_INPUT:-0}" == "1" ]]; then\n'
                        '    printf "TRAINING_READY\\n"\n'
                        '    IFS= read -r answer\n'
                        '    cp -- "$2" "$JOB_TEST_AFTER_SNAPSHOT"\n'
                        '  fi\n'
                        '  exit "${JOB_TEST_TRAIN_FAIL:-0}"\n'
                        'fi\n')
        self.script = Path(__file__).resolve().parents[2] / "rosie/train.sbatch"

    def executable(self, path, text):
        path.write_text(text)
        path.chmod(0o700)

    def job_environment(self, **settings):
        environment = {key: value for key, value in os.environ.items() if not key.startswith("SIM_")}
        environment.update(PATH=str(self.tools) + os.pathsep + os.environ["PATH"],
                           SLURM_SUBMIT_DIR=str(self.root), SLURM_JOB_ID="77", JOB_TEST_LOG=str(self.log),
                           SLURM_TMPDIR=str(self.scratch), JOB_TEST_SNAPSHOT=str(self.snapshot),
                           JOB_TEST_BUILD=str(self.root / "simulator/build-rosie-cuda-75"),
                           MCSIM_CUDA_ARCHITECTURES="75")
        environment.update(settings)
        return environment

    def launch(self, **settings):
        return subprocess.run(["bash", str(self.script)], env=self.job_environment(**settings), capture_output=True,
                              text=True, timeout=10)

    def test_defaults_use_plan_learner_without_legacy_flags(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self.log.read_text().splitlines()
        self.assertIn("--plan", arguments)
        self.assertEqual(arguments[arguments.index("--plan") + 1],
                         "simulator/setups/curricula/shield-axe-curriculum.json")
        self.assertEqual(arguments[arguments.index("--envs") + 1], "256")
        for removed in ("--schema", "--environment", "--seed", "--epochs", "--hidden-dim", "--minibatch-size"):
            self.assertNotIn(removed, arguments)

    def test_resume_does_not_override_checkpoint_settings(self):
        checkpoint = self.root / "latest.pt"
        checkpoint.touch()
        result = self.launch(SIM_RESUME=str(checkpoint))
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self.log.read_text().splitlines()
        self.assertIn(str(checkpoint), arguments)
        for omitted in ("--plan", "--envs", "--rollout-steps", "--hidden-dim", "--epochs"):
            self.assertNotIn(omitted, arguments)

    def test_removed_configuration_fails_before_build(self):
        result = self.launch(SIM_ENVIRONMENT="old-scene.json")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("were replaced", result.stdout)
        self.assertFalse(self.log.exists())

    def test_named_output_directory_reaches_trainer(self):
        output = "simulator/training/runs/lr-test-.0001"
        result = self.launch(SIM_OUTPUT=output)
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self.log.read_text().splitlines()
        self.assertEqual(arguments[arguments.index("--output") + 1], output)

    def test_private_library_is_copied_and_cleaned_after_training(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self.log.read_text().splitlines()
        library = Path(arguments[arguments.index("--library") + 1])
        self.assertEqual(library.parent.parent, self.scratch)
        self.assertTrue(library.parent.name.startswith("cambium-77."))
        self.assertEqual(self.snapshot.read_bytes(), b"compiled simulator")
        self.assertFalse(library.parent.exists())
        self.assertTrue((self.root / "simulator/build-rosie-cuda-75/minecraft_sim_training.so").is_file())
        self.assertTrue((self.root / "simulator/build-rosie-cuda-75/.build.lock").is_file())

    def test_build_failure_never_starts_training_and_cleans_scratch(self):
        result = self.launch(JOB_TEST_BUILD_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_training_failure_still_cleans_private_library(self):
        result = self.launch(JOB_TEST_TRAIN_FAIL="9")
        self.assertEqual(result.returncode, 9, result.stderr)
        self.assertTrue(self.snapshot.exists())
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_lock_failure_never_starts_build_or_training(self):
        self.executable(self.tools / "flock", "#!/usr/bin/env bash\nexit 12\n")
        result = self.launch()
        self.assertEqual(result.returncode, 12, result.stderr)
        self.assertNotIn("Building native CUDA library", result.stdout)
        self.assertFalse(self.log.exists())
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_missing_library_aborts_and_cleans_private_directory(self):
        (self.root / "simulator/build-rosie-cuda-75/minecraft_sim_training.so").unlink()
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("native training library was not produced", result.stdout)
        self.assertFalse(self.log.exists())
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_private_library_uses_tmpdir_when_slurm_scratch_is_unset(self):
        fallback = self.root / "fallback scratch"
        fallback.mkdir()
        result = self.launch(SLURM_TMPDIR="", TMPDIR=str(fallback))
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self.log.read_text().splitlines()
        self.assertEqual(Path(arguments[arguments.index("--library") + 1]).parent.parent, fallback)
        self.assertEqual(list(fallback.iterdir()), [])

    def test_custom_build_directory_uses_its_own_lock_and_snapshot(self):
        build = self.root / "custom build"
        build.mkdir()
        (build / "minecraft_sim_training.so").write_bytes(b"custom simulator")
        result = self.launch(SIM_BUILD_DIR=str(build), JOB_TEST_BUILD=str(build))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((build / ".build.lock").exists())
        self.assertEqual(self.snapshot.read_bytes(), b"custom simulator")
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_second_job_builds_while_first_trains_from_unchanged_snapshot(self):
        after = self.root / "first-library-after-rebuild"
        environment = self.job_environment(JOB_TEST_VERSIONED_BUILD="1", JOB_TEST_WAIT_FOR_INPUT="1",
                                           JOB_TEST_AFTER_SNAPSHOT=str(after))
        with subprocess.Popen(["bash", str(self.script)], env=environment, stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as first:
            try:
                assert first.stdout is not None
                for line in first.stdout:
                    if line.strip() == "TRAINING_READY":
                        break
                else:
                    self.fail("First job never reached training")
                second_snapshot = self.root / "second-library"
                second = self.launch(SLURM_JOB_ID="78", JOB_TEST_VERSIONED_BUILD="1",
                                     JOB_TEST_LOG=str(self.root / "second-arguments"),
                                     JOB_TEST_SNAPSHOT=str(second_snapshot))
                self.assertEqual(second.returncode, 0, second.stderr)
                self.assertIsNone(first.poll(), "First job must still be training when the second completes")
                self.assertEqual(self.snapshot.read_bytes(), b"77")
                self.assertEqual(second_snapshot.read_bytes(), b"78")
                _, errors = first.communicate("continue\n", timeout=10)
                self.assertEqual(first.returncode, 0, errors)
                self.assertEqual(after.read_bytes(), b"77", "A later build must not change the first job's library")
            finally:
                if first.poll() is None:
                    first.kill()
                    first.communicate(timeout=10)
        self.assertEqual(list(self.scratch.iterdir()), [])


@unittest.skipUnless(os.name != "nt" and shutil.which("bash"), "Requires a POSIX shell")
class SubmitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "project with spaces"
        (self.root / "rosie").mkdir(parents=True)
        source = Path(__file__).resolve().parents[2] / "rosie/submit.sh"
        self.script = self.root / "rosie/submit.sh"
        shutil.copyfile(source, self.script)
        plans = self.root / "simulator/setups/curricula"
        plans.mkdir(parents=True)
        definitions = {
            "arenas/test.json": {
                "format": "minecraft_sim_arena", "version": 1, "minecraft": "26.3", "name": "test-arena",
                "bounds": {"min": [-3, 0, -3], "size": [7, 4, 7]},
                "blocks": [{"from": [-3, 0, -3], "to": [3, 0, 3], "block": "minecraft:stone"}],
            },
            "environments/test.json": {
                "format": "minecraft_sim_environment", "version": 2, "minecraft": "26.3", "name": "test-duel",
                "arena": "../arenas/test.json",
                "players": {"shared": {"loadout": {"weapons": ["minecraft:iron_sword"]}},
                            "overrides": [{"position": [-1, 1, 0]}, {"position": [1, 1, 0]}]},
                "task": {"max_ticks": 100, "rewards": {"doing_damage": 1}},
            },
            "policies/test.json": {
                "format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3", "name": "test-policy",
                "ticks_per_action": 1, "requires": ["movement", "melee"],
                "observations": [{"feature": name} for name in
                                 ("target.distance_fraction", "target.yaw_error", "target.pitch_error")],
                "actions": [{"name": "movement", "values": ["neutral", "forward"]},
                            {"name": "attack", "values": [False, True]},
                            {"name": "yaw", "values": [-5, 0, 5]},
                            {"name": "pitch", "values": [-5, 0, 5]}],
            },
        }
        for relative, definition in definitions.items():
            path = self.root / "simulator/setups" / relative
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(definition))
        for name, budgets in (("axe-shield-fighting", [1500]),
                              ("shield-axe-curriculum", [100, 200, 200, 1500])):
            self.write_plan(plans / f"{name}.json", budgets)
        python = self.root / ".venv-rosie/bin/python"
        python.parent.mkdir(parents=True)
        python.write_text(f"#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} \"$@\"\n")
        python.chmod(0o700)
        self.tools = self.root / "tools"
        self.tools.mkdir()
        self.log = self.root / "submission.json"
        self.history = self.root / "submissions.jsonl"
        sbatch = self.tools / "sbatch"
        sbatch.write_text(f"#!{sys.executable}\n" + '''import json
import os
from pathlib import Path
import sys

submission = {
    "arguments": sys.argv[1:], "cwd": os.getcwd(),
    "inherited": [name for name in os.environ if name.startswith(("SIM_", "SBATCH_"))],
}
Path(os.environ["SUBMIT_TEST_LOG"]).write_text(json.dumps(submission))
history = Path(os.environ["SUBMIT_TEST_HISTORY"])
count = len(history.read_text().splitlines()) if history.exists() else 0
with history.open("a") as stream:
    stream.write(json.dumps(submission) + "\\n")
if os.environ.get("SUBMIT_TEST_FAIL") == "1" or os.environ.get("SUBMIT_TEST_FAIL_AT") == str(count + 1):
    raise SystemExit(9)
print(f"{12345 + count};test-cluster")
''')
        sbatch.chmod(0o700)

    def write_plan(self, path, budgets):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "format": "minecraft_sim_training_plan", "version": 2, "name": path.stem, "seed": 42,
            "policy": str(self.root / "simulator/setups/policies/test.json"),
            "environment": str(self.root / "simulator/setups/environments/test.json"),
            "opponents": {"target": {"type": "scripted", "controller": "stationary_target"}},
            "mixtures": {"targets": [{"id": "target", "weight": 1}]},
            "stages": [{"name": f"stage-{index}", "opponents": "targets", "advance": {"updates": budget}}
                       for index, budget in enumerate(budgets)],
            "learner": {"hidden_dim": 512, "epochs": 2, "minibatch_size": 16384, "lr": 0.0003},
        }))

    def launch(self, answers=None, dry_run=True, **settings):
        environment = {**os.environ, "PATH": str(self.tools) + os.pathsep + os.environ["PATH"],
                       "PYTHONPATH": str(Path(__file__).resolve().parents[2]) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                       "SUBMIT_TEST_LOG": str(self.log), "SUBMIT_TEST_HISTORY": str(self.history), **settings}
        arguments = ["bash", str(self.script)] + (["--dry-run"] if dry_run else [])
        return subprocess.run(arguments, input="\n".join(answers or [""] * 17) + "\n",
                              cwd=self.temporary.name, env=environment,
                              capture_output=True, text=True, timeout=10)

    def test_enter_defaults_use_h100_and_current_plan_budget(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in ("--partition=dgxh100", "--gres=gpu:h100:1", "MCSIM_CUDA_ARCHITECTURES=90",
                         "SIM_ENVS=8192", "SIM_UPDATES=2000", "SIM_ROLLOUT_STEPS=128",
                         "SIM_HIDDEN_DIM=512", "SIM_EPOCHS=2", "SIM_MINIBATCH_SIZE=16384",
                         "SIM_LR=0.0003", "Validated 1 plan(s); 0 failed", "Dry run: nothing submitted"):
            self.assertIn(expected, result.stdout)
        self.assertFalse(self.log.exists())

    def test_invalid_advancement_blocks_preview_and_submission(self):
        path = self.root / "simulator/setups/curricula/shield-axe-curriculum.json"
        value = json.loads(path.read_text())
        value["stages"][0]["advance"] = {"updates": 0}
        path.write_text(json.dumps(value))
        for dry_run in (True, False):
            with self.subTest(dry_run=dry_run):
                result = self.launch([""] * 16 + ["y"], dry_run=dry_run)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("validation failed; nothing submitted", result.stderr)
                self.assertIn("ERROR", result.stderr)
                self.assertNotIn("Training settings", result.stdout)
                self.assertNotIn("sbatch --parsable", result.stdout)
                self.assertFalse(self.log.exists())

    def test_missing_reference_blocks_submission(self):
        (self.root / "simulator/setups/policies/test.json").unlink()
        result = self.launch([""] * 16 + ["y"], dry_run=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("test.json", result.stderr)
        self.assertIn("validation failed; nothing submitted", result.stderr)
        self.assertFalse(self.log.exists())

    def test_malformed_json_blocks_submission(self):
        (self.root / "simulator/setups/curricula/shield-axe-curriculum.json").write_text("{broken")
        result = self.launch([""] * 16 + ["y"], dry_run=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("validation failed; nothing submitted", result.stderr)
        self.assertFalse(self.log.exists())

    def test_unselected_invalid_plan_does_not_block_valid_submission(self):
        (self.root / "simulator/setups/curricula/unselected.json").write_text("{broken")
        result = self.launch([""] * 16 + ["y"], dry_run=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Validated 1 plan(s); 0 failed", result.stdout)
        self.assertTrue(self.log.exists())

    def test_other_gpu_presets_and_fighting_plan(self):
        for gpu, partition, arch, worlds, rollout in (("v100", "dgx", 70, 2048, 128),
                                                      ("teaching", "teaching", 75, 256, 64)):
            with self.subTest(gpu=gpu):
                result = self.launch([gpu, "1"] + [""] * 14)
                self.assertEqual(result.returncode, 0, result.stderr)
                for expected in (f"--partition={partition}", f"MCSIM_CUDA_ARCHITECTURES={arch}",
                                 f"SIM_ENVS={worlds}", f"SIM_ROLLOUT_STEPS={rollout}",
                                 "SIM_UPDATES=1500", "axe-shield-fighting.json"):
                    self.assertIn(expected, result.stdout)
                self.assertFalse(self.log.exists())

    def test_nested_plan_learning_rate_sweep_preview(self):
        relative = "simulator/setups/curricula/experiments/comparison.json"
        destination = self.root / relative
        self.write_plan(destination, [300, 1200])
        answers = self.sweep_answers(10, [".0001", ".0003", ".0006"])
        answers[1] = "2"
        answers[2] = "comparison-lr"
        result = self.launch(answers)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("experiments/comparison.json", result.stdout)
        commands = [shlex.split(line) for line in result.stdout.splitlines()
                    if line.lstrip().startswith("sbatch --parsable")]
        self.assertEqual(len(commands), 3)
        for rate, command in zip((".0001", ".0003", ".0006"), commands):
            self.assertIn(f"--job-name=comparison-lr-{rate}", command)
            export = next(argument for argument in command if argument.startswith("--export="))
            settings = dict(item.split("=", 1) for item in export.removeprefix("--export=ALL,").split(","))
            self.assertEqual(settings["SIM_UPDATES"], "1500")
            self.assertEqual(settings["SIM_PLAN"], relative)
            self.assertEqual(settings["SIM_OUTPUT"], f"simulator/training/runs/comparison-lr-{rate}")
            self.assertEqual(settings["SIM_LR"], rate)
        self.assertFalse(self.log.exists())

    def test_recursive_buckets_show_distinct_names_and_select_nested_default(self):
        plans = self.root / "simulator/setups/curricula"
        first = plans / "v1"
        first.mkdir()
        for plan in plans.glob("*.json"):
            plan.rename(first / plan.name)
        nested = plans / "v2/aim practice/shield-axe-curriculum.json"
        nested.parent.mkdir(parents=True)
        value = json.loads((first / nested.name).read_text())
        value["stages"] = [{**value["stages"][0], "advance": {"updates": 25}}]
        nested.write_text(json.dumps(value))
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ("v1/axe-shield-fighting.json", "v1/shield-axe-curriculum.json",
                     "v2/aim practice/shield-axe-curriculum.json"):
            self.assertIn(name, result.stdout)
        self.assertIn("Plan number [3]", result.stderr)
        self.assertIn("SIM_UPDATES=25", result.stdout)
        self.assertFalse(self.log.exists())
        result = self.launch([""] * 16 + ["y"], dry_run=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        submission = json.loads(self.log.read_text())
        export = next(argument for argument in submission["arguments"] if argument.startswith("--export="))
        self.assertIn("SIM_PLAN=simulator/setups/curricula/v2/aim practice/shield-axe-curriculum.json,", export)

    def test_empty_plan_tree_fails_without_submission(self):
        plans = self.root / "simulator/setups/curricula"
        for plan in plans.glob("*.json"):
            plan.unlink()
        (plans / "empty/deeply/nested").mkdir(parents=True)
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No training plans found", result.stderr)
        self.assertFalse(self.log.exists())

    def test_invalid_choices_reprompt_and_overrides_are_applied(self):
        answers = ["bad-gpu", "v100", "99", "1", "custom", "2-00:00:00", "0", "1024",
                   "50", "32", "256", "3", "4096", "0", "1e-4", "2", "32G", "3", "2", "0", "77:78"]
        result = self.launch(answers)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Invalid value", result.stderr)
        self.assertIn("listed plans", result.stderr)
        self.assertIn("must not exceed", result.stderr)
        for expected in ("--job-name=custom", "SIM_ENVS=1024", "SIM_UPDATES=50", "SIM_THREADS=2",
                         "SIM_LR=1e-4", "SIM_REPLAY_EVERY=0", "--dependency=afterany:77:78"):
            self.assertIn(expected, result.stdout)

    def test_confirmation_submits_once_and_clears_stale_settings(self):
        result = self.launch([""] * 16 + ["y"], dry_run=False,
                             SIM_RESUME="old.pt", SIM_OUTPUT="old-run", SIM_BUILD_DIR="old-build",
                             SIM_ENVIRONMENT="old.json", SBATCH_PARTITION="wrong")
        self.assertEqual(result.returncode, 0, result.stderr)
        submission = json.loads(self.log.read_text())
        self.assertEqual(submission["cwd"], str(self.root))
        self.assertEqual(submission["inherited"], [])
        self.assertEqual(submission["arguments"][-1], "rosie/train.sbatch")
        self.assertIn("--partition=dgxh100", submission["arguments"])
        self.assertTrue((self.root / "rosie/logs").is_dir())
        self.assertIn("Submitted job 12345", result.stdout)
        self.assertIn("runs/shield-axe-curriculum", result.stdout)

    def test_custom_name_sets_job_and_dashboard_run_directory(self):
        result = self.launch(["", "", "lr-experiment"] + [""] * 13 + ["y"], dry_run=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = json.loads(self.log.read_text())["arguments"]
        self.assertIn("--job-name=lr-experiment", arguments)
        export = next(argument for argument in arguments if argument.startswith("--export="))
        self.assertIn("SIM_OUTPUT=simulator/training/runs/lr-experiment,", export)
        self.assertIn("Run: simulator/training/runs/lr-experiment", result.stdout)

    def test_enter_at_confirmation_cancels(self):
        result = self.launch(dry_run=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Cancelled", result.stdout)
        self.assertFalse(self.log.exists())

    def sweep_answers(self, index, values):
        answers = [""] * 16
        answers[2] = "sweep-test"
        answers[index:index + 1] = ["multiple", *values, ""]
        return answers + ["y"]

    def test_learning_rate_sweep_submits_only_one_changed_setting(self):
        rates = [".0001", ".0003", ".0006"]
        answers = self.sweep_answers(10, rates)
        answers[-2] = "77:78"
        result = self.launch(answers, dry_run=False, SIM_RESUME="old.pt", SBATCH_PARTITION="wrong")
        self.assertEqual(result.returncode, 0, result.stderr)
        submissions = [json.loads(line) for line in self.history.read_text().splitlines()]
        self.assertEqual(len(submissions), 3)
        common = None
        for rate, submission in zip(rates, submissions):
            arguments = submission["arguments"]
            self.assertEqual(submission["inherited"], [])
            self.assertIn(f"--job-name=sweep-test-{rate}", arguments)
            export = next(argument for argument in arguments if argument.startswith("--export="))
            settings = dict(item.split("=", 1) for item in export.removeprefix("--export=ALL,").split(","))
            self.assertEqual(settings.pop("SIM_LR"), rate)
            self.assertEqual(settings.pop("SIM_OUTPUT"), f"simulator/training/runs/sweep-test-{rate}")
            if common is None:
                common = settings
            self.assertEqual(settings, common)
            self.assertIn("--dependency=afterany:77:78", arguments)
            self.assertIn(f"Run: simulator/training/runs/sweep-test-{rate}", result.stdout)

    def test_sweep_dry_run_previews_every_job_without_submitting(self):
        result = self.launch(self.sweep_answers(10, [".0001", ".0003", ".0006"]))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("sbatch --parsable"), 3)
        self.assertIn("3 jobs", result.stdout)
        self.assertIn("concurrently", result.stdout)
        self.assertNotIn("--dependency=", result.stdout)
        self.assertNotIn("previous-job-id", result.stdout)
        self.assertFalse(self.log.exists())

    def test_sweep_without_explicit_dependencies_submits_independent_jobs(self):
        result = self.launch(self.sweep_answers(10, [".0001", ".0003", ".0006"]), dry_run=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        submissions = [json.loads(line) for line in self.history.read_text().splitlines()]
        self.assertEqual(len(submissions), 3)
        for submission in submissions:
            self.assertFalse(any(argument.startswith("--dependency=") for argument in submission["arguments"]))

    def test_second_multiple_reprompts_for_single_value(self):
        answers = self.sweep_answers(4, ["1024", "2048"])
        answers[13:14] = ["multiple", "0.0001"]
        result = self.launch(answers, dry_run=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Only one setting can use multiple", result.stderr)
        submissions = [json.loads(line) for line in self.history.read_text().splitlines()]
        self.assertEqual(len(submissions), 2)
        for submission in submissions:
            self.assertIn("SIM_LR=0.0001,", " ".join(submission["arguments"]))

    def test_sweep_rejects_invalid_empty_and_duplicate_values(self):
        result = self.launch(self.sweep_answers(10, ["", "0", "bad", ".0001", ".0001", ".0003"]), dry_run=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("at least one value", result.stderr)
        self.assertIn("Invalid value", result.stderr)
        self.assertIn("already included", result.stderr)
        self.assertEqual(len(self.history.read_text().splitlines()), 2)

    def test_cpu_sweep_keeps_threads_valid_and_fixed(self):
        result = self.launch(self.sweep_answers(11, ["8", "2"]), dry_run=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        submissions = [json.loads(line) for line in self.history.read_text().splitlines()]
        self.assertEqual(len(submissions), 2)
        for cores, submission in zip((8, 2), submissions):
            self.assertIn(f"--cpus-per-gpu={cores}", submission["arguments"])
            self.assertIn("SIM_THREADS=2,", " ".join(submission["arguments"]))

    def test_thread_sweep_rejects_values_above_allocated_cores(self):
        result = self.launch(self.sweep_answers(13, ["2", "9", "4"]), dry_run=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("must not exceed", result.stderr)
        self.assertEqual(len(self.history.read_text().splitlines()), 2)

    def test_existing_run_name_aborts_entire_sweep_before_submission(self):
        directory = self.root / "simulator/training/runs/sweep-test-.0003"
        directory.mkdir(parents=True)
        result = self.launch(self.sweep_answers(10, [".0001", ".0003"]), dry_run=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already exists", result.stderr)
        self.assertFalse(self.log.exists())

    def test_sweep_submission_failure_stops_remaining_jobs(self):
        result = self.launch(self.sweep_answers(10, [".0001", ".0003", ".0006"]), dry_run=False,
                             SUBMIT_TEST_FAIL_AT="2")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout.count("Submitted job"), 1)
        self.assertEqual(len(self.history.read_text().splitlines()), 2)
        self.assertIn("no further jobs submitted", result.stderr)

    def test_canceled_sweep_submits_nothing(self):
        answers = self.sweep_answers(10, [".0001", ".0003"])
        answers[-1] = ""
        result = self.launch(answers, dry_run=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Cancelled", result.stdout)
        self.assertFalse(self.log.exists())

    def test_closed_sweep_input_submits_nothing(self):
        result = self.launch([""] * 10 + ["multiple", ".0001"], dry_run=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Input closed", result.stderr)
        self.assertFalse(self.log.exists())

    def test_wall_time_sweep_uses_filename_safe_suffixes(self):
        result = self.launch(self.sweep_answers(3, ["00:30:00", "01:00:00"]), dry_run=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        submissions = [json.loads(line) for line in self.history.read_text().splitlines()]
        self.assertEqual(len(submissions), 2)
        for time, submission in zip(("00:30:00", "01:00:00"), submissions):
            self.assertIn(f"--time={time}", submission["arguments"])
            self.assertIn(f"--job-name=sweep-test-{time.replace(':', '-')}", submission["arguments"])

    def test_sweep_name_length_includes_value_suffix(self):
        answers = self.sweep_answers(10, [".0001", ".0003"])
        answers[2] = "a" * 118
        result = self.launch(answers, dry_run=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("120 filename-safe characters", result.stderr)
        self.assertFalse(self.log.exists())

    def test_closed_input_does_not_submit(self):
        result = self.launch(["h100"], dry_run=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Input closed", result.stderr)
        self.assertFalse(self.log.exists())

    def test_submission_failure_is_not_reported_as_success(self):
        result = self.launch([""] * 16 + ["y"], dry_run=False, SUBMIT_TEST_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Submitted job", result.stdout)


@unittest.skipUnless(os.name != "nt" and shutil.which("bash"), "Requires a POSIX shell")
class DashboardDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.tools = Path(self.temporary.name)
        script = Path(__file__).resolve().parents[2] / "rosie/dashboard-connect.sh"
        self.discovery = script.read_text().split("<<'REMOTE'\n", 1)[1].split("\nREMOTE\n", 1)[0]
        self.project = self.tools / "project with spaces"
        dashboard = self.project / "simulator/training/dashboard.py"
        dashboard.parent.mkdir(parents=True)
        dashboard.write_text('"""Test dashboard marker."""\n', encoding="utf-8")
        self.calls = self.tools / "slurm-calls"
        self.jobs = {
            "77": {"name": "shield-axe-curriculum", "state": "RUNNING", "work_dir": "/project",
                   "command": "/project/rosie/train.sbatch", "node": "test-node"},
            "78": {"name": "axe-shield-fighting", "state": "RUNNING", "work_dir": "/project",
                   "command": "/project/rosie/train.sbatch", "node": "test-node"},
        }
        mock = f"#!{sys.executable}\n" + '''import json
import os
from pathlib import Path
import sys

with Path(os.environ["DASHBOARD_TEST_CALLS"]).open("a") as calls:
    calls.write(Path(sys.argv[0]).name + "\\n")
jobs = json.loads(os.environ["DASHBOARD_TEST_JOBS"])
arguments = sys.argv[1:]
if Path(sys.argv[0]).name == "squeue":
    assert "--states=RUNNING" in arguments
    selected = next((value.split("=", 1)[1] for value in arguments if value.startswith("--jobs=")), None)
    for job_id, job in jobs.items():
        if job["state"] != "RUNNING" or (selected is not None and job_id != selected):
            continue
        if "--format=%i" in arguments:
            print(job_id)
        else:
            print(f'{job_id}|{job["name"]}|{job["node"]}')
elif arguments[:2] == ["show", "job"]:
    job = jobs[arguments[-1]]
    print(f'JobId={arguments[-1]} Command={job["command"]} WorkDir={job["work_dir"]}')
elif arguments[:2] == ["show", "hostnames"]:
    print(arguments[-1])
else:
    raise SystemExit(2)
'''
        for name in ("squeue", "scontrol"):
            executable = self.tools / name
            executable.write_text(mock)
            executable.chmod(0o700)

    def discover(self, job_id="", project_dir=None, **settings):
        environment = {**os.environ, "PATH": str(self.tools) + os.pathsep + os.environ["PATH"],
                       "DASHBOARD_TEST_JOBS": json.dumps(self.jobs), "USER": "test-user",
                       "DASHBOARD_TEST_CALLS": str(self.calls), **settings}
        return subprocess.run(["bash", "-s", "--", job_id,
                               str(self.project) if project_dir is None else project_dir], input=self.discovery,
                              env=environment, capture_output=True, text=True, timeout=10)

    def test_default_uses_management_node_without_querying_jobs(self):
        self.jobs["76"] = {**self.jobs["77"], "command": "/other/job.sbatch"}
        result = self.discover()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().split("\t"), ["-", "-", str(self.project)])
        self.assertFalse(self.calls.exists())

    def test_no_jobs_are_required(self):
        self.jobs.clear()
        result = self.discover()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().split("\t"), ["-", "-", str(self.project)])
        self.assertFalse(self.calls.exists())

    def test_default_remote_checkout(self):
        home = self.tools / "home"
        checkout = home / "minecraftlol/Cambium/gpu-sim-minecraft"
        checkout.parent.mkdir(parents=True)
        checkout.symlink_to(self.project, target_is_directory=True)
        result = self.discover(project_dir="", HOME=str(home))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(result.stdout.strip().split("\t")[-1]).resolve(), self.project)

    def test_home_relative_checkout(self):
        result = self.discover(project_dir="~/project with spaces", HOME=str(self.tools))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().split("\t"), ["-", "-", str(self.project)])

    def test_missing_checkout_reports_override(self):
        result = self.discover(project_dir=str(self.tools / "missing"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Dashboard not found", result.stderr)
        self.assertIn("ROSIE_PROJECT_DIR", result.stderr)
        self.assertFalse(self.calls.exists())

    def test_explicit_custom_named_job(self):
        result = self.discover("78")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().split("\t"), ["78", "test-node", "/project"])

    def test_default_name_still_works(self):
        self.jobs["77"]["name"] = "cambium-sim"
        result = self.discover("77")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_pending_jobs_are_not_selected(self):
        for job in self.jobs.values():
            job["state"] = "PENDING"
        self.assertEqual(self.discover().returncode, 0)
        result = self.discover("77")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("running", result.stderr)

    def test_non_training_job_is_rejected_even_with_default_name(self):
        self.jobs = {"77": {**self.jobs["77"], "name": "cambium-sim", "command": "/other/job.sbatch"}}
        self.assertEqual(self.discover().returncode, 0)
        result = self.discover("77")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rosie/train.sbatch", result.stderr)

    def test_other_job_directories_do_not_affect_management_hosting(self):
        self.jobs["78"]["work_dir"] = "/other-project"
        result = self.discover()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().split("\t"), ["-", "-", str(self.project)])
        self.assertFalse(self.calls.exists())
        explicit = self.discover("78")
        self.assertEqual(explicit.returncode, 0, explicit.stderr)
        self.assertIn("/other-project", explicit.stdout)


class RosieTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.game = self.root / "Minecraft with spaces"
        self.remote = self.root / "remote runs"
        self.run = self.remote / "rosie-123"
        self.run.mkdir(parents=True)
        (self.run / "latest.pt").write_bytes(b"fake checkpoint\x00\xff")
        (self.run / "config.json").write_text('{"environment":"test"}', encoding="utf-8")
        (self.run / "replays").mkdir()
        (self.run / "replays" / "sample.jsonl").write_bytes(b'{"frame":0}\n')
        configure(self.game, "rosie-test", "student@ad.msoe.edu", str(self.remote))
        self.rosie = Rosie(self.game)

        def local_command(action, **values):
            request = {"action": action, "runs_directory": str(self.remote), **values}
            encoded = base64.b64encode(json.dumps(request).encode()).decode()
            return [sys.executable, "-c", REMOTE_SCRIPT, encoded]

        self.local_command = local_command
        self.command_patch = patch.object(self.rosie, "ssh_command", side_effect=local_command)
        self.command_patch.start()
        self.addCleanup(self.command_patch.stop)

    def test_list_select_and_replay(self):
        self.assertEqual(self.rosie.select("latest", checkpoint=True), "rosie-123")
        manifest = self.rosie.manifest("rosie-123")
        destination = self.rosie.replay("rosie-123", manifest)
        self.assertEqual(destination.parent, self.game / "sim-replays")
        self.assertEqual(destination.read_bytes(), b'{"frame":0}\n')
        with patch.object(self.rosie, "ssh_command", side_effect=AssertionError("cached transfer")):
            self.assertEqual(self.rosie.replay("rosie-123", manifest), destination)

    def test_checkpoint_refresh_and_offline(self):
        manifest = self.rosie.manifest("rosie-123")
        checkpoint = self.rosie.checkpoint("rosie-123", manifest)
        self.assertEqual(checkpoint.read_bytes(), (self.run / "latest.pt").read_bytes())
        (self.run / "latest.pt").write_bytes(b"new weights")
        manifest = self.rosie.manifest("rosie-123")
        self.rosie.checkpoint("rosie-123", manifest)
        offline = Rosie(self.game, offline=True)
        self.assertEqual(offline.select("latest", checkpoint=True), "rosie-123")
        self.assertEqual(offline.checkpoint("rosie-123", offline.manifest("rosie-123")).read_bytes(), b"new weights")
        with self.assertRaisesRegex(ValueError, "not cached"):
            offline.replay("rosie-123", manifest)

    def test_named_snapshot_sync_and_persistent_local_library(self):
        import torch
        from backend.fast_rl_model import ActorCritic
        from simulator.training.contract import from_legacy
        from simulator.training.environment import ROOT, SCHEMA

        environment = json.loads((ROOT / "simulator/environments/duel.json").read_text())
        saved = {"format": "minecraft_sim_ppo", "version": 1, "update": 100,
                 "metadata": {"schema": SCHEMA, "hidden_dim": 8, "max_ticks": 300,
                              "players_per_environment": 2, "learner": "both_shared_policy", "environment": environment},
                 "model": ActorCritic(22, 8, 9).state_dict()}
        torch.save(saved, self.run / "latest.pt")
        (self.run / "snapshots").mkdir()
        torch.save({**saved, "update": 50}, self.run / "snapshots/stage-50.pt")
        (self.run / "config.json").write_text(json.dumps({"experiment": {"policy": from_legacy(SCHEMA),
            "stages": [{"name": "practice", "scenario": {"arenas": [{"environment": environment},
                                                                   {"environment": environment}]}}]}}))
        manifest = self.rosie.manifest("rosie-123")
        self.assertEqual(manifest["snapshots"][0]["name"], "stage-50.pt")
        self.rosie.checkpoint("rosie-123", manifest)
        self.rosie.sync_snapshots("rosie-123", manifest)
        library = self.rosie.index_run("rosie-123")
        self.assertEqual([entry["update"] for entry in library["policies"]], [100, 50])
        self.assertEqual(len(library["environments"]), 4)
        self.assertEqual([entry["name"] for entry in library["environments"][2:]],
                 ["rosie-123 / practice / arena 1", "rosie-123 / practice / arena 2"])
        library["policies"][0]["name"] = "My reliable fighter"
        (self.rosie.cache / "library.json").write_text(json.dumps(library))
        with patch.object(self.rosie, "ssh_command", side_effect=AssertionError("offline library contacted Rosie")):
            self.assertEqual(self.rosie.index_run("rosie-123")["policies"][0]["name"], "My reliable fighter")
            self.rosie.sync_snapshots("rosie-123", manifest)
        for invalid in ("snapshots/../outside.pt", "snapshots/a/b.pt", "snapshots/file.json"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.rosie.fetch("rosie-123", invalid, self.game / "outside", {})

    def test_truncated_transfer_does_not_replace_previous_file(self):
        manifest = self.rosie.manifest("rosie-123")
        checkpoint = self.rosie.checkpoint("rosie-123", manifest)
        before = checkpoint.read_bytes()
        broken = 'import sys;sys.stdout.buffer.write(b\'{"size":20,"mtime_ns":1,"sha256":"bad"}\\nshort\')'
        with patch.object(self.rosie, "ssh_command", return_value=[sys.executable, "-c", broken]):
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                self.rosie.fetch("rosie-123", "latest.pt", checkpoint, {"size":20,"mtime_ns":1})
        self.assertEqual(checkpoint.read_bytes(), before)
        self.assertEqual(list(checkpoint.parent.glob("*.part")), [])

    def test_hash_mismatch_rejected(self):
        broken = 'import sys;sys.stdout.buffer.write(b\'{"size":3,"mtime_ns":1,"sha256":"bad"}\\nabc\')'
        destination = self.game / "bad.pt"
        with patch.object(self.rosie, "ssh_command", return_value=[sys.executable, "-c", broken]):
            with self.assertRaisesRegex(ValueError, "integrity"):
                self.rosie.fetch("rosie-123", "latest.pt", destination, {"size":3,"mtime_ns":1})
        self.assertFalse(destination.exists())

    def test_names_rejected(self):
        for name in ("../outside", "--option", "a/b", "..", "run;echo bad", "x\n"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                valid_name(name)

    def test_remote_symlink_escape_rejected(self):
        outside = self.root / "outside.pt"
        outside.write_bytes(b"secret")
        (self.run / "latest.pt").unlink()
        try:
            (self.run / "latest.pt").symlink_to(outside)
        except (OSError, NotImplementedError) as error:
            self.skipTest("Symlinks unavailable: " + str(error))
        with self.assertRaisesRegex(RuntimeError, "SSH failed"):
            self.rosie.request("manifest", run="rosie-123")

    def test_remote_error_leaves_no_partial_download(self):
        with patch.object(self.rosie, "ssh_command", return_value=[sys.executable, "-c", "raise SystemExit('denied')"]):
            with self.assertRaisesRegex(RuntimeError, "denied"):
                self.rosie.fetch("rosie-123", "latest.pt", self.game / "missing.pt", {})
        self.assertEqual(list(self.game.glob("*.part")), [])

    def test_ssh_request_is_encoded_and_batch_only(self):
        rosie = Rosie(self.game)
        command = rosie.ssh_command("manifest", run="rosie-123")
        self.assertIn("BatchMode=yes", command)
        self.assertEqual(command[-2], "rosie-test")
        self.assertNotIn(str(self.remote), command[-1])
        self.assertIn("student@ad.msoe.edu", command)

    def test_terminal_session_reuses_reader_and_requires_token(self):
        configuration = {"runs_directory": str(self.remote), "token": "test-session-token", "port": 0,
                         "reader": base64.b64encode(REMOTE_SCRIPT.encode()).decode()}
        encoded = base64.b64encode(json.dumps(configuration).encode()).decode()
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen([sys.executable, "-u", "-c", SESSION_SCRIPT],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors)
            try:
                process.stdin.write(encoded.encode() + b"\n")
                process.stdin.flush()
                ready = json.loads(process.stdout.readline())
                session = {"port": ready["port"], "token": configuration["token"]}
                with session_response(session, {"action": "runs"}) as response:
                    self.assertEqual(json.load(response)[0]["name"], "rosie-123")
                with session_response(session, {"action": "manifest", "run": "rosie-123"}) as response:
                    self.assertIn("latest.pt", json.load(response)["files"])
                with session_response(session, {"action": "fetch", "run": "rosie-123", "file": "latest.pt"}) as response:
                    header = json.loads(response.readline())
                    content = response.read()
                    self.assertEqual(header["size"], len(content))
                    self.assertEqual(content, (self.run / "latest.pt").read_bytes())
                with self.assertRaisesRegex(RuntimeError, "Unauthorized"):
                    with session_response({**session, "token": "incorrect"}):
                        self.fail("Unauthenticated request succeeded")
                with self.assertRaisesRegex(RuntimeError, "Unsupported"):
                    with session_response(session, {"action": "shell"}):
                        self.fail("Unexpected operation succeeded")
                with session_response(session, {"action": "runs", "runs_directory": "/wrong-root"}) as response:
                    self.assertEqual(json.load(response)[0]["name"], "rosie-123")
                session["connection_id"] = connection_id(self.rosie.configuration)
                session_path(self.game).write_text(json.dumps(session), encoding="utf-8")
                connected = Rosie(self.game)
                with patch.object(connected, "ssh_command", side_effect=AssertionError("new SSH connection")):
                    self.assertEqual(connected.select("latest"), "rosie-123")
                    manifest = connected.manifest("rosie-123")
                    checkpoint = connected.checkpoint("rosie-123", manifest)
                    self.assertEqual(checkpoint.read_bytes(), (self.run / "latest.pt").read_bytes())
                    self.assertEqual(connected.replay("rosie-123", manifest).read_bytes(), b'{"frame":0}\n')
                self.assertIsNone(Rosie(self.game, offline=True).session)
            finally:
                process.stdin.close()
                try:
                    process.wait(timeout=5)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                    process.stdout.close()
            self.assertEqual(process.returncode, 0)

    def test_closed_terminal_connection_does_not_silently_fall_back_to_ssh(self):
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            descriptor = {"port": reserved.getsockname()[1], "token": "expired",
                          "connection_id": connection_id(self.rosie.configuration)}
            session_path(self.game).write_text(json.dumps(descriptor), encoding="utf-8")
            connected = Rosie(self.game)
            with patch.object(connected, "ssh_command", side_effect=AssertionError("password fallback")):
                with self.assertRaisesRegex(RuntimeError, "-Connect"):
                    connected.runs()

    def test_terminal_connect_uses_interactive_ssh_and_cleans_descriptor(self):
        observed = []
        def wait(timeout=None):
            observed.append(json.loads(session_path(self.game).read_text(encoding="utf-8")))
            return 0

        with patch("sys.stdin.isatty", return_value=True), patch("simulator.training.rosie.secrets.randbelow", return_value=10000), \
                patch("simulator.training.rosie.session_response", return_value=contextlib.nullcontext()), \
                patch("simulator.training.rosie.subprocess.Popen") as spawn, contextlib.redirect_stdout(io.StringIO()):
            spawn.return_value.stdout.readline.return_value = b'{"ready":true,"port":30000}\n'
            spawn.return_value.wait.side_effect = wait
            connect_terminal(self.game)
        command = spawn.call_args.args[0]
        self.assertIn("BatchMode=no", command)
        self.assertIn("ExitOnForwardFailure=yes", command)
        self.assertIn("-L", command)
        self.assertNotIn("stderr", spawn.call_args.kwargs)
        self.assertTrue(observed)
        self.assertEqual(set(observed[0]), {"port", "token", "connection_id"})
        self.assertNotIn(observed[0]["token"], " ".join(command))
        payload = spawn.return_value.stdin.write.call_args.args[0]
        self.assertEqual(json.loads(base64.b64decode(payload))["token"], observed[0]["token"])
        self.assertFalse(session_path(self.game).exists())

    def test_terminal_connect_rejects_nonterminal_invocation(self):
        with patch("sys.stdin.isatty", return_value=False):
            with self.assertRaisesRegex(ValueError, "not in Minecraft"):
                connect_terminal(self.game)

    def test_setup_keeps_virtual_environment_interpreter_path(self):
        interpreter = self.root / "venv" / "bin" / "python"
        interpreter.parent.mkdir(parents=True)
        try:
            interpreter.symlink_to(sys.executable)
        except (OSError, NotImplementedError) as error:
            self.skipTest("Symlinks unavailable: " + str(error))
        with patch("sys.executable", str(interpreter)):
            path = configure(self.game, "rosie-test", "student", str(self.remote))
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["python"], str(interpreter))

    def test_ntfs_timestamp_precision_reuses_cache(self):
        timestamp = 1700000000123456789
        os.utime(self.run / "latest.pt", ns=(timestamp, timestamp))
        manifest = self.rosie.manifest("rosie-123")
        original_utime = os.utime

        def ntfs_utime(path, *, ns):
            original_utime(path, ns=tuple(value // 100 * 100 for value in ns))

        with patch("simulator.training.rosie.os.utime", side_effect=ntfs_utime):
            checkpoint = self.rosie.checkpoint("rosie-123", manifest)
        with patch.object(self.rosie, "ssh_command", side_effect=AssertionError("cached transfer")):
            self.assertEqual(self.rosie.checkpoint("rosie-123", manifest), checkpoint)

    def test_encoded_remote_command_executes_with_paths_containing_spaces(self):
        command = Rosie(self.game).ssh_command("runs")
        remote = shlex.split(command[-1])
        result = subprocess.run([sys.executable, *remote[1:]], capture_output=True, check=True)
        self.assertEqual(json.loads(result.stdout)[0]["name"], "rosie-123")

    def test_cli_sync_and_offline_replay(self):
        output = io.StringIO()
        with patch.object(Rosie, "ssh_command", side_effect=self.local_command), \
            patch.object(Rosie, "index_run"), contextlib.redirect_stdout(output):
            result = main(["sync", "--minecraft-directory", str(self.game), "--run", "rosie-123", "--json"])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue())["replays"], 1)
        output = io.StringIO()
        with patch.object(Rosie, "ssh_command", side_effect=AssertionError("offline SSH")), contextlib.redirect_stdout(output):
            result = main(["replay", "--minecraft-directory", str(self.game), "--offline", "--json"])
        self.assertEqual(result, 0)
        self.assertTrue(Path(json.loads(output.getvalue())["path"]).is_file())

    def test_cli_emits_machine_readable_errors(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["replay", "--minecraft-directory", str(self.root / "missing"), "--json"])
        self.assertEqual(result, 1)
        self.assertEqual(json.loads(output.getvalue())["event"], "error")

    def test_replays_only_run_and_empty_run_selection(self):
        (self.run / "latest.pt").unlink()
        self.assertEqual(self.rosie.select("latest"), "rosie-123")
        with self.assertRaisesRegex(ValueError, "No saved"):
            self.rosie.select("latest", checkpoint=True)


if __name__ == "__main__":
    unittest.main()