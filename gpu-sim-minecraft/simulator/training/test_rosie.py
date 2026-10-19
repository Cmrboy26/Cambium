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
        (self.root / "simulator/build-rosie-cuda-75/minecraft_sim_training.so").touch()
        (self.root / ".venv-rosie/bin").mkdir(parents=True)
        self.log = self.root / "arguments"
        for name in ("module", "nvcc", "nvidia-smi"):
            self.executable(self.tools / name, "#!/usr/bin/env bash\nexit 0\n")
        self.executable(self.tools / "cmake",
                        '#!/usr/bin/env bash\nif [[ "$1" == "--version" ]]; then echo "cmake version 3.25.0"; fi\n')
        self.executable(self.root / ".venv-rosie/bin/python",
                        '#!/usr/bin/env bash\nif [[ "$1" == "-B" ]]; then printf "%s\\n" "$@" > "$JOB_TEST_LOG"; fi\n')
        self.script = Path(__file__).resolve().parents[2] / "rosie/train.sbatch"

    def executable(self, path, text):
        path.write_text(text)
        path.chmod(0o700)

    def launch(self, **settings):
        environment = {key: value for key, value in os.environ.items() if not key.startswith("SIM_")}
        environment.update(PATH=str(self.tools) + os.pathsep + os.environ["PATH"],
                           SLURM_SUBMIT_DIR=str(self.root), SLURM_JOB_ID="77", JOB_TEST_LOG=str(self.log),
                           MCSIM_CUDA_ARCHITECTURES="75", **settings)
        return subprocess.run(["bash", str(self.script)], env=environment, capture_output=True,
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
        with patch.object(Rosie, "ssh_command", side_effect=self.local_command), contextlib.redirect_stdout(output):
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