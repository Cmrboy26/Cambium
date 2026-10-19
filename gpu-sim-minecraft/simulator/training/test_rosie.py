"""SSH artifact protocol tests using local subprocesses instead of a cluster."""

import base64
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from simulator.training.rosie import REMOTE_SCRIPT, Rosie, configure, main, valid_name


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