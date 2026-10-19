import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from simulator.training.dashboard import RunStore, make_server


class DashboardChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.run = self.root / "runs" / "first"
        self.run.mkdir(parents=True)
        (self.run / "config.json").write_text('{"options":{"device":"cpu","envs":4}}', encoding="utf-8")
        self.metrics = self.run / "metrics.jsonl"
        self.metrics.write_bytes(b'{"update":1,"reward_mean":0.25,"samples":16}\n')
        self.store = RunStore(self.root / "runs")

    def test_discovery_custom_output_and_missing_root(self):
        self.assertEqual(len(self.store.list_runs()), 1)
        identity = self.store.list_runs()[0]["id"]
        self.assertEqual(self.store.read(identity)["config"]["options"]["envs"], 4)
        self.assertEqual(RunStore(self.root / "missing").list_runs(), [])
        custom = RunStore(self.root / "missing", [self.run, self.run])
        self.assertEqual(custom.list_runs()[0]["id"], identity)
        self.assertEqual(len(custom.list_runs()), 1)
        with self.assertRaises(FileNotFoundError):
            self.store.read("../../config.json")

    def test_partial_tail_and_live_append(self):
        identity = self.store.list_runs()[0]["id"]
        with self.metrics.open("ab") as output:
            output.write(b'{"update":2,"reward_mean":')
        self.assertEqual(len(self.store.read(identity)["metrics"]), 1)
        with self.metrics.open("ab") as output:
            output.write(b'0.5}\n{"update":3}\n')
        self.assertEqual([row["update"] for row in self.store.read(identity)["metrics"]], [1, 2, 3])
        self.metrics.write_bytes(b'{"update":1,"reward_mean":-1}\n')
        self.assertEqual(self.store.read(identity)["metrics"][0]["reward_mean"], -1)

    def test_invalid_records_and_config_only_run(self):
        identity = self.store.list_runs()[0]["id"]
        with self.metrics.open("ab") as output:
            output.write(b'bad\n[]\n{"update":true}\n{"update":2,"loss":NaN}\n{"update":3}\n')
        result = self.store.read(identity)
        self.assertEqual([row["update"] for row in result["metrics"]], [1, 3])
        self.assertIn("Skipped 4", result["warnings"][0])
        (self.run / "config.json").write_text('{"options":', encoding="utf-8")
        self.assertIsNone(self.store.read(identity)["config"])
        self.metrics.unlink()
        self.assertEqual(self.store.read(identity)["metrics"], [])

    def test_resume_replaces_updates_after_an_older_checkpoint(self):
        with self.metrics.open("ab") as output:
            output.write(b'{"update":2,"samples":32}\n{"update":3,"samples":48}\n')
            output.write(b'{"update":2,"samples":64}\n')
        identity = self.store.list_runs()[0]["id"]
        rows = self.store.read(identity)["metrics"]
        self.assertEqual([row["update"] for row in rows], [1, 2])
        self.assertEqual(rows[-1]["samples"], 64)

    def test_reward_history_and_logged_contributions(self):
        self.metrics.write_text(
            '{"update":1,"samples":16,"reward_mean":0.25}\n'
            '{"update":2,"samples":48,"reward_mean":-0.5}\n'
            '{"update":3,"samples":64,"reward_mean":0.5,"reward_total":7.5,'
            '"reward_types":{"doing_damage":10,"taking_damage":-2.5}}\n', encoding="utf-8")
        identity = self.store.list_runs()[0]["id"]
        rows = self.store.read(identity)["metrics"]
        self.assertEqual([row["reward_total"] for row in rows], [4, -16, 7.5])
        self.assertEqual(rows[-1]["reward_types"], {"doing_damage": 10, "taking_damage": -2.5})
        self.metrics.write_text('{"update":10,"samples":160,"reward_mean":1}\n'
                                '{"update":12,"samples":192,"reward_mean":1}\n', encoding="utf-8")
        self.assertTrue(all("reward_total" not in row for row in self.store.read(identity)["metrics"]))
        self.metrics.write_text('{"update":1,"reward_types":{"death":NaN}}\n', encoding="utf-8")
        self.assertEqual(self.store.read(identity)["metrics"], [])

    def test_http_read_only_routes(self):
        with make_server(self.store, 0) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                with urlopen(base + "/api/runs", timeout=3) as response:
                    identity = json.load(response)["runs"][0]["id"]
                    self.assertEqual(response.headers["Cache-Control"], "no-store")
                with urlopen(base + f"/api/runs/{identity}/metrics", timeout=3) as response:
                    self.assertEqual(json.load(response)["metrics"][0]["samples"], 16)
                for route, content_type in (("/", "text/html"), ("/dashboard.css", "text/css"),
                                            ("/dashboard.js", "text/javascript"),
                                            ("/chart.umd.min.js", "text/javascript")):
                    with urlopen(base + route, timeout=3) as response:
                        self.assertEqual(response.headers.get_content_type(), content_type)
                        self.assertGreater(len(response.read()), 100)
                for request, status in ((base + "/api/runs/unknown/metrics", 404),
                                        (base + "/../config.json", 404),
                                        (Request(base + "/api/runs", b"{}"), 405)):
                    with self.assertRaises(HTTPError) as failure:
                        urlopen(request, timeout=3)
                    self.assertEqual(failure.exception.code, status)
                    failure.exception.close()
            finally:
                server.shutdown()
                worker.join(timeout=3)


if __name__ == "__main__":
    unittest.main()