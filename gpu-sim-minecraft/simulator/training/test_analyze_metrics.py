import json
from pathlib import Path
import tempfile
import unittest

from simulator.training.analyze_metrics import read_run, summarize_run


class AnalyzeMetricsChecks(unittest.TestCase):
    def test_summarizes_recent_rewards_and_timeout_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run-one"
            run.mkdir()
            (run / "config.json").write_text(json.dumps({
                "options": {"device": "cpu", "envs": 2, "updates": 20},
                "metadata": {"environment": {"name": "test-duel"},
                             "reward_weights": {"won_duel": 100, "death": -40}},
            }), encoding="utf-8")
            rows = [
                {"update": 1, "samples": 8, "reward_mean": 0.1, "terminated_duels": 1,
                 "timeouts": 1, "entropy": 2.0, "reward_types": {"won_duel": 4.0}},
                {"update": 2, "samples": 16, "reward_mean": 0.3, "terminated_duels": 3,
                 "timeouts": 1, "entropy": 1.0, "reward_types": {"won_duel": 8.0}},
            ]
            (run / "metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            report = summarize_run(read_run(run), windows=(2,))

        window = report["windows"][0]
        self.assertEqual(report["environment"], "test-duel")
        self.assertEqual(window["reward_mean"], 0.2)
        self.assertEqual(window["ended_duels"], 4)
        self.assertEqual(window["timeouts"], 2)
        self.assertEqual(window["timeout_fraction"], 1 / 3)
        self.assertEqual(window["average_metrics"]["entropy"], 1.5)
        self.assertEqual(window["reward_per_player_transition"]["won_duel"], 0.75)


if __name__ == "__main__":
    unittest.main()