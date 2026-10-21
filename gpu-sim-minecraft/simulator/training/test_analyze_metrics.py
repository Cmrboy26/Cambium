import json
from pathlib import Path
import tempfile
import unittest

from simulator.training.analyze_metrics import format_timing_report, read_run, summarize_run, summarize_timings


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

    def test_timing_profiles_keep_nested_phases_and_separate_throughput(self):
        profile = {"version": 1, "seconds": {"environment_reset": 0.6, "environment_reset.sample": 0.4},
                   "native_seconds": {"reset_validate": 0.1}, "cuda_stream_seconds": {},
                   "counts": {"reset_worlds": 4}, "synchronize_seconds": 0.01, "unaccounted_seconds": 0.02}
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()
            (run / "config.json").write_text(json.dumps({"options": {}}))
            rows = [{"update": update, "samples": update * 8, "iteration_seconds": seconds,
                     "collection_seconds": seconds - 0.1, "update_seconds": 0.1,
                     "stage": "aim" if update <= 2 else "combat",
                     **({"collection_profile": profile} if update == 2 else {})}
                    for update, seconds in ((1, 1), (2, 2), (3, 1))]
            (run / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows) + '{"update":4')
            self.assertNotIn("collection_profile", read_run(run)["metrics"][1])
            loaded = read_run(run, include_profiles=True)
            report = summarize_timings(loaded, skip_updates=0)
            warm = summarize_timings(loaded, skip_updates=1)
            combat = summarize_timings(loaded, skip_updates=0, stage="combat")
        self.assertEqual(report["profiled_updates"], 1)
        self.assertEqual(report["groups"]["profiled"]["transitions_per_second"], 4)
        self.assertEqual(report["groups"]["unprofiled"]["transitions_per_second"], 8)
        self.assertEqual(report["seconds"], profile["seconds"])
        self.assertEqual(report["native_seconds"], profile["native_seconds"])
        self.assertEqual(report["counts_per_profile"]["reset_worlds"], 4)
        self.assertEqual(warm["groups"]["unprofiled"]["updates"], 1)
        self.assertEqual(combat["groups"]["unprofiled"]["updates"], 1)
        self.assertEqual(combat["profiled_updates"], 0)
        self.assertEqual(report["stages"]["combat"]["unprofiled"]["transitions_per_second"], 8)
        self.assertIn("dot-named children", format_timing_report(report))

    def test_timing_report_handles_old_runs_and_rejects_invalid_profiles(self):
        run = {"name": "old", "metrics": [{"update": 1, "samples": 8}], "warnings": []}
        report = summarize_timings(run, skip_updates=0)
        self.assertEqual(report["profiled_updates"], 0)
        self.assertIsNone(report["groups"]["unprofiled"]["transitions_per_second"])
        self.assertIn("No collection profiles", format_timing_report(report))
        run["metrics"][0]["collection_profile"] = {"version": 999}
        with self.assertRaisesRegex(ValueError, "version 1"):
            summarize_timings(run, skip_updates=0)
        run["metrics"][0]["collection_profile"] = {"version": 1, "seconds": {"bad": float("nan")}}
        with self.assertRaisesRegex(ValueError, "finite nonnegative"):
            summarize_timings(run, skip_updates=0)


if __name__ == "__main__":
    unittest.main()