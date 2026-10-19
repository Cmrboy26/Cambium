"""Summarize a standalone simulator training run for quick review."""

import argparse
import json
from pathlib import Path

from simulator.training.dashboard import RUNS, RunStore


METRIC_NAMES = ("reward_mean", "loss", "policy_loss", "value_loss", "entropy",
                "player_transitions_per_second")


def read_run(path=None):
    if path is None:
        store = RunStore()
        runs = store.list_runs()
        if not runs:
            raise FileNotFoundError(f"No training runs found under {RUNS}")
        selected = Path(runs[0]["path"])
    else:
        selected = Path(path).expanduser().resolve()
        if selected.is_file() and selected.name == "metrics.jsonl":
            selected = selected.parent
        if not selected.is_dir():
            raise FileNotFoundError(f"Training run directory not found: {selected}")
        store = RunStore(selected.parent, [selected])

    runs = store.list_runs()
    if not runs:
        raise FileNotFoundError(f"No run config or metrics found under {selected}")
    return store.read(runs[0]["id"])


def _average(rows, name):
    values = [row[name] for row in rows if name in row]
    return sum(values) / len(values) if values else None


def _sample_deltas(rows):
    deltas = {}
    previous = None
    previous_update = None
    for row in rows:
        samples = row.get("samples")
        update = row["update"]
        if samples is not None and previous is not None and previous_update == update - 1:
            deltas[update] = max(0, samples - previous)
        elif samples is not None and update == 1:
            deltas[update] = samples
        previous = samples
        previous_update = update
    return deltas


def summarize_run(run, windows=(10, 50)):
    rows = run["metrics"]
    sample_deltas = _sample_deltas(rows)
    window_reports = []
    for width in windows:
        recent = rows[-width:]
        if not recent:
            continue
        terminated = sum(row.get("terminated_duels", 0) for row in recent)
        timeouts = sum(row.get("timeouts", 0) for row in recent)
        endings = terminated + timeouts
        transitions = sum(sample_deltas.get(row["update"], 0) for row in recent)
        reward_types = {}
        for row in recent:
            for name, value in row.get("reward_types", {}).items():
                reward_types[name] = reward_types.get(name, 0.0) + value
        if transitions:
            reward_types = {name: value / transitions for name, value in reward_types.items()}
        else:
            reward_types = {}

        window_reports.append({
            "updates": len(recent),
            "first_update": recent[0]["update"],
            "last_update": recent[-1]["update"],
            "reward_mean": _average(recent, "reward_mean"),
            "positive_reward_updates": sum(row.get("reward_mean", 0) > 0 for row in recent),
            "ended_duels": terminated,
            "timeouts": timeouts,
            "timeout_fraction": timeouts / endings if endings else None,
            "average_metrics": {name: _average(recent, name) for name in METRIC_NAMES},
            "reward_per_player_transition": reward_types,
        })

    config = run.get("config") or {}
    metadata = config.get("metadata") or {}
    options = config.get("options") or {}
    latest = rows[-1] if rows else {}
    return {
        "run": run["name"],
        "path": run["path"],
        "environment": (metadata.get("environment") or {}).get("name") or options.get("environment"),
        "device": options.get("device"),
        "worlds": options.get("envs"),
        "configured_updates": options.get("updates"),
        "updates_recorded": len(rows),
        "first_update": rows[0]["update"] if rows else None,
        "latest_update": latest.get("update"),
        "player_transitions": latest.get("samples"),
        "completed_rounds": latest.get("completed_rounds"),
        "reward_weights": metadata.get("reward_weights"),
        "windows": window_reports,
        "warnings": run.get("warnings", []),
        "interpretation_note": (
            "Shared self-play reward is not a win rate. Compare checkpoints against a fixed opponent "
            "and report both player seats separately."
        ),
    }


def format_report(report):
    lines = [
        f"Run: {report['run']}",
        f"Environment: {report['environment'] or 'unknown'} | Device: {report['device'] or 'unknown'} | "
        f"Worlds: {report['worlds'] or 'unknown'}",
        f"Updates: {report['updates_recorded']} recorded, latest {report['latest_update']} / "
        f"{report['configured_updates'] or 'unknown'} | Player transitions: {report['player_transitions'] or 0:,}",
    ]
    if report["completed_rounds"] is not None:
        lines.append(f"Completed duels: {report['completed_rounds']:,} (cumulative)")

    for window in report["windows"]:
        averages = window["average_metrics"]
        timeout_rate = window["timeout_fraction"]
        timeout_text = f"{timeout_rate:.1%}" if timeout_rate is not None else "n/a"
        lines.extend([
            f"Last {window['updates']} updates ({window['first_update']}-{window['last_update']}): "
            f"mean reward {window['reward_mean']:.5f}, positive {window['positive_reward_updates']}/"
            f"{window['updates']}, ended duels {window['ended_duels']:,}, "
            f"timeouts {window['timeouts']:,} ({timeout_text})",
            "  PPO: " + ", ".join(
                f"{name} {averages[name]:.4g}" for name in METRIC_NAMES if averages[name] is not None),
        ])
        if window["reward_per_player_transition"]:
            contributions = sorted(window["reward_per_player_transition"].items(),
                                   key=lambda item: abs(item[1]), reverse=True)
            lines.append("  Reward / player step: " + ", ".join(
                f"{name} {value:+.4g}" for name, value in contributions[:8]))

    lines.append("Note: " + report["interpretation_note"])
    lines.extend(f"Warning: {warning}" for warning in report["warnings"])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", help="Run directory or its metrics.jsonl; defaults to newest run")
    parser.add_argument("--windows", nargs="+", type=int, default=[10, 50],
                        help="Recent update window sizes to report (default: 10 50)")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args(argv)
    if any(width < 1 for width in args.windows):
        parser.error("window sizes must be positive")

    report = summarize_run(read_run(args.run), args.windows)
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())