"""Overlay a sweep's trial metrics on a single headless PNG chart."""

import argparse
import json
from pathlib import Path

from simulator.training.analyze_metrics import read_run
from simulator.training.sweep import load_manifest


METRICS = ("win_rate", "evaluation_reward", "reward_mean", "entropy", "loss",
           "player_transitions_per_second", "collection_seconds", "update_seconds")
AXES = ("elapsed_seconds", "samples", "update")


def series(path, metric="win_rate", suite=None, axis="elapsed_seconds", runs=None):
    if metric not in METRICS or axis not in AXES:
        raise ValueError("Unsupported sweep metric or x-axis")
    manifest = load_manifest(path)
    trials = []
    warnings = set()
    available = set()
    for trial in manifest["trials"]:
        output = Path(runs) / Path(trial["output"]).name if runs else Path(trial["output"])
        try:
            result = read_run(output)
            rows = result["metrics"]
            warnings.update(result["warnings"])
        except FileNotFoundError:
            rows = []
        elapsed = 0
        for row in rows:
            elapsed += row.get("iteration_seconds", 0)
            if "elapsed_seconds" not in row:
                row["elapsed_seconds"] = elapsed
                if axis == "elapsed_seconds":
                    warnings.add("Legacy elapsed time is estimated from recorded iterations.")
            available.update(row.get("evaluations", {}))
        trials.append((trial, rows))
    if suite is None:
        suite = "combined_benchmark" if "combined_benchmark" in available else next(iter(sorted(available)), None)
    if metric in ("win_rate", "evaluation_reward") and suite is None:
        warnings.add("No evaluation results yet; curves appear after the first benchmark.")
    curves = []
    for trial, rows in trials:
        points, seen = [], set()
        updates = {row["update"]: row for row in rows}
        for row in rows:
            anchor = row
            if metric in ("win_rate", "evaluation_reward"):
                evaluation = row.get("evaluations", {}).get(suite)
                if not evaluation:
                    continue
                snapshot = evaluation.get("snapshot_update")
                if snapshot is None or snapshot in seen or snapshot not in updates:
                    continue
                seen.add(snapshot)
                anchor = updates[snapshot]
                value = evaluation["win_rate"] * 100 if metric == "win_rate" else evaluation["reward_mean"]
            else:
                value = row.get(metric)
            if value is not None and axis in anchor:
                points.append((anchor[axis], value))
        curves.append({"name": trial["name"], "label": f"{manifest['parameter']}={trial['value']}", "points": points})
    return {"name": manifest["name"], "suite": suite, "metric": metric, "axis": axis,
            "curves": curves, "warnings": sorted(warnings)}


def plot(path, output=None, metric="win_rate", suite=None, axis="elapsed_seconds", runs=None):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    report = series(path, metric, suite, axis, runs)
    figure = Figure(figsize=(9, 5.5), dpi=140)
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots()
    for curve in report["curves"]:
        points = curve["points"]
        axes.plot([point[0] for point in points], [point[1] for point in points],
                  marker="o" if metric in ("win_rate", "evaluation_reward") else None,
                  markersize=3, linewidth=1.5, label=curve["label"] + (" (no data yet)" if not points else ""))
    axes.set_title(report["name"] + (f" / {report['suite']}" if report["suite"] and metric in ("win_rate", "evaluation_reward") else ""))
    axes.set_xlabel({"elapsed_seconds": "Elapsed training time (seconds)", "samples": "Learner transitions", "update": "PPO update"}[axis])
    axes.set_ylabel("Benchmark win rate (%)" if metric == "win_rate" else "Benchmark mean reward" if metric == "evaluation_reward" else metric.replace("_", " ").capitalize())
    if metric == "win_rate":
        axes.set_ylim(0, 100)
    axes.grid(alpha=.25)
    axes.legend(loc="best")
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(rect=(0, .09 if report["warnings"] else 0, 1, 1))
    if report["warnings"]:
        figure.text(.02, .02, " ".join(report["warnings"]), fontsize=8, wrap=True)
    destination = Path(output) if output else Path(path).resolve().parent / "comparison.png"
    if destination.suffix.lower() != ".png":
        raise ValueError("Sweep comparison output must be a PNG")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    canvas.print_png(str(temporary))
    temporary.replace(destination)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--output")
    parser.add_argument("--metric", choices=METRICS, default="win_rate")
    parser.add_argument("--suite", help="Benchmark name; defaults to combined_benchmark, otherwise first available")
    parser.add_argument("--x", choices=AXES, default="elapsed_seconds")
    parser.add_argument("--runs", help="Local run cache root when plotting a copied sweep manifest")
    options = parser.parse_args(argv)
    print(plot(options.manifest, options.output, options.metric, options.suite, options.x, options.runs))


if __name__ == "__main__":
    main()