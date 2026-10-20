"""Opt-in validation for authored or resolved experiment JSON files."""

import argparse
from pathlib import Path
import sys

from simulator.training.environment import ROOT
from simulator.training.plans import load_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path,
                        help="JSON files or folders; defaults to all files under simulator/setups/curricula")
    options = parser.parse_args(argv)
    paths = []
    if not options.paths:
        paths.extend(sorted((ROOT / "simulator/setups/curricula").rglob("*.json")))
    for path in options.paths:
        paths.extend(sorted(path.rglob("*.json")) if path.is_dir() else [path])
    paths = list(dict.fromkeys(path.resolve() for path in paths))
    failed = 0
    for path in paths:
        try:
            plan = load_plan(path)
        except (OSError, ValueError, TypeError, KeyError) as error:
            print(f"ERROR {path}: {error}", file=sys.stderr)
            failed += 1
        else:
            print(f"OK {path} ({len(plan['stages'])} stages)")
    print(f"Validated {len(paths)} plan(s); {failed} failed.")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())