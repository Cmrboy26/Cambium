"""Generate the small cross-language policy tables; --check never writes files."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render():
    catalogue = json.loads((ROOT / "contracts/policy_catalogue.json").read_text(encoding="utf-8"))
    tables = {"feature_names": catalogue["features"],
              "action_names": [action["name"] for action in catalogue["actions"]],
              "action_kinds": [action["kind"] for action in catalogue["actions"]],
              "movement_names": catalogue["movements"], "capability_names": catalogue["capabilities"],
              "armor_slot_names": catalogue["armor_slots"]}
    for name, values in tables.items():
        if not values or not all(isinstance(value, str) for value in values):
            raise ValueError(f"Invalid catalogue table: {name}")
        if name != "action_kinds" and len(values) != len(set(values)):
            raise ValueError(f"Duplicate catalogue entry: {name}")
    if set(tables["action_kinds"]) - {"movement", "boolean", "look", "weapon"}:
        raise ValueError("Unknown action value kind")
    if any(type(value) is not int or value < 1 for value in catalogue["limits"].values()):
        raise ValueError("Catalogue limits must be positive integers")
    digest = hashlib.sha256(json.dumps(catalogue, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    cpp = ['#pragma once', '#include <array>', '', 'namespace minecraft_sim::app {',
           f'inline constexpr const char* catalogue_hash = "{digest}";']
    java = ['package net.cambium;', '', 'import java.util.List;', '', 'final class PolicyCatalogue {',
            f'\tstatic final String HASH = "{digest}";']
    for name, values in tables.items():
        quoted = ", ".join(json.dumps(value) for value in values)
        cpp.append(f'inline constexpr std::array<const char*, {len(values)}> {name}{{{quoted}}};')
        java.append(f'\tstatic final List<String> {name.upper()} = List.of({quoted});')
    for name, value in catalogue["limits"].items():
        cpp.append(f'inline constexpr int policy_max_{name} = {value};')
        java.append(f'\tstatic final int MAX_{name.upper()} = {value};')
    cpp.extend(['}', ''])
    java.extend(['\tprivate PolicyCatalogue() {}', '}', ''])
    return {ROOT / "simulator/apps/policy_catalogue.generated.hpp": "\n".join(cpp),
            ROOT / "cambiummod/src/main/java/net/cambium/PolicyCatalogue.java": "\n".join(java)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    options = parser.parse_args()
    stale = []
    for path, content in render().items():
        if options.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.write_text(content, encoding="utf-8", newline="\n")
    if stale:
        parser.exit(1, "Stale policy tables: " + ", ".join(stale) + "\nRun python scripts/generate-contract.py\n")
    print("Policy catalogue tables are current.")


if __name__ == "__main__":
    main()