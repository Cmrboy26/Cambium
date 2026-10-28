"""Build the platform-specific standalone helper embedded in the Fabric mod."""

import argparse
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def platform_name():
    system = {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}.get(platform.system())
    machine = platform.machine().lower()
    architecture = "x86_64" if machine in ("amd64", "x86_64") else "arm64" if machine in ("arm64", "aarch64") else None
    if system is None or architecture is None:
        raise ValueError(f"Unsupported build platform: {platform.system()} {machine}")
    return f"{system}-{architecture}"


def build_command():
    output = ROOT / "cambiummod/build/inference-build"
    return [sys.executable, "-m", "PyInstaller", "--noconfirm", "--onefile", "--name", "cambium-inference",
            "--paths", str(ROOT), "--add-data", f"{ROOT / 'contracts/policy_catalogue.json'}{os.pathsep}contracts",
            "--add-data", f"{ROOT / 'simulator/training/reward_weights.json'}{os.pathsep}simulator/training",
            "--distpath", str(output / "dist"), "--workpath", str(output / "work"), "--specpath", str(output),
            str(ROOT / "simulator/training/rosie.py")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    target = ROOT / "cambiummod/build/runtime-bundle/cambium-runtime" / platform_name()
    executable = "cambium-inference.exe" if sys.platform == "win32" else "cambium-inference"
    command = build_command()
    print(subprocess.list2cmdline(command), flush=True)
    print(f"Bundle: {target / executable}", flush=True)
    if args.dry_run:
        return 0
    import torch
    if torch.version.cuda is not None:
        raise ValueError("Build the inference helper using a CPU-only PyTorch installation")
    subprocess.run(command, cwd=ROOT, check=True)
    target.mkdir(parents=True, exist_ok=True)
    destination = target / executable
    shutil.copyfile(ROOT / "cambiummod/build/inference-build/dist" / executable, destination)
    digest = hashlib.sha256()
    with destination.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    destination.with_name(executable + ".sha256").write_text(digest.hexdigest() + "\n", encoding="ascii")
    print(f"Bundled {destination.stat().st_size} bytes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())