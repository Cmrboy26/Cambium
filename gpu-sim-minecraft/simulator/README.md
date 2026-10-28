# Shared Minecraft Physics Core

C++20 physics for Minecraft Java **26.3**, shared by CPU and optional CUDA
execution. Includes synthetic duels, replay export, movement comparison and
regression checks. Synthetic scenarios need no Minecraft installation.

## Windows Build

Requires Git, CMake 3.24+ and a C++20 compiler. From a Visual Studio 2022
compiler-enabled terminal at the repository root:

```powershell
cmake -S simulator -B simulator/build -G "Visual Studio 17 2022" -A x64
cmake --build simulator/build --config Release --parallel
ctest --test-dir simulator/build -C Release --output-on-failure
.\simulator\build\Release\minecraft_sim_runner.exe --duel --ticks 80
```

First configuration fetches host-only CSV/JSON dependencies. For MinGW or
offline builds, see the [build reference](AGENTS.md#windows-build).

## Linux Build

From the repository root with CMake, a C++20 compiler and Make or Ninja:

```sh
cmake -S simulator -B simulator/build -DCMAKE_BUILD_TYPE=Release
cmake --build simulator/build --parallel
ctest --test-dir simulator/build --output-on-failure
./simulator/build/minecraft_sim_runner --duel --ticks 80
```

Use separate build directories when changing generators or toolchains.

## CUDA

On a configured Windows CUDA machine:

```powershell
.\scripts\run-gpu.ps1 -Mode All
```

This builds, requires passing device parity checks, runs a synthetic duel and
benchmarks physics. It needs an NVIDIA GPU, a C++20 CUDA Toolkit and supported
host compiler. [Manual CUDA/Linux setup](AGENTS.md#cuda-build).
Physics benchmark rates are not end-to-end PPO training rates.

`CudaBatch::step(k)` launches one kernel per tick and computes arrow summaries
on the final tick. A device-side tick loop produced a corrupted loop counter
with CUDA 12.0.140 on H100 when threads diverged into summary work; retain this
ordering unless that toolchain and workload are verified with an alternative.

## Use It

- [Train a policy](training/README.md) with the Python adapter and PPO trainer.
- [Edit an arena](environments/research-duel.json) using the
  [scenario contract](../docs/simulation_contract.md).
- [Export and view a replay](../cambiummod/README.md#simulation-playback).
- [Compare a movement recording](AGENTS.md#recorded-physics-comparison) with the
  native comparator.

## Development

[AGENTS.md](AGENTS.md) covers implemented mechanics, tick/API ordering,
equipment/food rules, recording formats, benchmark methodology and checks.
Use the [physics specification](../docs/minecraft_simulation_specs.md) and matching
26.3 source when changing behavior.

This is a growing movement/combat subset, not complete Minecraft physics.
Synthetic regression checks, CPU/CUDA parity and real-game captures validate
different things; passing one does not establish full game parity.