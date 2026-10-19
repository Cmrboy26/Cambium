# Rosie GPU Simulator Workflow

Follow the repository [development guide](../AGENTS.md). This folder owns
cluster-specific launchers; general simulator and training behavior remains in
the simulator guides.

## Scope And Prerequisites

This workflow trains the standalone Minecraft Java 26.3 simulator. It does not
start the legacy 1.8.9 server, HeadlessMC clients, or FastAPI backend. Submit
jobs from `gpu-sim-minecraft` so `SLURM_SUBMIT_DIR`, relative plan paths,
and `rosie/logs` resolve as expected.

`setup.sh` creates `.venv-rosie` and installs CUDA-enabled PyTorch 2.5.1/cu124
plus the training requirements. It runs on the management node and does not
compile CUDA. The first training job configures CMake and builds on its allocated
compute node, which needs network access for the pinned FetchContent dependencies.

Setup reuses an existing virtual environment after checking its interpreter and
pip; it does not rewrite activation scripts. This permits reruns when activation
files are read-only. An incomplete or broken environment fails with an explicit
diagnostic and is never deleted automatically. Move that specific environment
aside before rerunning setup if it needs recreation; do not copy virtual
environments between machines.

The training job checks `.venv-rosie/bin/python`. On `dh-dgxh100-*`, the venv's
management-node interpreter symlink may be unavailable. In that case, the job
uses the compute-node `python3` and adds the venv's matching
`lib/python<major>.<minor>/site-packages` to `PYTHONPATH`. The Rosie H100 setup
was verified with system Python 3.12.3, PyTorch 2.5.1+cu124, and an H100 80GB.
If the system Python minor version differs from the venv packages, this fallback
fails explicitly because the matching site-packages directory is absent.

## Training Resources And Build Trees

`train.sbatch` defaults to `teaching`, one T4, 8 CPUs, 24 GB and a two-hour
limit. It builds CUDA architecture 75 by default. Override resources on the
`sbatch` command line for another partition. H100 requires architecture 90:

```sh
sbatch --partition=dgxh100 --gres=gpu:h100:1 \
  --export=ALL,MCSIM_CUDA_ARCHITECTURES=90,SIM_ENVS=8192,SIM_UPDATES=500,SIM_ROLLOUT_STEPS=128,SIM_EPOCHS=2,SIM_MINIBATCH_SIZE=16384,SIM_HIDDEN_DIM=512 \
  rosie/train.sbatch
```

One requested H100 is used by this single-process trainer; requesting all eight
does not enable multi-GPU training. Builds are isolated by architecture:
`simulator/build-rosie-cuda-75` for T4 and
`simulator/build-rosie-cuda-90` for H100. This avoids different architectures
reconfiguring a shared CMake tree. Do not run concurrent jobs against the same
architecture/build directory.

Supported training environment variables:

- `SIM_ENVS`: parallel worlds; default 256 for new runs, restored on resume.
- `SIM_UPDATES`: optimizer updates; default 1.
- `SIM_ROLLOUT_STEPS`: steps per rollout; default 16 for new runs, restored on resume.
- `SIM_EPOCHS`, `SIM_MINIBATCH_SIZE`, `SIM_HIDDEN_DIM`, `SIM_LR`: optional
  overrides of the plan's learner settings; unset by default.
- `SIM_THREADS`: CPU threads; default 4. The plan owns the seed.
- `SIM_PLAN`: version-2 training plan path relative to `gpu-sim-minecraft` or
  absolute; defaults to `simulator/setups/curricula/shield-axe-curriculum.json`
  for new runs. References inside authoring files resolve relative to their
  containing file, not the submission directory.
  `SIM_UPDATES` caps this invocation; a completed curriculum stops early.
- `SIM_RESUME`: resolved-experiment checkpoint; restores embedded configuration.
- `SIM_INITIALIZE`: compatible checkpoint weights for a new plan, not a resume.
- `SIM_ENVIRONMENT`, `SIM_SCHEMA`, `SIM_SEED` are removed; nonempty values fail
  explicitly. Configure the environment, policy and seed through the plan.
- `SIM_OUTPUT`: run directory; default `simulator/training/runs/rosie-<JOB_ID>`.
- `SIM_REPLAY_EVERY`: replay interval in completed duels; default 0 (disabled).
- `SIM_REPLAY_DIR`: optional replay destination; default `<run>/replays` when replay is enabled.

For the four-stage shield/axe curriculum, submit from `gpu-sim-minecraft`:

```sh
sbatch --partition=dgxh100 --gres=gpu:h100:1 --time=1-00:00:00 \
  --export=ALL,MCSIM_CUDA_ARCHITECTURES=90,SIM_PLAN=simulator/setups/curricula/shield-axe-curriculum.json,SIM_ENVIRONMENT=,SIM_SCHEMA=,SIM_SEED=,SIM_ENVS=8192,SIM_UPDATES=1500,SIM_ROLLOUT_STEPS=128,SIM_REPLAY_EVERY=50000 \
  rosie/train.sbatch
```

This clears previously exported legacy settings. Learner settings come from the
plan. Runtime is unmeasured; the requested day is a wall-time limit, not a
promise of completion. Use a one-update/256-world smoke job before scaling up.
The job rebuilds native CUDA code; CUDA execution remains a target-device gate.

For another task, create an environment and reference it from a version-2 plan;
do not pass historical concrete scenario files directly to the trainer.

Windows DLL paths and `%APPDATA%`/`$env:APPDATA` do not exist on Rosie. With
`SIM_REPLAY_EVERY=50000`, files go to
`simulator/training/runs/rosie-<JOB_ID>/replays` unless `SIM_REPLAY_DIR` is set.
Replay cadence counts completed duels, not optimizer updates. CUDA training
still stages observations and rewards through the CPU.
The hidden size is part of checkpoint compatibility; changing it starts a new
model rather than resuming a checkpoint with a different hidden dimension.

SLURM output and error logs are `rosie/logs/gpu-sim-<JOB_ID>.out` and `.err`.
The trainer writes metrics and checkpoints under its run directory. A one-update
smoke job is a wiring check, not a useful training duration; check its logs before
scaling up.

## SSH Key Setup

On Windows, run `.\rosie\setup-ssh-key.ps1 -SshHost 2 -SshUser <account>` from
`gpu-sim-minecraft` with GlobalVPN connected. Windows OpenSSH Client (`ssh` and
`ssh-keygen`) must already be installed. Host/user defaults come from
`ROSIE_SSH_JUMP` and `ROSIE_SSH_USER`; missing values prompt in the terminal.
Hosts accept a node number (`1`-`4`), hostname or SSH alias. Bare account names
get `@ad.msoe.edu` appended. This does not change launcher or SSH configuration.

The helper reuses `$HOME/.ssh/id_ed25519` or creates it interactively, and can
recover a missing public key from the existing private key. It never overwrites
an existing key pair or transmits the private key. Passwords, key passphrases
and host-key confirmations are handled directly by OpenSSH in the terminal;
host-key checking stays enabled. Only the public key goes to Rosie, encoded
separately from the remote shell script to avoid Windows argument-quoting issues.
Installation preserves existing `authorized_keys` entries, backs up that file
before adding a missing entry, avoids duplicate entries on reruns, and sets
remote `.ssh`/`authorized_keys` permissions to `700`/`600`.

The final check uses public-key-only authentication with `BatchMode=yes`, so
success verifies that noninteractive Rosie helpers can authenticate. For a
passphrase-protected key, load it into a running SSH agent using
`ssh-add "$HOME/.ssh/id_ed25519"`, then rerun setup. Agent installation or
service configuration is not performed automatically and the passphrase is
not removed. `-DryRun` previews the target and key choice without creating files
or opening connections; specify `-SshHost` to avoid prompting. GlobalVPN is
still required after key setup.

## Dashboard And SSH Tunnel

With a `cambium-sim` training job running, start
`bash ./rosie/dashboard-connect.sh`
from PowerShell in the local `gpu-sim-minecraft` directory (Git Bash must be
installed and available as `bash`), or `bash rosie/dashboard-connect.sh` from
Git Bash.
Connect to GlobalVPN first. The script finds the unique running training job,
or selects one automatically when several share a working directory; pass the
job ID when they use different working directories.
It starts the read-only Python dashboard with `srun --jobid ... --overlap` on
the allocated compute node, so it does not submit another job or request
another node. Both the remote and local loopback ports are selected
automatically; the script prints and opens the local URL. Ctrl+C stops the
dashboard step and tunnel, not the training job.

The dashboard binds only to loopback and is reached through SSH `ProxyJump` via
the assigned `dh-mgmt1` through `dh-mgmt4` host; do not expose its port to the
network. If `ROSIE_SSH_JUMP` is unset, the script derives the host when run on
a Rosie management node; from a local computer it prompts for the node number
(`1`-`4`) or full hostname (for example, `dh-mgmt1.hpc.msoe.edu`). The variable
accepts either form. The local computer needs Git Bash, SSH and Python 3
(`python3` or `python`). Set
`ROSIE_SSH_USER` to specify/override the Rosie account, `ROSIE_SSH_JUMP` to
avoid the host prompt, or `ROSIE_JOB_ID` to select a training job without a
positional argument. The default SSH account appends `@ad.msoe.edu` to the local
username. If SSH policy blocks direct
compute-node access through `ProxyJump`, use the site's approved tunneling
method.

## Local Artifact Launcher

`scripts/rosie-play.ps1 -Setup` configures the **local** Minecraft instance;
it does not start a Rosie job or modify the remote installation. The account,
management host/SSH alias, remote runs directory, local repository and selected
Python executable are saved in game `config/cambium-rosie.json`. Use the local
checkout and a Python environment with training requirements for duels. Re-run
setup after moving the checkout or changing interpreters. No SSH passwords are
stored. Keep the config trusted:
the mod executes its Python interpreter from its repository directory.

Connect GlobalVPN. For password authentication, run
`scripts/rosie-play.ps1 -Connect -MinecraftDirectory <game>` in a **local terminal**
and leave it open. Python starts interactive native SSH (`BatchMode=no`) with
terminal stderr; SSH handles passwords, MFA, key passphrases and host-key prompts
directly through the terminal. Never enter credentials in Minecraft or store them
in the config. Ctrl+C or SSH disconnection ends this session.

One SSH connection forwards a local loopback port to a temporary, loopback-only
Python artifact reader on the management node. An ephemeral bearer token protects
the reader; it is bootstrapped over encrypted SSH stdin, never command arguments.
The local port/token/connection fingerprint are atomically saved in game
`config/cambium-rosie-session.json` (owner-only temp-file permissions on POSIX),
not the SSH password. The descriptor is removed on normal shutdown. Other local
helpers automatically use it and stream the original length/SHA-256 transfer
protocol over the tunnel. The reader shuts down on SSH stdin EOF and only accepts
the three read-only artifact operations under the configured run root. Do not
share the session descriptor; it grants temporary artifact access.

Stale connections fail with `-Connect` instructions rather than silently trying
password-less SSH again. Close the old connection before changing setup settings.
With no descriptor, existing key/agent use still opens noninteractive SSH
(`BatchMode=yes`) per request, with a ten-second connect timeout and keepalives.
The SSH command/agent must then be available to Minecraft's process. Offline
mode ignores the descriptor. Do not disable host-key checking.

The default remote root is
`~/minecraftlol/Cambium/gpu-sim-minecraft/simulator/training/runs`; override with
`-RemoteRunsDirectory` at setup. SSH runs a bundled, standard-library Python 3
artifact reader on the management host. Only saved files are read, not training
or inference. No public port or active SLURM allocation is needed. The password-
session reader runs only while `-Connect` is open; key-based requests run the
reader separately per operation. This assumes that the run root is visible on
that management host. Custom
`SIM_REPLAY_DIR` locations must be collected into the run's `replays` directory
before using this launcher.

The helper is `python -B -m simulator.training.rosie <action>
--minecraft-directory <game>`; actions are `setup`, `connect`, `runs`, `replays`,
`replay`, `sync`, `duel`. `connect` requires a terminal, not `--json`/`--offline`.
`--run` and `--replay` default to `latest`; `--offline` uses only
cached artifacts. `--json` emits newline-separated events for the mod. Run/replay
basenames are limited to 120 ASCII letters, digits, dots, underscores or hyphens,
starting with a letter/digit. Paths and remote arguments are encoded, not shell
interpolated; resolved remote files must remain inside the configured run root.

`replay` downloads only the selected sample to game `sim-replays`, prefixing its
filename with the run ID. `duel` caches `latest.pt` and available config/metrics
in game `sim-rosie-cache/<run>`, then serves CPU inference on an automatically
allocated loopback port. `sync` caches the checkpoint and **all** replay files
for that run; consider disk usage before syncing large runs. Transfers reuse
matching size/mtime files, stream to unique `.part` files, verify length/SHA-256,
then atomically replace destinations. A failed transfer does not replace an
existing artifact. Remote reads pin a file descriptor across atomic trainer
checkpoint replacements. Replay limit is 64 MiB, other artifact limit 512 MiB.

For offline duels: `scripts/rosie-play.ps1 -Action duel -Run <run> -Offline`;
use `/simduel arena <printed-port>`. Ctrl+C stops the terminal-owned service.
Cached replays can be loaded normally with `/simreplay load latest`. The cache
is not a historical model archive: syncing newer weights replaces cached
`latest.pt`, and neither local nor remote `latest` means best-performing.

Tests: `python -B -m unittest simulator.training.test_rosie -v` is stdlib-only
and uses local subprocesses to exercise the actual remote reader. The existing
`test_training.InferenceChecks` also launches a real offline policy process.
Neither proves local Windows SSH authentication or in-game behavior. See
[in-game commands and lifecycle](../cambiummod/AGENTS.md#rosie-artifact-commands)
and [launcher parameters](../scripts/AGENTS.md#rosie-sync-and-play).

## Local Linux Checks

The PowerShell scripts under `scripts/` target Windows. To build and run a small
CPU check on Linux from `gpu-sim-minecraft`:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r simulator/training/requirements.txt
cmake -S simulator -B simulator/build -DMCSIM_ENABLE_CUDA=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build simulator/build --parallel
ctest --test-dir simulator/build --output-on-failure
python -B -m simulator.training.train --device cpu --envs 4 --updates 1 --rollout-steps 16
```

Do not use `sudo`; use an unprivileged venv and build folder. For local CUDA,
install CUDA-enabled PyTorch and configure with `-DMCSIM_ENABLE_CUDA=ON` and the
GPU's `CMAKE_CUDA_ARCHITECTURES` value.