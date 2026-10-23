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
reconfiguring a shared CMake tree. Concurrent jobs may share an architecture's
build cache: the launcher holds an exclusive `flock` on `<build>/.build.lock`
through CMake configure, build and the library snapshot copy. It canonicalizes
the build path before opening the lock, and releases/closes the lock before
training. Keep this persistent lock file; deleting it can split lock ownership.

Each job copies the completed `minecraft_sim_training.so` into a unique private
directory under `SLURM_TMPDIR`, falling back to `TMPDIR` and then `/tmp`. The
copy is read-only and Python's `--library` points to it, not the shared output.
Later builds cannot change an already-running job's library. The private
directory is removed on normal/error exit and handled termination; forced
SIGKILL/node failure may leave scratch for site cleanup. Build/copy/lock failures
abort rather than starting training from an unprotected shared file.

This needs util-linux `flock` and a shared filesystem with working cross-node
advisory locks. Local process-level lock/snapshot tests do not certify cluster
mount behavior. All writers must cooperate: manual builds and older submitted
batch scripts do not automatically use this lock. Wait for older jobs to finish,
or explicitly depend on them, before overlapping new jobs on that build tree.
Do not change native sources while a build is running. Slurm GPU/resource/account
limits still determine whether eligible training jobs actually run concurrently.

### Bounded Diagnostic Jobs

`diagnostics.sbatch` runs evaluation or PPO diagnostics, not training. It
requests one H100, eight CPUs, 24 GB and 30 minutes. Submit from project root
with `MCSIM_DIAGNOSTIC_MODE=evaluation|ppo`, `MCSIM_DIAGNOSTIC_SOURCE` pointing
to a frozen checkout snapshot, `MCSIM_EVALUATION_BUNDLE` to a prepared immutable
input bundle, and `MCSIM_DIAGNOSTIC_OUTPUT` to a new output path. PPO mode also
requires a frozen full `MCSIM_PPO_CHECKPOINT`. Do not point it at a running
run's mutable `latest.pt`. Ensure `rosie/logs` exists before submission.

Each job builds in its private scratch tree, reading cached FetchContent source
dependencies when available; it never reconfigures a training build. Evaluation
runs 256 episodes per matchup with balanced seats. PPO mode runs native/CPU/CUDA
checks, captures 8192 x 128 real transitions once, compares four minibatch sizes
with identical initial model/optimizer state, and writes a short profiler trace.
Artifacts are separate from training runs. Failed jobs may retain diagnostic
output; use a fresh output path when retrying and preserve failed logs.

For bounded replay counts, direct batch submissions can set
`SIM_REPLAY_EVERY_UPDATES=N` instead of episode-based `SIM_REPLAY_EVERY`.
These modes are mutually exclusive. One recording is exported per matching
global PPO update, independent of world count, episode completions or stage
changes. A fresh 1,500-update run with `SIM_REPLAY_EVERY_UPDATES=30` produces
at most 50 recordings (fewer if training ends early). The trainer equivalent
is `--replay-every-updates 30`; global update numbering is retained on resume.

### Interactive Submission

Run `bash rosie/submit.sh` from any working directory using the script's path.
Use `bash rosie/submit.sh --profile-every 10` to record collection profiles every
ten global updates for all jobs in that invocation; combine with `--dry-run`
to preview without submitting. Profiling defaults off, ignores inherited
`SIM_PROFILE_EVERY` in the wizard, and does not add or shift prompts. Choose a
unique baseline run name. See the [timing schema and report commands](../simulator/training/AGENTS.md#collection-profiling)
before comparing profiled and ordinary update throughput.
The wizard changes to the project root. Choose GPU and plan first (Enter accepts
their defaults), then use the editable terminal form: Up/Down or Tab/Shift+Tab
select a row, type to edit, Ctrl+U clears its value, and Ctrl+C/Ctrl+D cancel.
Values start populated; use the usual Left/Right/Home/End/Backspace editing keys.
Enter moves to the next row; Enter on the bottom **Submit** row authorizes the
selected jobs, without a second confirmation prompt. Errors leave the form open.
The form scrolls on shorter terminals and keeps Submit visible. `--dry-run` uses
the same form but only previews commands, submitting/writing nothing. Install the
updated training requirements for `prompt-toolkit` (or rerun `rosie/setup.sh`).
`--line-mode` keeps the original sequential prompts and explicit `y` confirmation;
piped/non-terminal input also uses that mode. The new PPO fields can be supplied
there through the override flags described below.
Plan discovery includes all nested folders under `simulator/setups/curricula`.
The menu shows paths relative to that directory so bucketed plans with the same
filename remain distinguishable; submission retains the full selected path.
After selection, the wizard runs `python -m simulator.training.validate_plans`
on that file before reading defaults or prompting for training settings, including
in `--dry-run` mode. Invalid JSON, missing references or invalid plan semantics
abort with the validator's diagnostic and submit nothing. Only the selected
plan is checked, so another person's unfinished curriculum does not block yours.
This uses `.venv-rosie/bin/python`, falling back to `python3` on `PATH`; that
interpreter needs the training Python requirements. Validation does not load the
native library, train, compile CUDA or submit a job. Run the validator without
paths separately to check all curricula.

| Preset | Partition | GPU / CUDA architecture | Worlds | Rollout | Wall-Time |
| --- | --- | --- | --- | --- | --- |
| `h100` (default) | `dgxh100` | H100 / 90 | 8,192 | 128 | 1 day |
| `v100` | `dgx` | V100 / 70 | 2,048 | 128 | 1 day |
| `teaching` | `teaching` | T4 / 75 | 256 | 64 | 2 hours |

World counts are starting points, not benchmarked optima. Each job requests one
GPU, eight CPUs and 24 GB host RAM by default. Other prompts cover plan selection,
job name, update cap, learner settings, CPU threads and replay cadence. Learner
defaults come from the selected plan; the default update cap sums plain update
budgets, or falls back to 1,500 for other advancement rules. All values are editable.
The wizard requires Bash 4+ and Python 3 with the training requirements to preview a submission.

The job/run name sets both the Slurm name and `SIM_OUTPUT` under
`simulator/training/runs/<name>`, so the dashboard and artifact launcher show
the chosen name. It defaults to the plan filename without `.json`. Names must
start with an ASCII letter/digit, use only letters, digits, dots, underscores or
hyphens, and fit within 120 characters including any sweep suffix. Existing run
paths and duplicate generated names abort the whole setup before submission;
choose a new name rather than overwriting an earlier test.

In the terminal form, enter comma-separated values in exactly one training or
resource field for a single-variable sweep, e.g. entropy `0,0.005,0.01`.
Equivalent numeric values are rejected. `THREADS=auto` chooses up to four threads,
bounded by the smallest core allocation and active shared-GPU trial count;
an explicit value must satisfy the same total CPU budget. All learner values
start from the selected plan's effective configuration.

Submit CLI flags `--gamma`, `--gae-lambda`, `--clip-eps`, `--vf-coef`, `--ent-coef`
and `--max-grad-norm` set the corresponding form defaults. A comma-separated flag
value also defines a sweep in line mode, for example:

```sh
bash rosie/submit.sh --line-mode --dry-run --ent-coef 0,0.005,0.01
```

The [learner reference](../simulator/training/AGENTS.md#learner-configuration)
defines defaults, numeric ranges, persisted metadata and resume constraints.
For line-mode sweeps in the original settings, type `multiple` at one
training/resource prompt (wall time, worlds, update cap,
rollout, learner settings, CPU cores/threads, RAM or replay interval) for a
single-variable sweep. Enter one value per line, then an empty line to continue
to the next setting. Values use the normal validation, require at least one
entry and ignore exact duplicates. A second `multiple` is rejected and that
prompt requires a single value; GPU preset, plan, name and dependency remain
single-valued. For example, name `lr-test` with learning-rate values `.0001`,
`.0003` and `.0006` generates `lr-test-.0001`, `lr-test-.0003` and `lr-test-.0006`,
each with the same other settings. Time suffixes replace colons with hyphens;
positive exponent suffixes omit `+`. CPU sweeps keep threads fixed at a value
valid for the smallest core allocation. The wizard previews every command;
line mode then asks once to submit the complete setup. Closed input or cancellation
submits nothing. A later submission failure stops the remainder without
canceling jobs already submitted.

The wizard starts fresh runs and clears inherited `SIM_*` and `SBATCH_*` variables
before invoking `sbatch`; use the batch interface below for resumes/initialization.
The optional dependency prompt accepts colon-separated
job IDs and applies the same `afterany` dependency to every job in the setup.
Without explicit dependencies, sweep jobs are independently scheduled: only
their build/snapshot steps serialize, not training. New wizard invocations can
share the locked build cache. Use explicit dependencies when overlapping older,
unlocked launchers; existing queued jobs retain their submitted scripts and
dependency chains and are not migrated automatically. Direct `sbatch` submissions keep the
`rosie-<JOB_ID>` output default unless `SIM_OUTPUT` is supplied.
No jobs are submitted by the mocked checks:
`python -B -m unittest simulator.training.test_rosie.SubmitTests -v`.

### Local WSL2 Training

The shared wizard supports direct local execution from a Linux/WSL checkout:

```sh
bash rosie/setup.sh
bash rosie/submit.sh --local
```

Install a Windows NVIDIA driver with WSL GPU support and a CUDA toolkit/compiler
inside WSL; `nvidia-smi`, `nvcc`, CMake 3.24+ and `flock` must be on `PATH`.
The setup script installs PyTorch 2.5.1/cu124 and the training requirements.
The local launcher defaults to CUDA architecture 86, 256 worlds and 64 rollout
steps, builds under `simulator/build-local-cuda-86`, and trains in the foreground.
Set `MCSIM_CUDA_ARCHITECTURES` to override the architecture. `multiple` sweeps
run sequentially on the one local GPU; `--shared-gpu` is not supported locally.
Ctrl+C stops the active run. Its output is written under the normal
`simulator/training/runs/<name>` directory for the local dashboard.

The checkout location determines where training runs. This remote VS Code
workspace is on Rosie; use a separate checkout inside WSL's Linux filesystem
(not `/mnt/c`) and pull/sync the changes there to use a home GPU.

### Shared-GPU Sweeps

`bash rosie/submit.sh --shared-gpu` uses the same form (or line-mode `multiple`
entry) but submits exactly one allocation. All selected trial values launch
concurrently by default; `--max-parallel N` optionally queues excess trials
inside the job (`0` means all). Ordinary submissions remain separate jobs.
Only training settings may be swept: worlds, update cap, rollout length, width,
PPO epochs, minibatch size, learning rate, gamma, GAE lambda, clipping epsilon,
value/entropy coefficients, maximum gradient norm, trainer threads or replay interval.
Allocation CPU cores, host RAM and wall-time remain single-valued. One seed and
one allocated GPU are supported; this is independent training, not multi-GPU PPO.

Allocated CPUs/RAM are totals for the entire job. The wizard defaults trainer
threads to at most `allocated CPUs / active trials`, bounded by four. It checks
the sum of the largest simultaneously active thread counts before submission;
the runner repeats the check against Slurm's allocation. Leave headroom for
Python, native and evaluation overhead. RAM/VRAM demand is not predicted or
partitioned: concurrent optimization/evaluation can exhaust memory. Every child
inherits Slurm's GPU visibility and explicitly bounded OpenMP/BLAS threads.
Use only sharing permitted by the cluster. No CUDA MPS or automatic packing is
enabled, and a job-count limit does not override resource/account limits.

After confirmation, preparation freezes resolved experiments and a version-1
`minecraft_sim_sweep` manifest under `simulator/training/sweeps/<name>`. Dry-run
and cancellation create no artifacts. Existing sweep/trial paths or equivalent
numeric values abort preparation. Submission failure retains the prepared files
and prints their location. `SIM_SWEEP=<absolute-path>/sweep.json` selects this
mode in `train.sbatch`; do not combine it with plan/resume/initialize settings.

The batch script builds once under the existing lock, snapshots the native
library once, then runs `python -B -m simulator.training.sweep run <manifest>
--library <private-library>`. Python sources run from the checkout; revision and
source digests are recorded at preparation and launch, not archived. Do not edit
training code while queued trials still need to launch. Each trial's plan is
frozen, including referenced scenes/policy; external checkpoint files referenced
by opponents are not copied. Treat those checkpoints as immutable.

Trials remain ordinary flat `runs/<name>-<value>` directories for existing
dashboard/artifact discovery. The sweep directory contains frozen plans,
`sweep.json`, atomic `status.json`, separate `logs/<trial>.out/.err`, and
`comparison.png`. Status records pending/running/succeeded/failed/interrupted,
exit codes, launch/end times, commands and source/library provenance. Failure
of one trial does not stop siblings; any failed trial makes the job fail. TERM/INT
stops active process groups and prevents pending trials from starting; five
seconds later remaining workers are killed. SIGKILL cannot save final status.
Automatic retries/resume and memory scheduling are not implemented.

After the supervisor completes (including normal trial failures), the job renders
one headless Matplotlib PNG with all trial curves overlaid on the same axes.
Regenerate it during or after training without starting a server:

```sh
.venv-rosie/bin/python -B -m simulator.training.plot_sweep simulator/training/sweeps/<name>/sweep.json
.venv-rosie/bin/python -B -m simulator.training.plot_sweep simulator/training/sweeps/<name>/sweep.json --suite combined_benchmark --x samples
```

`--metric` selects win_rate (default), evaluation_reward, reward_mean, entropy,
loss, player_transitions_per_second, collection_seconds or update_seconds.
`--x` selects elapsed_seconds (default), samples or update. Default evaluation is
combined_benchmark if available, otherwise the first named suite. With no results
yet, the PNG retains labeled empty curves; no fake zero scores are introduced.
`--output` chooses another PNG. Copying the manifest/config/metrics to a local
machine is sufficient; use `--runs <local-cache-root>` to remap run paths. The
existing artifact client does not automatically fetch sweep manifests or plots.
Re-run setup once to install Matplotlib and its dependencies. Plot errors are
reported separately; a successful training sweep with a failed plot exits nonzero.

Compare fixed evaluation scores rather than raw rewards across reward sweeps.
For world-count sweeps, update-based curriculum budgets also change experience
per stage; stage progression and GPU/CPU contention remain comparison confounds.
Measure total sweep completion time as well as per-trial learning speed.

### Batch Environment

Supported training environment variables:

- `SIM_ENVS`: parallel worlds; default 256 for new runs, restored on resume.
- `SIM_UPDATES`: optimizer updates; default 1.
- `SIM_ROLLOUT_STEPS`: steps per rollout; default 16 for new runs, restored on resume.
- `SIM_EPOCHS`, `SIM_MINIBATCH_SIZE`, `SIM_HIDDEN_DIM`, `SIM_LR`: optional
  overrides of the plan's learner settings; unset by default.
- `SIM_GAMMA`, `SIM_GAE_LAMBDA`, `SIM_CLIP_EPS`, `SIM_VF_COEF`, `SIM_ENT_COEF`,
  `SIM_MAX_GRAD_NORM`: optional PPO learner overrides; unset by default so the
  plan/checkpoint owns their effective values. Changed resume settings fail.
- `SIM_THREADS`: CPU threads; default 4. The plan owns the seed.
- `SIM_PROFILE_EVERY`: collection profiling interval in global updates; default
  0 (off). Positive intervals add `collection_profile` to sampled metric rows.
- `SIM_BUILD_DIR`: optional shared build cache override; defaults to the
  architecture-specific directory above. The same lock/snapshot rules apply.
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

Start `bash ./rosie/dashboard-connect.sh`
from PowerShell in the local `gpu-sim-minecraft` directory (Git Bash must be
installed and available as `bash`), or `bash rosie/dashboard-connect.sh` from
Git Bash.
Connect to GlobalVPN first. By default, the script starts the read-only,
standard-library Python dashboard directly on the management node without
querying Slurm or requiring/submitting an allocation. It reads saved run files,
so training can start or stop independently. Set `ROSIE_PROJECT_DIR` to the
remote checkout when connecting from another computer without a job ID;
there is no assumed checkout location. When invoked on a management node,
the default is the script's own checkout. Missing or unspecified checkouts
fail with an explicit configuration diagnostic.

An explicit job ID (positional or `ROSIE_JOB_ID`) retains compute-node hosting
with `srun --jobid ... --overlap`, using that job's working directory and SSH
`ProxyJump` through the management host. It must be running and submitted with
`rosie/train.sbatch`, regardless of job name; pending/unrelated jobs fail.
No additional allocation is submitted, and stopping the dashboard does not
cancel training.

Both ports are selected automatically and bound only to loopback. The default
tunnel goes directly to the assigned `dh-mgmt1` through `dh-mgmt4` host; do not
expose its port to the network. The script verifies an HTTP run-list response
through the tunnel before printing/opening the local URL. The server's SSH
connection owns a remote PTY so closing it terminates the dashboard. Ctrl+C
stops both SSH connections; losing the server also ends the tunnel with an
error. Run the command locally to put the forwarded localhost port on your
computer, not in a remote management-node terminal.

If `ROSIE_SSH_JUMP` is unset, the script derives the host when run on
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

Connect GlobalVPN. Minecraft remote operations pass `--connect` to the helper:
it health-checks and reuses an existing matching session. On Windows, if none is
working, it opens the configured Python `connect` command in a new visible console
and waits up to two minutes for login. Closing that console before connection
fails the operation; a timeout leaves it open so login can finish before retrying.
The terminal stays open across completed operations and is separate from the
inference process. Stop/disconnect does not close an already established terminal
session. Close it explicitly with Ctrl+C when finished.
On other platforms, open
`scripts/rosie-play.ps1 -Connect -MinecraftDirectory <game>` in a **local terminal**
first. This manual command remains available on Windows too.
Python starts interactive native SSH (`BatchMode=no`) with
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

Launcher `--connect` operations reopen stale sessions in a visible Windows terminal,
never as a hidden password prompt. A connection fingerprint mismatch fails rather
than replacing another configuration's session. Close the old connection before
changing setup settings. Direct CLI calls without `--connect` retain their existing
behavior: stale connections fail with `-Connect` instructions, and with no descriptor,
key/agent use still opens noninteractive SSH
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
`replay`, `sync`, `duel`, `local`. `connect` requires a terminal, not `--json`/`--offline`.
`--connect` enables automatic terminal startup/reuse for remote actions; offline
and local actions never launch a connection. Health probes time out after two
seconds; artifact requests retain their ninety-second timeout.
`--run` and `--replay` default to `latest`; `--offline` uses only
cached artifacts. `--json` emits newline-separated events for the mod. Run/replay
basenames are limited to 120 ASCII letters, digits, dots, underscores or hyphens,
starting with a letter/digit. Paths and remote arguments are encoded, not shell
interpolated; resolved remote files must remain inside the configured run root.

`replay` downloads only the selected sample to game `sim-replays`, prefixing its
filename with the run ID. `duel` caches `latest.pt` and available config/metrics
in game `sim-rosie-cache/<run>`, then serves CPU inference on an automatically
allocated loopback port. `sync --scope policy` (default) caches the latest policy,
retained snapshots and config; `--scope latest` adds the latest replay, and
`--scope all` includes all replays and metrics. `--workers 1..4` controls parallel
downloads (default 3), including requests through the terminal tunnel. Progress
events include aggregate bytes/files/cache hits, active filenames and preparation,
transfer, verification and indexing phases. The index preserves local display
aliases for runs and policy versions. Transfers reuse
matching size/mtime files, stream to unique `.part` files, verify length/SHA-256,
then atomically replace destinations. A failed transfer does not replace an
existing artifact. Remote reads pin a file descriptor across atomic trainer
checkpoint replacements. Replay limit is 64 MiB, other artifact limit 512 MiB.

For offline duels: `scripts/rosie-play.ps1 -Action duel -Run <run> -Offline`;
use `/simduel arena <printed-port>`. Ctrl+C stops the terminal-owned service.
Cached replays can be loaded normally with `/simreplay load latest`. The cache
retains explicit named policy snapshots, but syncing newer weights replaces
cached `latest.pt`; neither local nor remote `latest` means best-performing.
Library duels use action `local --duel <file> --offline`, without SSH access.
The mod can reload that fixed configuration in the same authenticated local
process to avoid repeated PyTorch imports; cold startup still imports/loads once.

Tests: `python -B -m unittest simulator.training.test_rosie -v` uses local
subprocesses to exercise the actual remote reader; library-index tests also need
the training Python dependencies. The existing
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