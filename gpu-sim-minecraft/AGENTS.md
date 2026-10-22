# Development Guide

## Documentation Standard

- READMEs are short, human-facing entry points: purpose, prerequisites, common
	commands, essential warnings and links. Aim for 30-80 lines; keep them below
	100 lines unless a necessary setup walkthrough genuinely requires more.
- Put architecture, implementation details, schemas, protocol references,
	invariants, exhaustive options and developer troubleshooting in the nearest
	`AGENTS.md`, not in a README. Link to that guide from the README.
- Keep this root guide focused on cross-project rules. Read the component guide
	before modifying its code; keep component-specific detail there.
- Preserve useful information when shortening documentation. Move technical
	material rather than deleting it, and update relative links and anchors.
- Keep commands and contracts current when changing behavior. Do not duplicate
	long technical explanations across guides. Existing specifications in `docs/`
	remain the detailed source of truth; link to them rather than copying them.
- Exclude generated builds, caches, dependencies and supplied Minecraft reference
	trees from documentation cleanups.

## Component Guides

Read the applicable guide before working in a component. This root guide applies
throughout the repository; component guides add local implementation references.

Use the [contribution playbook](CONTRIBUTING.md) for source-derived mechanics,
observations/actions, controllers, state lifecycle, rewards and verification.
For setup-only authoring, load the workspace's
[GPU simulator setup skill](../.github/skills/gpu-sim-setup/SKILL.md) before
changing definitions. It is also available as `/gpu-sim-setup` in supporting clients.

| Scope | Guide |
| --- | --- |
| Native physics, CPU/CUDA, recording comparison | [simulator/AGENTS.md](simulator/AGENTS.md) |
| Environment adapter, PPO, checkpoints, dashboard, inference | [simulator/training/AGENTS.md](simulator/training/AGENTS.md) |
| Fabric recording, replay, live duels | [cambiummod/AGENTS.md](cambiummod/AGENTS.md) |
| PowerShell launchers and option reference | [scripts/AGENTS.md](scripts/AGENTS.md) |
| Rosie training, SSH dashboards and local artifact sync | [rosie/AGENTS.md](rosie/AGENTS.md) |
| Shared policy network and reward configuration | [backend/AGENTS.md](backend/AGENTS.md) |

## Architecture And Ownership

The goal is a faithful Minecraft Java 26.3 physics subset for an RL environment.
Minecraft 26.3 is an intentional compatibility requirement; launcher names,
instance names, usernames and checkout locations are not.

- `simulator/`: C++20 physics shared by CPU and CUDA, native tools, environments
	and regression checks. Movement, collisions, targeting, damage and equipment
	mechanics belong here, not in the policy encoder.
- `simulator/setups/`: central authoring definitions in `arenas/`,
	`environments/`, `policies/` and `curricula/`. Legacy concrete scenario
	fixtures remain separate in `simulator/environments/`.
- `simulator/training/`: Python adapter over a small native C ABI, shared
	two-player PPO policy, checkpoints, dashboard and inference service. Policy
	schemas define observations and action heads separately from physics.
- `backend/`: shared actor-critic model and PPO constants. Authored environment
	tasks define rewards explicitly; omitted terms are zero. The historical
	`reward_weights.json` defaults apply only to legacy concrete fixtures.
	The legacy backend server and tooling have been removed.
- `cambiummod/`: Fabric recording, replay comparison and real-game policy actors.
	Live actors use Minecraft mechanics, not embedded simulator physics.
- `scripts/`: Windows PowerShell entry points for these workflows. Scripts
	support `-DryRun`; they do not install external toolchains.
- `recorded_sessions/`: captured sessions and comparison artifacts for validation.

The current training path stages controls and state through host memory.
Observations and rewards are computed on the CPU even with CUDA physics; this is
not a fully GPU-resident PPO pipeline.

## Behavioral References

- `minecraft-26.3/`: supplied `.class` files, assets and data.
- `minecraft-26.3-src/`: matching decompiled Java source. Inspect the controlling
	code when establishing exact Minecraft behavior.
- [Physics specification](docs/minecraft_simulation_specs.md): exact functions,
	parameters and procedures derived from Minecraft.
- [Scenario contract](docs/simulation_contract.md): portable experiment format.
- [Environment design](docs/environment_interface_next_steps.md): RL interface
	and validation direction.

## Cross-Component Contracts

- Training authoring separates arena geometry, environment tasks/initialization,
	policy layout and version-2 training plans. Paths resolve relative to the
	file containing them. The compiler produces one self-contained resolved
	experiment for training, checkpoints and the dashboard.
- Training plans provide scripted/frozen checkpoint/self-play opponent pools, constrained
	curricula and reset-time scenario sampling. The shared catalogue defines 89
	named features, including world-axis velocity in blocks per tick; relative
	velocity is opponent minus self. Equipment features include shield presence,
	own slot metadata and visible opponent equipment (not hidden opponent inventory).
	Legacy 22/31-feature presets remain isolated for historical artifact readers
	and regression fixtures, not a second public training configuration path.
	Version-1 authoring plans must be rewritten; old checkpoint inference/replay
	remains supported. Semantic descriptors are the common execution path.
- Custom experiments select/reorder observations and action choices, weapon
	slots and per-player armor/shields in JSON without a new preset ID.
- Supported layouts/loadouts are data-driven across training, replay and live
	inference. Install the updated mod once; configuration-only changes need no
	rebuild. Checkpoints and replays carry resolved experiment conditions.
- Native armor and opt-in food/regeneration APIs do not imply adapter support.
	The portable scenario contract requires frozen food and disabled natural
	regeneration. Fluids, special blocks, inventory/durability, enchantments and
	general world simulation are outside the implemented scope.

## Workflows And Validation

- Run `./scripts/check.ps1 -Build` for native and Python checks.
- Use `-Scope Native|Training|Policy|Rewards|Dashboard|Mod` for focused checks;
	see the [validation matrix](simulator/training/AGENTS.md#extension-recipes).
- Root [CI](.github/workflows/checks.yml) runs CPU/native/Python, Java fixtures
	and the HTTP/replay bridge. CUDA and in-game checks remain separate gates.
- Source-derived regression checks, CPU/CUDA parity and real-game captures are
	separate validation layers. Passing one does not prove complete game parity.
	Physics benchmark throughput is not end-to-end PPO throughput.
- Recreate `.venv`, native build directories and Gradle caches per machine. Do
	not copy generated toolchain/build folders. Runs can be shared separately.
- `latest` selects the most recently modified checkpoint, not the best policy.
- Scripted replay export is not a learned policy sample; configure trainer
	replay exports for learned-policy samples.
- Minecraft output selection uses `-MinecraftDirectory`, then
	`MCSIM_MINECRAFT_DIRECTORY`, then a single detected standard launcher folder.
	Multiple candidates require a choice; never assume an instance named `26.3`.
- Live inference binds to loopback and loads trusted checkpoints. Use private,
	disposable singleplayer worlds for mod experiments.

## Shared Changes And Collaboration

- [policy_catalogue.json](contracts/policy_catalogue.json) owns primitive names,
	value kinds, capabilities and limits. Python reads it; C++/Java tables are
	generated by `python scripts/generate-contract.py`. Never edit generated
	tables manually. Commit catalogue, generated outputs and fixture updates
	together; CI runs `--check` without writing files.
- Follow the [extension recipes](simulator/training/AGENTS.md#extension-recipes)
	before adding primitives or changing rewards, schemas, checkpoints or replay.
	Catalogue generation does not implement mechanics or prove parity.
- Use one worktree/build tree/run directory per contributor or agent. Agree
	on ownership before concurrent edits to the catalogue, ABI or shared schema.
	Do not share mutable CMake/Gradle caches or training outputs across worktrees.
- Keep contract changes in reviewable PRs with all affected adapters and focused
	tests. Split unrelated cleanup from behavior changes; do not revert another
	contributor's work. Record commands, results and untested gates in the handoff.