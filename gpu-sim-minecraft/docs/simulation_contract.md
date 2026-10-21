# Research Experiment Contract

## Design

Authoring separates four responsibilities:

1. **Arena** (`minecraft_sim_arena`, version 1): bounds, ordered block fills and
   physical rules. No gear, observations, rewards or learner settings.
2. **Environment** (`minecraft_sim_environment`, version 2): arena selection,
   player initialization, gear/spawn distributions and task limits/rewards.
3. **Policy** (`minecraft_sim_policy`, version 1): ordered observations and
   semantic action choices, independent of which items an episode samples.
4. **Training plan** (`minecraft_sim_training_plan`, version 2): learner,
   controllers, named opponent mixtures, curriculum stages and evaluation.

References resolve relative to the file containing them. The compiler resolves
these into a self-contained `minecraft_sim_experiment`, version 1, before native
allocation. Stages independently resolve their declared environment and typed
patches; they do not inherit the previous stage. Arrays replace, unknown keys
fail, and explicit null disables optional sampling.

Environment rewards are complete task definitions: unspecified terms are zero.
A stage reward patch replaces the reward map rather than inheriting hidden
global defaults. One policy layout remains fixed throughout the experiment.
Weapon choices describe nine stable hotbar slots; absent gear is air, not a
different network architecture.

Training/checkpoints/dashboard use the resolved experiment. Native physics,
replays and live Minecraft consume **concrete episodes**, using the internal
version-1 scenario envelope described below. They never chase authoring
references or execute curriculum schedules. Replays additionally carry semantic
controls and resulting state, so playback needs neither weights nor source files.
See the [training-plan reference](../simulator/training/AGENTS.md#research-training-plans).

In the [example environment](../simulator/setups/environments/example/sword-duel.json),
`players.shared` holds constant initialization and `players.overrides` contains
only each actor's spawn differences. To add gear sampling, use `equipment`:

```json
{
  "sample": "world",
  "players": "independent",
  "loadout": {
    "offhand": {"choices": [
      {"value": "minecraft:shield", "weight": 3},
      {"value": "minecraft:air", "weight": 1}
    ]}
  }
}
```

This gives each player a shield with probability 75%, sampled once per world.
Use `sample: episode` to reroll on reset; literal values do not consume RNG.
The policy still has a shield control when no shield is present, and observations
describe actual ownership. Stages patch only changed conditions, for example
`"task_override": {"rewards": {"good_aim": 5}}` for an aiming-only reward task.

The native simulator, Python trainer/service and mod implement the same bounded
primitive catalogue. Changing supported loadouts, observation order or action
bins needs no mod code. A genuinely new mechanic still needs physics and adapter
implementation. Unknown required capabilities/features/items fail explicitly.

## Catalogue And Fixtures

[policy_catalogue.json](../contracts/policy_catalogue.json) is the source of
primitive names/order, action value kinds, capabilities, armor slots and limits.
Python reads it directly. Run `python scripts/generate-contract.py` from root
to regenerate committed C++/Java constants; CI uses `--check` to reject drift.
Appending a primitive also requires handwritten encoders/decoders/mechanics.
Do not reorder existing feature/action indices or change legacy preset meanings.

[policy_fixtures.json](../contracts/policy_fixtures.json) supplies independent
base observation expected values (1e-6 tolerance), reordered/scaled projection,
all nine semantic controls and invalid descriptor mutations. Native and Java
run the observation/decoder cases; all three languages run descriptor rejection
and projection cases. Equipment extraction, raycasting and real tick behavior
need their own component/source-derived and in-game tests.

Catalogue identity is SHA-256 of sorted-key compact JSON, distinct from each
resolved policy contract's fingerprint. Current Python/native catalogues must
match before stepping; Java checks the service hash when provided. Checkpoints
remain descriptor-driven, so older supported layouts can still be served.
Native/mod source hashes identify artifacts; they are not semantic versions or
proof of game parity. See the [extension recipes](../simulator/training/AGENTS.md#extension-recipes).

## Quick Start

All authoring files live centrally in [simulator/setups](../simulator/setups/).
Start with the [setup creation guide](../simulator/setups/README.md) and adapt
[the example plan](../simulator/setups/curricula/example/melee.json).
It references only the [example arena](../simulator/setups/arenas/example/flat-duel.json),
[environment](../simulator/setups/environments/example/sword-duel.json) and
[policy](../simulator/setups/policies/example/melee.json). Keep personal definitions
in person-named subfolders; category roots are reserved for intentionally shared files.
Edit the environment to change item presence; edit the plan to change training,
not arena geometry or policy dimensions.

For a new machine, follow the [setup walkthrough](../scripts/README.md#first-time-setup)
first. Before exporting to Minecraft, [select its game folder](../scripts/README.md#minecraft-folder)
with `-MinecraftDirectory` or `MCSIM_MINECRAFT_DIRECTORY`, or use unambiguous
standard launcher detection. No fixed launcher instance name is required.
After building the native library, start training:

```powershell
.\scripts\train.ps1 -Plan simulator/setups/curricula/example/melee.json `
  -Updates 1500 -ReplayEvery 100 -ReplayToMinecraft
```

CUDA machines can add `-Device cuda -Environments 8192`.
Replay samples are separate complete evaluation
duels every N completed training rounds, with the same resolved conditions.

Build/install the updated mod once, closing Minecraft before installation:

```powershell
.\scripts\build-mod.ps1 -Install
.\scripts\serve-policy.ps1 -Checkpoint latest
```

Use `/simreplay load latest`, then `/simreplay restart` for saved replays, or
`/simduel arena` for live inference in private singleplayer. `/simduel pos1`,
`/simduel pos2`, `/simduel world` relocate live spawns onto existing terrain while
retaining checkpoint equipment/layout. Use a disposable world.

Supported configuration changes need no further mod rebuild. Start a new run for
a changed training task/layout; restart the service when switching checkpoints.
Active sessions do not hot-swap contracts, equipment or weights.
The live service exposes experiment/stage provenance in `/info`; the mod shows
that deployment context. Both live actors use the same served learner and fixed
checkpoint conditions. Scripted/checkpoint/self-play/frozen-self-play opponent pools, curriculum advancement
and reset-time distribution sampling are trainer functionality, not live-duel
features. The dashboard reads resolved experiments and metrics; it does not edit
plans or launch training.

## Scenario

The **internal concrete episode** envelope stays `minecraft_sim_environment`,
version 1, Minecraft `26.3`; it is not the version-2 environment authoring schema.
Use `profile: configured`; old profiles remain legacy defaults. Bounds, ordered
inclusive air/stone fills and two ordered players use the
[existing limits](../simulator/AGENTS.md#json-environments).

Each player's optional `loadout` resolves to an explicit shape:

```json
{
  "weapons": ["minecraft:iron_sword", "minecraft:netherite_axe", "minecraft:air"],
  "selected": 0,
  "offhand": "minecraft:shield",
  "armor": {
    "boots": "minecraft:air", "leggings": "minecraft:air",
    "chestplate": "minecraft:iron_chestplate", "helmet": "minecraft:diamond_helmet"
  }
}
```

- `weapons`: 1..9 IDs; non-air IDs must be distinct, air may repeat.
  `selected`: zero-based initial slot. Air is empty
  hand. Sword/axe materials: wooden, stone, copper, golden, iron, diamond,
  netherite. Slots are a control mapping, not a general inventory.
- `offhand`: air or shield. Equipment is unenchanted and unbreakable.
- Armor slots: boots, leggings, chestplate, helmet. Materials: leather, copper,
  chainmail, golden, iron, diamond, netherite; turtle helmet also works. Omitted
  slots resolve to air. Armor is fixed for the episode.
- Initial `health`, `absorption`, `attack_strength_ticks` default to 20, 0, 0.
  Health must be in (0,20]. Existing position/velocity/look fields are unchanged.
- Bridge `rules`: `simulate_food: false`, `natural_regeneration: false`, both
  defaulting to false. True or unknown rules fail. Additional native food/regen
  APIs are not yet part of the portable training/replay/live contract.
- `requires` supports `movement`, `melee`, `equipment`, `shield`, `armor`.
  Resolved artifacts list the currently supported catalogue.

Both live and replay actors freeze food and natural regeneration, including
Peaceful healing, without changing world rules. PvP must be enabled. Other
entities, terrain and world side effects can still affect real-game behavior.

## Policy Layout

Policy descriptor fields: `format: minecraft_sim_policy`, `version: 1`,
`minecraft: 26.3`, `ticks_per_action: 1`, `observations`, `actions`, optional
`requires`. No preset ID is needed. Dimensions are derived; supplied dimension
metadata must agree with the descriptors. Authoring policies live separately;
historical concrete fixtures may still embed a descriptor.

Observations are ordered objects like `{"feature":"self.attack_charge"}` or
`{"feature":"relative.x","scale":2}`. Scale defaults to 1, multiplying the
normalized feature; it must be finite and in -1000..1000. There are 1..128 entries;
repeats are allowed.

| Primitive | Value before optional scale |
| --- | --- |
| `self.health_fraction`, `opponent.health_fraction` | Health / 20 |
| `self.yaw_fraction`, `opponent.yaw_fraction` | Wrapped yaw / 180 |
| `self.pitch_fraction`, `opponent.pitch_fraction` | Pitch / 90 |
| `opponent.alive` | 0 or 1 |
| `relative.x`, `relative.y`, `relative.z` | Opponent feet minus own feet, divided by 10 |
| `arena.ray_world_0`, `_45`, `_90`, `_135`, `_180`, `_225`, `_270`, `_315` | Horizontal world-relative eye ray / 50; no hit is 1 |
| `target.distance_fraction` | Feet distance / 10, capped at 1 |
| `target.aim_dot` | Look dot normalized feet-relative target direction |
| `target.yaw_error`, `target.pitch_error` | Desired-minus-current angle / 180 or / 90; yaw wrapped |
| `self.weapon_is_axe`, `opponent.weapon_is_axe` | 0 or 1 |
| `self.shield_using`, `opponent.shield_using` | Held shield use, including raise delay |
| `self.shield_disabled` | Own active axe-disable cooldown |
| `opponent.shield_disabled_observed` | Own observed disable event/countdown, not the opponent's hidden timer |
| `self.shield_cooldown_fraction` | Own remaining cooldown / 100, capped at 1 |
| `self.attack_charge` | Charge at partial tick 0.5, capped at 1 |
| `self.shield_raise_fraction` | Use ticks / 5 capped at 1, or 0 when not using |
| `self.weapon_slot`, `opponent.weapon_slot` | Raw zero-based slot in that actor's loadout |
| `self.has_shield`, `opponent.has_shield` | Equipped offhand shield, whether using it or not |
| `self.weapon_damage_fraction`, `opponent.weapon_damage_fraction` | Selected weapon base damage / 10, clamped 0..1 |
| `self.weapon_speed_fraction`, `opponent.weapon_speed_fraction` | Selected attack speed / 4, clamped 0..1 |
| `self.armor_defense_fraction`, `opponent.armor_defense_fraction` | Equipment defense / 20, clamped 0..1 |
| `self.armor_toughness_fraction`, `opponent.armor_toughness_fraction` | Equipment toughness / 12, clamped 0..1 |
| `self.weapon_slot_N.present`, `.is_axe`, `.damage_fraction`, `.speed_fraction` | N=0..8; own slot presence/type/damage/speed; absent/air slots are all zero |

The additive equipment features do not change v1/v2. Research `equipment_v3`
selects 78 fixed named features, excluding hidden opponent inventory and the
legacy opponent slot index. Selected empty-hand damage/speed are 0.1/1; empty
slot metadata is zero, allowing absence to be distinguished from a weapon.

Action heads are ordered objects like `{"name":"weapon","values":[2,1,0]}`.
There are 1..9 distinct named heads with 1..64 distinct categorical choices:

| Head | Semantic choices |
| --- | --- |
| `movement` | Any subset/order of neutral, forward, forward_right, right, backward_right, backward, backward_left, left, forward_left |
| `jump`, `sprint`, `attack`, `shield`, `sneak` | Booleans in either order |
| `yaw`, `pitch` | Finite degree deltas in -180..180; actual pitch clamps to -90..90 |
| `weapon` | Integer slot indices present in both players' loadouts |

The network outputs choice indices, not semantic values. Omitted heads are
neutral; omitted weapon selection preserves the current item. Same-item selection
preserves charge; switching resets it before combat. Shield use is held, not
toggled, and does not automatically lower for attacks. Raise delay, use/attack
arbitration and cooldown still apply.

## Artifacts

Checkpoints retain the standalone PPO envelope and legacy parameter names. They
embed the original environment for compatibility, a resolved scenario with
explicit loadouts/rules, and normalized policy descriptors. Resume validates the
layout and task. With no environment override, it restores the embedded definition,
not the original path. The wrapper recovers the legacy preset, hidden size and
round limit from adjacent run metadata; share that metadata for easy resume.

Replay envelope **3** carries the resolved scenario, Minecraft-ID gear,
shield/use/cooldown state and semantic controls. Policy metadata is provenance
only. Native scripted duels and trained samples use the same equipment serializer.
The updated mod also reads legacy sword-only version 2; version 1 needs re-export.

Inference `GET /info` returns `contract`, `contract_hash`, resolved `environment`
and round limits. `POST /act` sends `[2, observation_dim]` plus protocol/session/tick
and the negotiated hash; it returns ordered categorical actions and the same tags.
The mod validates capabilities/loadouts before spawning and decodes by descriptor.
Hashes catch accidental same-size/different-layout requests, not authentication.
Custom contracts require the hash; legacy API clients may omit it for old presets.

V1 (22 observations/six heads) and v2 (31/eight) remain convenience presets. Old
standalone checkpoints load without separate replay/live implementations. Format
versions describe structural compatibility, not each experiment; supported new
descriptor values do not bump them.

## Boundaries

This is two-player air/stone full-cube melee with supported equipment, not a
generic Minecraft executor. No arbitrary blocks/items, fluids, durability,
enchantments, effects, inventory operations, projectiles or arbitrary rule changes.
World mode can expose bots to unsupported terrain; that is out-of-distribution
evaluation, not newly simulated support.

CPU regression, custom PPO/resume/export and Java/Python artifact checks are
automated. They do not prove in-game input/rendering parity. The new gear adapter
still needs Minecraft conformance captures; near-coincident knockback RNG and
server-side/network timing remain limitations. Configured CUDA execution needs a
CUDA machine. See the [verification guide](../simulator/training/AGENTS.md#verification-and-limits).