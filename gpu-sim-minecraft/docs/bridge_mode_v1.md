# Bridge Mode Specification, Version 1

## Status And Scope

This document defines a proposed two-player Bridge-style mode for the
Minecraft Java 26.3 simulator. It is a contract for future implementation,
not evidence that Bridge, its setup fields, or its live controller exist.
Rules marked provisional are configurable design choices, not verified
Hypixel behavior. Vanilla 26.3 movement and combat are not 1.8.9 parity.

Confirmed simulator foundations are two players per world, a world-axis
coordinate system, shared CPU/CUDA tick mechanics, finite ordinary arrows,
and terminal-aware PPO. The existing duel episode ends on death or leaving
its bounds; Bridge must use its own lifecycle, not silently reuse that rule.
See [native tick mechanics](../simulator/include/minecraft_sim/core.hpp),
[projectile ordering](../simulator/include/minecraft_sim/projectiles.hpp),
and [PPO advantages](../simulator/training/ppo.py).

## Provisional Rules

| Setting | Default | Meaning |
| --- | --- | --- |
| Teams | 2, one player each | Player index 0 is team 0; index 1 is team 1. |
| Goals to win | 5 | A team scores by entering the other team's goal. |
| Match clock | 18,000 ticks | Includes cage ticks; higher score wins at expiry, tie draws. |
| Cage | 60 ticks | Applied at match start and after each nonwinning goal. |
| Terrain on goal | Persist | Optional explicit rule clears all placed blocks instead. |
| Kit | Sword, bow, pickaxe, 64 blocks, 2 golden apples, 1 arrow | Item types, slots and counts are explicit per-player setup data. |
| Arrow refill | 80 ticks while empty | Restore one arrow; reset the timer on refill or kit reset. |
| Death respawn | Immediate | Restore only the dead player at the end of the tick, without a new cage. |

After the arrow becomes empty, count consecutive subsequent match ticks with
no arrow; reaching the configured refill count restores one arrow at that
tick's end. A positive arrow count clears the timer. A kit reset restores the
configured count and clears the timer. A refill value of zero disables refill.

The actual Hypixel cage duration, kits, arrow timing, map geometry, build
rules, death/goal precedence, simultaneous scores and timeout outcome remain
unverified. The defaults above are not captures or source-derived claims.
Coordinates, goal and cage regions, void height, build limits, item IDs and
spawns must be supplied by each arena/environment; there is no universal
Hypixel map implied by this contract.

## Coordinates And Terrain

World coordinates are absolute X/Y/Z, with Y vertical. Integer block
coordinates identify unit cubes `[x,x+1) x [y,y+1) x [z,z+1)`. Bounds,
build zones and goal/cage regions are axis-aligned half-open boxes with finite
coordinates and strictly increasing limits. A player's body overlaps a goal
only when their AABB has positive-volume intersection with that goal; touching
a face alone is not a goal. Player 0 and player 1 use their own configured
spawn and defend their own goal. Team-relative observations must identify self
and opponent consistently; world geometry itself is not rotated or mirrored.

The arena defines a finite base grid, protected solid cells, two non-overlapping
goal regions, two cage regions, legal build volumes and a void Y threshold.
Spawns must be finite, collision-free, inside the intended cage and outside
both goals and lethal void. Each world shares an immutable protected base and
owns an independent editable overlay with empty, team-0 and team-1 placed
cells. A composite cell is solid when its base or overlay is solid. Reading
outside the base grid returns air; placing or breaking outside it is rejected.
Base solids cannot be placed over or broken; breaking a placed block clears
only that world's overlay. Placement must also respect build zones and player
collision. Same-cell edits are applied in player-index order, with each edit
validated against the result of earlier edits. The complete rules for block
targeting, repeat timing and tools remain later implementation work.

Overlay ownership and lifetime are per world, including on CUDA; the shared
base must outlive every view. A full reset clears only the selected overlay.
The implementation must check dimension multiplication and allocation sizes;
the existing scene limit still applies until the separate map-budget task.

## Tick And Outcome Ordering

Both players receive observations of the same pre-tick world. One match tick
uses the following provisional order, with existing 26.3-subset player and
projectile mechanics retaining their established internal ordering:

1. Select both controls. Apply slot selection before attack/use; cage-locked
   players may change look but cannot move, attack, use items or change slots.
2. Resolve enabled item and terrain actions in stable player-index order,
   then run the shared player and projectile tick. New building mechanics must
   specify their precise arbitration within this phase before implementation.
3. Determine both players' death/void status and goal intersections from the
   resulting state. A dead or voided player cannot score on that tick. Each
   living player intersecting the opposing goal scores once; apply both scores
   together before checking a winner.
4. Increment the match clock once, including while caged. Resolve a winning
   score first: if both reach the winning score on the same tick, the result
   is a draw; otherwise the qualifying team wins. If nobody wins by score and
   the clock has reached its cap, the higher score wins or equal scores draw.
5. On match end, emit the outcome and final observation without respawning or
   resetting. Otherwise, a goal causes one round reset of both players; if a
   death also occurred, emit the death event but do not perform a second reset.
   With no goal, respawn each dead player independently. Sample the resulting
   observation and emit score/death events before the next controls are chosen.

A cage locks exactly the next configured number of complete match ticks after
match initialization or a round reset. A duration of zero opens immediately;
the first unlocked controls apply on the next tick after the count reaches
zero. The clock does not pause. A death-only respawn does not restart the cage.
Goal detection is a post-tick AABB test, not a swept crossing test.

Examples: at 4-4 with a target of 5, simultaneous valid goals yield a 5-5
draw. If player 0 dies while intersecting player 1's goal, player 0 scores
nothing; player 1 may still score. At tick 18,000 a valid winning goal takes
precedence over the clock; otherwise the updated scores decide the result.

The match clock is part of the task: its result is `terminated`, and PPO does
not bootstrap it. A separate externally imposed training cutoff is
`truncated`, with bootstrap from the final pre-reset observation. A death or
nonwinning goal alone is neither. If both an external cutoff and a match end
occur on the same tick, match termination takes precedence.

## Reset Matrix

| State | Individual death | Nonwinning goal | Full match reset |
| --- | --- | --- | --- |
| Players | Restore only dead player at configured spawn/orientation; preserve the other | Restore both at their spawns | Restore both in the selected world |
| Health, effects, kit | Restore affected player to configured initial kit and health; clear transient effects | Restore both kits and transient player state | Restore both initial states |
| Controls and timers | Clear affected player's queued/held controls, item-use and refill timers | Clear both players' controls and timers | Clear both players' controls and timers |
| Projectiles | Remove projectiles owned by the dead player; preserve the other's | Clear all in the world | Clear all in the selected world |
| Scores and match clock | Preserve | Preserve | Reset to zero |
| Cage | Preserve existing phase; do not start a new cage | Start a new cage | Start initial cage |
| Terrain | Preserve | Preserve by default; optional rule clears overlay | Clear selected world's overlay |
| RNG | Continue existing per-world/player sequences | Continue existing sequences | Advance selected world's deterministic episode seed; do not affect other worlds |

Refill restores configured counts, including the initial arrow, and restarts
its timer. Spawn velocity and transient motion are reset to configured initial
values. Match-end observations and events must be consumed before a full
reset overwrites them. A reset of one world must not alter another world's
terrain, players, projectile state, timers, score or RNG.

## Observability And Setup Ownership

The proposed observation contract may expose each team's score, remaining
match/cage time, own inventory and refill state, visible player equipment,
and explicitly defined local terrain. It must not expose hidden opponent
inventory. The following structured observations are provisional design
choices, not accepted policy descriptors. Their geometry and omniscient
visibility are not claims about Hypixel or the current catalogue.

#### Experimental Host Observation Foundation

The experimental host trainer accepts the following exact descriptor in the
top-level `bridge_observation` field of an authored version-2 plan or a
resolved experiment. Omit it for the unchanged scalar model and fingerprints.
This is separate from the scalar policy contract, which still rejects that
field. Unknown fields, versions and altered values are rejected.

```json
{
   "format": "minecraft_sim_bridge_observation",
   "version": 1,
   "native_capabilities": {"bridge_host_batch_version": 2, "bridge_local_voxels_version": 1},
   "voxel_shape": [5, 9, 9],
   "voxel_dtype": "uint8",
   "voxel_categories": 5,
   "match_features": ["own_score", "opponent_score", "match_remaining", "cage_remaining",
      "own_blocks", "own_golden_apples", "own_arrows", "own_refill_progress",
      "refill_enabled", "own_selected_slot"],
   "match_scales": ["goals_to_win", "goals_to_win", "max_ticks", "max(1,cage_ticks)",
      "64", "64", "64", "max(1,arrow_refill_ticks)", "1", "8"],
   "voxel_perspective": "actor_feet_yaw_v1",
   "sampling": "current_host_state",
   "model": "bridge_voxel_actor_critic_v1"
}
```

All stages and evaluations use fixed Bridge scenarios; their arenas, match rules,
and task rewards may differ. Host collection supports mixtures of
`stationary_target`, `bridge_fixed_lane`, compatible structured
checkpoint, current-policy and frozen self-play opponents. Mixed Bridge/duel
plans, resident execution and live inference remain unsupported. The model's
descriptor is saved in checkpoint and snapshot metadata and v4 replay provenance.
Resume restores the model, optimizer, RNG and sampler/pool/reset counters, then
starts fresh episodes rather than restoring in-flight matches. Initialization and
resume reject scalar/structured mismatches; scalar consumers and live serving
reject structured checkpoints. Catalogue identity remains part of the separate
scalar policy contract.

Host self-play controllers receive the three structured inputs for the opponent's
own seat. Structured sampling uses checkpointed per-world RNG streams, and
frozen snapshots stay pinned until that world resets. Evaluation uses the latest
compatible snapshot supplied by the source pool and records its update, content
SHA-256, fixed actions and controller seed; a missing or incompatible snapshot
is an error. Replay verifies and records the assigned snapshot update, content
hash, fixed actions, controller seed and actor slot, and starts a fresh opponent
stream from that actor's seed. Replay
does not recreate the in-flight match or the advanced training RNG state.
Scalar self-play retains its existing sampling behavior.

The standalone host sampler takes the configured scalar observation
`float32[world,2,observation_dim]`, and returns copies of those scalars,
`uint8[world,2,5,9,9]` local voxels as specified below, and
`float32[world,2,10]` match features. Select one player's row from each tensor
before presenting inputs to an actor. The sampler reads current host state;
call it before reset to retain a final terminal or truncated observation.
It requires native Bridge host batch version 2 or later and local voxel version
1. The model one-hot encodes the five voxel categories, applies a 3D convolution
and global average pooling, and concatenates the result with the scalar and
match features. No global lane tensor is included.

The ten match features, in order, are own score and opponent score divided by
`goals_to_win`; remaining match ticks divided by `max_ticks` (remaining ticks
floored at zero); remaining cage ticks divided by `max(1,cage_ticks)`; own
blocks, golden apples and arrows each divided by 64; own arrow refill elapsed
ticks divided by `max(1,arrow_refill_ticks)`; 1 if arrow refill is enabled,
otherwise 0; and own selected slot divided by 8. Values are not clipped. Only
the actor's own inventory and refill progress are included. The scalar catalogue
continues to govern separately configured visible opponent features.

Version 2 retains those ten values and appends the actor's active golden-apple
eating progress and mining progress, yaw and pitch error and distance to the
center of the opposing goal, and use and mining cooldown fractions. Eating is
elapsed ticks divided by 32 while active, otherwise zero; mining is the current
breaking fraction. Angles are divided by 180 and 90 degrees respectively, and
distance is clipped at 50 blocks and divided by 50. Use and mining delays are
divided by four and five ticks. Only the actor's own item state is exposed.
The v2 model has 17 match features and cannot load v1 weights. Its actor pose and
item-state API and objective geometry encoder are task-neutral, and the shared
native item tick can be called without Bridge match rules. Ordinary authored duel
environments do not currently enable these item actions or automatically append
the task features to their policies.

Optional `task_rewards.opponent_score` pays when the opposing player scores.
Nonnegative `toward` and `away` independently weight distance progress and
retreat per 50 blocks; `away` is subtracted. Toward rewards only improvement
beyond the actor's previous best distance to the current goal. Death and round
teleports never pay progress, and a death does not reset that best distance;
scoring starts a new progress budget for the scorer. Positive-only shaping is
not potential-based and may change the optimal policy. Curriculum gates still
use actual goal or match outcomes, not shaped return. The four per-rollout score,
opponent-score, toward and away contributions are logged separately.

Generic combat and shaping rewards may also be specified in `task.rewards` or
replaced per stage or evaluation with `task_override.rewards`. They add to the
Bridge task rewards; `won_duel` and `win_speed` are rejected. Damage events are
sampled before respawns. Death pays once per death, while timeout and no-damage
penalties apply only when the match clock or external cutoff expires. The
no-damage flag spans the entire match. Shaping uses the external tick limit as
its budget and is skipped on terminal, truncation, respawn and goal-reset ticks.

### Initial Bridge Terrain

An authored Bridge environment may opt in to `terrain_generation`, for example:

```json
"terrain_generation": {"center_x": 0, "floor_y": 0, "start_z": -7, "end_z": 8, "segments": 4}
```

See `simulator/setups/environments/bridge/progression-narrow-generated.json` for
a complete example. This rule requires Bridge `building` and its explicit kits.
It supports a straight, one-block-wide protected center lane along the Z axis,
from `start_z` inclusive to `end_z` exclusive. Both side columns out to two
blocks from the center must be empty at `floor_y` and the level above, inside
build zones and arena bounds, and clear of both goals and player spawns.
`segments` is an integer from 1 to 16. The other fields are bounded integers;
the Z span must be 1 to 10000 blocks.

Each match reset creates short, connected side extensions, sometimes two blocks
wide with a rail. The original center lane stays clear. Blocks are ordinary
team-owned overlays and can be mined; players do not pay for them from their
kits. Generation uses a separate deterministic random stream derived from the
batch seed, world and reset counter. Selective resets affect only selected
worlds, and restored reset counters reproduce the next terrain. Player respawns
preserve the overlay. A nonterminal goal clears generated and player-placed
blocks when `persist_blocks_on_goal` is false; it does not regenerate them.
Full match resets regenerate the terrain. CPU and host-staged CUDA use the same
generator; CUDA-resident Bridge remains unsupported. Native libraries advertise
`bridge_terrain_generation_version: 1`.

### Proposed Structured Terrain Observations

The two observations below are independently optional. Each actor receives its
own perspective of the same per-world composite terrain snapshot. Shapes below
are per actor, without batch or player axes; a two-player batch may present
them as `[batch, 2, ...]`, with player 0 then player 1. Axis order is exactly as
listed, and the last index varies fastest when flattened. These tensors are
separate from scalar inputs: the current scalar policy contract allows at most
128 expanded inputs and cannot simply flatten these grids into that contract.
Their descriptors, transport, model projection and compatibility identity
require later coordinated work. Unsupported consumers must reject them.

Sample both players from one consistent pre-control, pre-tick snapshot, after
any prior tick's ordered edits, scoring and applicable respawns or round reset.
At match initialization sample after initialization; at match end sample the
final post-tick state before a full reset overwrites it. An external truncation
similarly retains its final pre-reset snapshot. Sampling must not mutate terrain
or let one actor's sampling see a different tick than the other's.

Both views are terrain-state observations, not camera visibility: they include
occluded blocks, including blocks behind walls. The local view is omniscient
within its window; the lane view is **global and omniscient** over the entire
finite base grid, including beyond the local window. Neither reveals entities,
items or hidden opponent inventory. A future live adapter must obtain the same
authoritative terrain and placed-block ownership, including unloaded or
otherwise unavailable areas, or reject the observation; it must not replace
unknown data with air or claim camera-limited equivalence. These proposals
cover the Bridge full-cube terrain subset only; unsupported block states or
unknown ownership must be rejected, not silently mapped to a category.

For both tensors, a protected base solid has no team owner. A placed overlay
solid belongs to the team that placed it and is `self` for that team and
`opponent` for the other. The base is immutable, each overlay is per world,
and an overlay on a protected base is invalid. The categories describe the
composite after all prior ordered edits; empty in-bounds cells are air.

#### Local Voxel Window

Proposed shape is `[5, 9, 9]`, axes `[up, right, forward]`, with `uint8`
category values. Indices `(u, r, f)` have offsets `(u-2, r-4, f-4)` in blocks.
The origin is the integer block containing the actor's feet position:
`(floor(x), floor(y), floor(z))`, with mathematical floor even for negatives.
Interpret finite feet coordinates as their exact stored binary64 values and
perform cell-coordinate arithmetic without overflow.
Each output is exactly one whole world cell, not a resampled volume or ray.
Pitch is ignored. The actor's finite, unwrapped yaw is the stored binary32
degree value, interpreted exactly; quantize to the nearest multiple of 90
degrees, with exact halfway ties toward increasing yaw:
`q = floor(yaw / 90 + 1/2) mod 4`, using mathematical arithmetic and a
nonnegative modulo. Thus 45 degrees selects 90, while -45 selects 0. No
epsilon, implementation-dependent trigonometry or integer overflow may change
the result, including for large finite yaw values.

| `q` | Forward `(dx, dz)` | Right `(dx, dz)` |
| --- | --- | --- |
| 0 | `(0, 1)` | `(1, 0)` |
| 1 | `(-1, 0)` | `(0, 1)` |
| 2 | `(0, -1)` | `(-1, 0)` |
| 3 | `(1, 0)` | `(0, -1)` |

For origin `(ox, oy, oz)`, the sampled cell is
`(ox + (r-4)*right.x + (f-4)*forward.x, oy + u-2,
oz + (r-4)*right.z + (f-4)*forward.z)`. The array rotates, never the world.
Team perspective changes ownership labels but does not add a team rotation;
the player's own yaw determines the frame.

| Value | Cell category |
| --- | --- |
| 0 | In-bounds air |
| 1 | Protected base solid |
| 2 | Self-team placed solid |
| 3 | Opponent-team placed solid |
| 4 | Outside the finite base grid |

Value 4 is an observation boundary marker, not a collision solid: the terrain
contract still reads outside as air and rejects edits there. Do not clamp,
wrap or treat an outside cell as in-bounds air. Grid bounds for this proposal
are integer half-open block coordinates `[minX,maxX)`, `[minY,maxY)`,
`[minZ,maxZ)` with positive extents. Out-of-bounds applies on any axis.

#### Global Lane Summary

Proposed shape is `[3, 4, 7, 16]`, axes `[category, up, right, forward]`,
with unsigned `uint32` raw counts, not normalized fractions. Category channels
0, 1 and 2 count protected base, self-placed and opponent-placed solid cells,
respectively. Air contributes no count; each valid solid contributes exactly
one count to exactly one bin. There is no outside channel: only finite-grid
cells are iterated. Bins without cells or solids are zero. With the current
262144-cell ceiling, even a bin containing the entire grid fits `uint32`.
This summarizes block counts, not occupied fraction, highest surface, goal
state, entities or line of sight; arena dimensions are needed to distinguish
empty bins from bins containing only air.

The fixed frame is independent of player position, yaw and pitch. Let `D` be
the horizontal vector from the center of the actor team's own goal box to the
center of the opposing goal box. Select the signed world cardinal axis with
the larger absolute component of `D` as forward; on an exact nonzero tie
select X, and if both components are zero select +Z. Goal centers and
comparisons are the exact mathematical values of the configured finite
coordinates, with no epsilon. Set right `(rx, rz) = (forward.z, -forward.x)`
and up to +Y. This is a rotation by a multiple of 90 degrees, not a reflection
or a rotation of arena geometry. Each team computes its own frame; opposite
nonzero goal vectors give opposite forward and right axes.

For each in-bounds integer cell, convert its X/Z coordinates to zero-based
oriented offsets. For an oriented axis along +X use `x-minX`, along -X use
`maxX-1-x`, along +Z use `z-minZ`, and along -Z use `maxZ-1-z`. Its extent is
the corresponding X or Z grid extent. Let these offsets and extents be
`(R, W)` for right and `(F, L)` for forward; let `U = y-minY` and
`H = maxY-minY`. The cell's indices are
`(floor(4*U/H), floor(7*R/W), floor(16*F/L))`. All calculations are exact
nonnegative integer arithmetic with sufficient width; a bin boundary goes to
the index given by the formula, never to two bins. Small extents can leave
some bins empty. No cells outside the grid contribute, even though outside
terrain reads as air. Cell iteration order must not affect these exact counts.

#### Worked Examples

- Local window: feet `(-1.2, 64.7, -2.01)` give origin `(-2, 64, -3)`.
   Index `(2, 5, 6)` has offsets `(up=0, right=1, forward=2)`. At yaw 0,
   right is +X and forward +Z, yielding cell `(-1, 64, -1)`. At yaw 45,
   the halfway rule selects 90: right is +Z and forward -X, yielding
   `(-4, 64, -2)`. Yaw 90 yields that same cell. At yaw -45, the tie selects
   0 and yields `(-1, 64, -1)`. Lookups use the category of each resulting
   cell, or 4 if any coordinate lies outside the grid.
- Local boundary: for the grid in the next example, feet `(-6.2,-3.1,-15.1)`
   have origin `(-7,-4,-16)`. At yaw 0, index `(2,3,4)` samples `(-8,-4,-16)`
   and is 4; `(2,4,4)` samples the in-bounds origin and is 0 if that cell is
   air. The grid's lower boundary is inclusive.
- Lane: let grid bounds be `[-7,7) x [-4,4) x [-16,16)` and goal centers
   `(0,0,-12)` for team 0 and `(0,0,12)` for team 1. A team-0 placed block
   at `(-6,-3,-13)` has, for team 0, `(U,R,F) = (1,1,3)` and extents
   `(H,W,L) = (8,14,32)`, giving bin `(0,0,1)` and channel 1. For team 1,
   forward is -Z and right is -X: `(U,R,F) = (1,12,28)`, giving bin
   `(0,6,14)` and channel 2. Thus the same block contributes one to
   `[1,0,0,1]` for team 0 and `[2,0,6,14]` for team 1; it is not counted
   twice within either observation. For example, `x=7` is outside the grid
   and contributes nothing to either lane summary.

The following are proposed responsibilities and fields, not accepted JSON:

| Layer | Proposed Bridge data |
| --- | --- |
| Arena | Bounds/base, protected cells, goal and cage boxes, build volumes, void threshold and safe spawns. |
| Environment | Two teams, spawn orientations, kits, goal target, clock, cage/refill timers, terrain persistence and task rewards. |
| Policy | Ordered supported observations, semantic action heads, perspective and future structured-grid descriptors. |
| Curriculum | Learner/model descriptor and options, stages, opponents, controls and evaluations. |

Proposed field shapes, subject to coordinated schema versions, are:

- Arena: `bounds` and base `blocks`, `goals` and `cages` as two ordered
   `{min: [x,y,z], max: [x,y,z]}` boxes, `build_zones` as an array of such
   boxes, and a finite `void_y`. Base solids are protected; placed cells exist
   only in per-world overlays.
- Environment: `mode: {name: "bridge", version: 1}`, two ordered players with
   absolute `spawn` position and yaw/pitch and an explicit kit, and `match`
   settings `goals_to_win`, `max_ticks`, `cage_ticks`,
   `persist_blocks_on_goal` and `arrow_refill_ticks`. Ticks and item counts are
   nonnegative integers; goals and the match cap are positive integers.
- Policy: the existing ordered scalar observations/actions plus separately
   versioned optional structured observation descriptors. Model architecture,
   factory and options are separate from the semantic policy descriptor.
- Curriculum: references to the environment, policy and learner/model plus
   stage-specific opponents, controls, evaluation and seeds. Each stage's
   resolved values, not just authoring paths, identify effective conditions.

These names are a design contract, not claims about the existing parser.
Version the Bridge mode independently from existing arena/environment/policy
envelopes, and require explicit Bridge capabilities for each implemented
mechanic, observation, action and adapter. The compiler must check the union
of requirements against each selected native, training, replay and live
consumer; a supported mode version alone must not imply all capabilities.
A future resolved experiment must embed the effective mode version, rules,
geometry, kits and policy/model identity, including capability requirements;
checkpoints and replays must retain that identity. Unsupported versions,
capabilities, invalid regions/spawns/kits and incompatible model or policy
descriptors must be rejected, never silently relabeled or defaulted to duel
behavior. Existing supported duel setups and checkpoints keep their current
interpretation. The current [compiler](../simulator/training/compiler.py) and
[plans](../simulator/training/plans.py) accept a fixed host-staged subset; the
ordinary [native scene loader](../simulator/apps/scene.hpp) deliberately rejects
Bridge configuration, which is supplied through the dedicated API. Full
capability validation remains a later checklist gate.

## Bridge Replay Contract

The host-staged website replay uses JSON Lines with `format` set to
`minecraft_sim_replay` and `version` set to 4. The first line is a header with
two players, tick rate, the initial ordinary scene in `environment`, and the
resolved Bridge configuration in `bridge`. Policy and training provenance are
included for policy samples; scripted recordings identify themselves as such.
Versions 2 and 3 retain their existing duel meaning. Unsupported replay
versions and Bridge consumers must be rejected explicitly.

Each following line is a single-tick, post-step frame with the existing actor
and projectile fields and a `bridge` object. An initial tick-zero frame is
recorded before controls are applied. The Bridge object contains `scores`,
`ticks`, `cage_remaining`, `outcome`, `terminated`, `truncated`, `scored`,
`died`, `placed`, `broke`, `respawn` and `round_reset` as in the native Bridge
state API. Its `terrain_edits` array contains the successful edits for that
tick in player order, each with `player` (0 or 1), `action` (`place` or
`remove`) and integer `position` `[x, y, z]`. `clear_overlay` indicates a
completed goal reset that clears placed blocks. Apply the edits in order, then
clear the overlay when this flag is true. The initial frame has no edits and
no clear operation. When generated terrain is enabled, it contains an
`initial_terrain` array of `{player, position}` team-owned blocks; apply this
snapshot before subsequent edits. Older replays omit this optional field.
Simultaneous goals are valid. A terminal or truncated
frame preserves the final pre-full-reset state; playback ends there.

Frame ticks increase by one; recorders must not discard intermediate events
when a policy holds an action across multiple ticks. A viewer reconstructs
the selected frame from the initial base terrain and ordered changes, including
after backward seeks and restart. Respawn and round reset are discontinuities,
not movement to interpolate through. Files retain the dashboard's 64 MiB and
100001-frame limits. This contract does not by itself imply Fabric replay,
resident collection or live Bridge support.