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
choices, not implemented primitives or accepted policy descriptors. Their
geometry, dimensions and omniscient visibility are assumptions to agree on
before implementation, not claims about Hypixel or the current catalogue.

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
interpretation. The current [compiler](../simulator/training/compiler.py),
[plans](../simulator/training/plans.py), [scenario contract](simulation_contract.md)
and [native scene loader](../simulator/apps/scene.hpp) do not accept these
proposed Bridge fields. Implementation and positive setup validation belong to
later checklist tasks.