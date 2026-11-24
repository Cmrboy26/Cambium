# HBridge1.0

HBridge1.0 trains from scratch on a symmetric Bridge-inspired arena. Both
7-by-7 spawn platforms are four blocks above their 7-by-7 lower platforms.
Each lower platform has a centered 3-by-3 goal hole and an 11-by-3 lip facing
the bridge. The protected bridge is one block wide, 41 blocks long and nine
blocks deep. The build zone extends from the bridge bottom to nine blocks above
its surface and excludes both bases and their holes.

The clear stage teaches crossing and entering the hole against a stationary
target. Later stages add generated obstacles, scripted runners, building and
frozen self-play. The final stage includes non-gating evaluations with separate
seeds; repeatedly inspecting those results is not a substitute for a final
unseen-seed test. There are no CBridge checkpoints in training or evaluation.

The straight and jumping runners reach both goals on the clear map. The simple
block-placing runner is a weak baseline on this layout and may finish on the
match clock instead of scoring five goals. Goal geometry and timings are
Bridge-inspired, not verified Hypixel replicas. The structured observation
clips objective distance at 50 blocks, so the far end of this longer arena is
initially beyond that distance.

Use the [setup guide](../../README.md) to validate and submit the plan. Start a
fresh run without initialization or resume; the fixed policy layout does not
mean that older CBridge weights are used.
