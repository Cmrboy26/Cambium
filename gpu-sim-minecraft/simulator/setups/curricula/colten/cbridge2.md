# CBridge2

Start from the copied CBridge1 weights at the final stage with:

```sh
bash rosie/submit.sh --initialize simulator/setups/curricula/colten/cbridge1-latest.pt --start-stage obstacle-building
```

Select `colten/cbridge2.json` in the submission wizard. Initialization starts a
new optimizer and progress history; selecting the plan without these arguments
starts from fresh weights at the first stage. No job is submitted until the wizard
is confirmed.

Each stage gives both players -5 for a draw, plus -0.1 on a match-clock
timeout or external cutoff. Wins earn up to 0.1 extra for finishing early. These
are additive with scoring and other rewards. Rebuild the native simulator and
start a new run rather than resuming an older CBridge2 run.

The final stage samples past self, a jumping runner, a jumping block placer, and
the frozen CBridge1 checkpoint with equal probabilities. Separate evaluations
against the block placer and CBridge1 report per-seat results but do not gate
completion. The randomized top pattern selects each bridge column independently,
so occasional obstacle-free samples are possible. Side patterns create wings.
See [Bridge terrain patterns](../../../../docs/bridge_mode_v1.md#initial-bridge-terrain)
for the format and composition rules.
