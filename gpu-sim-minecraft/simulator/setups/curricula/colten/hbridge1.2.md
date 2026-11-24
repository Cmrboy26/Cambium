# HBridge1.2

Fine-tune from the HBridge1.1 `mixed-builders` snapshot at update 2760 using
`--initialize`. This snapshot precedes training against the attacking builder.
The policy, clear arena, Bridge task rewards, and learner settings match 1.1;
direct damage reward is zero.

The single stage trains against equal weights of frozen self-play and the
sprinting jumping block placer. The frozen opponent starts with the initialized
weights and refreshes every 20 updates. The stage requires at least 600 updates
and gates completion on clear-arena target and runner evaluations.

The attacking builder is evaluation-only and never gates completion. Its new
seed is a diagnostic rather than a pristine final holdout if its results guide
tuning. Hard-terrain target and runner evaluations are also non-gating. Normal
training metrics include goals and match outcomes; no new instrumentation is
required. Keep additional seeds and attacker variants for final assessment.
