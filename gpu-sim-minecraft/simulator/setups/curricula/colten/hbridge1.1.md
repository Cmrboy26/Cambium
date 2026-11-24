# HBridge1.1

This separate, fresh-weight curriculum shares the HBridge1.0 arena, clear and
hard environments, and policy. It does not use CBridge checkpoints.

1. Clear crossing against a stationary target, without item use or building.
2. Simple generated one-block-high bridge obstacles against a stationary target,
   without item use or building.
3. Hard generated obstacles with building and mining enabled, against a walking,
   jumping block placer.
4. The same hard terrain, with an equal mixture of walking and sprinting jumping
   block placers.
5. The clear default arena, with equal frozen self-play, sprinting jumping block
   placer, and sprinting jumping attacking block placer opponents. The attacking
   variant stops to aim and swing when a living opponent is nearby and ahead;
   otherwise it resumes placing blocks and following its spawn heading.

The hard stages gate advancement on crossing against a stationary target as
well as on the scripted opponents. Stage 5 evaluates the attacking opponent and
has separate, non-gating hard-terrain evaluations. Evaluation seeds are fixed;
use additional unseen seeds for final assessment.

These are fixed-heading scripts, not obstacle navigators. They can stall or
score only once on hard terrain, and knockback can displace them off center.
Sixteen sampled simple-terrain matches were won by a jumping runner, eight in
each seat; this is not a guarantee for every generated terrain seed. The
objective-progress weight remains 3, matching 1.0 for comparison. The policy's
objective-distance observation clips at 50 blocks.
