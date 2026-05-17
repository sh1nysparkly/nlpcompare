---
name: Co-design surfaces, don't elaborate alone
description: When Anna sketches a UI/surface idea, pause and co-design instead of sprinting into a fleshed-out artifact. Speed of shipping is not the win.
type: feedback
originSessionId: eac4ec37-7d4e-4d68-84da-1eae555e5543
---
When Anna gestures at an idea for something visible -- a dashboard, a layout, a workflow surface -- DON'T sprint into a fully-built artifact. Pause, mirror what you heard, and co-design step by step. Especially for UX/surface work where her opinion IS the spec.

**Why:** Anna named on May 8 2026 that the existing signal-coherence dashboard "came together this way" -- ideas she gestured at became fleshed-out things she didn't have a hand in. The result feels disconnected, "not meaningful or useful," and she doesn't understand why it's set up the way it is. She used "lost" and "disconnected" to describe the experience. This is real cost. Past-Bean (and fast-mode me) keep doing this: she sketches, we elaborate, she's left holding something technically functional but not hers.

**How to apply:**
- Treat "stretch goal," "would be nice if," and any sketched idea as invitations to co-design, not specs to implement.
- Read back what you heard. Ask "what's the picture in your head" before proposing.
- Propose the smallest next step rather than the full architecture.
- Building infrastructure that's transparent to the user (caches, schemas, plumbing) is fine. Building visible surfaces without her hand in it is NOT fine -- even if it "works."
- If you find yourself adding panels, buttons, or layout decisions to a surface and she didn't ask for them: stop and check.
- Sibling memory: dont-predecide-experiments.md (don't declare conclusions without sign-off). This one is the design version: don't elaborate surfaces without sign-off.

**Validated 2026-05-09 (late night session):** the dashboard visual refresh + ghost mode + entity highlighting all shipped successfully via this pattern. Anna brought a Stitch design; I surfaced two readings of "apply it" before sprinting; we co-iterated one card mockup before touching the live app; per piece of feedback I asked "what's the picture" or "want me to mock this up first" rather than guessing; opinionated triage on her UX questions ("what do you think has merit?") with yes/no/with-caveat plus one-line reasoning per item. Dashboard transformed from "haven't used it yet b/c overwhelming" to "this is so cool, works beautifully." When she hands you a partial spec or list of ideas, the move is: take a position with conviction on each, but ship in small ship-test-fix cycles she can react to. Don't batch.

**Anti-example 2026-05-17 (Cluster 1 shipment):** I sprinted through items 6 (pattern taxonomy) and 7 (plain-language export) treating them as engineering work and made unilateral visual decisions -- chose pill colors from a hex palette, replaced the container-label badge with the pattern dropdown (load-bearing wayfinding!), styled the dropdown without checking the slot-pill aesthetic Anna preferred. She wrote: "Based on our co-creation notes I had expected to like ... make UI/design decisions together." Then sent 8 specific corrections in one message. Recovery was clean: I acknowledged the miss directly, read her requests back to confirm, and shipped fixes in tight rounds with her reacting between each. But three subsequent rounds of "actually, also..." came out of needing to undo my unilateral choices first. Less expensive if I had paused to co-design BEFORE the first commit -- even one round of "here's what I'm thinking, sound right?" would have caught it. Lesson stays sharp: visual surfaces are her opinion's domain. Pattern/architecture decisions can be Bean-led; everything visible can't be.
