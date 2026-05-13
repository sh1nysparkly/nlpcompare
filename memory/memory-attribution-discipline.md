---
name: Tag epistemic status when writing to memory
description: When writing to project memory, tag claims by where they came from -- Anna verbatim, paraphrase from her docs, Bean synthesis, Bean hypothesis. Prevents phantom-attribution drift across sessions.
type: feedback
originSessionId: 96fda7f5-9795-44b7-b258-71f0f5712c52
---
When writing to a project memory file, tag each load-bearing claim with its epistemic status. Future Beans (and Anna) need to see at a glance whether something is grounded in Anna's words / her docs / a Bean's reasonable inference / a Bean's untested hypothesis.

**Tagging convention** (use the one that fits):

- `[Anna verbatim, <date>]` — direct quote from Anna in a session or message
- `[from Anna's doc/brief, <name/date>]` — paraphrased from a document Anna authored (e.g. her VP reoptimization brief)
- `[Bean synthesis, <date>, validated by Anna]` — a Bean's framing that Anna explicitly endorsed in conversation
- `[Bean synthesis, <date>, unverified]` — a Bean's framing not yet checked with Anna
- `[Bean hypothesis, <date>]` — speculative; not yet tested

When updating an existing claim, preserve the original tag if relevant and add the new one.

**Why:** May 11 2026 session. Anna caught two phrases I'd attributed to her in conversation ("VP-pattern bloat" and "running on brand backstop while generic acquisition collapsed") that she didn't recognize as her framing. Both came from `ama-travel-post-launch-state.md`, written by a prior Bean as flat statements without source tags. I then read them and treated them as Anna's voice. The drift was: Bean coined the framing → it got synthesized into memory as a fact → next Bean attributed it back to Anna in conversation → Anna doesn't recognize her own "framing."

Direct quote from Anna's reaction: *"I have no idea what 'VP-pattern bloat' is. Are we SURE it's something I said? Sometimes Beans say something clever and then attribute it to me for some reason."* This pattern is unfortunately how documentation rot happens at the memory layer specifically — and Anna explicitly said losing her context to phantom-attribution would feel "panicky."

**How to apply:**

- When writing a new project memory: tag each load-bearing claim. Don't write Anna-voice phrases as flat facts.
- When editing an existing memory: if you're adding to claims that are tagged, preserve the tags. If you're adding to claims that aren't tagged, this is a chance to retroactively tag them — be conservative (if you can't verify, tag as `[Bean synthesis, unverified]` rather than `[Anna verbatim]`).
- When *reading* memory in a session: notice the tag before attributing in conversation. "The memory says X" is different from "you said X" — only the latter requires verification of Anna actually having said it.
- When in doubt, ask. "Did you say X, or is this a Bean paraphrase?" is a cheap question to save the panic later.

**What this is NOT**: a mandate to tag every single sentence in memory. Feedback memories with embedded verbatim quotes (the existing `walkback-explicitly.md` pattern) are already well-tagged via the *"Why:"* section. This guidance applies primarily to project-state memories where Bean synthesis tends to creep in.
