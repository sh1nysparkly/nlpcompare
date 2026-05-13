---
name: Operate under worst-case dev scenarios
description: Don't frame work assuming dev-side fixes will land; build within the constraint as if permanent
type: feedback
originSessionId: de1b3dfa-125b-44e0-ac30-b1b9c2dade69
---
Phase B and adjacent SEO/structural work on AMA Travel is downstream of an active dev-side hostility — the senior dev (95% of the dev-side problem per Anna) is actively arguing AGAINST SSR adoption (i.e. for keeping the current CSR setup), not just passively deprioritizing the fix. Past dev conversations have included laptop-turn "GOTCHA" moments grounded in wildly-wrong citations (e.g., an AI Overview about a product deprecated since 2018). The institutional posture is: this will not get fixed.

**Why:** when work is framed as "let's flag this and the devs will address it," it sets up Anna for repeated exhausting fights she's already had — and implicitly downgrades the within-constraint work, which is the actual deliverable. The "we'll fix the rendering" frame is a fantasy under the current dev posture. Anna's words: "We're continuing to operate under the assumption of worst-case scenarios."

**How to apply:**
- Frame Phase B output as within-CSR optimization, not as "flag the rendering issue + supporting work."
- GSC-faithful baselines are the working reality, not a "what if" scenario.
- The GSC-vs-SF audit / rendering-failure documentation is background context for *us*, not a headline finding aimed at devs.
- Don't bank on upstream fixes when scoping or sequencing work.
- The same principle generalizes to other AMA institutional blockers — assume the obstruction is permanent, build the workaround, treat the obstruction as a constraint not a TODO.
- If a future Bean drifts back into "Track A is the dev flag, Track B is supporting work" framing, redirect immediately: the dev-flag isn't an action item, it's context.
