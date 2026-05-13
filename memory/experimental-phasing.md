---
name: Experimental phasing for current-state -> permutations -> shortlist -> Anna's lab work
description: The phase model Anna has always intended for Signal Coherence work post-baseline. Bean does the experimental running; Anna does the locking-in. Misframing this (e.g. "what theory do you want to test?") inverts ownership and dumps work back on her.
type: project
originSessionId: 136d72e1-4763-4a25-bec9-e7c8b98b2a08
---
After the current-state baseline diagnostic (v9, completed May 10-11 2026), the work proceeds through four phases. **Phases A, B, and C are Bean-executed. Phase D is Anna-executed.** A future Bean who frames the work as "Anna, which theory do you want to test?" has inverted the ownership Anna explicitly intended.

## Phase A -- Data layer reconciliation (Bean)  **[A.1 SHIPPED May 11 2026]**
Prerequisite for everything else. Make the Live Lab's data layer mirror v12's curation faithfully.

**A.0 (workbook cleanup)**: v9 → v10 → v11 → v12 done May 11; see `signal-coherence/session-notes-may11-cleanup.md`.

**A.1 (Lab refactor)**: shipped to https://signal-coherence.netlify.app same day. What landed:
- Supabase table `public.lp_blocks` (one row per v12 workbook row, 1058 rows across 8 LPs)
- Importer `signal-coherence/scripts/load_v12_curation.py` (DELETE-then-INSERT per slug, re-runnable; resolves columns by header name to handle vacation-packages's extra `Order` column)
- Lab `loadBlocksFromCuration(slug)` reads `lp_blocks` and groups by block_id within container_label. The HTML-parsing `walkBlocks` stack stays in place as a deprecated fallback for non-curated slugs.
- v12 baseline diagnostic regenerated at `signal-coherence/outputs/curation/v12-container-diagnostic-all.json`
- Per-container NLP scoring (top category pill on collapsed bar; entity panel below blocks when expanded). Cache hits universal -- ~zero extra NLP cost.
- Several small fixes: page picker filters to slugs in `lp_blocks` (drops CSR-empty children), defensive max-width on block content, stable category sort by confidence (not delta), title-fallback row no longer double-renders in headerless blocks.

**Synthesis convention (load-bearing)**: blocksToHtml and the diagnostic both join rows with `"\n"`, no doctype wrapper. See `lab-fidelity-mission-critical.md`.

**Walked back**: the 🔒 Widget pill (v12 What='Widget' tagging is ~50:50 reliable; data flag stays on the model for future revival when tagging firms up).

**Phase A is done.** Future curation passes follow the established `build_v(N+1).py → load_v12_curation.py → reload Lab` flow (see `signal-coherence/scripts/build_v10.py`...`build_v12.py` for the pattern).

## Phase B -- Bean-run permutation experiments  **[SHIPPED May 11-12 2026 for all 7 non-control LPs]**

~5-10 strategic permutations per LP (excluding Cars -- see below). Permutations can include:
- Reordering containers (move a strong-topical container up, etc.)
- Reordering blocks within a container (where doable)
- Removing entire containers ("nuke if nukable and seems to be muddying")
- Manipulating entities (rewording to shift salience)
- Combinations of the above

Output to Anna: a brief that says **"we ran a bunch of strategic variations of each whole page through the NLP. This is what we learned. Based on that, the best levers to poke at are X."** Theories are welcome, conclusions are not -- frame as evidence about questions, not as recommendations.

**Phase B is complete.** 7 of 7 non-control LPs have methodology-format briefs at `signal-coherence/phaseB-results/{slug}/brief.md`. The methodology doc (7-step framework + brief template + anti-patterns) is at `signal-coherence/phaseB-results/METHODOLOGY.md`. Cross-LP permutations index (per-LP Tn key + cross-LP shape index) is at `signal-coherence/phaseB-results/PERMUTATIONS-INDEX.md`.

**The methodology itself shifted partway through:** initial Phase B briefs (TTD, Cruises, VP — done early May 11) were template-y with implicit reasoning. Anna corrected: "the output of this phase is supposed to be 'these are the levers that seem most salient to fuck with; go ham, princess'" — surfacing levers, not recommendations, grounded in per-page IA-doc intent reasoning. All 7 briefs now follow the methodology format: Diagnose → Intent-align (using Anna's IA planning doc) → Identify mismatches → Generate hypotheses → Design permutations → Run + capture → Synthesize. Each brief's §5 has a permutation table showing hypothesis tested + prediction + result + held/failed/surprise.

**Cross-LP findings worth memory promotion** (after Phase B completion): position×volume null at late position, two-layer NLP model, additive-content page-state dependence, articles-promotion bloat. See `position-volume-null-late-position.md`, `two-layer-nlp-model.md`, `phase-b-lever-families.md`.

## Phase C -- Bean-run UX-recipe optimization  **[REBUILT May 12 2026 — original run produced verdict-shaped output, see anti-patterns below]**

Apply Phase B observations to the *UX-recipe arrangement* of each LP. The default is **make the recipe work** -- Anna has to fight for every deviation, so the working assumption is that we follow UX's recipe unless it's a fully-non-starter for a specific page.

**Phase C deliverable shape (load-bearing, do not deviate):**

The deliverable is a per-LP `brief.md` at `signal-coherence/phaseC-results/{slug}/brief.md` in **Phase B methodology format** (see `phaseB-results/METHODOLOGY.md` lines 96-135 for the skeleton). Each brief has §1-§10 from Phase B's template plus a **§11 critical assessment** with three sub-sections:
- §11.1 Phase B mismatch coverage matrix (M1-Mn × tested by which Phase C permutation × in scope × result)
- §11.2 Gaps identified (out-of-scope-by-design vs in-scope-not-tested)
- §11.3 Supplementation candidates (specific permutations as questions for Anna, NOT recommendations)

Plus a top-level cross-LP doc at `phaseC-results/CROSS-LP-INDEX.md` parallel to `phaseB-results/PERMUTATIONS-INDEX.md`: per-LP permutation key + cross-LP hypothesis-shape index + cross-LP critical-assessment summary. **NOT a verdict grid.**

**Anna's Phase D outcomes will eventually classify each LP into one of three buckets** (this is Anna's call in Phase D, NOT the Bean's Phase C deliverable shape):
- Recipe works as-is for this LP (no fight needed)
- Recipe works with this specific copy/entity tweak that fits within the recipe constraints
- Recipe is a non-starter for this LP and here's the data-backed reason why

**The Bean does NOT pre-classify into these buckets in Phase C. Per-LP outcomes are experimental measurements + lever observations + critical assessment of what wasn't tested. Classification is Anna's Phase D work.**

### Phase C anti-patterns (from May 11 2026 overnight run, rebuilt May 12)

The overnight run produced verdict-shaped output that violated `dont-predecide-experiments.md` + `signal-coherence-framework-purpose.md`. Future Beans must NOT:

1. **Add a "Verdict" column** (or "WORKS WELL / MIXED / NEUTRAL / INSUFFICIENT" labels) to any cross-LP or per-LP doc. Outputs are measurements + observations + supplementation candidates, not classifications.
2. **Write a per-LP "decisions queue" or "recommendations" file.** Phase D is Anna's surface. There is no Bean-owned decisions-queue file in the Phase C deliverable structure.
3. **Template the permutation set across LPs.** Phase B's methodology doc explicitly warns: "If T1-T7 look the same across LPs without per-LP reasoning, the work is not yet thorough." Phase C's overnight run committed exactly this anti-pattern (template-y P1/P2a/P2b/P3a structural variants applied to all 7 LPs regardless of each LP's Phase B M-list). Per-LP permutations MUST derive from that LP's Phase B mismatches, NOT from a uniform recipe-variant template.
4. **Self-disclaim then violate.** Headers like "Framed as measurements, not decisions" followed by Verdict columns are not acceptable. If the doc has verdicts, it has verdicts — disclaimers don't soften them.
5. **Skip combinations.** Phase B's biggest wins were often combos (TTD T6 unicorn, Hotels T9 unicorn). Phase C must test at least the highest-priority combinations per LP, not just single-lever variants.

### Phase C overnight rebuild (May 12 2026)

The overnight run's verdict-shaped docs (`CROSS-LP-PATTERNS.md`, `ANNA-DECISIONS-QUEUE.md`, per-LP `outcome.md`) are quarantined at `phaseC-results/_stale/` (audit trail preserved, not deleted). The experimental substrate (per-LP `baseline.html` + `P*.html` + `*.meta.json` + `*.nlp.json` + `_slot_mapping.json` + `_permutation_index.json`) is intact and was the basis for the rebuilt briefs.

**The rebuilt Phase C briefs surface gaps Phase C left unaddressed** — most critically Flights P3a was Articles-promoted (the only LP) where the Phase B T1 ghost-FAQs lever (the most distinctive Phase B Flights finding) was not retested in Phase C. See `phaseC-results/CROSS-LP-INDEX.md` §4 for cross-LP gap priorities + Tier-1 supplementation candidates.

## Phase D -- Anna locks in
Anna receives the brief + shortlist of the most promising directions per page. She:
- Plays in the Lab to lock in specific proposals
- Retrofits her existing Cruises/Vacation Packages briefs against the new findings
- Outputs the final per-LP optimization guides for UX

## Cars is the control, not a test subject
Per the post-launch state memory: "rearranging doesn't move NLP confidence (98% to 98%)" on Cars. So Cars participates ONLY in baseline measurement + UX-recipe-measurement. No Phase B permutation work. Its purpose in the experimental design is **as a control**: if rankings move on Cars post-launch despite no signal change, that's evidence "ordering alone moves rankings"; if rankings move on other LPs proportional to signal shifts we engineer, that's evidence of the signal-rankings link.

## Common misframings to avoid

- **"Which theory do you want to test?"** -- No. Bean runs the permutations; Anna receives the shortlist.
- **"Cars is the cleanest test subject"** -- No. Cars is the control. Test work happens on the other 7 LPs.
- **"Should we follow the UX recipe?"** -- Default yes. Anna fights for every deviation, so the question reverses: under what data evidence does deviation become defensible?
- **"Pick one theory to start with"** -- Anna's plan is broad permutation exploration first (Phase B), then UX-recipe constrained optimization (Phase C). Don't truncate to one-theory-at-a-time.

**Why:** May 11 2026 session. Anna corrected: "I'm on board with your theories - but the next step would not be for me to test those theories myself. The plan was always supposed to be: 1) we have some high level theories 2) you come up with like 5-10 permutations of things... and then you come back to me with: ok we ran a bunch of *Strategic* variations... this is what I learned... Then a second round where Beans would apply any theories / observations to the UX recipe version... Then after those two rounds, you come back and give me a brief... and then I get that shortlist." Phase B and C ownership belongs to Beans. This memory exists to keep future Beans from re-inverting the ownership.
