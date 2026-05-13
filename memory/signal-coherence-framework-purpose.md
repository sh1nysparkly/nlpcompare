---
name: Signal Coherence Framework -- experimental apparatus, not decision support
description: What the Signal Coherence Framework is for, why Anna built it, and how to use its outputs (hypothesis evidence, not verdicts). Includes May 8 2026 Phase 1 reset.
type: project
originSessionId: cbca5962-0e2d-4116-8f62-e96b9488c9c7
---
The Signal Coherence Framework is experimental apparatus. Anna built it to generate predictions she can later regress against actual GSC outcomes -- Phase 6 of the original build spec was "empirical weight validation from GSC data." The 4 lenses (Zone, Page, Funnel, Site), the Roche Limit concept, the rearrangement lab in the dashboard, the NLP MCP -- all of it is mad-scientist instruments designed to test hypotheses about what makes a page topically coherent.

The framework is currently theoretical. Validation only happens once restructured pages have generated 30/60/90-day GSC performance data. Until then, framework outputs are hypotheses, not verdicts.

**Phase 1 reset, May 8 2026 (load-bearing context):**

The original Build Spec and Methodology docs elaborated past Anna's actual kernel. She named on May 8 that:
- The 4-lens scaffold (Zone / Page / Funnel / Site), Roche Limit, KL Divergence / Information Gain, and funnel-aware coherence are HERS. These are real Phase 2 concepts and survive intact.
- The specific Phase 1 implementation (positional entity weighting methodology with hand-curated entity lists, 3:2:1 zone weighting, heading slot scoring, the skill, the elaborated Supabase tables, the Pages tab) was elaborated past her -- about 90% not her ideas. This was set aside.
- Her Phase 1 kernel, in her words: "we know higher on the page IS important for ranking signals, so it's important to make sure the right shit is in the right parts of the page. And if we can get the NLP API to pull the list of entities per page we could make sure things are being arranged optimally so that the things that SHOULD be the salient entities are placed prominently and in a way that improves their salience scores."

So Phase 1 = entities + salience + position + rearrange. The skill is deprecated. The elaborated tables (analysis_runs, zone_scores, heading_slot_scores, etc.) are dropped. The Pages tab is removed from the dashboard. The conceptual framework (lenses + Roche + KL + funnel) is intact for Phase 2.

**Categories are NOT out** (correction May 9): when Anna said "I just want entities and salience," she meant "drop the elaborated apparatus (3:2:1, hand-curated lists, manual entity sets, the skill)" -- not "drop categories." Categories remain useful as (a) directional diagnostic, (b) the stakeholder-facing language people have glommed onto, and (c) the layer that entities+salience actually feed INTO. The kernel-aligned Lab should show categories AND entities+salience together so Anna can see how salience shifts move category perception. The Stitch mockup that surfaced this -- right pane shows Content Categories block above the entity Base/WIP table -- is on-spec, not a deviation.

Reference docs:
- Status doc (current state, post-reset): `signal-coherence/SIGNAL-COHERENCE-STATUS.md`
- Build spec (PARTIALLY OBSOLETE -- the lenses + Roche + KL + funnel-aware survive; the Phase 1 implementation does not): `Signal Coherence Framework - Build Spec.md`
- Positional Entity Weighting methodology (DEPRECATED -- this is the elaborated Phase 1 that got set aside; do not implement): `Positional Entity Weighting - Methodology.md`

**How to apply:** When using NLP classification, salience, or any framework metric: the result is a measurement, not a verdict. Frame as "the page becomes measurably more focused on X" (clarity diagnostic, defensible) not "this will rank better" (unvalidated chain). For each restructure that ships, capture pre-restructure GSC baseline (90-day snapshot) so prediction-vs-outcome data accumulates over time. When picking up Lab work, the kernel is entities + salience + position -- co-design any UI elaboration with Anna (see `co-design-surfaces-with-anna.md`), do NOT pre-decide a UI surface. The Build Spec's specific Phase 1 implementation is not the spec to follow.
