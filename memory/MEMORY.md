# Memory index

## Feedback (how to work with Anna on this project)
- [Concert of stupidity](concert-of-stupidity.md) -- resist single-cause diagnosis; expect multiple compounding failures
- [Walk back explicitly](walkback-explicitly.md) -- when caught wrong, name it directly; she treats clean walkbacks as partnership
- [Save tool outputs during work](save-tool-outputs-during-work.md) -- write each NLP/Ahrefs/GSC result to a file at the time, not later
- [Test the actual claim](test-the-actual-claim.md) -- evidence must come from the entities the hypothesis is about, not adjacent ones
- [Cross-check before declaring tanking](cross-check-before-declaring-tanking.md) -- layer rankings + clicks + sales + qualitative before calling a page broken or healthy
- [Don't pre-decide experiments](dont-predecide-experiments.md) -- treat hypothesis-tests as hypothesis-tests, not findings; no conclusions in status docs without sign-off
- [Co-design surfaces, don't elaborate alone](co-design-surfaces-with-anna.md) -- pause and co-design when Anna sketches a UI/surface idea; don't sprint into a fleshed-out artifact
- [Lab fidelity is mission-critical](lab-fidelity-mission-critical.md) -- v9 curation is structural ground truth; never propose having the Lab re-parse HTML and merge with curation. Anna has explicitly rejected lossy measurements.
- [Memory attribution discipline](memory-attribution-discipline.md) -- tag claims in memory by source (Anna verbatim / from her doc / Bean synthesis / Bean hypothesis); prevents phantom-attribution drift across sessions.
- [Worst-case dev scenarios](worst-case-dev-scenarios.md) -- senior dev actively hostile to CSR fixes; frame work as within-constraint optimization, not "let's flag this and they'll fix it"

## Project (active AMA Travel context)
- [Post-launch state (May 8, 2026)](ama-travel-post-launch-state.md) -- current diagnosis, active levers, recipe-rollout workstream, per-page experimental status
- [SEO technical facts](seo-technical-facts.md) -- established context (template links discounted, JS redirects, AI Overview history, link-spec pattern, brand decline)
- [Signal Coherence Framework purpose](signal-coherence-framework-purpose.md) -- experimental apparatus for hypothesis testing, not decision support; outputs are measurements not verdicts
- [Recipe spec canonical](recipe-spec-canonical.md) -- the six top-level Sections of the UX recipe + the Whats inside each. Read before any LP curation/section diagnostic; keeps getting flattened, which makes everything end up tagged Hero.
- [Experimental phasing](experimental-phasing.md) -- Phases A (data)/B (permutations) [DONE]/C (recipe-optimization) [REBUILT May 12 with anti-patterns]/D (Anna locks in). Phase C deliverable = per-LP brief.md in Phase B format + §11 critical assessment + cross-LP index. NO verdict columns, decisions-queue files, or templated permutations.
- [Phase B lever families (7-LP refresh)](phase-b-lever-families.md) -- 5 lever families (category-steering / commercial-profile guarding / cannibalization-resolution / position-of-commercial-content / category-profile simplification) + cross-cutting principles. Read as hypotheses, not laws.
- [Position×volume null at late position](position-volume-null-late-position.md) -- AMA Travel LPs' whole-page category measurement is determined by ~first half of page. n=11 null tests across 7 LPs; structural property.
- [Two-layer NLP model](two-layer-nlp-model.md) -- categories driven by position-weighted content; entities driven by mention frequency. Different lever types affect different layers. Confirmed n=5 LPs.
- [Cross-LP stacking regimes](cross-lp-stacking-regimes.md) -- three compound shapes when stacking within-recipe levers: multiplicative (same-metric), linear-additive (different-layer), under-additive (same-entity-space). Classify levers BEFORE predicting compound outcomes. Confirmed n=3 LPs.

## Reference
- [signal-coherence/ directory](signal-coherence-directory.md) -- where SEO investigation artifacts live
- [Lab dashboard state and followups](lab-state-and-followups.md) -- what's in `lab/index.html` + `lab/styles.css`, what's deployed at signal-coherence.netlify.app, what looks-dead-but-isn't (V0.5, `is_snap_target`, `kind: "recipe"`), and open work. **Read before touching the Lab code.**
- [May 15 widget-fix patch](signal-coherence-may15-widget-patch.md) -- record of the surgical swap (raw_html in `crawled_pages` + card rows in `lp_blocks`, marked `curation_version='v20'`) that replaced widget-error placeholders on hotels/cruises/flights and the empty Activities Albertans Loved carousel on things-to-do. Includes the row_order shift technique (high-range parking to dodge the unique constraint) and the RLS-toggle pattern for bulk HTML upload.
