---
name: Phase B lever families (refreshed from 7-LP sample, May 11 2026)
description: Lever family hypotheses from Phase B permutation experiments across 7 non-control LPs (Hotels, VP, TTD, Cruises, Destinations, Flights, Travel-Insurance). Three families validated, two new families surfaced, two cross-cutting principles confirmed.
type: project
originSessionId: de1b3dfa-125b-44e0-ac30-b1b9c2dade69
---
[Bean synthesis from full Phase B methodology run on 7 LPs. Cross-LP empirical findings; hypotheses, not laws. Anna's prior corroboration on profile-simplification is noted inline.]

## Validated lever families (from 3-LP initial + refined to 7-LP sample)

### Family 1 — Category steering (wrong-category pages)

**Applies when:** dominant category is *wrong* for what the page should be about. Page reads as one thing in NLP but the IA target keywords say it should be another.

**Strongest example:** /things-to-do — page reads as Tourist Destinations 0.89, IA target is Sightseeing Tours / Things to Do. Hero copy rewording (drop "near me"/Alberta/Canada localizers; add activity vocab) shifted top-cat -0.115 and lifted Sightseeing Tours +0.155.

**Lever:** rewrite **hero copy** (H1, intro paragraph, search-prompt language). Strip localizer/destination signal from hero; inject activity/product-type vocab.

**Why it works:** hero is position 1 — the most heavily weighted position in the category-layer measurement. Per `two-layer-nlp-model.md`, hero changes punch above their volume weight class.

### Family 2 — Commercial-profile guarding (right-category pages with rich secondary profiles)

**Applies when:** top-cat is correct and strong (0.85+) AND the page has a commercial-intent secondary profile to protect (Vacation Offers, Low Cost, Luxury Travel).

**Strongest example:** /cruises — Cruises & Charters 0.87 top-cat with secondary commercial profile (Travel Agencies, Vacation Offers 0.30, Low Cost 0.19, Luxury Travel 0.15). The commercial profile is load-bearing for the page's transactional intent.

**Lever:** **don't promote informational content to high slots.** On /cruises, promoting articles to slot 2 destroyed the commercial profile (Vacation Offers / Low Cost / Luxury fell out of top 5). Same content at low slot is fine; high slot kills commercial intent.

### Family 3 — Cannibalization resolution (sibling-page overlap)

**Applies when:** a page is reading as content a SIBLING page should own.

**Original example (now refined):** /vacation-packages was competing with /vacation-packages/all-inclusives for "all inclusive vacations" 53K MSV. Anna's prior brief prescribed strip all-inclusive language so the AI sibling can hold the intent.

**Note on the v13 "T7 recipe" finding:** the v13 brief found a recipe (MVP copy strip + commercial-first reorder + density-reinforcement section) recovered Vacation Offers to 0.78 with 7× lift on `vacation packages` entity. **This finding was largely measuring v13's SF-inflated phantom-cards content.** On v15 GSC-faithful baseline, the recipe's structural components behave differently — Vacation Offers is already at 0.76 before any intervention; additive content shifts mass AWAY from primary target. The recipe still has cannibalization-resolution value at the entity layer (target-entity Σ rises), but the dramatic category-layer numbers from v13 don't replicate. See `signal-coherence/phaseB-results/vacation-packages/brief.md` for full v15-baseline reanalysis.

### Family 4 (NEW from 7-LP sample) — Position-of-commercial-content (the "unicorn lever")

**Applies when:** the page has a small late-position container that scores aligned with target category AND contains target-keyword brand/specific entities.

**Strongest example:** /hotels T7 (promote "Albertans Love Our Trusted Hotel Partners" from position 10 → 2). Single move affected four measurement dimensions simultaneously:
- Top-cat Hotels & Accommodations +0.035
- Category count 6 → 4 (Luxury Travel + Low Cost dropped out)
- Vacation Offers cannibalization -0.07
- Brand-entity salience (Best Western + Fairmont + Choice Hotels + Marriott) tripled

**Why it works:** the container's content was target-aligned AND contained discrete brand-entity content. Promoting it lifted BOTH layers (category via position; entity via making brand names more salient).

**Caveat:** the brand-partner content lever may not generalize. Hotels is the cleanest n=1 instance. Other LPs' "promote a small container" tests (TTD T5 Value Props, cruises T2 Cruise Style) produced minimal effect.

### Family 5 (NEW from 7-LP sample) — Category-profile simplification

**Applies when:** the page has top-cat correct but a bloated/distracting secondary profile (multiple competing sub-cats).

**Anna's prior corroboration:** "I'd brought it up before — limited competitor research has shown a few examples." Validates as a candidate family even though n=1-2 in this data.

**Example:** /hotels T7 (above) — also collapsed category count from 6 → 4. /destinations T6 (additive aligned vocab) lifted top-cat without bloat.

**Lever:** promote target-aligned content OR add precisely-target-vocab content. The mechanism is that the boost to top-cat displaces secondary-cat mass below the detection threshold.

## Cross-cutting principles (confirmed across 7 LPs)

### Position×volume null at late position

Now a confirmed structural property — promoted to its own memory: see `position-volume-null-late-position.md`. n=11 null results across 7 LPs at volumes from 137w to 4,461w. Late-position content is invisible to whole-page category measurement.

### Two-layer NLP measurement model

Categories driven by position-weighted content; entities driven by mention frequency. Promoted to its own memory: see `two-layer-nlp-model.md`. Confirmed on 5 LPs via T1-style late-position-ghost tests.

### Additive content lever is PAGE-STATE-DEPENDENT (refined hypothesis)

**Refinement of the v13 "additive content is a high-risk lever" hypothesis.** With 7-LP data, the lever has THREE distinct behaviors depending on baseline state + vocab-target alignment:

| Page-state | Vocab alignment | Result type | Examples |
|---|---|---|---|
| Sub-cat bloated baseline | Misaligned vocab (introduces new categories) | **Bloat** — top-cat unchanged, new sub-cats emerge | Hotels T6 (Hotel Types & Amenities → Luxury Travel jumps 0.11 → 0.22) |
| Focused single-target baseline | Partial match (overlaps Secondary KW more than Primary) | **Mass-rebalance** — top-cat drops, secondary target lifts | VP T5, Flights T6 (Low Cost surges) |
| Wrong-category baseline | Target-aligned vocab | **Target sub-cat lift + mild bloat** | TTD T7 (Sightseeing Tours +0.115, cat count +2) |
| Strong baseline + tight vocab match | Tightly matches Primary KW | **Clean lift, no bloat** | Destinations T6 (+0.015, no new cats), TI T5 (+0.005, single cat preserved) |

**Methodology implication:** before designing an additive-content permutation, classify the baseline state (bloated / focused / wrong-cat / strong+aligned) and design vocab against the expected behavior. The simple "anti-pattern" framing is wrong; this is a multi-state lever.

### Articles-promotion bloat (NEW anti-pattern, n=4 LPs)

Promoting an article-carousel container to position 2 ALMOST ALWAYS lifts top-cat (article content is roughly target-aligned) but ALWAYS introduces secondary-cat bloat (article titles carry diverse vocabulary that surfaces secondary signals).

| LP | Articles container | Top-cat Δ | Cat count Δ | Bloat type |
|---|---|---:|---:|---|
| Cruises T4 | Related Cruise Articles | +0.066 | 4 → 8 | Adventure Travel + Travel Guides + 4 more "Other" |
| Destinations T4 | Uncover the Best Places | FLIPPED top-cat | 6 → 2 | (different shape — top-cat flipped TO Travel Guides) |
| Flights T3 | Articles to inspire | +0.034 | 4 → 6 | Product Reviews 0.41 + Travel Guides 0.17 |
| TI T3 | Related articles | +0.030 | 1 → 2 | Product Reviews 0.19 |

**Methodology implication:** don't propose article-carousel promotion as a clean top-cat lever. The bloat is the cost.

## Where this lives

- Per-LP briefs: `signal-coherence/phaseB-results/{slug}/brief.md` for each of 7 LPs
- Methodology doc: `signal-coherence/phaseB-results/METHODOLOGY.md`
- Cross-LP permutations index: `signal-coherence/phaseB-results/PERMUTATIONS-INDEX.md` (Tn key per LP + cross-LP shape index)
- GSC-vs-SF audit: `signal-coherence/outputs/curation/gsc-vs-sf-audit.md`

## Caveats

- **n=7 LPs**, which is the full non-control set but still small in absolute terms.
- Levers are measured against Google V2 NLP classifier output. **Ranking performance is downstream** and depends on link equity, brand demand, competitive SERPs, indexing, etc. NLP shifts are necessary-not-sufficient for ranking shifts.
- **CSR rendering failures are a constraint we operate within** per `worst-case-dev-scenarios.md`. The Phase B levers are all WITHIN that constraint.
- Each LP's results are best understood by reading that LP's brief, not by generalizing from the lever-family abstraction. The cross-LP patterns are real but the LP-specific findings are where the deliverable lives.
