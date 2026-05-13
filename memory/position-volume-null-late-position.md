---
name: Position×volume null at late position (cross-LP structural property)
description: In Phase B testing across 7 LPs (n=11 null tests at volumes from 137w to 4,461w), removing late-position content produced null category-layer change. Suggests AMA Travel LP whole-page category measurement is position-weighted to approximately the first half of the page in this test set.
type: project
originSessionId: de1b3dfa-125b-44e0-ac30-b1b9c2dade69
---
[Bean synthesis from running Phase B methodology across 7 of 7 non-control LPs (May 11 2026). Cross-LP empirical pattern, not a per-LP curiosity.]

## The pattern

Across 11 individual permutation tests on 7 LPs, removing late-position content produced **zero measurable change** in whole-page top-cat confidence regardless of volume removed (tested from 137 words up to 4,461 words — including extreme tests at 73% and 86% of page volume).

| LP | Container ghosted | Words | Position | Top-cat Δ |
|---|---|---:|---:|---:|
| Hotels | Articles to Inspire | 685 | 11 of 13 | 0 |
| VP | Articles | 137 | 8 of 10 | 0 |
| VP | FAQs | 455 (41%) | 9 of 10 | 0 |
| TTD | My Travel Dream | 4,461 (73%!) | 8 of 12 | +0.003 |
| Cruises | Related Cruise Articles | 1,278 (43%) | 14 of 16 | 0 |
| Destinations | AI Curated Trips | 4,447 (86%!) | 6 of 10 | +0.003 |
| Flights | Flights FAQs | 483 (40%) | 12 of 13 | 0 |
| Flights | Articles to inspire | 291 | 11 of 13 | 0 |
| TI | Travel Insurance FAQs | 468 | 8 of 11 | 0 |
| TI | Related articles | 644 (33%) | 10 of 11 | 0 |
| TI | T4 combo (T1+T2) | 1,112 (57%) | — | 0 |

## What this means for Phase B work

**Don't bank on ghosting late-position content as a category-layer lever.** Removing entire late containers — even ones representing >50% of page volume — won't shift the top-cat confidence on AMA Travel LPs. The page's category classification comes from approximately the first half of the page; whatever's in the back half is essentially invisible to Google's V2 classifier at the category measurement layer.

**The entity layer is a DIFFERENT story.** See `two-layer-nlp-model.md`. Ghosting late-position content with dominant-entity content (e.g., the "Get Inspired by AI Curated Trips" container that mentions AMA Travel 19 times) DOES dramatically rebalance the entity profile, even though it doesn't move category confidence. Entity measurements are mention-frequency-driven; categories are position-weighted.

**Methodology implication:** when designing Phase B permutations, weight EARLY-position interventions for category-layer goals. Late-position interventions are for entity-layer goals (or for non-NLP reasons like reducing user confusion or AI Overview citation contamination).

## Where the null doesn't apply

- **Mid-position content can still affect top-cat.** Cruises' rebuild dropped top-cat -0.082 because the cruise-deal cards (positions 6 and 9) WERE carrying category mass. Position 6-9 is "mid-page" not "late"; the null applies above position ~10 of ~13.
- **Promoting late content to early position DOES move top-cat** (see articles-promotion bloat pattern in `phase-b-lever-families.md`). The null is asymmetric — removing from late is null; relocating to early is a lever.

## Where to find the data

Per-LP briefs at `signal-coherence/phaseB-results/{hotels,vacation-packages,things-to-do,cruises,destinations,flights,travel-insurance}/brief.md`. Cross-LP shape table in `signal-coherence/phaseB-results/PERMUTATIONS-INDEX.md` (Shape 1).
