---
name: Two-layer NLP measurement model — categories vs entities respond to different levers
description: Google's V2 NLP API on AMA Travel LPs measures categories and entities via different mechanisms. Categories are position-weighted; entities are mention-frequency-driven. Interventions can move one layer without moving the other.
type: project
originSessionId: de1b3dfa-125b-44e0-ac30-b1b9c2dade69
---
[Bean synthesis from running Phase B across 7 LPs (May 11 2026). Cross-LP empirical observation; mechanism is hypothesized, not verified against Google's docs.]

## The model

Google's NLP API on AMA Travel LPs behaves as if it has TWO measurement layers that respond to DIFFERENT lever types:

**Category layer (top-cat + secondary cats):** driven by **position-weighted content**. Approximately the first half of the page determines category classification. Late-position content is essentially invisible to category measurement (see `position-volume-null-late-position.md`). Lever to move category: rearrange or rewrite early-position content.

**Entity layer (entities + salience + mention counts):** driven by **mention frequency** anywhere on the page. An entity with 20 mentions anywhere will be salient; an entity with 1 mention won't be. Lever to move entity profile: remove content that's saturating a specific entity, or add content that surfaces a target entity.

## Empirical evidence (cross-LP)

The "ghost a large late-position container" test on multiple LPs:

| LP | Container ghosted | Category-layer change | Entity-layer change |
|---|---|---|---|
| TTD T1 | My Travel Dream (4,461w, pos 8) | top-cat unchanged | AMA Travel entity collapsed 0.357 → 0.004 (-99%); target entities +286% |
| Cruises T1 | Related Cruise Articles (1,278w, pos 14) | top-cat unchanged | "Your Complete Guide" entity collapsed 0.503 → 0.013 (-97%); cruise entity Σ +122% |
| Destinations T1 | AI Curated Trips (4,447w, pos 6) | top-cat unchanged | AMA Travel entity collapsed similarly |
| Flights T1 | Flights FAQs (483w, pos 12) | top-cat unchanged | AMA Travel entity collapsed 0.085 → 0.005 (-94%); "flights" entity became #1 entity |
| Hotels (v13→v14 corrections) | All cruise-deal carousels (9 of 9 missing in GSC) | top-cat unchanged | "Best Western Cedar Park Inn Edmonton" entity collapsed 0.136 → undetectable |

In each case, the category-layer was unaffected by the removal; the entity-layer was dramatically rebalanced.

## What this means for Phase B work

**When designing a permutation, name the layer you're trying to move.**
- "Lift top-cat" or "remove sub-cat noise" → category layer → look at early-position interventions
- "Reduce AMA Travel entity dominance" or "lift target keyword entity" → entity layer → look at mention-frequency interventions (anywhere on page)
- "Surface specific brand-partner entities" (e.g., hotels Trusted Partners) → BOTH layers if the container has both mass + position
  - Hotels T7 (promote Trusted Partners to pos 2): lifted top-cat +0.035 (category-layer via position) AND brand-entity Σ tripled (entity-layer via making those entities more visible). This is the "unicorn" pattern.

**Anti-pattern: assuming a category-layer intervention moves entities, or vice versa.**
- Late-position content removal will NOT lift target sub-categories (category layer doesn't respond)
- Hero copy reword will NOT shift entity dominance if the dominant entity lives mid-page (mention frequency unchanged)

## Caveats

- 5 LPs of confirmation; pattern may not hold in different rendering contexts (different sites, different NLP versions, different page structures)
- "First half of page" is an approximation — exact position weighting cutoffs are unverified
- The mechanism is hypothesized from output behavior, not verified against Google's documentation
