# Tours LP Sightseeing Category Experiments

**Date:** May 26, 2026
**Target:** /vacation-packages/tours
**Goal:** Reduce Sightseeing Tours category confidence without harming Guided Tours & Escorted Vacations

## Methodology

Assembled page text from the V4 optimization guide, then ran 35+ variations through the Google NLP Content Classification API via the bridge endpoint. Tested individual section changes, combinations, and sensitivity variants for UX fallbacks.

**Caveat:** Page text is a plain-text approximation of the actual rendered page. Absolute confidence numbers differ from the Lab's V4 scores (which use the full rendered HTML). Relative changes between variants should transfer.

## Key Results (simplified text, consistent scaffolding)

| Variant | Guided | Sightseeing | vs V4-orig |
|---|---|---|---|
| V4-orig (from optimization guide) | 90.6% | 48.5% | -- |
| **WINNER: "Find Your Guided Tour Style" + WhyBook B + Compare** | **90.1%** | **42.5%** | **-6.0pp** |
| BONUS: same + "hosted group tours" in subtext | 90.3% | 43.1% | -5.4pp |
| Sensitivity: "Guided Tour Styles" (label only) | 90.5% | 43.7% | -4.8pp |
| Sensitivity: "Explore Guided Tour Styles" | 91.3% | 44.6% | -3.9pp |
| Sensitivity: keep "Multi-Day" in WhyBook H3 | 89.6% | 43.8% | -4.7pp |
| Sensitivity: "Match Your Guided Tour Style" | 90.3% | 44.0% | -4.5pp |

## What Moves the Needle (ranked by impact)

### 1. Kill "Journeys" in the H2 heading (biggest single lever)
"Explore Multi-Day Journeys by Style" → "Find Your Guided Tour Style"

"Journeys" strongly feeds the Sightseeing category. Dropping it + adding "Guided" was the most impactful single change across all tests.

### 2. Replace "Multi-Day Tours" with "Guided Tours" in WhyBook H3
"Why Book Multi-Day Tours with AMA Travel" → "Why Book Guided Tours with AMA Travel"

"Multi-Day" doesn't disambiguate from sightseeing (multi-day sightseeing tours exist). "Guided" does.

### 3. Tighten WhyBook body copy
- "travel tour companies" → "tour operators" (light touch only!)
- "bring Albertans offers" → "build itineraries"
- "destination experts" → "travel experts"
- "week-long Europe tour packages" → "European tour itineraries"
- "tour providers" → "tour operators"
- "right trip" → "right itinerary"
- "your style and interests" → "your travel style"

### 4. Curated section: "Browse" → "Compare"
Small (approx -1.2pp) but free -- no UX cost.

## What DOESN'T Work (tested and rejected)

| Intervention | Effect on Sightseeing | Why |
|---|---|---|
| Loading "itinerary" into headings | **+3-7pp increase** | "Itinerary" feeds both categories equally; position-weighted heading makes it worse |
| Replacing card names (Solo → Small Group, Adventure → Hosted) | +1.1pp increase | Card names aren't the signal source |
| Heavy "tour operators" framing in body copy | +7.5pp increase | "Tour operators" is also sightseeing industry vocabulary |
| "Coach tour" in body copy | +7.5pp increase | Same -- "coach tour" is a sightseeing term |
| "Hosted group trip" replacing "small-group adventure" | +2.0pp increase | "Hosted" doesn't disambiguate; "adventure" isn't the culprit |

## UX Negotiation Fallbacks

All variants below still beat V4-orig by 3.9-5.4pp:

| If UX says... | Use this instead | Sightseeing cost vs winner |
|---|---|---|
| "We need a verb in the heading" | "Find Your Guided Tour Style" (the winner) | 0pp |
| "We like label-style headings" | "Guided Tour Styles" | +1.2pp |
| "We want to keep 'Explore'" | "Explore Guided Tour Styles" | +2.1pp |
| "'Match' feels more on-brand than 'Find'" | "Match Your Guided Tour Style" | +1.5pp |
| "'Multi-Day' must stay in WhyBook H3" | "Why Book Multi-Day Guided Tours with AMA Travel" | +1.3pp |

## Mechanism Notes

Per the two-layer NLP model:
- **Categories are position-weighted.** Heading changes (H2, H3) have outsized impact vs body copy changes. This is why the heading word swaps moved the needle more than body copy rewrites.
- **"Journeys" is the smoking gun.** It's strongly coded toward sightseeing/travel narrative and was sitting in a high-weight position (H2).
- **"Guided" disambiguates from sightseeing.** In heading position, "Guided" strongly activates the Guided Tours category without feeding Sightseeing.
- **"Multi-Day" is neutral.** It doesn't hurt, but it doesn't help disambiguate -- multi-day sightseeing tours are a real concept.
- **"Itinerary" is a wash in headings.** In body copy (mid-page, lower weight), "build itineraries" helps slightly. In headings (high weight), it feeds both categories.
