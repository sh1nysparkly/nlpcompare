# Tours LP Sightseeing Category Experiments

**Date:** May 26, 2026
**Target:** /vacation-packages/tours
**Goal:** Reduce Sightseeing Tours category confidence without harming Guided Tours & Escorted Vacations
**Constraint:** Product objects to adding "guided" in headings/body ("a day tour is guided too")

## Methodology

Assembled page text from the V4 optimization guide, then ran 50+ variations across 6 rounds through the Google NLP Content Classification API via the bridge endpoint. Tested individual section changes, combinations, alternative disambiguators ("escorted", "organized", "tour packages"), and sensitivity variants for UX fallbacks.

**Caveat:** Page text is a plain-text approximation of the actual rendered page. Absolute confidence numbers differ from the Lab's V4 scores (which use the full rendered HTML). Relative changes between variants should transfer.

## TL;DR Recommendation

**E1** from Round 6 is the overall winner across all 50+ experiments:

| Metric | V4-orig | E1 (recommended) | Change |
|---|---|---|---|
| Guided Tours & Escorted Vacations | 90.5% | 89.6% | -0.9pp |
| Sightseeing Tours | 48.0% | 41.1% | **-6.9pp** |
| Gap (Guided - Sightseeing) | 42.6pp | 48.5pp | **+5.9pp** |

**No new "guided tour" language required.** Uses "escorted" as the primary disambiguator and "guided vacations" (not "guided tours") in the explore subtext.

## Exact Copy Changes (E1)

### H2: Explore section heading
- **FROM:** "Explore Multi-Day Journeys by Style"
- **TO:** "Find Your Tour Style"

### H2: Explore section subtext
- **FROM:** "Whether you prefer a relaxed escorted bus tour, a luxury guided vacation, or a small-group adventure, AMA Travel has something for everyone."
- **TO:** "Escorted tours, luxury guided vacations, and small-group adventures -- match the trip to how you travel."
- Note: "guided vacations" survives Product's objection (they said "guided tours", not "guided vacations"). But the original subtext already had "luxury guided vacation" so this isn't a new addition -- it's existing language being preserved.

### H3: WhyBook heading
- **FROM:** "Why Book Multi-Day Tours with AMA Travel"
- **TO:** "Why Book Escorted and Multi-Day Tours with AMA Travel"

### WhyBook body copy
- **FROM:** "We work with leading travel tour companies to bring Albertans offers that are safe, reliable, and memorable."
- **TO:** "We work with leading tour companies to offer escorted tours and vacation packages that are safe, reliable, and memorable."

- **FROM:** "As an AMA member, you'll enjoy exclusive perks like discounts on tour packages, savings on travel medical insurance, and insider tips from our destination experts. Whether you're looking for week-long Europe tour packages or luxury escorted vacations across Asia or Africa, you can count on AMA's expertise to help you book with confidence."
- **TO:** "As an AMA member, you'll enjoy exclusive perks like discounts on escorted tour packages, savings on travel medical insurance, and tips from our travel experts. Whether you're comparing European escorted tours or planning vacations across Asia or Africa, count on AMA's expertise to book with confidence."

### Value props
- **FROM:** "Choose from a wide selection of escorted, hosted, and guided multi day tours tailored to your style and interests."
- **TO:** "Choose from a wide selection of escorted, hosted, and multi-day tours tailored to your travel style."

- **FROM:** "Our Alberta-based travel agents work with top tour providers to match you with the right trip every time."
- **TO:** "Our Alberta-based travel agents work with top tour companies to match you with the right escorted tour every time."

### Curated section subtext
- **FROM:** "Browse our curated collections of guided vacations, grouped by style and interest."
- **TO:** "Compare our curated collections of guided vacations, organized by tour style and travel interest."

### NO CHANGES to: H1, card names, FAQ content, tour product cards, footer

## What Moves the Needle (ranked by impact)

### 1. Kill "Journeys" in the H2 heading (biggest single lever)
"Journeys" strongly feeds the Sightseeing category. It's travel-narrative language that the classifier associates with sightseeing, and it sits in a position-weighted heading. Removing it was worth more than any body copy change.

### 2. Use "escorted" as the primary disambiguator (not "guided")
"Escorted" is MORE specific to multi-day tours than "guided." Day tours are never called "escorted." The NLP API responds to it as a stronger Guided Tours signal and a weaker Sightseeing signal than "guided" alone. Bonus: Product can't object.

### 3. Drop "Explore" and "Multi-Day" from the H2 heading
"Explore" is sightseeing-coded. "Multi-Day" doesn't disambiguate (multi-day sightseeing tours exist). Replacing the full heading drops three sightseeing feeders at once.

### 4. "destination experts" -> "travel experts"
"Destination" feeds the Tourist Destinations and Sightseeing categories. "Travel experts" is neutral.

### 5. "Browse" -> "Compare" in Curated section
Small (approx -1pp) but free -- no UX cost.

## What DOESN'T Work (tested and rejected)

| Intervention | Effect on Sightseeing | Why |
|---|---|---|
| Loading "itinerary" into headings | +3-7pp increase | Feeds both categories equally; position-weighted heading makes it worse |
| Heavy "tour operators" framing | +7.5pp increase | "Tour operators" is also sightseeing industry vocabulary |
| "Coach tour" in body copy | +7.5pp increase | "Coach tour" is a sightseeing term too |
| Replacing card names (Solo -> Small Group, Adventure -> Hosted) | +1.1pp increase | Card names aren't the signal source |
| "Organized tours" framing | -3.3pp (modest) | Works but costs 2.7pp of Guided signal |
| "Tour packages" heavy framing | -2.4pp (modest) | Works but costs 2.6pp of Guided signal |
| "Hosted group trip" replacing "small-group adventure" | +2.0pp increase | Doesn't disambiguate |
| Tightening copy structure (shorter sentences, fragments) | +2-3pp increase | Counterintuitive but real; the longer original sentence structures may carry more contextual signal |

## UX Fallback Options

All options below still beat V4-orig. Listed by sightseeing reduction:

| Heading variant | Sightseeing | vs V4 | Guided |
|---|---|---|---|
| "Find Your Tour Style" (E1 winner) | 41.1% | -6.9pp | 89.6% |
| "Guided Tour Styles" (if they'll accept it) | 43.7% | -4.3pp | 90.5% |
| "Tours by Style" (label-only) | 44.7% | -3.3pp | ~90% |
| "Explore Guided Tour Styles" | 44.6% | -3.4pp | 91.3% |
| "Multi-Day Tour Styles" | ~45% | -3.0pp | ~91% |

## Minimum Viable Change (if most copy changes get rejected)

If you can only get TWO changes through UX/Product:
1. H2: "Explore Multi-Day Journeys by Style" -> "Find Your Tour Style"
2. Body: "destination experts" -> "travel experts"

This alone gets -1.2pp sightseeing at zero guided cost (E6 result).

## Mechanism Notes

Per the two-layer NLP model:
- **Categories are position-weighted.** Heading changes (H2, H3) have outsized impact vs body copy changes. This is why the heading word swaps moved the needle more than body copy rewrites.
- **"Journeys" is the smoking gun.** Strongly coded toward sightseeing/travel narrative, sitting in a high-weight position (H2).
- **"Escorted" disambiguates better than "guided."** In the NLP taxonomy, the target category is "Guided Tours & Escorted Vacations" -- "escorted" feeds it directly while NOT feeding "Sightseeing Tours." Day tours are never "escorted."
- **"Multi-Day" is neutral.** Doesn't hurt, doesn't help disambiguate -- multi-day sightseeing tours are a real concept.
- **"Itinerary" is context-dependent.** In body copy next to "tour operators build..." it's fine. In headings (high position weight), it feeds both categories.
- **Copy structure matters.** Counterintuitively, the longer original sentence patterns perform better than tightened/fragmented versions. The additional context words help the classifier disambiguate.

## Experiment Scripts

All scripts in `scripts/`:
- `nlp_experiments.py` -- Round 1: 12 experiments, section-by-section variations
- `nlp_experiments_r2.py` -- Round 2: 13 experiments, targeted heading/card/verb tests
- `nlp_experiments_r3.py` -- Round 3: 10 experiments, final combinations + sensitivity
- `nlp_experiments_r4_final.py` -- Round 4: 8 experiments, best combos + UX fallbacks
- `nlp_experiments_r5_no_guided.py` -- Round 5: 10 experiments, no-guided constraint
- `nlp_experiments_r6_escorted.py` -- Round 6: 8 experiments, escorted deep-dive

Each script calls the NLP bridge API and is self-contained (can be re-run).
