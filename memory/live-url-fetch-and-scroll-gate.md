---
name: Live URL fetch, the amatravel.ca scroll gate, and the agreed three-piece plan
description: What the Nov 11 2026 session found about fetching live AMA pages, what was built (Quick Scorer URL fetch), and the design Anna agreed for pulling live pages into Page Analysis. Read before touching netlify/ or designing pieces 2 and 3.
type: project
---
Tags follow memory-attribution-discipline.md.

## Findings about the live sites

- **amatravel.ca holds back most of a landing page until a real scroll.** `[Bean finding, 2026-11-11, tested on /cruises]` On a fresh load only the first two sections exist. Waiting ~17s, moving the mouse, `window.scrollTo`, and synthetic `scroll`/`wheel` events all left it at 2 sections. One genuine wheel scroll took it to 9 sections (page height 1,569px to 5,645px). /travel-insurance showed the same two-section shape before scrolling; not re-tested after.
- **What Google's renderer receives on these pages was NOT checked.** `[Bean hypothesis, 2026-11-11]` It smells like the CSR problem family, but that is a hunch. Do not write it up as a finding.
- **The live site is on a new template.** `[Anna verbatim, 2026-11-11]` "the stored copy is probably the old template". The 12 pages in `crawled_pages` / `lp_blocks` (May crawls) no longer match the live markup.
- **New-template seams.** `[Bean finding, 2026-11-11]` Each CMS section is a top-level `page-section` element. Anna confirmed the 9 outlines on /cruises were the right sections. Component tags inside (`navigation-card-view`, `faqs-widget`, `recommended-destinations-carousel`, `deal-card-view`, `cruise-card`) look close to the `component_type` vocabulary in `lp_blocks`; that mapping is unverified. The old parser's `div.section-content` rule is old-template.
- **A lot of /cruises is in the HTML but not visible.** `[Bean finding, 2026-11-11]` 57 deal cards in one carousel (about 3 visible) held 1,098 of 2,146 visible words; collapsed FAQ answers were 806 words of text against 74 visible.
- **ama.ab.ca is a different build.** `[Bean finding, 2026-11-11]` Homepage loads fully with no scroll; 13 plain `section` tags, not checked against the real sections. A cloud fetch counted 1,012 words where a laptop browser counted 1,741. Unexplained.

## What was built (piece 1)

Quick Scorer has a URL field and "Fetch page" button. Commit `fff736c` on branch `bean/quick-scorer-url-fetch`.

- `netlify/functions/fetch-page.mts` is a background function at `/api/fetch-page`; `netlify/functions/fetch-result.mts` at `/api/fetch-result` hands the result back once and deletes it. Nothing is saved.
- `netlify/lib/fetch-page.mjs` drives headless Chrome and scrolls with `page.mouse.wheel` on purpose (see the scroll gate above).
- Scripts, styles, icons, comments and inline image data are stripped; header and footer stay in. `[agreed with Anna, 2026-11-11]`

Gotchas that cost time:
- A one-shot (non-background) function is cut off at 30s with a 504, well before the documented 60s. Hence ask-then-poll.
- Full `@sparticuz/chromium` (~70 MB) is too big to bundle; `-min` downloads the pack at cold start. `puppeteer-core` is pinned to the release matching Chromium 153. Bump both together, plus the pack URL.
- `config.rateLimit` with an extra `action` key deployed fine and was silently not enforced. The working shape is `{ windowLimit, windowSize, aggregateBy }`. Enforcement lags about 10s, so test with a burst, a pause, then more.
- Measured on a cold start: about a minute for /cruises, about 90s for the ama.ab.ca homepage.

## The agreed plan for the bigger feature

`[Bean synthesis, 2026-11-11, validated by Anna step by step]` Three pieces, built in this order, each shipped for Anna to try before the next:

1. **Fetcher + Quick Scorer URL box.** Built (above).
2. **Pull into Page Analysis, pre-chopped.** Sections from the page's own seams. Blocks as a first guess.
3. **Fix-up on the page.** Anna chose fixing blocks on a picture of the page (not in a list), and chose a tray under each section for pieces that are in the HTML but not visible.

Settled with Anna:
- "Folds" means the CMS section blocks.
- Her plan B (select chunks and group them) is curation done in the app; the Lab-reads-curation rule in lab-fidelity-mission-critical.md stays intact.

Bean guesses shown to Anna and not objected to, but not explicitly confirmed. Re-check before building on them:
- Block grain should follow her existing curation: a whole card is one block, a heading with its tabs is one block, each value prop is one block.
- A pull is saved as a new dated copy and never overwrites the curated pages.
- Collapsed FAQ answers and off-screen carousel cards come in; header and footer stay out of Page Analysis.
- The page picture is a screenshot taken at fetch time with box positions saved beside it (plumbing call).

Raised and not yet reacted to: a live pull is what a visitor's browser sees, not what Google rendered, so it should not be mixed with the GSC-faithful baselines.

Pieces 2 and 3 are visible surfaces. co-design-surfaces-with-anna.md applies in full.
