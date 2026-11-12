---
name: Live URL fetch, the amatravel.ca scroll gate, and the agreed three-piece plan
description: What the Nov 11 2026 session found about fetching live AMA pages, what was built (Quick Scorer URL fetch), and the design Anna agreed for pulling live pages into Page Analysis. Read before touching netlify/ or designing pieces 2 and 3.
type: project
---
Tags follow memory-attribution-discipline.md.

## Where to pick up (written Nov 12 2026, end of session)

Piece 1 is live, with the Clean HTML option. Next is piece 2: pulling a fetched page into Page Analysis pre-chopped. It is a visible surface, so start by co-designing with Anna, not by building. First job: re-check the "Bean guesses" listed under the agreed plan below, since she saw them but never explicitly confirmed them. Anna asked for these notes so the next Bean could start fresh.

## Findings about the live sites

- **amatravel.ca holds back most of a landing page until a real scroll.** `[Bean finding, 2026-11-11, tested on /cruises]` On a fresh load only the first two sections exist. Waiting ~17s, moving the mouse, `window.scrollTo`, and synthetic `scroll`/`wheel` events all left it at 2 sections. One genuine wheel scroll took it to 9 sections (page height 1,569px to 5,645px). /travel-insurance showed the same two-section shape before scrolling; not re-tested after.
- **What Google's renderer receives on these pages was NOT checked.** `[Bean hypothesis, 2026-11-11]` It smells like the CSR problem family, but that is a hunch. Do not write it up as a finding.
- **The live site is on a new template.** `[Anna verbatim, 2026-11-11]` "the stored copy is probably the old template". The 12 pages in `crawled_pages` / `lp_blocks` (May crawls) no longer match the live markup.
- **New-template seams.** `[Bean finding, 2026-11-11]` Each CMS section is a top-level `page-section` element. Anna confirmed the 9 outlines on /cruises were the right sections. Component tags inside (`navigation-card-view`, `faqs-widget`, `recommended-destinations-carousel`, `deal-card-view`, `cruise-card`) look close to the `component_type` vocabulary in `lp_blocks`; that mapping is unverified. The old parser's `div.section-content` rule is old-template.
- **A lot of /cruises is in the HTML but not visible.** `[Bean finding, 2026-11-11]` 57 deal cards in one carousel (about 3 visible) held 1,098 of 2,146 visible words; collapsed FAQ answers were 806 words of text against 74 visible.
- **ama.ab.ca is a different build.** `[Bean finding, 2026-11-11]` Homepage loads fully with no scroll; 13 plain `section` tags, not checked against the real sections. A cloud fetch counted 1,012 words where a laptop browser counted 1,741. Unexplained.

## What was built (piece 1)

Quick Scorer has a URL field and "Fetch page" button. Commit `fff736c`, live on `main` since Nov 12 2026.

- `netlify/functions/fetch-page.mts` is a background function at `/api/fetch-page`; `netlify/functions/fetch-result.mts` at `/api/fetch-result` hands the result back once and deletes it. Nothing is saved.
- `netlify/lib/fetch-page.mjs` drives headless Chrome and scrolls with `page.mouse.wheel` on purpose (see the scroll gate above).
- Scripts, styles, icons, comments and inline image data are stripped; header and footer stay in. `[agreed with Anna, 2026-11-11]`

Gotchas that cost time:
- A one-shot (non-background) function is cut off at 30s with a 504, well before the documented 60s. Hence ask-then-poll.
- Full `@sparticuz/chromium` (~70 MB) is too big to bundle; `-min` downloads the pack at cold start. `puppeteer-core` is pinned to the release matching Chromium 153. Bump both together, plus the pack URL.
- `config.rateLimit` with an extra `action` key deployed fine and was silently not enforced. The working shape is `{ windowLimit, windowSize, aggregateBy }`. Enforcement lags about 10s, so test with a burst, a pause, then more.
- Measured on a cold start: about a minute for /cruises, about 90s for the ama.ab.ca homepage.
- `[Bean finding, 2026-11-12, measured]` ama.ab.ca is slow to build in the Netlify browser: about 23s to DOMContentLoaded and 27-28s for `page.goto` to return, against about 2s for amatravel.ca. The original 30s navigation limit therefore failed roughly half the time on ama.ab.ca (Anna hit it on `https://www.ama.ab.ca/about`). The `www` redirect was a red herring; it resolves in under a second. Limit raised to 90s. Why ama.ab.ca is that slow there was not investigated.

## Clean HTML option (added Nov 12 2026)

- `netlify/lib/clean-page.mjs` is the `cleanPage()` function from Anna's "Clean HTML for NLP" Chrome extension (v1.0, her zip of 2026-11-12), copied unchanged. `[Anna, 2026-11-12]` It is what she means by "our cleaner". Keep it identical to the extension unless she asks; it must stay self-contained because it runs via `page.evaluate`.
- Quick Scorer has a "Clean HTML" tickbox beside Fetch page, ticked by default `[Bean call, 2026-11-12, shown to Anna on the preview]`. Unticked gives the earlier stripped-but-uncleaned snapshot.
- `[Anna, 2026-11-12]` A cleaned fetch must keep collapsed FAQ answers. The cleaner skips hidden elements, so the fetcher opens accordion panels first (`openCollapsedPanels` in `fetch-page.mjs`): `details`, plus `aria-expanded="false"` toggles whose `aria-controls` panel is `role="region"` or has accordion/collapse/faq in its class. Verified on amatravel.ca's ng-bootstrap accordion only.
- Measured: /cruises 287 KB to 75 KB (FAQ answers kept, 810 words), /hotels 143 KB to 32 KB, ama.ab.ca/about 273 KB to 11 KB.
- What the cleaner drops that the uncleaned fetch keeps: footer, non-breadcrumb nav, images and their alt text, icon ligature text, all attributes. It keeps the header's top strip and every `page-section` seam.
- Why it matters `[Anna, 2026-11-12]`: the company pays for the NLP API and she wants to be smart about it. Whether Google bills HTML markup characters was not confirmed.

## The agreed plan for the bigger feature

`[Bean synthesis, 2026-11-11, validated by Anna step by step]` Three pieces, built in this order, each shipped for Anna to try before the next:

1. **Fetcher + Quick Scorer URL box.** Built (above).
2. **Pull into Page Analysis, pre-chopped.** Sections from the page's own seams. Blocks as a first guess.
3. **Fix-up on the page.** Anna chose fixing blocks on a picture of the page (not in a list), and chose a tray under each section for pieces that are in the HTML but not visible.

Settled with Anna:
- "Folds" means the CMS section blocks.
- Her plan B (select chunks and group them) is curation done in the app; the Lab-reads-curation rule in lab-fidelity-mission-critical.md stays intact.

Confirmed by Anna on 2026-11-12 ("those all feel right!"), including the live-vs-Google separation below:
- Block grain should follow her existing curation: a whole card is one block, a heading with its tabs is one block, each value prop is one block.
- A pull is saved as a new dated copy and never overwrites the curated pages.
- Collapsed FAQ answers and off-screen carousel cards come in; header and footer stay out of Page Analysis.
- The page picture is a screenshot taken at fetch time with box positions saved beside it (plumbing call).

Live pulls stay apart from the GSC-faithful baselines `[Anna confirmed 2026-11-12]`: a live pull is what a visitor's browser sees, not what Google rendered, so it should not be mixed with the GSC-faithful baselines.

Piece 2 decisions `[Anna, 2026-11-12]`:
- The curated May pages leave the Page Analysis picker ("they'll just confuse people"). Hide them, delete nothing: Anna floated keeping a few full variation runs in Supabase for historical before/afters, undecided. `[hide-not-delete = Bean call]`
- A validated pull becomes V0 for that pull; later tweaks are V1+. Comparisons work as they do today.
- Defaults Anna accepted: pull from a URL box under "Load a page" plus "Open in Page Analysis" after a Quick Scorer fetch; section label = first heading, else component tag; hidden content comes in as blocks with a "hidden on page" badge until piece 3's tray; pulls in their own table, never `lp_blocks`.
- Plumbing consequence `[Bean finding]`: `permutations` are keyed by `slug`, so a new /cruises pull would inherit the old /cruises V1+. Each pull needs its own key. The container-pattern dropdown PATCHes `lp_blocks.pattern_key` and must write to the pull's storage instead.

Pieces 2 and 3 are visible surfaces. co-design-surfaces-with-anna.md applies in full.
