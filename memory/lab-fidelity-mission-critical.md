---
name: Lab fidelity is mission-critical -- v9 curation is structural ground truth
description: The Live Lab must measure exactly what Anna's curation says is on the page, not whatever an HTML parser infers. Drift between Lab and workbook is not acceptable. Future Beans must NOT propose "we'll just use what the Lab parses" -- Anna has explicitly rejected that as lossy.
type: feedback
originSessionId: 136d72e1-4763-4a25-bec9-e7c8b98b2a08
---
The Live Lab dashboard is the experimental apparatus Anna built so she can move blocks/containers around and measure signal shifts. For that apparatus to be trustworthy, **what the UI shows as a "block" must correspond exactly to what's actually on the page, in the groupings Anna has manually curated**.

Today (May 11, 2026) the Lab's data layer parses HTML out of `crawled_pages.html_content` directly. That parsing is lossy in three ways:
1. **Container groupings come from DOM structure inference, not from Anna's screenshot-derived visual containers.** Often these disagree.
2. **Some visible text gets dropped** -- `<div>` value-prop subheads, `<img alt>` brand names on Trusted Partners carousels, sub-head/intro paragraphs that the parser misses.
3. **Some hidden HTML gets included** -- e.g. the destinations page has FAQ markup the devs added that isn't visually rendered; an HTML parser doesn't know to skip it.

Anna's v9 workbook (`signal-coherence/outputs/curation/lp-curation-v9-all-pages.xlsx`) had all of that reconciled by hand against the actual page screenshots. **Post-May-11 cleanup pass, the current canonical version is `lp-curation-v12-all-pages.xlsx`** (v9 → v10 → v12 via Bean-run scripts; see [session-notes-may11-cleanup.md](signal-coherence/session-notes-may11-cleanup.md)). Future curation versions follow the same `v{N}` naming. **That curation is the structural ground truth.** The Lab must consume v{latest}-equivalent data, not re-parse HTML and hope.

**Why:** May 11 2026 session. Anna, after I floated "the Lab measurements will be slightly thinner than v9 measurements": "no I'm not ok with the measurements being thinner that's what all of this is FOR, Bean. So I'd want to do whatever we needed to do to the schemas in supabase or whatever so that each block on the ui side is tied to the corresponding html 'block'. the whole point is so we can see what happens if we move things around at a container level and/or a block level with as much accuracy as possible." This was directly after a 14-hour curation slog with repeated parser/sweep failures producing lossy data. Lab fidelity is non-negotiable downstream of that work.

**How to apply:**

- When proposing Lab improvements, the workbook (latest `v{N}` — v12 as of May 11) is canonical for: container groupings, block content, tag-per-block, container labels, what/section annotations.
- The HTML (in `crawled_pages.html_content` for May 8 baseline, or fresher `signal-coherence/<slug>.html` for later pulls) is canonical for: the text that the rendered DOM actually contains (i.e. what Googlebot sees). But the Lab does NOT consume HTML directly going forward -- it consumes Anna's curation, which is itself a faithful (and corrected) representation of what's visible in the rendered HTML.
- Never propose "we'll just have the Lab re-parse the HTML and merge with curation" -- that's the lossy path. The Lab parses the curation table, full stop.
- A Phase A workstream covers the migration: design Supabase schema for blocks + containers that mirrors v{latest}; populate from it; refactor Lab to read it. **[A.1 shipped May 11 2026]** -- see `experimental-phasing.md` for the deployed state and `signal-coherence/outputs/curation/v12-lab-parity-handoff.md` for the parity table.
- If a Bean is about to ship a Lab change that breaks parity with v{latest}, stop. Anna will treat lossy measurements as a step backwards, not progress.

**Synthesis convention (load-bearing for parity)**: as of cycle 3 (May 11 2026 PM), both the Lab's `blocksToHtml` AND the diagnostic's `synthesize_html` join rows with `"\n"` (not empty string), no doctype/html wrapper. This matters because:

1. Adjacent inline tags across block boundaries (e.g. `<a>...11 min read</a><a>ARTICLE Oceania...</a>` in the Related Articles carousels) get their text glued by NLP if there's no whitespace between them -- the `\n` separator forces a word boundary.
2. The diagnostic and Lab MUST use the same separator or content_hash parity breaks -- cache lookup is content-hash keyed, so any synthesis divergence means cache misses and divergent NLP responses.

If a Bean changes the separator (in either layer), the OTHER layer must change too, and the baseline diagnostic must be regenerated. Do not change without explicit consideration of this coupling.
