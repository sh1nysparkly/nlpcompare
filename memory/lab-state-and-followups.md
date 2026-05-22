# Lab dashboard: state and followups (as of May 17, 2026)

Source-of-truth for what's currently in `lab/index.html` + `lab/styles.css`, what's deployed at signal-coherence.netlify.app, and what's queued for follow-up work. Update as fixes ship / new threads open.

This doc exists because the May 14 cleanup session uncovered several spots where things that LOOK like dead code are actually load-bearing. Future Beans (and Anna across sessions) need a place to find the "don't delete this even though it looks dead" notes before grepping leads them astray.

## Deployment model

- Netlify site (signal-coherence.netlify.app) is now **git-connected** (Anna wired it up after the May 14 session). Pushes to `main` should auto-deploy. Site ID: `cab32293-702d-4f95-9358-d293c666ff96`.
- **Publish directory: `lab/`** (discovered May 17). Netlify serves `lab/index.html` at the site root, not `lab/` -- `/` returns it, `/lab/` returns 404. Anything that needs to be served (favicons, manifest, future static assets) must live under `lab/`, not at repo root. Favicons currently exist in both places (`/favicon.ico` and `/lab/favicon.ico`) -- the deploy reads the lab/ copy; the root copies are dead weight unless the publish dir ever changes.
- If a deploy doesn't fire on push, the git connection has drifted -- fall back to Netlify CLI/MCP deploy from inside `lab/` and reconnect via the Netlify dashboard.
- Pre-May-15 history: the site was deploy-by-API-only and the repo + Netlify drifted on May 14 when Cowork Beans landed fixes only on Netlify. Keep an eye out for that pattern recurring; the git connection is the fix but the older deploys remain api-sourced.

## Architecture facts to preserve (read these before deleting anything that looks dead)

### V0.5 is a load-bearing concept

V0.5 = the UX-proposed baseline state for a page. One per page. Anna validates UX redesigns by comparing V0 (live production) against V0.5 (UX proposal). V1, V2, ... are experimental perms layered on top.

DB column: `permutations.is_ux_baseline`. Version-list `kind`: `"ux_baseline"`. (Both were renamed May 15 from the prior legacy `is_snap_target` / `kind: "recipe"`, which referenced the long-deleted snap-to-recipe ordering feature.)

Hot paths that read V0.5:
- `getVersionList()` -- inserts V0.5 between V0 and V1 if a perm with `is_ux_baseline=true` exists, AND excludes those perms from the regular V1+ numbering via `.filter(p => !p.is_ux_baseline)`
- `getCommitOverwriteTarget()` -- recognizes `kind: "ux_baseline"` for the overwrite-V0.5 commit path
- `commitOverwriteV05()` -- PATCHes the existing V0.5 row when user is on V0.5 in the active picker + the other slot is blank and hits Commit

### Scorecard columns map directly to picker slots

LEFT column = slot `"a"`'s data. RIGHT column = slot `"b"`'s data. The two pickers ARE the columns.

The "active vs pinned" distinction is **separate** and controls only:
- Which slot's content drives `labBlocks` (the editor) -- the active slot's version is what's loaded for editing
- Which column reflects fresh WIP scores -- if the active slot's content has been scored via Score WIP since loading (matched via `labRearrangedNlpVersionLabel`), that column uses `labRearrangedNlp`; otherwise uses the perm's saved `nlp_result`

Cycling either picker only changes its own column. Cycling the pinned picker does NOT reload the editor (intentional: user can flip through comparison versions without losing their edit-in-progress).

`getColumnNlp(slot)` in `renderNlpDeltas` is the single source of truth for column data. Do not reintroduce "active = baseline" mediation -- it inverts the mental model in the common case where the user keeps their baseline in the LEFT picker and cycles comparisons on the RIGHT.

### Per-container NLP must re-run on version change

Container IDs are positional (`"c0"`, `"c1"`, ...). The same id can hold different content in V0 vs V0.5 vs V_n. `scoreContainersForLab()` must be called whenever a version is loaded so the entities panel reflects the loaded version, not whichever version was loaded at page-load time.

The function takes `{ blocks, versionTag }`. `loadVersionIntoLab` calls it with `labBlocks` and a version-specific cache tag. HTML synthesis falls back to `regenerateBlockHtml(b)` for trimmed-perm blocks that don't carry `original_html`. The `cacheMeta.source` includes the version tag so different versions cache separately.

### Container pattern dropdown is the canonical container affordance (May 17)

The pattern dropdown (`<select class="container-pattern-select">`) replaced the `+ assign slot` button + recipe-slot pill UI in the container header. The slot data structures (`labSections`, `labRecipeSlots`, `openSlotPickerForContainer`) are still in code for back-compat but no UI exposes them anymore. See `memory/pattern-taxonomy-and-export.md` for the full substrate writeup.

Ghost is now a pattern option (`pattern_key="ghost"`, displayed as "🫥 Misc" after Anna's relabel). Selecting it flips `block.is_ghost=true` on every block in the container, so the existing scoring-exclusion path picks it up. Legacy `slotInfo.is_ghost` still honored on load for any pre-pattern data.

Container left-edge color now reads from `pattern.badge_color` instead of `slotInfo.color`. Empty pattern = no border.

### `slimBlockForSnapshot()` is now self-contained (May 17)

All three commit paths (save-new, V0.5 overwrite, V0 overwrite) use this helper. As of May 17 it ALWAYS includes `original_html`, `block_id`, `curation_block_id`, and `pattern_key` so saved permutation rows are self-contained -- reloading + re-scoring doesn't need the underlying curation/crawl data.

Legacy V0/V0.5 permutations saved before this fix don't have these fields. The load-time V0 swap hydrates missing `original_html` via `regenerateBlockHtml()`; the plain-language export's diff engine falls through gracefully (everything reads as NEW/REMOVED for the missing-id case). Re-saving upgrades them.

### V0 load swap (May 17)

`loadSelectedPage` checks for an `is_live_baseline=true` row in `labPermutations` and, when found, replaces `labOriginalBlocks` / `labBlocks` / `labSections` / `labOriginalNlp` with the row's data. Hydrates missing `original_html` per block. Skips the auto `scoreOriginal()` call when the saved NLP has data (uses the saved score directly -- it's the authoritative V0 score, computed against the original parsed HTML, not the regenerated one).

**Tripwire:** if `scoreOriginal()` runs against snapshot blocks without hydrated `original_html`, it sends empty HTML to the NLP API, gets back nothing, and clobbers `labOriginalNlp`. `renderNlpDeltas()` then early-returns blank. This was the May 17 "scorecard is COMPLETELY blank" bug.

### Other live concepts not to confuse with dead snap code

- `loadRecipeSlots()` + `labRecipeSlots` -- still loads recipe slot data even though no UI surfaces it (the pill-click-to-choose-template-slot interaction was removed when the pattern dropdown replaced the slot UI). Container header rename + auto-inherit logic still uses `labSections` for ghost-state checking. **Don't remove without auditing every `labSections` / `labRecipeSlots` reference.**
- The colloquial word "snap" in comments (e.g. `// Reset snaps the cursor back to V0`) is unrelated to the cut feature.
- `container_label` rename UI (double-click, `onContainerLabelBlur`, etc.) still works. Auto-inherit (sets label to first heading text on render) still fires silently. Anna had Bean nuke the badge in initial work and immediately wanted it back; treat as load-bearing wayfinding.

## Recent commits (May 14, 2026, branch `claude/cleanup-index-html-Ml9b0`)

**Honesty note (added May 15 2026):** This list previously claimed "All synced to Netlify" -- which conflated "Netlify has it" with "the repo has it." Three of the entries below (`c29f073`, `fee08ec`, `ff336dc`) were applied to Netlify via Cowork-Bean sessions but **never committed to the repo**. They're listed here as historical breadcrumbs, NOT as commits a future Bean can check out. When the Netlify git-connection landed (May 15) and an auto-deploy from `main` ran, the Netlify-only fixes got overwritten and had to be rebuilt (see below). **Don't add an entry to this list unless you've verified `git rev-parse <sha>` resolves.**

In repo:
- `ff103bf` -- Cleanup: removed snap-to-recipe machinery, restyled "Showing V*" status chip to harmonize with `.outline-btn`, extracted inline `<style>` to `lab/styles.css`
- `5545b54` -- Restored V0.5 picker entry (regression fix from ff103bf where V0.5 got bundled into the snap cleanup); removed vestigial `setAsBaseline` / `labBaselineVersionRef` / `"Pinned"` machinery
- `a09d7d6` -- Reset to baseline now also resets dual pickers + chip (pre-existing bug surfaced during the session)
- `31dd4e8` -- Synced Cowork Bean fixes from Netlify (`await loadPermutations`, real `updateFlipperUI` rebuild, `commitSavePermutation` always reads live editor state); fixed per-container entity attribution to be version-aware via the parameterized `scoreContainersForLab`

Netlify-only (NEVER committed; lost when git auto-deploy from `main` overwrote prod May 15):
- `c29f073` -- Added `.gitignore` (`.netlify/` + common OS noise). Not currently in the repo; recreate if/when desired.
- `fee08ec` -- Initial delta-baseline-from-pinned-picker fix (was superseded by ff336dc on Netlify).
- `ff336dc` -- Scorecard columns map directly to picker slots (LEFT=a, RIGHT=b). The lab-state section above describes the intended behavior. **Rebuilt fresh May 15** as `getColumnNlp(slot)` -- see the May 15 entry below.

## Recent commits (May 15, 2026, branch `claude/update-supabase-widgets-YSP6I` + main)

- `675e4d3` -- Per-block `</>` toggle for HTML-authored block bodies + container-level `</>` diagnostic popover. `html_authored` flag + `body_html` field; `regenerateBlockHtml` short-circuits to authored HTML verbatim so NLP receives real structure. Persisted via both save paths.
- `3d30ce1` -- Rebuild of the lost `ff336dc` fix: `getColumnNlp(slot)` routes scorecard columns to picker slots (LEFT=a base, RIGHT=b comparison). Active-slot WIP override via `labRearrangedNlpVersionLabel` match on shortLabel.

## Recent commits (May 17, 2026, branch `claude/explore-and-plan-phase-1-G9PZq` → merged to main)

Cluster 1 of the Phase 1 plan + extensive design-pass iteration with Anna. Eight commits total.

- `ab01812` -- Favicon + "Rearrangement Lab" → "Page Lab" rename (T5 + T6). Lab title kept as "Signal Coherence Dashboard" (suite-level name preserved).
- `36ca8a4` -- Captured-ideas batch: auto-inherit container heading from top H-tag text (when label source is `fallback-heading` or `auto-inherited`), entity counts in scorecard column headers, ToC sidebar (first pass as slide-in from toolbar button).
- `6f9bf6b` -- Pattern taxonomy substrate (item 6): `lp_blocks.pattern_key` column, `LAB_PATTERNS` enum with 11 canonical patterns, container header dropdown replacing the label badge. Plain-language export (item 7): modal with V0.5/V0 diff target radio, markdown outline + entity/category TSV tables, ARIA notes for tabs/accordion patterns. **See `memory/pattern-taxonomy-and-export.md` for the full substrate writeup.**
- `8dbee36` -- T10 V0 overwrite: `permutations.is_live_baseline` column, `commitOverwriteV0()` paralleling `commitOverwriteV05()`, V0 picker entry sources from flagged row when present, load-time swap of `labOriginalBlocks` / `labOriginalNlp` from the flagged row.
- `2893de1` -- Design-pass after first review: container label badge restored (Bean over-eagerly removed it), pattern dropdown moved to where `+ assign slot` was (slot UI decommissioned), pattern dropdown styled to match slot-pill aesthetic, "(rare)" / "Grid" / "(general)" suffixes stripped from pattern names, scorecard column headers show V0/V0.5/V1 labels instead of Base/Now, "Entity" column header removed, unique-entity counts moved to scorecard footer, ToC redesigned as binder-divider tab on the left edge, favicon files copied into `lab/` so the deploy actually picks them up.
- `db516f3` -- Second design-pass: ToC tab repositioned to align with panel-head divider line, container names suppressed in ToC body (pure heading outline now), Ghost re-added as pattern entry (selecting it flips `block.is_ghost`), "Assign pattern" capitalized, container left-edge color tied to `pattern.badge_color`, ENTITIES title moved into first `<th>` of the scorecard table.
- `a4e6d1d` -- Shrink pattern pill: `field-sizing: content` so the pill fits the SELECTED option instead of widest (was forcing every pill to ~280px wide). Tighter padding, smaller letter-spacing, font shaved 0.5pt. "Ghost (exclude from scoring)" shortened to "Ghost". Block-count display removed from container header. Anna's correctly-cropped favicon set copied into lab/.
- `8f0963a` -- Binder tab color swap (idle = old hover blue tint; hover = warm yellow + purple text echoing the Flights pill). Pattern pill text centered via `text-align-last: center` (the property that actually targets `<select>` closed-state rendering; standard `text-align` doesn't catch it).
- `83c8c90` -- Bug fix: pages with committed V0 baselines (`/flights`, `/destinations`, `/things-to-do`) loaded with blank scorecards because legacy V0 rows lacked `original_html` -- `scoreOriginal()` was sending empty HTML to NLP and nulling `labOriginalNlp`. Fix: `slimBlockForSnapshot` now always saves `original_html`; load-time swap hydrates missing `original_html` via `regenerateBlockHtml()`; auto-`scoreOriginal()` skipped when V0 row has authoritative saved NLP. See "V0 load swap" section above.

**Schema migrations applied via Supabase MCP** (project `signal-coherence` / `ghzfrxxevjjfgpxvmahy`):
- `add_pattern_key_to_lp_blocks` (May 17) -- `lp_blocks.pattern_key text` (nullable)
- `add_is_live_baseline_to_permutations` (May 17) -- `permutations.is_live_baseline boolean NOT NULL DEFAULT false`

## Open followups

### Renames / migrations

- **`ux_recipe_section` column**: still exists in `lp_blocks` but no code reads it after the May 14 cleanup. **Investigated May 15: holds 995 rows of hand-curated v12 section archetypes (Hero, FAQs & Resources, Travel Product Carousel, etc.) -- intentionally kept as curation reference data even though no code reads it. Do NOT drop without explicit go-ahead from Anna.**

### Half-wired flows

- ~~**V0 overwrite**: stubbed.~~ **SHIPPED May 17 (commit `8dbee36`).** `is_live_baseline` column on `permutations`, `commitOverwriteV0()` wired. Confirmation prompt before overwrite. INSERTs on first commit, PATCHes thereafter.
- **Stale `labRearrangedNlp` on version switch**: still open. `loadVersionIntoLab` doesn't clear `labRearrangedNlp`, so if user scores V7 then switches active to V8 without re-scoring, V8's column shows V7's old score. The freshness check in `getColumnNlp` compensates by routing to perm's `nlp_result` when `labRearrangedNlpVersionLabel` doesn't match `ver.shortLabel`. Band-aid in place; cleaner fix is to clear both `labRearrangedNlp` and `labRearrangedNlpVersionLabel` in `loadVersionIntoLab`. (Listed as T14 in Phase 1 Plan.)
- **Slot UI scaffolding still in code, no longer surfaced**: the May 17 pattern dropdown replaced `+ assign slot` in the container header, but `labSections` / `labRecipeSlots` / `loadRecipeSlots()` / `openSlotPickerForContainer()` are all still present. Future Bean could do a sweep to remove if confident, but audit every `labSections` reference first (some live code paths still touch it for ghost-state checking).
- **Container rename UI is functionally orphaned**: `enterContainerLabelEdit` / `onContainerLabelBlur` / `onContainerLabelKeydown` still exist and work, but the new pattern-dropdown-first layout makes the rename less prominent. Auto-inherit fires on render so labels usually look right without rename. Keep or remove depending on Anna's preference after using it for a bit.

### Architectural threads (May 14 audit + May 17 reground; lean scope kept us out of these)

Counts refreshed May 17 (drift from May 14 audit noted in parens):

- ~50 global `let`/`const` state variables; no dispatch pattern. Real brittleness -- easy to mutate state and forget to call the right render function.
- `renderBlockList()` is **236 lines** (was 204); mixes section headers, blocks, container headers, and entity panels. Split per concern. `renderContainerHeader` is **133 lines** -- separate split target.
- **48 inline `onclick=` attributes** (was 34); replace with event delegation on `#block-list`.
- Drag-drop handlers all live in the same file; could extract to a module.
- **21 `!important` declarations** (was 19, mostly GridJS overrides).
- Button base styles duplicated across `.outline-btn` / `.reset-btn` / `.group-btn` / `.clear-btn`.
- No tests anywhere. Highest-leverage targets: NLP unwrapping (defensive against MCP response shapes), version-list builder, commit-overwrite-target detection.
- 26 console statements with no DEBUG gate; 3 fetches with no AbortController/timeout; magic constants (truncation lengths, entity display limits) inline rather than promoted.

**May 17 framing:** This is real work but it's long-game eng investment, NOT a prereq for the captured-ideas backlog. Time-box "just enough" before piling more features on top -- event delegation + stale `labRearrangedNlp` clear are the just-in-time pair (see "May 17 execution staging" below). Full refactor only worth it if 6+ more months of building on this codebase are planned.

### May 17 execution staging

Per the north star §10 May 17 recalibration: **V0 overwrite + captured-ideas backlog are the immediate-sprint Lab leverage** -- they make Anna's next round of Track 2 work (optimization-guide building + UX validation) materially less painful. Recommended staging for Bean-led work picking these up:

**~~Cluster 1~~ -- SHIPPED May 17.** V0 overwrite + all six captured ideas in scope (favicon, Lab rename, sidebar ToC as binder-tab, entity count per version, auto-inherit container heading, version-win plain-language export). Plus the pattern-taxonomy substrate that wasn't originally in Cluster 1 but became foundational for the export. Eight commits, two schema migrations, ~16 hours of Anna-in-the-loop design iteration.

**Cluster 2 -- Light refactor, just-in-time.** *Still open.* Event delegation (replace the now-50+ inline handlers with one delegated listener on `#block-list`) + stale `labRearrangedNlp` clear in `loadVersionIntoLab` (cleaner fix for the band-aid noted in "Half-wired flows"). ~1 session. NOT the full state-centralization rabbit hole.

**Cluster 3 -- Resume captured ideas on cleaner ground.** *Still open.* Marking winning version, entity colour-coding by type, click-entity-highlight-containers, permutation-aware edit state, card duplication, block deletion, in-line comments. Each gets its own thread when promoted -- they're not a single deliverable.

**Cluster 4 (optional) -- Full refactor.** *Still open.* Full state centralization (~50 globals → labState + dispatch + subscribe), render-function split (renderBlockList 236L + renderContainerHeader now 150+L per-concern), drag-drop module extraction, test infrastructure. Worth it only if 6+ more months of building on this codebase are planned. Otherwise skip.

Reading: don't do the full refactor first. Time-box to "just enough" (cluster 2) to keep velocity. Cluster-4 work is long-game eng investment, not a prereq for the captured-ideas backlog.

## Other fixes to capture

*Anna: add your other fixes / observations here so they don't get lost. Each entry: short heading + 1-3 lines on what's broken and the rough fix. Future Beans should be able to pick up any of these without back-channel context.*

### Shipped May 17 (cross-referenced from idea list)

- ~~Add a favicon~~ -- SHIPPED in `ab01812`; later swapped to Anna's correctly-cropped artwork in `a4e6d1d`. Lives in `lab/` (the deploy root), duplicated at repo root for completeness.
- ~~Slide-in sidebar ToC for tag structure/flow validation~~ -- SHIPPED as binder-divider tab on the left edge (`db516f3`). Click expands; vertical text, warm-yellow hover.
- ~~Count of entities per version / per container~~ -- SHIPPED in `2893de1` as a scorecard footer (per-version totals; container-level NLP already showed per-container counts).
- ~~Container heading auto-inherited from top text~~ -- SHIPPED in `36ca8a4`. `container_label_source` set to `auto-inherited` when the first heading text overrides the imported label.
- ~~"Rearrangement Lab" rename~~ -- SHIPPED as "Page Lab" in `ab01812`.
- ~~Version-win plain-language export~~ -- SHIPPED in `6f9bf6b`. Modal with V0.5 default / V0 fallback diff target, markdown outline + entity/category TSV tables. See `memory/pattern-taxonomy-and-export.md`.
- ~~`</>` button to inspect API payload~~ -- DONE May 15 in `675e4d3`.

### Still open from original list

- Marking certain versions as the "winning" version for later tracking
- What if the matrix was just like, side-by-side versions of the delta sidebar (intersects T23)
- Adding in keyword lists/highlighting as another reference layer (T12)
- Bootstrap-style container layout to mimic Figma
- Entity + topic data → topic map / IA (stretch)
- Click entity in sidebar → highlight relevant containers
- Entity colour-coding by type (PLACE vs PERSON)
- Permutation-aware edit state tracking (edited-from-which-version, not just edited-from-base)
- Card duplication
- Save reusable containers/blocks (global-ish components)
- In-line comments / notes
- Block deletion (Anna's later clarification: needed; touches drag-drop logic)

### New May 17 (discovered during the design pass)

- **Container left-edge color for ghost** is currently the same neutral grey as no-pattern (per `pattern.badge_color` lookup). Might want distinct ghost styling on the left edge so ghost containers visually stand out from "no pattern assigned yet" containers.
- **Load path leaks null `block_id`** -- root cause of the V0/V0.5 export diff bug. `slimBlockForSnapshot` (lab/index.html:4954) correctly preserves block_id when present, but the editor's `labBlocks` hydration path doesn't assign UUIDs to blocks coming in without IDs. As of May 19 every existing row was patched by migration `backfill_permutations_block_ids` (all 81 rows, 5351 blocks); but any future load of source data lacking IDs will round-trip nulls back through save. Concrete fix: in `loadVersionIntoLab` and the initial-load path, generate a UUID for any block lacking all three of `block_id` / `curation_block_id` / `id` before it touches `labBlocks`. Not blocking once the data is patched -- but the next genuinely-new-from-curation page load will resurrect the bug.
- **`field-sizing: content` browser support**: Chrome 123+ / Safari 17.4+ / Firefox 123+. Pattern pills will bloat to widest-option width on older browsers. If a stakeholder ever views the Lab on something old, swap to a JS measurer. Not urgent.
- **Slot machinery cleanup**: `+ assign slot` UI is gone but the supporting code (`labSections`, `labRecipeSlots`, `loadRecipeSlots()`, `openSlotPickerForContainer()`, `assignSlotToContainer()`) is all still in place. Future Bean could do a sweep if confident -- audit every `labSections` reference first, some live code paths still touch it for ghost-state checking and for the section-grouping multi-select.
- **`scoreOriginal()` should not nuke a valid `labOriginalNlp`** even on its own merits. If the NLP call errors or returns empty, the existing labOriginalNlp (e.g., from V0 row) should remain. The May 17 fix routes around this by skipping `scoreOriginal()` when we already have authoritative data; the underlying brittleness still exists for the no-V0-row case. Worth a defensive check.
