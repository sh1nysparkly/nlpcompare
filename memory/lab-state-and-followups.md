# Lab dashboard: state and followups (as of May 14, 2026)

Source-of-truth for what's currently in `lab/index.html` + `lab/styles.css`, what's deployed at signal-coherence.netlify.app, and what's queued for follow-up work. Update as fixes ship / new threads open.

This doc exists because the May 14 cleanup session uncovered several spots where things that LOOK like dead code are actually load-bearing. Future Beans (and Anna across sessions) need a place to find the "don't delete this even though it looks dead" notes before grepping leads them astray.

## Deployment model

- Netlify site (signal-coherence.netlify.app) is **NOT git-connected**. Deploy rows show `commit_ref: null`, `branch: null`, `deploy_source: "api"`. Pushes to the repo do not auto-deploy.
- To deploy: from inside `lab/`, run the Netlify CLI/MCP. Site ID is `cab32293-702d-4f95-9358-d293c666ff96`. `.netlify/state.json` (gitignored) carries the local binding.
- Repo and Netlify can drift if changes are made on either side without the other. **Always reconcile both directions** before assuming the repo represents truth -- May 14 had a case where Cowork Beans had landed three fixes only on Netlify and the repo was behind.

## Architecture facts to preserve (read these before deleting anything that looks dead)

### V0.5 is a load-bearing concept, not a snap-to-recipe artifact

V0.5 = the UX-proposed baseline state for a page. One per page. Anna validates UX redesigns by comparing V0 (live production) against V0.5 (UX proposal). V1, V2, ... are experimental perms layered on top.

The DB column is named `is_snap_target` and the version-list `kind` is `"recipe"`. **Both names are legacy** -- they reference the cut snap-to-recipe ordering feature, not the concept they actually serve. Future-Bean trap: grepping "snap" will surface V0.5 code and look like dead snap residue. **Don't delete it.**

Hot paths that read V0.5:
- `getVersionList()` -- inserts V0.5 between V0 and V1 if a perm with `is_snap_target=true` exists, AND excludes those perms from the regular V1+ numbering via `.filter(p => !p.is_snap_target)`
- `getCommitOverwriteTarget()` -- recognizes `kind: "recipe"` for the overwrite-V0.5 commit path
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

### Other live concepts not to confuse with dead snap code

- `loadRecipeSlots()` + `labRecipeSlots` -- populates the container-header slot picker UI (the pill-click-to-choose-template-slot interaction). Has nothing to do with snap-to-recipe ordering. **Don't remove.**
- `kind: "recipe"` in the version list = V0.5, not the snap feature.
- The colloquial word "snap" in comments (e.g. `// Reset snaps the cursor back to V0`) is unrelated to the cut feature.

## Recent commits (May 14, 2026, branch `claude/cleanup-index-html-Ml9b0`)

All synced to Netlify.

- `ff103bf` -- Cleanup: removed snap-to-recipe machinery, restyled "Showing V*" status chip to harmonize with `.outline-btn`, extracted inline `<style>` to `lab/styles.css`
- `5545b54` -- Restored V0.5 picker entry (regression fix from ff103bf where V0.5 got bundled into the snap cleanup); removed vestigial `setAsBaseline` / `labBaselineVersionRef` / `"Pinned"` machinery
- `a09d7d6` -- Reset to baseline now also resets dual pickers + chip (pre-existing bug surfaced during the session)
- `31dd4e8` -- Synced Cowork Bean fixes from Netlify (`await loadPermutations`, real `updateFlipperUI` rebuild, `commitSavePermutation` always reads live editor state); fixed per-container entity attribution to be version-aware via the parameterized `scoreContainersForLab`
- `c29f073` -- Added `.gitignore` (`.netlify/` + common OS noise)
- `fee08ec` -- Initial delta-baseline-from-pinned-picker fix (superseded by ff336dc)
- `ff336dc` -- Scorecard columns now map directly to picker slots (LEFT=a, RIGHT=b)

## Open followups

### Renames / migrations

- **`is_snap_target` -> `is_ux_baseline`**: DB column rename + JS references update (`kind: "recipe"` -> `kind: "ux_baseline"`, comments). Removes the future-Bean trap of "this looks like snap residue."
- **`ux_recipe_section` column**: still exists in `lp_blocks` but no code reads it after the May 14 cleanup. Drop the column if you want a clean schema.

### Half-wired flows

- **V0 overwrite**: stubbed. `getCommitOverwriteTarget()` returns `{ kind: "v0" }` but `showSavePermInput` shows a toast `"V0 overwrite needs a schema migration -- coming next pass. Falling back to 'save as new'."` and falls through. Real work: add `is_live_baseline` (or similar) column to `permutations`, then add `commitOverwriteV0()` paralleling `commitOverwriteV05`.
- **Stale `labRearrangedNlp` on version switch**: `loadVersionIntoLab` doesn't clear `labRearrangedNlp`, so if user scores V7 then switches active to V8 without re-scoring, V8's column shows V7's old score. The freshness check in `getColumnNlp` compensates by routing to perm's `nlp_result` when `labRearrangedNlpVersionLabel` doesn't match `ver.shortLabel`. Band-aid in place; cleaner fix is to clear both `labRearrangedNlp` and `labRearrangedNlpVersionLabel` in `loadVersionIntoLab`.

### Architectural threads (from the May 14 audit, lean scope kept us out of these)

- ~50 global `let`/`const` state variables; no dispatch pattern. Real brittleness -- easy to mutate state and forget to call the right render function.
- `renderBlockList()` is 204 lines; mixes section headers, blocks, container headers, and entity panels. Split per concern.
- 34 inline `onclick=` attributes; replace with event delegation on `#block-list`.
- Drag-drop handlers all live in the same file; could extract to a module.
- 19 `!important` declarations (mostly GridJS overrides).
- Button base styles duplicated across `.outline-btn` / `.reset-btn` / `.group-btn` / `.clear-btn`.
- No tests anywhere. Highest-leverage targets: NLP unwrapping (defensive against MCP response shapes), version-list builder, commit-overwrite-target detection.

## Other fixes to capture

*Anna: add your other fixes / observations here so they don't get lost. Each entry: short heading + 1-3 lines on what's broken and the rough fix. Future Beans should be able to pick up any of these without back-channel context.*

-not fixes just ideas: marking certain versions as the "winning" version for later like, tracking
- what if the matrix was just like, side-by-side versions of the um, delta sidebar
- can we add a favicon?
- adding in keyword lists/highlighting as another reference layer
- I want to add the slide-in sidebar ToC on the lefthand side to help validate tag structure/flow of the page
- show a count of entities (might be informative to see if the count of unique entities goes up or down significantly with copy changes
- it'd be cool if each container operated like a bootstrap container and I could arrange the blocks in a way that mimics the Figma layout (easier to do audits/comparisons)
- what about a little </> button inside containers or at the page level so I can look at exactly what's passed to the API (could help w troubleshooting/diagnostics)
- could the container heading/text automatically be inherited from the top text contained within?
- as a stretch goal might be cool to use this entity and topic data to create like a topic map/IA
- what if when I click on an entity in the sidebar it sort of highlights the relevant container(s)
- what about colour-coding for entities? like in the NLP API? Like PLACE vs PERSON kinda thing?

