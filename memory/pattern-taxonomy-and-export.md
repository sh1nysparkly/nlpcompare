# Pattern taxonomy + plain-language export (Page Lab substrate, May 17 2026)

The Lab gained two related substrates this session that future Beans need to understand before touching the container header or the scorecard. Both originated in Anna's "captured ideas" backlog and got threaded together because the export reads pattern metadata.

Status tag: **Bean synthesis informed by Anna's PR-by-PR review** -- every decision below was either Anna-confirmed or Anna-corrected during the May 17 design pass.

---

## What replaced what

| Was | Is | Why |
|---|---|---|
| `+ assign slot` button → recipe slot pill (orange "Hotels & Accommodations") | Pattern dropdown (slate "Assign pattern" when empty, colored pill when assigned) | Anna found slots redundant with the auto-NLP category pill that sits beside them; she wanted a single curator-controlled affordance. Slot data structures (`labSections`, `labRecipeSlots`, `openSlotPickerForContainer`) are still in code for back-compat. |
| Container left-edge color from `slotInfo.color` | Container left-edge color from `pattern.badge_color` | Visual continuity with the new badge. Empty pattern = no border. |
| Slot `is_ghost` flag as the only ghosting path | Pattern key `"ghost"` (Anna re-labeled to "🫥 Misc" in the dropdown) flips `block.is_ghost` on every block in the container | Slot UI was gone but ghost is still a needed action -- moved into the pattern enum. Legacy `slotInfo.is_ghost` still honored on load for any pre-pattern data. |
| No way to mark V0 as "saved state" | `permutations.is_live_baseline = true` row | Anna's commit-V0-overwrite path. |

---

## The pattern enum (`LAB_PATTERNS` in `lab/index.html`)

**Hardcoded, not a table.** Lives at ~line 695. Twelve entries (eleven canonical AMA Travel patterns + Ghost/Misc). Each entry:

```js
{
  key: "deal_cards_tabs",        // text, persisted to lp_blocks.pattern_key
  name: "Deal Cards (Tabs)",     // dropdown label (and rendered badge text)
  structure: "H3 tabs + ...",    // tooltip on hover
  editable: ["h3"],              // for Bean code-reading context; not enforced
  aria_pattern: "role='tab'...", // emitted as "  ARIA: ..." line in export when non-null
  badge_color: "#f59e0b",        // tailwind hex; paints pill + container left edge
}
```

**Why hardcoded:** Anna wanted a tight curated list, not a free-form text field or a Supabase-backed lookup that would invite drift. The enum IS the API contract -- adding/renaming patterns means editing the file.

**Order matters in the dropdown** (users scan top-to-bottom). Current order roughly matches page-flow frequency.

**`editable` field is documentation, not enforcement.** It tells future Beans which elements of a pattern are safe to suggest changes to in optimization briefs. E.g., FAQ pattern says `["h2", "question", "answer"]` because the accordion structure itself is hardwired.

**`aria_pattern`** is consumed by the plain-language export -- emits a line like:
```
  ARIA: role='tab' with aria-controls + aria-selected on H3 headings.
```
only when non-null. Most patterns don't need ARIA notes; the ones that do (tabs, accordions) get them.

---

## Schema additions

Two migrations applied via Supabase MCP (project `ghzfrxxevjjfgpxvmahy` aka `signal-coherence`):

### `lp_blocks.pattern_key` (text, nullable)

Denormalized: every block in a container carries the same `pattern_key`. The load code (`loadBlocksFromCuration`, ~line 1413) reads the first non-null row in a container and propagates it to all `flatBlocks` in that container.

Persisted via `onContainerPatternChange`: PATCH `/lp_blocks?slug=eq.X&block_id=in.(...)` with `{pattern_key: newKey}`.

### `permutations.is_live_baseline` (boolean, NOT NULL, default false)

Marks the saved V0 state for a slug. Convention: at most one true per slug.

Wired into:
- `getVersionList()` -- V0 entry sources from the flagged row when present, falls back to "original" (page-load parse) when absent.
- `getCommitOverwriteTarget()` / `commitOverwriteV0()` -- INSERT first time, PATCH thereafter.
- `loadSelectedPage()` -- on load, if a flagged row exists, replaces `labOriginalBlocks` / `labBlocks` / `labSections` / `labOriginalNlp` with the row's data.

**Regular Vn filter excludes both flags:** `labPermutations.filter(p => !p.is_ux_baseline && !p.is_live_baseline)`. Forget this and your V0/V0.5 rows will double up as V1/V2.

---

## `slimBlockForSnapshot()` -- now self-contained

The block-trimming helper used by all three commit paths (save-new, V0.5 overwrite, V0 overwrite). After the May 17 fix, **always** includes:

- `block_id`, `curation_block_id` -- so the plain-language export diff engine can match blocks across versions
- `pattern_key` -- so reloaded permutations remember their pattern assignments
- `original_html` -- so reloaded permutations can re-score against the same HTML that was scored originally

The `b.edited` gate that used to wrap `original_html` is gone. Permutation rows are now self-contained: re-loading + re-scoring works without needing the underlying curation/crawl data.

**Historical gotcha (May 17-22):** existing V0/V0.5 rows committed BEFORE May 17 lacked `original_html` and `block_id`. May 19 backfill (`b6687ac`) injected `block_id` UUIDs but used INDEPENDENT random values per row, so logically-identical blocks got different IDs across rows -- the ID-based diff couldn't match them, export read everything as NEW. **Resolved May 22** at the matcher layer: `diffBlockSets` is now multi-pass (ID lookup → structural alignment by `(container_id, level)` bucket → text-similarity fallback) and no longer depends on cross-row ID overlap. The backfilled random UUIDs are harmless under the new matcher. Legacy `original_html` is still hydrated via `regenerateBlockHtml()` at load time for re-scoring.

---

## Plain-language export (`openPlainLanguageExport`, modal at ~line 2860)

Trigger: "Export brief" button in `lab-score-panel` action row.

Modal shows:
- Radio for diff target (V0.5 default if it exists, V0 fallback with explicit note)
- Markdown preview (outline + ARIA notes + TSV tables)
- Copy all / Copy tables only buttons

Diff engine (`diffBlockSets`, rewritten May 22): three-pass matcher tolerant of ID-namespace drift across permutation rows.

1. **Stable-ID lookup** via `blockStableId` (`block_id || curation_block_id || id`). Catches the cases where saved IDs do line up across versions (e.g. unchanged rows derived from the same load-time IDs).
2. **Positional alignment within `(container_id, level)` buckets.** For each unmatched active block at bucket position N, match to baseline block at the same bucket position N. Catches title rewrites in place -- the dominant edit pattern in optimization briefs even when both title and body were replaced.
3. **Text similarity (Jaccard on word tokens of title + body_text)** for remaining unmatched. Catches moves across containers and heavier rewrites that share most vocabulary. Threshold 0.35.

Output categories:
- **NEW** -- active block with no match (truly added content, including blocks from `+ Block` / `+ Container` that have no IDs).
- **REMOVED** -- baseline block with no match (rendered as a `### Removed (from baseline)` trailer).
- **CHANGED** -- matched but title/level/body differ (with `~~old~~ → new` strikethrough markers).
- **MOVED** -- matched, identical content, position changed (emits anchor like `MOVED above 'Best Price Guarantee'`).

Ghosted blocks (`b.is_ghost === true`, set by the `"ghost"` pattern dropdown or block-level ghost toggle) are filtered upstream in `buildPlainLanguageExport` before reaching the matcher. Their baseline counterparts fall into REMOVED via the standard diff path -- Anna's "ghost == REMOVED in proposal" workflow.

Container headers in the outline emit pattern name + ARIA line when `pattern.aria_pattern` is defined.

TSV tables: entity salience deltas + category confidence deltas, sorted by `|delta|` descending. Designed for paste-into-Sheets.

**Acknowledged ceiling:** the matcher INFERS what changed by comparing snapshots. Heavy-edit-AND-move-AND-level-change of the same block will read as NEW + REMOVED instead of CHANGED + MOVED. Anna catches these on QA. The durable fix is the structured change log -- see `memory/lab-state-and-followups.md` § "Structured change log architecture."

---

## Tripwires for future Beans

1. **Don't reintroduce the slot UI.** Anna explicitly decommissioned `+ assign slot` and the orange recipe-slot pill. The pattern dropdown is the canonical container affordance. The orange auto-NLP category pill that sits next to it (`.container-nlp-pill`) is a different thing -- keep it.

2. **Don't remove `container_label` or the rename handlers.** Anna had Bean remove the badge in the initial pattern-dropdown work and immediately wanted it back -- container names are load-bearing wayfinding. The auto-inherit logic (sets `container_label` to the top H-tag text on render) still runs silently. The rename UX (double-click) still works.

3. **`field-sizing: content` on `.container-pattern-select`** is what makes the pill shrink-fit to selected text instead of widest option. Chrome 123+ / Safari 17.4+ / Firefox 123+. If you see all pills rendering at "GHOST (EXCLUDE FROM SCORING)" width on an older browser, that's why -- swap to a JS measurer.

4. **`text-align-last: center`** (not `text-align`) is what centers the displayed text in a `<select>`. Standard `text-align` doesn't catch the closed state.

5. **`renderNlpDeltas` early-returns if `labOriginalNlp` is falsy** -- if the scorecard goes blank, that's almost always why. Check whether `scoreOriginal()` is overwriting a valid `labOriginalNlp` with null/empty. The May 17 fix was: skip `scoreOriginal` when the V0 row already has authoritative saved NLP.

6. **Pattern keys are namespace-stable** (snake_case lowercase). Don't rename existing keys -- they're persisted in `lp_blocks.pattern_key`. Add new ones; deprecate via comment if needed.

7. **RLS still disabled on `public.permutations`** -- pre-existing security issue surfaced during this work. Not blocking; flag if you're already doing schema work in that area.
