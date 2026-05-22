# RLS on `public.lp_blocks` -- the intentional permissive posture (May 22)

Stop the loop. If a future Bean reads this and thinks "RLS is misconfigured on lp_blocks, I should fix it" -- read the whole doc first.

## Current state

- **RLS is ENABLED** on `public.lp_blocks`
- **One policy:** `anon_all_access` -- `FOR ALL TO anon, authenticated USING (true) WITH CHECK (true)`
- Migration name: `lp_blocks_rls_permissive_solo_user`
- Mirrors the posture documented in `memory/rls-permutations-intentional.md` for the `permutations` table

## Why this is necessary (the bug that surfaced it)

Until May 22, `lp_blocks` had RLS enabled with **no policies** -- the default-deny posture for non-owner roles. The Lab at `/lab/index.html` calls `/rest/v1/lp_blocks` directly with the anon key, not through the `bridge` edge function. That call has been silently returning `[]` for every slug **since `loadBlocksFromCuration` shipped in Phase A.1 (May 11)**.

The Lab's `loadSelectedPage` falls back to the HTML parser (`extractMainContent` → `walkBlocks`) when `loadBlocksFromCuration` returns no rows. On the original 8 LPs the SF-crawled HTML has the class structure (`div.section-content`, `div.widget-wrapper`, full angular component tag names) that `walkBlocks` is built around, so the parser-fallback output is *close enough* to the v19 curation that nobody noticed the divergence. The v9→v19 hand-curation work and the GSC-faithful corrections (v14-v19) have **not actually been driving the live Lab**.

The divergence finally exposed itself on May 22 when Anna landed `/vacation-packages/tours` -- a class-stripped, hand-rendered HTML page where `walkBlocks` finds no `div.section-content` anchors and falls through to `extractHeadingBlocks`. That produces "one container per `<h>` tag," which rendered the Tours page as 10 separate single-block containers instead of 1 container with 9 blocks. The lp_blocks rows for the page were correct; the Lab just couldn't see them.

The previous version of `rls-permutations-intentional.md` (line 62 pre-this-doc) claimed `lp_blocks` anon-deny was "correct" because "service-role-only via the bridge edge function." That was wrong about lp_blocks specifically -- the Lab's load path was already on direct PostgREST. That line has been corrected in the permutations doc.

## What this changes

- `loadBlocksFromCuration(slug)` now returns the actual curated rows for all 8 LPs that have v19 curation + the 2 new v20-extrapolated pages
- The Lab will start rendering v19 lp_blocks data instead of parser-fallback output on the 8 original LPs. **Behavior may visibly change**: phantom cards dropped in v14-v19 will no longer appear, container labels will reflect the hand-curated text instead of the auto-detected one, etc.
- `onContainerPatternChange` (PATCH `/rest/v1/lp_blocks` with `pattern_key`) now actually persists; before, writes were silently failing under RLS too
- For the new v20-extrapolated pages (`/vacation-packages/tours`, `/travel-insurance/car-rental-insurance`), this fix is what makes them load at all

## What advisors will say -- and what NOT to do

After this migration, `get_advisors` will surface:

- **`rls_policy_always_true` (WARN)** -- "Table `public.lp_blocks` has an RLS policy `anon_all_access` for `ALL` that allows unrestricted access. This effectively bypasses row-level security for anon, authenticated."

**This is the intentional state.** Do NOT:

- **Disable RLS** to silence the WARN -- trips the louder `rls_disabled_in_public` ERROR
- **Tighten the policy** to restrict INSERT/UPDATE -- breaks `onContainerPatternChange` and any future Lab→lp_blocks write path. There is no row-level identity column to scope by; without restructuring the auth model first, any tightening either matches everything (no effect) or blocks everything (Lab breaks)
- **Re-route to bridge** as a "fix" without first verifying the Lab's read+write paths still work. The bridge SQL tool is SELECT-only; writes still need direct PostgREST

If `rls_policy_always_true` is the only finding on `lp_blocks` after a future change, **leave it alone**.

## Implications beyond this fix

The "Lab was silently parser-fallback for everything" finding has wider implications worth holding:

- **Phase B brief findings** were computed against parser-fallback output, not v19 curated lp_blocks. The brief NLP scores reflect what the parser saw (including phantom cards). After this fix, re-scoring the same V0 baselines in the Lab will produce slightly different numbers -- the v14-v19 GSC-faithful corrections will finally matter for live scoring.
- **Saved permutation `nlp_result` JSONB blobs** in the `permutations` table were also computed against parser-fallback HTML. The saved scores are accurate for what they were computed against, but they won't match a fresh re-score after this fix.
- **The "Score WIP" button on existing V_n permutations** will produce different scores than the stored `nlp_result` for the same blocks. Not a bug -- just a baseline shift.

None of this requires immediate action. Anna will likely notice differences as she pokes around. Flag this doc when she does.

## How to migrate to real security (if ever wanted)

Same as the permutations table -- requires Supabase Auth in the Lab + a `creator_id` column + scoped policies. See `rls-permutations-intentional.md` § "How to migrate to real security." Not in scope.
