# RLS on `public.permutations` -- the intentional permissive posture (May 19)

Stop the loop. If a future Bean reads this and thinks "RLS is misconfigured on permutations, I should fix it" -- read the whole doc first.

## Current state

- **RLS is ENABLED** on `public.permutations`
- **One policy:** `anon_all_access` -- `FOR ALL TO anon, authenticated USING (true) WITH CHECK (true)`
- Migration name: `permutations_rls_permissive_solo_user`
- Table comment documents the posture inline

## Why permissive (not restrictive)

The Lab at `/lab/index.html` writes directly to `/rest/v1/permutations` with the anon key. As of May 19, five sites do so (locate by function name -- line numbers shift):

- `loadPermutations()` -- SELECT
- `commitOverwriteV05()` -- PATCH
- `commitOverwriteV0()` -- PATCH (existing-row path) + POST (first-time-insert path)
- `commitSavePermutation()` -- POST

There is **no auth layer in the Lab**. A restrictive policy would require one of:

- (a) Real auth in the Lab (Supabase Auth sign-in + JWT), or
- (b) Routing all writes through the `bridge` edge function with service_role (like `crawled_pages` / `lp_blocks` / `nlp_cache` already do)

Both are larger refactors. For this **solo-user tool**, RLS-on with a permissive policy is functionally equivalent to RLS-off security-wise, but it **stops the recurring "RLS disabled" advisor warning** that prompts Beans to keep toggling RLS off (which then breaks reads, which then prompts another Bean to re-enable, ad nauseam).

## What advisors will say -- and what NOT to do

After this migration, `get_advisors` will surface:

- **`rls_policy_always_true` (WARN)** -- "Table `public.permutations` has an RLS policy `anon_all_access` for `ALL` that allows unrestricted access. This effectively bypasses row-level security for anon, authenticated."

**This is the intentional state.** The previous warning was `rls_disabled_in_public` (ERROR). We deliberately downgraded ERROR → WARN by accepting the "policy always true" trade-off. Do NOT:

- **Disable RLS** to silence the WARN -- you'll trip the louder ERROR and restart the loop
- **Tighten the policy** to restrict INSERT/UPDATE/DELETE -- you'll break Lab writes silently. There is no row-level identity column to scope by; without restructuring the auth model first, any tightening either matches everything (no effect) or blocks everything (Lab breaks)

If `rls_policy_always_true` is the only finding on `permutations` after a future change, **leave it alone**.

## How to migrate to real security (if ever wanted)

Not in scope today. If Anna ever wants a real auth posture:

1. Add Supabase Auth sign-in to the Lab (email magic link or similar)
2. Add `creator_id uuid REFERENCES auth.users(id)` to `permutations`
3. Backfill `creator_id` for existing rows (single user, so trivially uniform)
4. Replace the permissive policy with scoped versions:
   - SELECT: `USING (creator_id = auth.uid() OR creator_id IS NULL)` (the NULL clause keeps any back-compat-needed rows readable)
   - INSERT: `WITH CHECK (creator_id = auth.uid())`
   - UPDATE / DELETE: `USING (creator_id = auth.uid())`
5. Update the five Lab write sites to include `creator_id: <session user id>`

Estimate: 1-2 focused sessions. Not blocking anything.

## Adjacent advisor findings (separate threads, NOT this doc's concern)

The May 19 advisor sweep also flagged:

- **`public.exec_sql(query text)` is a SECURITY DEFINER function callable by `anon` and `authenticated`** -- via `/rest/v1/rpc/exec_sql`. This is materially worse than the permutations finding was: it lets anyone with the anon key run arbitrary SQL with elevated privileges. Worth a separate session to either revoke EXECUTE from anon or switch the function to SECURITY INVOKER.
- **`public.set_updated_at` has a mutable search_path** -- low-priority hygiene fix.
- **`crawled_pages`, `lp_blocks`, `nlp_cache`, `recipe_slots` have RLS-enabled-no-policies** -- this IS correct for those tables; they're service-role-only via the `bridge` edge function, so absence of policies blocks anon access (intended). The advisor INFO-level note is noise for those four.

If you're reading this and considering a security pass on the Supabase project, the `exec_sql` thing is where to start, not here.