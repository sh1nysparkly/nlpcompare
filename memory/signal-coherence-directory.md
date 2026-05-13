---
name: signal-coherence/ directory and live infrastructure
description: Where Signal Coherence work lives -- on disk, in Supabase, deployed on Netlify. Updated post-reset May 8 2026.
type: reference
originSessionId: cbca5962-0e2d-4116-8f62-e96b9488c9c7
---
All SEO/discoverability investigation work for amatravel.ca lives under `Travel Website/signal-coherence/`. Key subdirectories:

- `Pre-Launch Reference/all page source aug 2025/` -- Screaming Frog rendered HTML from Aug 2025 (pre-launch baseline; not in DB)
- `Post-Launch Reference/all page source apr 2026/` -- SF rendered HTML from Apr 2026 (older crawl; was briefly in DB, removed May 8)
- `Post-Launch Reference/new extraction may 8 2026/` -- SF rendered HTML from May 8 (54 files, the freshest crawl; loaded into Supabase `crawled_pages` on May 8 evening)
- `nlp-results-may8/` -- May 8 paid NLP results saved to disk (8 LPs benchmark + recipe comparison). Pre-cache, not yet in `nlp_cache`.
- `competitor-html-may8/` -- May 8 freshly fetched competitor pages, organized by category. Both raw `.html` and `.cleaned.html` (HTML-mode strip) saved for all 22 competitors.
- `old-page-stripped/` -- text-mode strips of pre-launch AMA pages (consistent prep)
- `new-page-stripped-may8/` -- text-mode strips of post-launch AMA pages (consistent prep)
- `diagnosis-may8/` -- May 8 master investigation: `_FINDINGS.md` (load-bearing synthesis), per-page `.md` writeups, `nlp-data/`, `gsc-data/`, `backlinks-data/`, `serp-snapshots/`
- `ablation-results-may8/` -- VP 4-way ablation comparison (Current / UX Recipe / Brief v2 / Brief Final), `vp-brief-comparison.html`
- `outputs/dashboard/` -- Cowork artifact source (legacy reference, not deployed)
- `outputs/dashboard-netlify/` -- **editable source for the live dashboard**; `index.html` is what gets deployed to Netlify
- `parser-redesign-may9/` -- May 9 PM two-tier parser redesign: Python verification prototype, per-page tree dumps, CSR-empty-page coverage scan. The JS port is in `index.html`; the Python prototype is verification-only (do not try to keep them in sync). **Now deprecated as the Lab's primary data path -- post-Phase-A.1 the Lab reads `lp_blocks` from Supabase. The parser stays in place as a fallback for non-curated slugs.**
- `outputs/curation/` -- v9 → v19 workbook lineage + diagnostics. **Current canonical: `lp-curation-v19-all-pages.xlsx`** (GSC-faithful corrections applied for all 5 SIGNIFICANT/MINOR GAP LPs). Per-version baseline diagnostics: `v{N}-container-diagnostic-all.{json,md}`. **GSC-vs-SF rendering audit: `gsc-vs-sf-audit.md`** (cross-LP card-survival analysis). Parity handoff: `v12-lab-parity-handoff.md`. Anna's editing canvas: `lp-curation-v{N}-formatted-anna.xlsx` (formatted-banded by Block ID for eyeball scanning).
- **`*-gsc.html` files in `signal-coherence/`** -- per-LP Search Console HTML pulled by Anna May 11 2026 (8 files: hotels, car-rentals, cruises, destinations, flights, things-to-do, travel-insurance, vacation-packages). These are Googlebot's actual rendered view; the audit script compares each against the SF crawl HTML to identify card-rendering gaps. They are the ground truth for "what does Googlebot actually see" — see `worst-case-dev-scenarios.md` for why we use GSC-faithful curation rather than waiting for the dev-side rendering fix.
- `phaseB-results/` -- Phase B per-LP outputs:
  - **`METHODOLOGY.md`** — 7-step methodology + brief template + anti-patterns. Read before doing Phase B on any new LP (or future re-runs).
  - **`PERMUTATIONS-INDEX.md`** — per-LP Tn key + cross-LP shape index. Read alongside any per-LP brief to disambiguate Tn references.
  - Per-LP folders: `hotels/`, `vacation-packages/`, `things-to-do/`, `cruises/`, `destinations/`, `flights/`, `travel-insurance/`. Each has `brief.md` + per-permutation JSONs (`baseline.json`, `T1_*.json`, etc.) + `all-permutations-summary.json` or `v15-all-permutations-summary.json`.
- `scripts/load_may8_crawl.py` -- loads SF rendered HTML files into `crawled_pages` via PostgREST (curl, sidesteps macOS Python SSL)
- `scripts/load_v{N}_curation.py` (N = 12..19) -- loads `lp-curation-v{N}-all-pages.xlsx` into `lp_blocks` (DELETE-then-INSERT per slug, idempotent). The companion `lp_container_diagnostic_v{N}.py` generates the baseline JSON; both use `\n` row/block separator (see `lab-fidelity-mission-critical.md`).
- `scripts/build_v{N}.py` (N = 10..19) -- per-pass workbook cleanup transforms. v10-v12 = Anna's hand-passes + ARTICLE prefix stripping + card insertion. **v13-v19 = GSC-faithful corrections per LP:** v14 hotels, v15 VP, v16 TTD, v17 cruises, v18 flights, v19 travel-insurance. Each script drops phantom card rows that don't appear in GSC HTML (identified by block_id or by missing-phrase audit).
- `scripts/audit_gsc_vs_sf.py` -- cross-LP audit comparing GSC HTML to SF crawl HTML. Uses curation-aware card detection (component_type ∈ {Card, Product Card, Tall Card Carousel - Programmatic}); reads SF-faithful curation from `lp-curation-v13-all-pages.xlsx`; checks whether each card's distinctive opening phrase survives in GSC HTML. Outputs `outputs/curation/gsc-vs-sf-audit.md`.
- `scripts/phaseB_{slug}.py` -- Phase B runner per LP. Pattern: fetch baseline rows from `lp_blocks`, apply permutation transforms, call bridge `nlp_analyze`, write per-permutation JSONs + summary. 7 scripts total (one per non-control LP). Per-LP archived legacy scripts `phaseB_vacation_packages_v13_legacy.py` preserve pre-methodology results.
- `scripts/verify_lab_parity.py`, `verify_lab_e2e_parity.py` -- offline parity checks (byte-equality and end-to-end-via-bridge)
- `lab-ui-followups.md` -- post-Phase-A.1 Lab UI wishlist Anna parked; future Beans pull from here
- `link score.csv` -- Screaming Frog Link Score export Anna provided

**Live infrastructure (post-Phase-A.1 state, May 11 2026):**
- Dashboard at https://signal-coherence.netlify.app (Netlify project `signal-coherence`, site ID `cab32293-702d-4f95-9358-d293c666ff96`, on Anna's Paper Doll team)
- Dashboard has TWO tabs only: Rearrangement Lab (default) and NLP Scorer. The Pages tab was removed in the May 8 Phase 1 reset.
- Supabase Edge Function `bridge` at `https://ghzfrxxevjjfgpxvmahy.supabase.co/functions/v1/bridge` (handles NLP classify, NLP analyze, SQL via tool dispatch; JWT-protected; transparent read-through cache via `nlp_cache` keyed on `(content_hash, content_mode, nlp_method)`)
- Postgres tables: `crawled_pages` (54 May 8 rows), `nlp_cache` (response cache), `recipe_slots` (11 slot definitions for top-level-lp), `permutations` (experiment log), **`lp_blocks` (added May 11 -- 1058 rows of v12 curation across 8 LPs; canonical data source for the Lab now)**. The 8 elaborated tables from past Phase 1 were dropped May 8.
- **Per-container NLP scoring** is built into page load (`scoreContainersForLab()` in the Lab) -- top category renders as a pill in the collapsed bar; entity panel renders below blocks when expanded.
- Dashboard writes permutations directly via PostgREST (`SUPABASE_URL/rest/v1/permutations`), NOT via bridge (bridge's `exec_sql` is SELECT-only).
- Postgres helper `public.exec_sql(query text)` (SELECT/WITH only; service-role-only execute)
- `GOOGLE_NLP_CREDENTIALS` set as Supabase secret
- Node 26 installed via Homebrew on Anna's machine for redeploys

**Deprecated (archived in place, do not use):**
- `.claude/skills/signal-coherence-bean/` -- the Signal Coherence skill writes to tables that no longer exist. Do NOT invoke.

**How to apply:** Start any SEO investigation by reading `diagnosis-may8/_FINDINGS.md` for current state. For dashboard work, read `signal-coherence/SIGNAL-COHERENCE-STATUS.md` for the current architecture (it's load-bearing for redeploys, env vars, and what's wired up). The `lp-category-diagnosis` skill expects HTML in the Pre/Post-Launch Reference folders. Maintain the diagnosis-may8/ pattern (with nlp-data/, gsc-data/, backlinks-data/, serp-snapshots/ subfolders) for future investigation runs -- datestamped to keep multiple runs separable.
