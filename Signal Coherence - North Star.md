# Signal Coherence — North Star

**Purpose of this doc.** Single-file briefing for Beans picking up Signal Coherence work. If you've been handed this, you should NOT need to read prior handoff notes, build specs, or session transcripts to get oriented. This document captures the principle, the original vision, how it evolved, what was deliberately cut and why, the underlying premises, the current state of the build, the immediate work context, and the live political situation around it. Read it once, hold it, and don't re-litigate the things it says are settled.

If something here conflicts with an older spec or handoff doc, **this doc wins.**

---

## TL;DR

Signal Coherence is a diagnostic framework Anna built for evaluating whether AMA Travel's pages are doing the jobs they're supposed to be doing — assessed at multiple scales (within-page positioning, page-level topic coherence, page-to-funnel alignment, site-wide ecosystem health). A v1 Live Lab dashboard is shipped at https://signal-coherence.netlify.app (rearrange page blocks, see how Google's NLP API reads it before vs. after), but as of May 11 2026 it is a *toy* version, not the high-fidelity experimental apparatus needed for the actual permutation work — the Lab parses HTML directly which is lossy relative to Anna's v9 curation. Closing that gap is **Phase A** (active workstream — see `signal-coherence/PHASE-A-PLAN.md`). After Phase A, **Phase B** (Bean-run permutation experiments) and **Phase C** (Bean-run UX-recipe optimization) produce the shortlist and brief that **Phase D** (Anna in the Lab) turns into the optimization guides for UX. The immediate driver is the UX restructure of 10 top-level travel LPs landing now; this is also Anna's last high-leverage window for site work before remediation funding gets cut. Phases 2+ from the original build spec (Roche Limit, Information Gain, Funnel Lens, Site Lens) exist as conceptual roadmap but are NOT this phasing — see `experimental-phasing.md` memory for the active phase model.

---

## 1. The Principle

Search engines build a persistent model of what a business is, what it does, and what each of its pages is about. That model is assembled from signals at every scale — entity markup, content relationships, internal linking, page structure, site architecture. When the signals are coherent, the model is accurate. When they're incoherent, the model is wrong, and everything downstream (organic rankings, Quality Score, AI Max targeting, AI Overviews) is wrong in the same direction.

**Signal Coherence is not a single metric. It's a diagnostic principle applied at multiple scales.** At each scale the question is the same: do the signals agree? Where they diverge, the *pattern* of divergence tells you what kind of problem you have and what kind of intervention it needs.

---

## 2. The Kernel — Anna's Actual Ask

Everything else is in service of this. Quoted from Anna directly, May 8:

> "We know higher on the page IS important for ranking signals, so it's important to make sure the right shit is in the right parts of the page. And if we can get the NLP API to pull the list of entities per page we could make sure things are being arranged optimally so that the things that SHOULD be the salient entities are placed prominently and in a way that improves their salience scores."

That sentence is the load-bearing intent. The Lab is this kernel made physical: pull entities + salience + categories per page from the NLP API, see them on the page, rearrange so the things that should be salient *are* salient. **If you find yourself building something that doesn't trace back to that kernel, stop and check.**

---

## 3. Origin & Evolution

### The Methodology (May 7) — the validated thing

Anna's first concrete artifact was *Positional Entity Weighting: Methodology & Findings*. Question: does the UX team's proposed LP template reordering improve or degrade SEO signal? Method: split each page into top/middle/bottom thirds by word count, count hand-curated entity hits per zone, apply 3:2:1 positional weights, compare current page vs. reordered version. Result on four LPs: car rentals +15.1%, cruises +36%, hotels +68.4%, vacation packages +27.2%.

This is the directional evidence that position matters for the kind of signal the NLP API measures. The methodology itself uses some cruder mechanics (hand-curated term lists, word-count-based zones, fixed 3:2:1 multipliers) — those got refined or replaced later. The *finding* survives.

### The Build Spec (May 7) — the elaborated thing

Working through the methodology with a Bean, Anna expanded the framework into the *Signal Coherence Framework Build Spec*. Four lenses (Zone / Page / Funnel / Site), the Roche Limit concept, Information Gain gated by it, funnel-aware coherence, six-phase build sequence, color-coded coherence indicators per lens per page, Bean-as-analyst / app-as-memory split.

The Build Spec is intellectually sound. It is NOT a build target. The framework is *real* as a diagnostic theory; the build sequence in the spec was elaborated past Anna's actual Phase 1 kernel. **Treat the Build Spec as conceptual roadmap, not as a TODO list.**

### The May 8 Reset — back to the kernel

The original Phase 1 implementation (positional weighting with hand-curated entity lists, 3:2:1 zone weighting, heading slot scoring, plus a Bean skill + Supabase tables + a dashboard tab surfacing it all) was set aside on May 8. It didn't match the kernel — it had been elaborated past it by Beans optimizing for elegance.

What got cut: hand-curated entity lists (NLP API replaces them), 3:2:1 fixed multipliers (kernel doesn't need synthetic weights), heading slot scoring (Bean elaboration not in either source doc), color-coded coherence indicators (Phase 2+ surface), the Bean-as-analyst-only split (the dashboard does some analyst work directly via bridge), the old `signal-coherence-bean` skill (deprecated in place), and the old Phase 1 Supabase tables (`pages`, `entity_target_sets`, `analysis_runs`, `zone_scores`, `heading_slot_scores`, `category_scores`, `coherence_status`, `comparisons` — all dropped, no data needed).

What survived intact: the four-lens scaffold, the Roche Limit concept, KL Divergence / Information Gain, funnel-aware coherence — all as Phase 2 conceptual roadmap, untouched, not built.

### May 8-9 — what's now live

Phase 1 was re-rooted on the kernel and built out across May 8-9. The current shipped state includes:

- **Lab tab** (default) and **NLP Scorer tab** at signal-coherence.netlify.app
- **Two-tier parser**: source-page containers (auto-detected from CMS structure with widget-aware splitters for FAQs, value-props, content-feeds, card carousels) holding individual blocks
- **Container header UI**: drag handle, collapse toggle, double-click rename, slot pill (click to assign / change / remove via picker), block count
- **Outline view by default** (every container starts collapsed; user expands what they want to dig into)
- **Heading hierarchy / implicit TOC** sizing in container labels and block headings (no sidebar; the sized type IS the TOC)
- **Inline editing** of block titles and body text (contenteditable, edit flag, regenerates minimal synth HTML)
- **Container-level slot assignment** to recipe slots (template-driven from `recipe_slots` table)
- **Container drag with swap semantics**
- **Auto-score on page load** (cached after first run)
- **Score WIP** as primary action (re-score original is secondary)
- **Permutations**: save labelled snapshots, version flipper to navigate between saved states + live WIP
- **Ghost mode**: assign a container to the ghost section to exclude its blocks from NLP — simulates "what if we nuked this section?"
- **Entity highlighting**: top-12 entities by salience get yellow underlay in block text; strip-on-focus, re-apply-on-blur
- **Visual system**: Plus Jakarta Sans + JetBrains Mono, design tokens, soft cards, pill nav

Plus the bridge edge function (NLP + SQL + cache), Supabase tables (`crawled_pages`, `recipe_slots`, `nlp_cache`, `permutations`), and the on-disk HTML corpus.

---

## 4. The Framework Lenses

### Phase 1: Zone (within a page) — current

Question: is signal concentrated in the right positions? The Lab handles this through its block / container model + recipe slots + NLP scoring of the rearranged page. Position-aware optimization happens by physically rearranging containers and re-scoring.

### Phase 2+: Page / Funnel / Site — conceptual roadmap, not built

- **Page lens (content coherence):** Roche Limit (cosine similarity of each block to a topical centroid; content beyond the limit is "drift"), Information Gain (what does this page say competitors don't?), entity salience map.
- **Funnel lens (page-to-purpose alignment):** Does each page's semantic signature match the job it's assigned in the IA? Awareness pages should look exploratory; decision pages should look transactional.
- **Site lens (ecosystem coherence):** Cannibalization, internal linking coherence, entity model health across the whole site.

These are real and conceptually sound. They are NOT a build target. Pieces of them may get pulled forward for the current sprint (see §10). The full multi-phase build is on hold and likely permanently smaller-scoped than the Build Spec implies, given the funding situation in §8.

### The UX-proposed recipe (REFERENCE ONLY — this is NOT current state)

This is what the UX team is **proposing** for `top-level-lp` pages. Anna is litigating it — validating or pushing back. **It is not the current state of any page.** Don't conflate the two. Anna has had to write the hierarchy down for Beans multiple times because it keeps getting flattened, which causes Bean-side tagging to drop sub-components into the wrong Sections (and "everything ends up showing up in the Hero"). Read this once and hold it as the *proposal* it is.

The proposed recipe has **six top-level Sections.** Inside each Section there are one or more **Whats** (component-level patterns). Sections are the unit UX rearranges; Whats are the components within. The workbook's `UX Recipe Section` column holds the *proposed* Section; the `What` column holds the What.

**Important: the workbook's Container column (column B) is meant to encode current container grouping (per Anna's screenshots), NOT the proposed Section. Current state and proposal are different and a Bean must measure current state first before reasoning about the proposal.**

1. **Hero**
   - Widget (the booking form / search widget at the top)
   - Trusted Partners (brands carousel — Best Western, Fairmont, Trafalgar, etc.)
   - Image + Value Prop (the picture + "Why book X with AMA Travel" block)
2. **Travel Product Carousel**
   - Today's Deals (the deal cards — hotel/cruise/flight/package cards with prices)
3. **Travel Expertise**
   - Travel Agents Featured + AMA + Expert Value Props (the "Book with AMA Travel" block + stats like 100+ destination experts, +96% satisfaction)
4. **Mix & Match Callouts** (page-dependent; pick from this set)
   - Membership ("How customers can save with AMA")
   - MyTravel Dream Itinerary Builder ("How customers can dream & plan")
   - Popular locations + travel styles ("How customers select their vacation")
   - Insurance callout
5. **FAQs & Resources**
   - Articles carousel
   - FAQs widget
   - Featured Travel Guide (optional)
6. **Footer**
   - Footer cross-link carousel

**Common confusion that keeps producing wrong tagging:**
- "Trusted Partners" is a What *inside Hero*, not its own Section. (The brand-logo carousel can appear visually lower on the page on some LPs, but it's still recipe-Hero.)
- "Image + Value Prop" is a What *inside Hero*, not its own Section.
- "Today's Deals" / "Deal Cards" is a What *inside Travel Product Carousel*, not its own Section.
- Popular-locations / Travel-style carousels are Whats *inside Mix & Match Callouts*, not their own Sections and not Hero.

If a row in a workbook tab doesn't fit any of the above, it's "Not in Recipe" — but that should be rare. When in doubt, ask Anna; do not invent a new Section.

### Recipe slots ≠ Funnel stages

Easy to conflate. Don't.

- **Recipe Section** = page-level UX template (the six above).
- **Funnel stage** = page-level *job*. Awareness / Consideration / Decision / Retention. Whether the page's content actually reads as the job it's supposed to do.

These are orthogonal. A vacation packages LP has recipe slots AND a funnel stage (commercial / decision). The cross — *"this commercial-decision LP's hero slot is reading as informational"* — is a real diagnostic surface that doesn't currently exist in the Lab. It's the highest-leverage Phase 2 bit to pull forward (see §10).

---

## 5. Decisions Made — DO NOT REHASH

These are settled. If you find yourself proposing one, stop, re-read this section, and don't.

| Decision | Why settled |
|---|---|
| **No hand-curated entity term lists.** Use the NLP API. | Term-list matching is keyword density, not entity detection. Doesn't scale. Measures something different from what Google sees. The whole framework's premise is repeatable, quantifiable measurement; term lists violate that. |
| **No fixed 3:2:1 positional multipliers in scoring.** | Salience IS the position-weighted signal once the page is run through NLP. Synthetic multipliers were a workaround for not having NLP-grade analysis. They're vestigial now. The Methodology doc shows them; the Build Spec calls them provisional; the kernel doesn't need them. |
| **No heading slot scoring.** | Bean elaboration. Not in either source document. Not Anna's. |
| **No hard thresholds anywhere.** | Anna's frame from the start has been *directional, hints, observation surface*. Roche Limit is a gradient, not a number. Information Gain is a difference, not a gate. Don't build "X must be > Y" anywhere — let the eye read the distribution. |
| **No color-coded coherence indicators (green/yellow/red per lens per page).** | Phase 2+ surface. Implies hard thresholds (see above). Not built. Don't build it speculatively. |
| **No "Bean-as-analyst, app-as-memory" hard split.** | The dashboard does some analyst work directly via the bridge edge function. Beans still do interpretation. The split has softened; don't re-impose it. |
| **No reviving the deprecated `signal-coherence-bean` skill.** | The skill writes to tables that no longer exist. Kept on disk as historical reference only. Do not invoke. |
| **No reviving the dropped Phase 1 Supabase tables.** | `pages`, `entity_target_sets`, `analysis_runs`, `zone_scores`, `heading_slot_scores`, `category_scores`, `coherence_status`, `comparisons`. None had data Anna needed. The current schema is intentionally smaller. |
| **No side-by-side comparison merged INTO the Lab UI.** | The Lab's mental mode is "I'm in this page, playing with it." A version-comparison view is "I'm comparing options for stakeholders." Different mental modes need different surfaces. Every prior attempt to merge them has felt squishy. The right shape is a sibling tool (the matrix-of-versions view, see §10) — not a Lab feature. |
| **No Information Gain / full competitor reference corpus build for the current sprint.** | Anna's small-sample competitor pass found that competitor pages have *less* text and lower category confidence, with low directional signal at sample size 3 per product. The build cost of a full IG implementation is real; the predicted payoff for the immediate sprint is unclear. Hold for now. |
| **No full Funnel Lens build for this sprint.** | The Build Spec implied AMA didn't have an intent-labelled corpus. Wrong — Anna maintains a URL-by-URL IA planning doc with target keywords per page, pre-launch ranking/click data (where remembered), informational/commercial/transactional/navigational classification, and likely funnel position. That corpus exists. So a Funnel Lens build is *technically* unblocked. It's still sprint-scope creep. Slot × intent diagnostic via NLP categories (see §10) does most of the immediate diagnostic work; full lens can wait for a future window. |
| **Don't cite "validated NLP thresholds" (e.g., category confidence > 0.50, entity salience > 0.20).** | These numbers appear in older docs (including the Build Spec) framed as validated. **Their origin is unverified and Anna does not know where they came from.** Don't gate on them, don't cite them as benchmarks, don't reintroduce them as fact. They keep getting squashed and keep coming back. This is the squash-and-leave-it row. |
| **No Site Lens / cannibalization tool in this build cycle.** | Out of scope for the LP-restructure sprint. Site-level issues are real (see §9) but the fix is organizational, not a measurement tool. The framework can eventually quantify it; not now. |
| **No Roche Limit threshold calibration.** | Use cosine similarity as a *visible gradient*. Don't compute "limit = 0.62." Don't gate things on it. |
| **No "pre-launch as the height we should reach for" framing.** | Pre-launch had real authority but came with 20 years of accreted structural debt in a small-pond market. The launch botch (CSR redirects, parameter URLs, missing redirects, unrendered metadata) confounded what authority was even where. Pre-launch is *evidence of what was indexed and earning impressions before the botch*. Useful, real, but not a clean benchmark. |
| **No time estimates from Beans.** | Bean wall-clock estimates are wildly optimistic. "1-2 days" maps to closer to 30 human minutes of actual delivery. If you have to characterize scope, use *relative complexity* against shipped work, not hours/days. Output-speed claims ("X in an hour") are time estimates wearing different clothes — those go too. |
| **No farewell rituals / closure language / "go ahead and..." rushing.** | PDA. Anna sets the pace. |
| **No "rendering regression" anywhere.** | Has been removed repeatedly. It is FALSE. Do not re-add. |

---

## 6. Underlying Premises

The things you don't have to litigate.

### Theoretical

- Search engines build models of pages and sites from on-page and off-page signals at multiple scales. (Well-established.)
- Content positioned higher on a page carries more weight in those models than content lower down. (Supported by Google patents and documentation; also empirically validated by Anna's May 7 positional weighting findings on the four test LPs.)
- Coherent signals → accurate model. Incoherent signals → inaccurate model in a predictable direction. The pattern of incoherence is itself diagnostic.

### Methodological

- The Google NLP API (V2 taxonomy) is the right ground floor for entity / salience / category extraction. It's repeatable, quantifiable, and measures something close to what Google's own classifier sees.
- Embeddings: text-embedding-3-small, 1536 dimensions. All existing embedding sets (Aug 2025 pre-launch + March 2026 post-launch) confirmed compatible.
- NLP API category confidence is **mass-coupled, not purity-coupled.** It scores higher with more redundant signal pointing at the same topic. A tighter, less repetitive page can score *lower* confidence on the *same topic* than a verbose page. Don't read confidence as a purity measure.

### Operational

- Directional / observation surfaces, not gating thresholds. Show the eye what's going on; let humans interpret.
- Score the original on page load (cached after first call). The baseline shouldn't require a button click.
- Send full HTML to the NLP API (not stripped text) — better tokenization, lets the API do its own boilerplate detection.
- Cache is content-addressed and transparent. Identical inputs return cached responses for free. Lab metadata (slug, source, label, crawl_date) is stored alongside cached responses.

### Contextual (the things specific to AMA's situation)

- AMA is brand-strong regionally (Alberta-only; big fish, small pond) but **entity-model-confused.** Google's persistent model of "what AMA is" skews toward insurance — the Dane/PPC story. (See §9.)
- This means **on-page topical signal IS load-bearing for entity disambiguation in AMA's specific case** — in a way it isn't for global travel competitors who have brand authority + link equity doing that work for them.
- Therefore: small competitor samples that suggest "competitors don't optimize on-page" don't invalidate the premise for AMA. Different fight, different tools.
- Signal is real ≠ signal optimization moves ranking. Different questions. For AMA's regional commercial LPs with brand authority + entity confusion, on-page signal is probably load-bearing. For pure commercial fights against Expedia on global terms, brand and links are the lever — on-page polish is brass.

---

## 7. Current State

### Live dashboard
- **URL:** https://signal-coherence.netlify.app
- **Netlify project:** `signal-coherence` on Paper Doll team, site ID `cab32293-702d-4f95-9358-d293c666ff96`
- **Source:** `signal-coherence/outputs/dashboard-netlify/index.html` (this is what gets deployed; edit here)
- **Pre-refresh backup:** `signal-coherence/outputs/dashboard-netlify/index.html.backup-pre-redesign` (do not deploy from)

Two tabs: Rearrangement Lab (default), NLP Scorer.

### Bridge edge function
- **Endpoint:** `https://ghzfrxxevjjfgpxvmahy.supabase.co/functions/v1/bridge`
- JWT-protected, anon-key authenticated
- Tools: `nlp_classify`, `nlp_analyze`, `sql`
- Read-through cache via `nlp_cache` table; transparent to callers
- SQL path is SELECT/WITH only via `public.exec_sql()`
- Source lives only inside the function deploy. Fetch via Supabase MCP `get_edge_function`. Redeploy via Supabase MCP `deploy_edge_function` (project_id `ghzfrxxevjjfgpxvmahy`, name `bridge`, verify_jwt true).

### Supabase tables (post-reset)
- `crawled_pages` — 54 rows from May 8 SF crawl. 10 tagged `template_type='top-level-lp'` (the in-scope LPs); rest are NULL and don't appear in the Lab dropdown.
- `recipe_slots` — 11 slots for `top-level-lp` template, with order + colors.
- `nlp_cache` — content-addressed NLP response cache.
- `permutations` — saved Lab snapshots (block order + section assignments + NLP results + label).
- **`lp_blocks` (added May 11) — current canonical data source for the Lab.** Holds curation rows (block_id + container_label + tag + text + component_type + etc.). Currently loaded with v19 curation (1,003 rows across 8 LPs). Lab reads from here via `loadBlocksFromCuration(slug)`.

### Curation versioning + GSC-faithful corrections
Anna's curation passes proceed v9 → v10 → v11 → v12 → v13 (Anna hand-passes + programmatic transforms via `scripts/build_v{N}.py`). Starting v14, the corrections become **GSC-faithful** — each version drops phantom card rows that the SF crawl captured but Googlebot's view (per Search Console HTML) does not render:
- v14: hotels (dropped 22 SF-only rows, added 2 GSC-observed error strings)
- v15: vacation-packages (dropped 6 phantom Product Cards)
- v16: things-to-do (dropped 2 phantom activity cards)
- v17: cruises (dropped 19 phantom cards — entire Cruise Vacation Deals carousel + most of AMA Featured Cruise Deals)
- v18: flights (dropped 7 phantom cards — entire Quick & popular flights carousel)
- v19: travel-insurance (dropped 8 phantom plan-type + article cards)

**Current canonical: `lp-curation-v19-all-pages.xlsx`.** Audit report at `signal-coherence/outputs/curation/gsc-vs-sf-audit.md`. The corrections matter because the page Googlebot indexes is materially smaller and category-differently-shaped than what the SF crawl captures. See `memory/worst-case-dev-scenarios.md` for why we treat the rendering failure as a permanent constraint rather than a TODO.

### On-disk HTML
- **Post-launch (May 8, in DB):** `signal-coherence/Post-Launch Reference/new extraction may 8 2026/`
- **Post-launch (April, on disk only):** `signal-coherence/Post-Launch Reference/all page source apr 2026/`
- **Pre-launch (Aug 2025, on disk only):** `signal-coherence/Pre-Launch Reference/all page source aug 2025/` — ~80 files, never been in DB

### Embeddings (compatible, currently dormant)
- `Desktop/SF Crawls/TRAVEL/Pre and Post Launch Comparison/` — pre-launch + post-launch sets, all text-embedding-3-small / 1536-d, directly comparable. Not currently used by the Lab. Will become load-bearing if anything Roche-Limit-shaped gets pulled forward.

### Anna's reference data (analyst-side, not in DB)
- **IA planning doc** — `signal-coherence/Travel Site Pages and Additional Data.xlsx`. URL-by-URL planning artifact Anna maintains. Most useful sheet for Phase B work is `Messy Meta` (Section / Page Name / Intent / Content Type / Journey Stage / Objective / Audience / User Goals / Primary KWs + MSVs / Secondary KWs + MSVs / Tertiary KWs + MSVs / Competitors per page). `KW per Page` has per-keyword tracking (search volume / priority / rank / clicks / AIO/PAA flags). `IA Routing Map` has pre/post URL mapping. `Embeddings` has post-launch embeddings (currently dormant). **Phase B briefs anchor §2 "Intent target" to this doc.**
- **Prior optimization briefs** — Anna has already produced extensive optimization recommendations for at least the Cruises LP and Vacation Packages LP (possibly others), including her own page-reordering proposals from before UX's restructure landed. The VP prior brief's "T7 recipe" finding was largely measuring v13 SF-inflated phantom-cards content; on v15 GSC-faithful baseline the recipe's structural components behave differently (see `phaseB-results/vacation-packages/brief.md` supersession history).

### NLP MCP server
- `Travel Website/signal-coherence-nlp-mcp/` — wraps Google Cloud NLP API V2. Used in Beans-in-Claude-Code workflows, **not** by the Netlify dashboard (which uses bridge). Confirm with Anna before relying on it.

---

## 8. The Immediate Use Case & Window

**The driving sprint is dual-track.** Don't conflate them.

**Track 1 — UX validation.** UX is rebuilding the 10 top-level travel LPs *now*, but the scope of UX's actual work is specifically **the order and appearance of recipe blocks** on each page. Nothing more. Anna needs to validate that UX's restructures don't make signal worse, and where data supports it, push back on specific block-ordering decisions. **The Lab is the tool for this track.**

**Track 2 — broader per-page optimization briefs.** Beyond UX's structural changes, Anna needs to produce optimization briefs per LP with recommendations UX isn't touching: contextual internal linking (where the widget-only-linking gap permits), copy improvements, keyword targeting, etc.

For Cruises and Vacation Packages (and possibly others), Anna already has extensive prior optimization recommendations — including her own page-reordering proposals from *before* UX's restructure landed. Those need to be **retrofitted**: cross-reference her prior reordering against UX's proposed reordering, resolve to best-options-overall, then layer broader recommendations on top.

For the remaining LPs, the briefs are first-pass new work.

The briefs are the deliverable for Track 2; Lab outputs are *one* input among several.

**Track 1 outputs feed Track 2.** Track 2 is where the optimization work actually lives. The Lab supports both but is built around Track 1.

In all cases, Anna needs **stakeholder-legible artifacts** that survive being forwarded around in meetings — see §10 for the matrix-of-versions sibling tool.

**Why this matters more than a normal sprint:** After this restructure ships (~May/June 2026), exec decision is to **cut "remediation and optimization" funding** for the AMA Travel website and reassign dedicated dev resources back to a shared business-unit pool. This is Anna's last high-leverage moment to influence the site for the foreseeable future.

**Implications for sprint scope decisions:**
- Bias toward what makes Anna's pushback case land *with execs*, not what's strategically elegant for some hypothetical future buildout.
- "Won't get worse" is the wrong target. Use the leverage window.
- Tools and instruments built during this window need to be self-sufficient afterwards — usable without ongoing dev support or further Bean elaboration.
- Legibility-to-execs matters more than analytical depth. Anna needs artifacts that are forwardable in meetings, not internal-tooling sophistication.
- "Phase 2 emerges from the next concrete pressure point after this" is acceptable. "Phase 2 should be partially built now to be ready for someday" is not — the someday Bean infrastructure won't exist.

Anna remains personally invested in the site doing well even after funding cut. The framing "leave a working instrument behind that whoever's still here can pull out next time there's a window" is real motivation, not a consolation prize.

---

## 9. Site Context You Need to Hold

Anna has been raising concerns about the AMA Travel site's technical posture for years. Most have been ignored, dismissed, or actively belittled by other stakeholders. The site is now tanking, and the framing some execs have landed on is "the site is failing, cut funding," as if the cause were detached from the choices that produced it. Hold this context — it's load-bearing for understanding both *why this sprint matters* and *why "the data already exists pre-launch" is a softer claim than it looks*.

### Known structural breakages (a partial list)

- **CSR / client-side rendering.** AMA Travel is a CSR Angular site. Google may not be processing the full rendered content of pages. Some pages (parent LPs) appear to be pre-rendered at build time via Angular Universal/SSG; child pages appear to fail hydration during Screaming Frog crawls (31 of 54 pages in the May 8 crawl are CSR-empty — only `<app-root>` + scripts in body content). JS rendering is enabled in the crawl with 25s render delay; doesn't help. Root cause unverified, hypothesis tracked separately. Affects: Lab can only ever work on parent LPs (~10) in the current crawl.
- **JS redirects at launch.** When the new site launched, devs implemented CSR JavaScript redirects for the ~70% of URLs that changed. Real (server-side) redirects were only implemented after execs got heat from the site tanking. **Result: months of Google indexing both versions, dividing clicks, no authority flowing to new URLs.**
- **404s never returned.** Pages that should have 404'd were returning 200s while customers saw "Oopsie!" pages instead of the actual destination (e.g., all-inclusive packages page). Soft-404 problem at scale.
- **Index bloat.** Pages Anna told the team to noindex pre-launch were left indexable. Parameter versions of URLs were indexable AND set to self-canonical (including the parameters), producing things like 17 distinct indexed versions of the Rome page. Currently ~4800 pages indexed when reality is ~2000.
- **Metadata was CSR until recently.** Page titles + meta descriptions were rendered client-side, meaning SERPs showed "AMA Travel Website" with no meta description for months.
- **Internal linking is essentially absent.** Contextual internal links (not duplicate widgets) don't exist except in the Travel Insurance section, which was put in place because the Insurance team had cachet to fight for at least basic SEO infrastructure. The rest of the site is linked through repeated widgets only.
- **Widget duplication = topic dilution.** ~80% of pages get flagged as duplicate in Screaming Frog crawls (both the tool's rendered-content assessment AND a separate per-page-embedding cosine similarity pass). Cause: the same widgets appear on every page and represent a huge proportion of total page text.
- **Entity model skew.** Google's persistent identity model of AMA skews heavily toward insurance — surfaced in the Dane/PPC investigation. Highest-scale incoherence; the technical fixes (SSR, schema, entity markup) that would address it have been descoped.

### Why pre-launch isn't a clean baseline

It's tempting to frame pre-launch as "what we lost — let's get back there." Don't. The pre-launch site had 20 years of accrued authority in a small-pond regional market and a lot of the same structural debt (widget-heavy, light internal linking, parameter-URL handling). The launch turned that authority into a *liability* rather than a clean inheritance — but pre-launch was not a high-water mark to be regained. Use pre-launch HTML/embeddings as **evidence of what was indexed and earning impressions before the botch**. Useful. Not aspirational.

### Working-with-Anna context

Your work-laptop project instructions / user preferences cover the operational pieces (PDA, demand-stacking, energy reads, communication patterns). This doc doesn't duplicate them. If you're missing them, ask Anna where to find them before proceeding — don't guess.

---

## 10. What's Actively Open

### Active workstream — the post-baseline phasing (May 11-12 2026)

The current-state baseline is done (v9 → v19 workbook chain + container diagnostic per version). The work proceeds through four phases — full detail in `experimental-phasing.md` memory:

- **Phase A — Lab Data Layer Reconciliation (Bean).** [SHIPPED May 11.] Lab consumes v19 curation from Supabase `lp_blocks` instead of inferring from raw HTML. Per-container NLP scoring built in. See `signal-coherence/PHASE-A-PLAN.md` for what landed.
- **Phase B — Bean-run permutation experiments.** [SHIPPED May 11-12.] 7 of 7 non-control LPs have methodology-format briefs at `signal-coherence/phaseB-results/{slug}/brief.md`. Methodology framework + brief template + anti-patterns at `signal-coherence/phaseB-results/METHODOLOGY.md`. Cross-LP permutations index (per-LP Tn key + cross-LP shape index) at `signal-coherence/phaseB-results/PERMUTATIONS-INDEX.md`. Each brief follows the 7-step methodology: Diagnose → Intent-align (from IA doc) → Identify mismatches → Generate hypotheses → Design permutations → Run + capture → Synthesize.
- **Phase C — Bean-run UX-recipe optimization.** Apply Phase B observations to the UX-recipe arrangement of each LP. Default is "make the recipe work" — Anna fights for every deviation. Per-LP outcomes: works as-is / works with X tweak / non-starter and here's why. **NOT YET STARTED.**
- **Phase D — Anna locks in.** Receives shortlist, plays in Lab, locks specific proposals, retrofits Cruises/VP briefs, outputs optimization guides for UX.

**Critical ownership note:** Phases A, B, C are Bean-executed. Phase D is Anna-executed. A Bean who asks Anna "what theory do you want to test" has inverted this. Run the permutations, bring the shortlist.

**Cars is the control, not a test subject.** Per the post-launch-state memory: rearranging doesn't move NLP confidence on Cars (98% → 98%). So Cars participates in baseline + UX-recipe measurement only. The 7 other LPs go through Phases B and C.

### Phase B cross-LP findings (n=7 LPs, May 11-12 2026)

Worth holding alongside the per-LP briefs (full detail in dedicated memory files):

- **Position×volume null at late position** (`memory/position-volume-null-late-position.md`). In Phase B testing across 7 LPs, removing late-position content (n=11 null tests, volumes from 137w to 4,461w) produced null category-layer change. Suggests AMA Travel LP whole-page category measurement is position-weighted to approximately the first half of the page; late content was invisible to category-layer measurement in these tests.
- **Two-layer NLP model** (`memory/two-layer-nlp-model.md`). Categories driven by position-weighted content; entities driven by mention frequency. Different lever types affect different layers. T1 (ghost large late container) tests routinely produce null at category-layer + dramatic rebalance at entity-layer.
- **Additive-content lever is page-state-dependent** (`memory/phase-b-lever-families.md`). Three distinct behaviors depending on baseline state + vocab-target alignment: bloat (Hotels) / mass-rebalance (VP, Flights) / target sub-cat lift (TTD) / clean lift (Destinations, TI). Not a uniform anti-pattern as previously thought.
- **Articles-promotion bloat pattern** (n=4 LPs). Promoting article-carousel container to position 2 always lifts top-cat but always introduces secondary-cat bloat. Anti-pattern for clean top-cat lift but valid lever if bloat is acceptable.
- **GSC rendering failure pattern** (n=6 of 8 LPs). The deal-carousel widget rendering fails for Googlebot consistently. Per `memory/worst-case-dev-scenarios.md`, we operate under the assumption that this won't get fixed; curation corrected on our side (v14-v19) to be GSC-faithful.

### Phase 2 bits worth pulling forward for the current sprint

These have real bite for the UX-restructure window. Priority order:

1. **Slot × intent diagnostic.** Run the NLP API on each *slot's* content separately (not just whole-page). Output per slot: dominant category + confidence. Surfaces things like *"this commercial LP's hero slot is reading 70% informational"* or *"the Travel Expertise slot on /vacation-packages is leaking 40% Insurance — your entity-model skew is showing up at the position most likely to influence intent."* Highest leverage / lowest cost combination. Bridge already does the necessary NLP calls; this is per-slot dispatch + UI surfacing. Exec-legible without statistical literacy. Directly actionable (the fix is in the slot).
2. **Pre-launch HTML loaded into the matrix-of-versions view.** Load the 10 LPs' Aug 2025 HTML into `crawled_pages` (files exist on disk; loader script `signal-coherence/scripts/load_may8_crawl.py` is straightforward to adapt with `source='pre-launch-aug2025'` and same template_type tagging). The matrix view becomes: pre-launch | current (post-launch) | UX proposal | optimized variant — for each page. Stakeholder artifact. Frames the case as "here's the trajectory" rather than "here's what's wrong with UX's proposal." Note: pre-launch is *evidence*, not benchmark (see §9).
3. **Block drift gradient (Roche Limit lite, no threshold).** Embed each block + a page centroid (H1 + meta + primary keyword cluster). Color blocks by cosine similarity to centroid — bright = on-topic, dim = drifting. **No number, no threshold, just visible gradient.** Argues *"this block isn't fixed by UX moving it — it shouldn't be there at all,"* which moves the output from safer-than-UX to better-than-UX. Embeddings infrastructure exists but is currently dormant; this would activate it.

### Sibling tool (separate surface, not in the Lab)

**Matrix-of-versions stakeholder view.** Read-only, presentation-optimized, side-by-side of multiple labelled permutations of the same page. Different mental mode from the Lab; needs its own surface. The `permutations` table + `exportPermutations()` function are already there as substrate. Design constraints: no edit / drag / score; emphasis on legibility and forward-ability. A heatmap-style category × version matrix is one possible shape; others on the table. Co-design with Anna when picked up.

### Track 2 analytical work-on-deck (briefs, not Lab features)

This is the work that actually ships per-page recommendations. It uses Lab outputs but isn't Lab work.

**Status update May 11-12 2026:** Phase B briefs for all 7 non-control LPs are done in methodology format. They surface levers per LP grounded in IA-doc intent — that's hypothesis-generation, not Track 2 deliverables. Track 2 (final per-page optimization guides for UX) still wants:

- **Retrofit Cruises LP brief** against Phase B findings (the v17 brief surfaces levers; doesn't yet retrofit Anna's prior reordering proposal against UX recipe + new GSC-faithful baseline)
- **Retrofit Vacation Packages LP brief** against Phase B findings (the v15 brief notes the v13 "T7 recipe" finding was largely measuring phantom cards; the prior recipe's structural value separate from carousel content needs re-evaluating)
- **Fresh per-LP optimization guides** for the other 5 LPs (Hotels, TTD, Cruises, Destinations, Flights, Travel-Insurance). Phase B briefs are inputs; Track 2 guides are deliverables.
- **Per-LP brief structure** for Track 2 is Anna's call — likely some combination of: Phase B brief findings + IA-doc target alignment + proposed block-order recommendation (cross-referenced with UX recipe) + broader recommendations layer (links, copy, keywords) + supporting data (GSC performance via GSC MCP).

Beans should support this work without absorbing it. Anna is the analyst on the Track 2 briefs; a Bean's job is to make Phase B outputs + IA-doc data readily usable, not to write the briefs unless explicitly asked.

### Sprint-side queued work (carryover from May 9 sessions)

- **External-widget visual treatment.** Blocks/containers tagged `external_widget: true` should render with locked / read-only visual (greyed background, no contenteditable, no drag, lock icon). Data flag is set; UI doesn't act on it yet.
- **Article-card titles for plain articles-carousel.** Some article cards show "(card)" because they don't use H-tags or title-class divs. Likely needs a per-component splitter for `articles-carousel`. Anna wanted to see live before deciding shape.
- **Container label persistence.** Container renames are session-only. Persistence would need a `container_renames` table keyed on (slug, container_id) wired into load + onContainerLabelBlur.
- **Target keyword highlighting.** Distinct from entity highlighting. Comes from a per-slug curated list (spreadsheet → `target_keywords` table); different underlay color. Entity-highlight infrastructure can be reused with new class + data source. **Load-bearing for editing workflow** — without it, easy to accidentally cut target keywords.
- **Toggle to disable entity highlighting.** Trivial — flag in the panel head that flips `labHighlightsOn` and re-renders. Add only if Anna finds the yellow noisy.
- **Log/history screen for permutations.** `exportPermutations()` exists in source, removed from the score panel as a different mental mode. Needs its own UI surface for browsing saved permutations per page.
- **Per-block ghost (vs current container-level only).** Deferred. Container-level covers the "nuke a section" use case; per-block adds complexity without obvious gain. Add only if Anna finds she needs it.

### Phase 2+ on the conceptual roadmap (not now)

The four-lens scaffold + Roche Limit + KL Divergence / Information Gain + funnel-aware coherence remain intact as conceptual scaffold. They are NOT a build target. If specific bits get pulled forward (as in "Phase 2 bits worth pulling forward" above), they go in *as observation surfaces, not as the framework in full*.

### Explicitly NOT pulling forward for this sprint

- Information Gain / full competitor reference corpus (small-sample finding suggests low payoff for time-to-build)
- Full Funnel Lens implementation. Anna's IA planning doc is the corpus that would enable this — it's no longer blocked on data — but it's still sprint scope creep. Slot × intent diagnostic does the immediate diagnostic work via NLP categories without needing the full lens.
- Site lens / cannibalization tooling (out of scope; separate fight)
- Any hard-threshold math (gradients and observations only)
- Roche Limit threshold calibration

### CSR investigation (load-bearing side branch)

Promised, not done. Affects scope (currently Lab works only on ~10 parent LPs; child pages would silently produce zero blocks). Investigation steps + reference data: `signal-coherence/parser-redesign-may9/page-coverage-may8-crawl.md`. Anna also has alternate paths to getting HTML if the crawl can't be fixed; CSR is "political and bigger and related but also not."

---

## 11. Source Documents

You should not need to read these to act on this brief. They exist if you want deeper context on specific points.

| Document | What's in it |
|---|---|
| `Positional Entity Weighting - Methodology.md` | The May 7 narrow-validated thing. Four LPs. The directional finding that position matters. |
| `Signal Coherence Framework - Build Spec.md` | The May 7 elaborated thing. Full four-lens framework, Roche Limit, IG, six-phase build. **Treat as conceptual roadmap, not as TODO.** |
| `Signal Coherence Skill - Handoff Notes May 2026.md` | May 7 validation pass. Largely superseded by the May 8 reset; the "What needs fixing" section's first item (NLP API) is what the post-reset build is. |
| `SIGNAL-COHERENCE-STATUS.md` | The most detailed running status doc, updated through May 9 late night. Captures every micro-decision in the build. Good for "where in the code does X live" questions; this North Star is for "why did we make these decisions" questions. |
| `signal-coherence/parser-redesign-may9/` | Parser prototype + per-page parsed trees + CSR coverage scan. Reference data, not a doc. |

---

## How to use this doc

- Read once, end-to-end, before touching anything.
- If you have an instinct to propose something, search §5 first. If it's listed as settled, don't propose it.
- If you're scoping work, tie it back to §2 (the kernel). If it doesn't trace, stop.
- If you're proposing Phase 2 work, check §8. The window is short and shrinking.
- If you find yourself wanting to ask "but why didn't you do X?", check §5 and §6. The answer is probably already there. If it isn't, ASK Anna before assuming the omission is an oversight.
- This doc evolves. If you settle a new decision Anna agrees on, add it to §5 with a why. Don't let the doc rot.
