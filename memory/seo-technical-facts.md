---
name: SEO technical facts learned working with Anna
description: Specific technical claims Anna has confirmed or that emerged from the May 8 investigation. Cite these instead of generic SEO conventional wisdom when they apply.
type: project
originSessionId: bff994f9-d48f-4cf1-99e5-e4b3b29402c5
---
Specific technical SEO facts that came up during the May 8, 2026 investigation that should inform future work:

1. **Google discounts template/global links; only contextual internal links count for ranking.** Anna confirmed this is well-studied. The Screaming Frog "Unique Inlinks = 19 across all top-level pages" pattern was the global nav/footer hitting every page identically — that 19 doesn't matter. Real signal is total inlinks minus the template count. Apply this when reading any internal-link-count data: subtract the template baseline.

2. **JS-rendered redirects don't transfer authority cleanly.** Even Google's renderer can struggle, and other crawlers (Ahrefs) ignore them entirely. AMA's launch redirects were JS-rendered for months before becoming proper 301s. During that window: split URL ratings between old + new URLs, both showing up as live pages in Ahrefs and getting GSC clicks. Recovery from this period takes time even after the 301s are fixed properly.

3. **AI Overviews on insurance queries pre-existed AMA's launch (Feb 2026).** Don't blame AI Overview rollout for losses on "best travel insurance" / "car rental insurance" — those features were already there in Feb 2026 and earlier. Travel/transactional queries (vacation packages, hotel deals) had no AI Overview on either Feb 17 or May 5, 2026.

4. **Contextual link spec → UX → builds.** AMA's pattern: SEO planning produces a contextual internal link spec, UX cuts most of it during build, only teams with org clout (Insurance) get their version implemented. Useful background when proposing link-graph changes — the constraint isn't usually "what's right" but "whose project has the leverage."

5. **Branded search volume is a separate signal from SEO performance.** Brand demand can decline independently of how the site ranks. AMA's "ama travel" search volume is materially down. This pulls overall traffic down regardless of SEO action and is a marketing/awareness question, not an SEO one.

**Why:** These were either confirmed by Anna directly or emerged from the May 8 data. They're load-bearing for future SEO discussions and prevent re-litigating questions she's already settled.

**How to apply:** When discussing SEO at AMA, treat these as established context. Don't generalize from generic SEO advice when one of these AMA-specific facts applies. If a future investigation produces evidence that contradicts one of them, flag it explicitly rather than quietly updating.
