---
name: AMA Travel LP recipe spec (UX PROPOSAL, not current state)
description: The UX team's PROPOSED recipe for top-level LPs. Anna is litigating it, not implementing it. Six Sections, each containing Whats. Keeps getting flattened, which makes Beans tag everything as Hero. Bigger confusion magnet — Beans also keep treating it as current state. It is not.
type: project
originSessionId: 136d72e1-4763-4a25-bec9-e7c8b98b2a08
---

**This is what UX is PROPOSING.** Anna is validating it or pushing back on it. **It is not what the pages look like right now.**

Before measuring "how a page is functioning currently," group by **current containers** (per Anna's screenshots and the workbook's Container column, when those are sane), **not** by this proposed Section structure. Grouping current-state content by proposed Sections measures the future arrangement, not the present one — that produces the wrong baseline.

The proposed `top-level-lp` recipe has **six Sections**. Inside each Section there are one or more **Whats** (component-level patterns). Sections are the unit UX rearranges; Whats are the components within.

**Workbook mapping:** `UX Recipe Section` column = the *proposed* Section; `What` column = the What; **Container column (B) = current container grouping** (when not muddled by Bean rework, e.g., widget decomposition spilling heading-text container labels into rows that previously had `cN-bM` labels).

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

**Common confusion that keeps producing wrong tagging in the proposed Section column:**
- "Trusted Partners" is a What *inside Hero*, not its own Section. (The brand-logo carousel can appear visually lower on the page on some LPs in current state, but it's still recipe-Hero in the proposal.)
- "Image + Value Prop" is a What *inside Hero*, not its own Section.
- "Today's Deals" / "Deal Cards" is a What *inside Travel Product Carousel*, not its own Section.
- Popular-locations / Travel-style carousels are Whats *inside Mix & Match Callouts*, not their own Sections and not Hero.

**Why:** May 10 2026 session. Two distinct corrections:

1. The recipe kept getting flattened in past sessions (a flat list-of-11 lived in the North Star §4) and that flatness misled Beans into treating sub-component Whats as separate Sections — Anna: "please put this somewhere that all beans will see it b/c it keeps getting lost and that's how we end up with everything showing up in the hero."
2. Beans (including this one) keep conflating the UX proposal with the current state of the pages. Anna corrected the North Star's language directly: "with recipes being 'The ordered list of structural slots a page is supposed to have' NO - not SUPPOSED to - *PROPOSED by* UX. I am validating or not. What I shared is what UX is proposing. It is not current state. It is a reference for us to have once we've got our shit together for real."

**How to apply:**
- **For "how is the page functioning NOW" work** (Step 1 of the diagnostic methodology): group by current containers (Column B / screenshots), not by proposed Section.
- **For "is the UX proposal better or worse than current" work** (later step): only then is grouping by proposed Section meaningful — and only as a comparison against the established current-state baseline.
- If the workbook's Section column uses values OTHER than the six above (plus "Not in Recipe" as a rare last resort) — that's a tagging error in the proposal mapping; flag, don't act.
- If the workbook's Container column has mixed conventions (some `cN-bM`, some heading-text labels) — that's the muddle Anna mentioned; resolve with her, don't guess.
