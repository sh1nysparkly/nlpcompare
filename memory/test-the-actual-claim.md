---
name: Test the actual claim, not the adjacent one
description: When testing a hypothesis, check it against the entities it actually concerns. Don't generalize from a related set.
type: feedback
originSessionId: bff994f9-d48f-4cf1-99e5-e4b3b29402c5
---
When testing a hypothesis, the evidence has to come from the actual entities the hypothesis is about — not adjacent ones. Generalizing from a related set produces wrong conclusions, even when the data itself is good.

**Why:** May 8, 2026 session. Hypothesis: "JS-rendered redirects broke external authority transfer." I checked URL Rating history for /vacation-packages, /cruises, /travel-insurance — and concluded the hypothesis was disproven because their URL ratings were stable. Anna pushed back: "/vacation-packages /cruises — didn't redirect. Their urls didn't change. And the TMI one changed months before the site launched. Did you look at any url ratings for pages that actually did change urls?" The hypothesis was about migrated URLs; I'd tested non-migrated URLs. When I re-tested on /cars, /activities, /rental-vehicle-insurance etc., the split-equity pattern was clearly visible.

**How to apply:** When designing a test for a hypothesis, name the entity class the hypothesis is about ("pages that had URL changes during the launch") and check that the test data is from that class. If it isn't, the test isn't a test. This is especially important when reaching for tidy "ruled out" conclusions — overreach in eliminating possibilities is as dangerous as overreach in confirming them.
