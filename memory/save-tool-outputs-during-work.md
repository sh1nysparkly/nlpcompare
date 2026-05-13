---
name: Save tool outputs to disk during investigation work
description: For multi-hour investigations involving NLP/Ahrefs/GSC pulls, write each result to a structured file at the time of the call. Don't rely on conversation history to preserve data.
type: feedback
originSessionId: bff994f9-d48f-4cf1-99e5-e4b3b29402c5
---
When running an investigation that involves multiple API/tool calls producing data Anna will want to slice and dice later (NLP results, Ahrefs pulls, GSC comparisons, SERP overviews), save each result to a structured file (JSON or CSV) AT THE TIME of the call. Don't defer to "I'll capture it later" — the conversation will compact, the data will be hard to reconstruct, and Anna will have to ask "did you save that?"

**Why:** May 8, 2026 diagnosis session. Made ~15 NLP API calls, didn't save any of them to disk during the work. When Anna asked at the end "did you save the NLP outputs?" I had to reconstruct them from conversation context, which took time and introduced at least one transcription error (mistakenly attributed AMA's category data to Intact). Same risk applied to GSC and backlinks data. Anna was kind about it but it was avoidable friction.

**How to apply:** During investigation work, after each significant tool call that produces data worth keeping, write a structured file. JSON for NLP/SERP data, CSV for tabular comparisons. Pick a directory naming convention up front (e.g., `diagnosis-may8/{nlp-data,gsc-data,backlinks-data,serp-snapshots}/`) and use it consistently. The cost of writing a file is tiny; the cost of reconstructing is large.
