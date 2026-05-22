#!/usr/bin/env bash
# Surface commits on origin/main whose SHA isn't mentioned in the canonical
# memory docs (memory/lab-state-and-followups.md or Phase 1 Plan). Advisory
# only -- read-only, no blocking. Output is wrapped as JSON so the harness
# injects it into Claude's startup context.
#
# Wired up via .claude/settings.json SessionStart hook. Lives in the repo so
# every Bean session in nlpcompare picks it up.
#
# Heuristic: a commit's SHA appearing in either canonical doc = "acknowledged"
# (doc has caught up, however that happened -- same commit, follow-up commit,
# subsequent session). A commit that touched code files AND has no SHA mention
# in either doc AND doesn't declare itself as docs/chore/WIP/merge = drift.

set -uo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$REPO_ROOT" ]; then
  exit 0
fi
cd "$REPO_ROOT"

git fetch origin main --quiet 2>/dev/null || true

if git rev-parse --verify origin/main >/dev/null 2>&1; then
  BASE_REF="origin/main"
elif git rev-parse --verify main >/dev/null 2>&1; then
  BASE_REF="main"
else
  exit 0
fi

LOOKBACK_DAYS=7
CODE_PATHS_REGEX='^(lab/|matrix\.html|signal-coherence/|netlify/|supabase/)'

LAB_STATE="memory/lab-state-and-followups.md"
PHASE_PLAN="Phase 1 Plan"

COMMITS=$(git log "$BASE_REF" --since="$LOOKBACK_DAYS days ago" --reverse --pretty=format:'%h|%s' 2>/dev/null)
if [ -z "$COMMITS" ]; then
  exit 0
fi

FLAGGED_LINES=""
FLAGGED=0

append_flagged() {
  if [ -z "$FLAGGED_LINES" ]; then
    FLAGGED_LINES="$1"
  else
    FLAGGED_LINES="$FLAGGED_LINES"$'\n'"$1"
  fi
}

while IFS='|' read -r SHA SUBJECT; do
  # Skip merges + self-declared doc/chore/WIP commits
  if [[ "$SUBJECT" =~ ^(docs:|chore:|WIP:|wip:|Merge\ ) ]]; then
    continue
  fi

  # Only consider commits that touched code paths
  FILES=$(git show --name-only --format='' "$SHA" 2>/dev/null)
  TOUCHED_CODE=0
  while IFS= read -r FILE; do
    [ -z "$FILE" ] && continue
    if [[ "$FILE" =~ $CODE_PATHS_REGEX ]]; then
      TOUCHED_CODE=1
      break
    fi
  done <<< "$FILES"

  if [ "$TOUCHED_CODE" -eq 0 ]; then
    continue
  fi

  # Acknowledged if SHA appears in either canonical doc
  ACKNOWLEDGED=0
  if [ -f "$LAB_STATE" ] && grep -q "$SHA" "$LAB_STATE" 2>/dev/null; then
    ACKNOWLEDGED=1
  fi
  if [ "$ACKNOWLEDGED" -eq 0 ] && [ -f "$PHASE_PLAN" ] && grep -q "$SHA" "$PHASE_PLAN" 2>/dev/null; then
    ACKNOWLEDGED=1
  fi

  if [ "$ACKNOWLEDGED" -eq 0 ]; then
    append_flagged "- \`$SHA\` $SUBJECT"
    FLAGGED=$((FLAGGED + 1))
  fi
done <<< "$COMMITS"

# Build report. If nothing flagged, emit nothing (silent success keeps the
# session-start signal sparse for the common case).
if [ "$FLAGGED" -eq 0 ]; then
  exit 0
fi

REPORT="## Possible doc drift: $FLAGGED commit(s) on $BASE_REF (last $LOOKBACK_DAYS days) touched code without SHA-acknowledgment in memory/lab-state or Phase 1 Plan

$FLAGGED_LINES

These commits' SHAs don't appear in the canonical docs. Before planning new work, consider:
- Reading the commits to see what shipped
- Updating \`memory/lab-state-and-followups.md\` (Recent commits log) or \`Phase 1 Plan\` to acknowledge them
- If they were intentionally not worth documenting (truly trivial), prepend their commit messages with \`docs:\` / \`chore:\` / \`WIP:\` next time to short-circuit this check
- If the drift is real and load-bearing, reconcile FIRST so the rest of the session works against accurate docs"

jq -n --arg ctx "$REPORT" '{
  hookSpecificOutput: {
    hookEventName: "SessionStart",
    additionalContext: $ctx
  }
}'
