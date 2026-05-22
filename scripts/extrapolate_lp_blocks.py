#!/usr/bin/env python3
"""
Extrapolate lp_blocks rows from cleaned (rendered + class-stripped) HTML.

For pages where Anna has hand-cleaned the rendered HTML and dropped it into
crawled_pages.raw_html, this script walks the structural tags she preserved
(<page-section>, widget tags) and emits one row per text-bearing element,
grouped into block_ids that match the conventions used in the existing v19
hand-curation (c{N}-b{N} for blocks, c{N}-h{N} for value-prop subheadings).

Provenance is honest: rows are tagged curation_version='v1-extrapolated' so
future readers (including future Anna) know they were not hand-curated and
weight accordingly.

Usage:
    python extrapolate_lp_blocks.py --slug /vacation-packages/tours [--dry-run|--commit]
    python extrapolate_lp_blocks.py --slug /vacation-packages/tours --html-file path/to.html

Default mode is --dry-run: prints rows as a table to stdout, no DB writes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from bs4 import BeautifulSoup, NavigableString, Tag


# ---------------------------------------------------------------------------
# Constants -- mirror the taxonomies used in the existing v19 lp_blocks data
# and the Lab's parser (lab/index.html ~lines 870-893).
# ---------------------------------------------------------------------------

CONTAINER_TAGS = {"page-section", "page"}

# Widget tags inside containers, with the child-card tag where known.
# Mirrors CARD_COMPONENTS in lab/index.html line 875.
#
# Some pages have had their HTML tag-suffix-stripped during Anna's manual
# render+cleanup (e.g. <insurance-product-carousel> -> <insurance>,
# <articles-carousel> -> <articles>). Those aliases live alongside the full
# tag names so detection works on either form.
CARD_WIDGETS = {
    "navigation-carousel": "navigation-card-view",
    "insurance-product-carousel": "insurance-product-card",
    "recommended-destinations-carousel": None,  # auto-detect
    "articles-carousel": None,
    "cruises-carousel": "cruise-card",
    "travel-dreams-carousel": "travel-dream-card",
    "content-feed-widget": "content-feed-card",
    "brands-carousel": None,
    "tri-card-content": None,
    # Cleanup-stripped aliases. For <insurance>, the same tag is both the
    # carousel wrapper and the card -- disambiguated at detection time by
    # checking for a <carousel> descendant (wrapper) vs not (card).
    "insurance": "insurance",
    "articles": None,
}
# Tags that can be value-prop-widget aliases (full + cleanup-stripped)
VALUE_PROP_WIDGET_TAGS = ("value-prop-widget", "value")
# Tags that can be FAQ-widget aliases
FAQ_WIDGET_TAGS = ("faqs-widget", "faqs")
# Legacy single-tag names preserved for any external callers/tests
VALUE_PROP_WIDGET = "value-prop-widget"
FAQ_WIDGET = "faqs-widget"
EXTERNAL_WIDGET_TAGS = {
    "tst-widget", "tst-control", "tabbed-tst-carousel",
    "vacation-packages-carousel", "form-activities",
}
# Chrome the parser strips before walking
CHROME_TAGS = {
    "script", "style", "noscript", "header", "footer", "nav", "aside",
    "site-footer", "travel-agent-contact-fab", "regular-footer",
    "breadcrumb", "trip-preparation",  # trailing nav-only containers
}
# Per CURATION_ALLOWED_TAGS in lab/index.html line 1375 -- these are the tags
# the Lab will emit HTML for when rehydrating curated rows. Anything else
# gets coerced to <div>.
ALLOWED_OUTPUT_TAGS = {
    "h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "a", "button", "div",
    # We also keep span at row level even though the curator HTML coerces unknown
    # to div -- but spans matter taxonomically (Badge etc.) and v19 includes them.
    "span",
}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

# Sentinel placeholder texts that v19 drops -- per CURATION_PLACEHOLDER_TEXTS
PLACEHOLDER_TEXTS = {"NA - TST WIDGET", "NA-TST WIDGET", "NA TST WIDGET", "GRAB IMAGE ALT TEXT"}


# ---------------------------------------------------------------------------
# Row shape (matches lp_blocks columns)
# ---------------------------------------------------------------------------

@dataclass
class Row:
    slug: str
    row_order: int
    block_id: str
    container_label: str
    tag: str
    text: str
    what: Optional[str] = None
    component_type: Optional[str] = None
    pattern_key: Optional[str] = None
    curation_version: str = "v20-extrapolated"

    def for_insert(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip() if s else ""


def element_text(el: Tag) -> str:
    """Direct text content collapsed, ignoring descendant block-level text."""
    return clean_text(el.get_text(separator=" ", strip=True))


def shallow_text(el: Tag) -> str:
    """Text from this element's immediate text nodes only (no descendants)."""
    parts = []
    for child in el.children:
        if isinstance(child, NavigableString):
            t = str(child).strip()
            if t:
                parts.append(t)
    return clean_text(" ".join(parts))


def strip_chrome(soup: BeautifulSoup) -> None:
    """Remove sidewide chrome before walking. Mutates in place."""
    for tag in CHROME_TAGS:
        for el in soup.find_all(tag):
            el.decompose()


def find_containers(root: Tag) -> list[Tag]:
    """Find ALL <page-section> / <page> elements that hold content.

    We include nested ones because Anna's cleaned HTML sometimes has malformed
    opening/closing tag pairs (e.g. /travel-insurance/car-rental-insurance opens
    <page> but closes </page-section>) which lxml interprets as a chain of
    deeply-nested <page> elements. Each one carries its own direct content; the
    nesting is an artifact of the parser, not a meaningful hierarchy.

    container_blocks() handles the "own content only" extraction so nested
    containers don't double-emit.
    """
    containers: list[Tag] = []
    seen: set[int] = set()
    for tag in CONTAINER_TAGS:
        for el in root.find_all(tag):
            if id(el) in seen:
                continue
            # Test whether this container has its OWN content (not just a nested
            # child container). own_content = direct text + direct non-container
            # element descendants up to the next nested container.
            own_text = direct_content_text(el)
            if own_text:
                containers.append(el)
                seen.add(id(el))
    # Preserve document order (find_all already does this within a tag; merge across tags)
    containers.sort(key=lambda c: _document_position(c, root))
    return containers


def direct_content_text(container: Tag) -> str:
    """Return text that belongs to `container` excluding nested CONTAINER_TAGS children."""
    parts = []
    for node in container.descendants:
        if isinstance(node, NavigableString):
            # Skip if inside a nested container
            p = node.parent
            inside_nested = False
            while p and p is not container:
                if p.name and p.name.lower() in CONTAINER_TAGS:
                    inside_nested = True
                    break
                p = p.parent
            if not inside_nested:
                t = str(node).strip()
                if t:
                    parts.append(t)
    return clean_text(" ".join(parts))


def _document_position(el: Tag, root: Tag) -> tuple:
    """Return a tuple representing the element's position in the document for sorting."""
    path = []
    cur = el
    while cur and cur is not root:
        parent = cur.parent
        if parent:
            siblings = [c for c in parent.children if isinstance(c, Tag)]
            try:
                idx = siblings.index(cur)
            except ValueError:
                idx = 0
            path.append(idx)
        cur = parent
    return tuple(reversed(path))


def iter_own_descendants(container: Tag):
    """Iterate descendants of `container` that aren't inside a nested CONTAINER_TAGS element."""
    for node in container.descendants:
        if not isinstance(node, Tag):
            continue
        # Skip nested-container elements themselves and their descendants
        p = node.parent
        inside_nested = False
        while p and p is not container:
            if p.name and p.name.lower() in CONTAINER_TAGS:
                inside_nested = True
                break
            p = p.parent
        if inside_nested:
            continue
        if node.name and node.name.lower() in CONTAINER_TAGS:
            continue
        yield node


def heading_level(tag_name: str) -> int:
    if tag_name and len(tag_name) == 2 and tag_name[0] == "h" and tag_name[1].isdigit():
        return int(tag_name[1])
    return 0


def container_label_for(container: Tag, fallback_idx: int) -> str:
    """Pick the highest-level heading inside the container."""
    for lvl in range(1, 7):
        h = container.find(f"h{lvl}")
        if h and element_text(h):
            return element_text(h)
    # Widget-named fallbacks
    if find_any(container, VALUE_PROP_WIDGET_TAGS):
        return "Value props"
    if container.find("articles-carousel") or container.find("articles"):
        return "Related articles"
    if container.find("content-feed-widget"):
        return "Related content"
    if container.find("trip-preparation"):
        return "Trip preparation"
    return f"(unlabeled section {fallback_idx + 1})"


def is_external_widget(container: Tag) -> bool:
    for tag in EXTERNAL_WIDGET_TAGS:
        if container.find(tag):
            return True
    return False


def find_any(container: Tag, tag_names) -> Optional[Tag]:
    """Find the first descendant matching any of the given tag names."""
    for t in tag_names:
        found = container.find(t)
        if found:
            return found
    return None


def detect_card_widget(container: Tag) -> Optional[tuple[str, Optional[str]]]:
    """Returns (widget_tag, card_tag) if container has a card widget, else None.

    Special-case: <insurance> can be both the carousel WRAPPER and each card.
    The wrapper has a <carousel> descendant; cards don't. We only treat the
    first <insurance> as a widget if it has a <carousel> descendant.
    """
    for widget_tag, card_tag in CARD_WIDGETS.items():
        found = container.find(widget_tag)
        if not found:
            continue
        # Disambiguate <insurance>: only treat as widget when it contains
        # a <carousel> descendant (otherwise it's a single-card scenario).
        if widget_tag == "insurance" and not found.find("carousel"):
            continue
        return widget_tag, card_tag
    return None


def detect_cards(widget_el: Tag, card_tag: Optional[str]) -> list[Tag]:
    """Find card-shaped repeating children inside widget_el.

    For the explicit card_tag case, prefer the shallowest set of cards so we
    don't cascade through nested wrappers (the <insurance> wrapper-tag ==
    card-tag case in Anna's cleaned HTML).
    """
    if card_tag:
        ambiguous = (card_tag == widget_el.name)
        # 1. Direct children of widget_el matching card_tag
        direct = [c for c in widget_el.children if isinstance(c, Tag) and c.name == card_tag]
        if len(direct) >= 2:
            return direct
        # 2. One level deeper (inside a wrapping element like <carousel>)
        for child in widget_el.children:
            if isinstance(child, Tag):
                nested = [c for c in child.children if isinstance(c, Tag) and c.name == card_tag]
                if len(nested) >= 2:
                    return nested
        # 3. Ambiguous case (wrapper-tag == card-tag, e.g. <insurance>): give up
        #    here rather than risk picking up cascaded nested wrappers that
        #    over-emit content. Caller's fallback emits one block instead.
        if ambiguous:
            return []
        # 4. Legacy fallback for unambiguous cases: any descendant.
        return widget_el.find_all(card_tag)
    # Auto-detect: any descendant with 3+ children sharing a tag name
    for el in widget_el.find_all(True):
        children = [c for c in el.children if isinstance(c, Tag)]
        if len(children) >= 3:
            counts: dict[str, int] = {}
            for c in children:
                counts[c.name] = counts.get(c.name, 0) + 1
            best_tag = max(counts, key=lambda k: counts[k])
            if counts[best_tag] >= 3 and best_tag not in ("br", "hr"):
                return [c for c in children if c.name == best_tag]
    return []


# ---------------------------------------------------------------------------
# Row emission: per-element walker
# ---------------------------------------------------------------------------

def emit_rows_for_element(
    el: Tag,
    rows_out: list[tuple[str, str]],  # (tag, text) tuples
) -> None:
    """Walk el and collect (tag, text) pairs for each text-bearing element.

    Strategy: for each descendant that has direct text content AND a relevant
    tag, emit a row. Headings and explicit container elements get their own
    rows; spans/buttons inside larger blocks are folded into the parent's text.

    We only emit rows for tags in ALLOWED_OUTPUT_TAGS + a handful of known-good
    structural tags. Custom angular tags are walked through, not emitted.
    """
    for descendant in el.descendants:
        if not isinstance(descendant, Tag):
            continue
        tname = descendant.name.lower() if descendant.name else ""
        if tname in CHROME_TAGS:
            continue
        if tname not in ALLOWED_OUTPUT_TAGS:
            continue
        # For containers like h1-h6, p, li, button: capture full text including
        # inline children (spans, etc.). For a, similar but trimmed.
        text = element_text(descendant)
        if not text:
            continue
        if text.upper() in PLACEHOLDER_TEXTS:
            continue
        # Skip if the same text was already captured by an ancestor we'll emit
        # for. Approximation: skip if the parent text equals this text (would
        # be folded). The cleaner check is parent-emitted-already, which we do
        # by tracking emitted text spans.
        parent_text = element_text(descendant.parent) if descendant.parent else ""
        if parent_text == text and descendant.parent.name.lower() in ALLOWED_OUTPUT_TAGS:
            continue
        rows_out.append((tname, text))


def emit_rows_dedup(el: Tag) -> list[tuple[str, str]]:
    """Walk el and emit (tag, text) pairs, deduping nested duplicates.

    Strategy: start from el itself. When we hit an ALLOWED_OUTPUT_TAGS element
    with non-empty text, emit it AND suppress its descendants (so an <a> card
    wrapping <h3>Title</h3><span>foo</span><h4>price</h4> emits ONE row with
    the joined text, not four overlapping rows).

    Container-style tags (those NOT in ALLOWED_OUTPUT_TAGS, e.g. <page-section>,
    custom angular tags) pass through without emitting, but their descendants
    still get walked.
    """
    out: list[tuple[str, str]] = []
    emitted_text_set: set[str] = set()

    def walk(node: Tag, suppress: bool) -> None:
        if not isinstance(node, Tag):
            return
        tname = (node.name or "").lower()
        if tname in CHROME_TAGS:
            return
        new_suppress = suppress
        if not suppress and tname in ALLOWED_OUTPUT_TAGS:
            text = element_text(node)
            if text and text.upper() not in PLACEHOLDER_TEXTS and text not in emitted_text_set:
                out.append((tname, text))
                emitted_text_set.add(text)
                new_suppress = True
        for child in node.children:
            if isinstance(child, Tag):
                walk(child, new_suppress)

    walk(el, False)
    return out


# ---------------------------------------------------------------------------
# Block-splitting strategies (per-widget)
# ---------------------------------------------------------------------------

def split_value_props(container: Tag) -> list[list[tuple[str, str]]]:
    """value-prop-widget (or cleanup-stripped <value>): each (icon-span,
    label, p) triplet is its own block.

    Anna's cleaned HTML for VPs is flat: <span>icon</span>label<p>copy</p>
    repeated. We split by detecting span->text->p triplets.
    """
    vp = find_any(container, VALUE_PROP_WIDGET_TAGS)
    if not vp:
        return []
    blocks: list[list[tuple[str, str]]] = []
    # First emit any container-level intro (heading + intro p before the VP)
    intro_rows = []
    for sib in container.children:
        if isinstance(sib, Tag) and sib.name.lower() in VALUE_PROP_WIDGET_TAGS:
            break
        if isinstance(sib, Tag):
            intro_rows.extend(emit_rows_dedup(sib))
    if intro_rows:
        blocks.append(intro_rows)

    # Walk VP children and group as triplets: span(icon) + heading-text + p(body)
    # Heading-text is a NavigableString sibling between the icon-span and the p.
    current: list[tuple[str, str]] = []
    current_has_label = False
    children = list(vp.children)
    i = 0
    while i < len(children):
        node = children[i]
        if isinstance(node, NavigableString):
            text = clean_text(str(node))
            if text:
                # This is the VP subheading (e.g. "Benefits and Savings")
                if current and current_has_label:
                    blocks.append(current)
                    current = []
                current_has_label = False
                current.append(("div", text))  # subheading as div per v19 pattern
                current_has_label = True
            i += 1
            continue
        if isinstance(node, Tag):
            tname = node.name.lower()
            if tname == "span":
                # icon, skip text content (it's just the icon name like "savings")
                # but use it as the boundary marker for a new VP starting
                if current and current_has_label:
                    blocks.append(current)
                    current = []
                    current_has_label = False
                i += 1
                continue
            elif tname == "p":
                text = element_text(node)
                if text:
                    current.append(("p", text))
                i += 1
                continue
            else:
                # Other tags inside VP: fall through to row emission
                current.extend(emit_rows_dedup(node))
                i += 1
                continue
        i += 1
    if current:
        blocks.append(current)
    return blocks


def split_card_widget(container: Tag, widget_tag: str, card_tag: Optional[str]) -> list[list[tuple[str, str]]]:
    """Card widget: intro (container content before the widget) + 1 block per card."""
    widget = container.find(widget_tag)
    if not widget:
        return []
    blocks: list[list[tuple[str, str]]] = []
    # Intro: container content before/around the widget
    intro_rows = []
    for sib in container.children:
        if isinstance(sib, Tag):
            if sib is widget or widget in sib.descendants:
                # Capture rows from siblings of the widget *within this parent*
                continue
            intro_rows.extend(emit_rows_dedup(sib))
    # Also pick up direct text children of the container
    for sib in container.children:
        if isinstance(sib, NavigableString):
            text = clean_text(str(sib))
            if text:
                intro_rows.append(("p", text))
    # Find the actual widget parent's pre-widget siblings (handle wrapping like
    # <page-section><h3>...</h3><widget>...</widget></page-section>)
    # Simpler: walk container, stop when we hit the widget.
    pre_widget_rows = []
    for el in container.find_all(True):
        if el is widget:
            break
        if el.name.lower() in ALLOWED_OUTPUT_TAGS:
            text = element_text(el)
            if text and text.upper() not in PLACEHOLDER_TEXTS:
                if not pre_widget_rows or pre_widget_rows[-1][1] != text:
                    pre_widget_rows.append((el.name.lower(), text))
    if pre_widget_rows:
        # Dedupe: drop rows whose text is a substring of a later row at the same level
        deduped = []
        seen = set()
        for tag, text in pre_widget_rows:
            if text in seen:
                continue
            deduped.append((tag, text))
            seen.add(text)
        blocks.append(deduped)

    # Cards
    cards = detect_cards(widget, card_tag)
    if cards:
        for card in cards:
            rows = emit_rows_dedup(card)
            if rows:
                blocks.append(rows)
    else:
        # No detectable card shape (e.g. ambiguous <insurance> wrapper-tag
        # case with cascaded nested wrappers). Emit the widget content as
        # a single block so we don't lose it.
        widget_rows = emit_rows_dedup(widget)
        if widget_rows:
            blocks.append(widget_rows)
    return blocks


def split_faq(container: Tag) -> list[list[tuple[str, str]]]:
    """faqs-widget (or cleanup-stripped <faqs>): alternating <button>Q</button><p>A</p> pairs."""
    widget = find_any(container, FAQ_WIDGET_TAGS)
    if not widget:
        return []
    blocks: list[list[tuple[str, str]]] = []
    # Intro before widget
    pre_widget_rows = []
    for el in container.find_all(True):
        if el is widget:
            break
        if el.name.lower() in ALLOWED_OUTPUT_TAGS:
            text = element_text(el)
            if text and text.upper() not in PLACEHOLDER_TEXTS:
                if not pre_widget_rows or pre_widget_rows[-1][1] != text:
                    pre_widget_rows.append((el.name.lower(), text))
    if pre_widget_rows:
        deduped, seen = [], set()
        for t, x in pre_widget_rows:
            if x not in seen:
                deduped.append((t, x))
                seen.add(x)
        blocks.append(deduped)

    # Pair up button + answer. The answer can be either a <p> sibling OR
    # one or more bare-text nodes between this <button> and the next.
    # Anna's cleaned HTML drops <p> wrappers on most FAQ answers, leaving
    # the text as direct children of the widget.
    children = list(widget.children)  # both Tags and NavigableStrings
    i = 0
    while i < len(children):
        node = children[i]
        if isinstance(node, Tag) and node.name.lower() == "button":
            q = element_text(node)
            answer_parts: list[str] = []
            answer_tags: list[tuple[str, str]] = []
            j = i + 1
            while j < len(children):
                nxt = children[j]
                if isinstance(nxt, Tag) and nxt.name.lower() == "button":
                    break
                if isinstance(nxt, Tag):
                    nm = nxt.name.lower()
                    if nm in ("p", "ul", "ol"):
                        t = element_text(nxt)
                        if t:
                            answer_tags.append(("p" if nm == "p" else "ul", t))
                    elif nm == "a":
                        t = element_text(nxt)
                        if t:
                            answer_tags.append(("a", t))
                else:
                    s = clean_text(str(nxt))
                    if s:
                        answer_parts.append(s)
                j += 1
            i = j
            if q:
                rows = [("button", q)]
                if answer_parts:
                    rows.append(("p", " ".join(answer_parts)))
                rows.extend(answer_tags)
                blocks.append(rows)
        else:
            i += 1
    return blocks


def split_by_headings(container: Tag) -> list[list[tuple[str, str]]]:
    """Default: split container by H3 headings (or H4 if no H3s at same depth)."""
    rows = emit_rows_dedup(container)
    if not rows:
        return []
    # Find heading boundaries (H3 or H4 -- whichever has 2+ at same level)
    h3_count = sum(1 for t, _ in rows if t == "h3")
    h4_count = sum(1 for t, _ in rows if t == "h4")
    if h3_count >= 2:
        split_tag = "h3"
    elif h4_count >= 2:
        split_tag = "h4"
    else:
        return [rows]  # Single block

    blocks: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    for tag, text in rows:
        if tag == split_tag:
            if current:
                blocks.append(current)
            current = [(tag, text)]
        else:
            current.append((tag, text))
    if current:
        blocks.append(current)
    return blocks


def trim_nested_containers(container: Tag) -> Tag:
    """Return a clone of container with nested CONTAINER_TAGS descendants removed.

    Used to isolate a container's "own" content before extraction. This way every
    downstream function (emit_rows_dedup, split_*, detect_*) can use the usual
    find_all/descendants APIs without worrying about nested-container leakage.
    """
    import copy
    clone = copy.copy(container)
    # bs4 .copy() is shallow; need a real deep copy via str round-trip OR via
    # BeautifulSoup's __copy__. Use the built-in deep clone via decomposing
    # nested containers from a string-parsed copy.
    soup = BeautifulSoup(str(container), "lxml")
    # The body wraps the clone; find the matching root tag inside body
    root = soup.find(container.name)
    if root is None:
        return container  # fallback: operate on original
    for tag in CONTAINER_TAGS:
        for nested in list(root.find_all(tag)):
            if nested is not root:
                nested.decompose()
    return root


def container_blocks(container: Tag) -> tuple[list[list[tuple[str, str]]], str]:
    """Returns (list of blocks, pattern_key for the container).

    pattern_key maps to LAB_PATTERNS entries in lab/index.html (~line 695).
    Caller is responsible for trimming nested CONTAINER_TAGS children (see
    trim_nested_containers) so this function can use find_all freely.
    """
    # 1. FAQ widget (or cleanup-stripped <faqs>)
    if find_any(container, FAQ_WIDGET_TAGS):
        return split_faq(container), "faqs"
    # 2. Value-prop widget (or cleanup-stripped <value>)
    if find_any(container, VALUE_PROP_WIDGET_TAGS):
        return split_value_props(container), "value_props"
    # 3. Card widget
    card_match = detect_card_widget(container)
    if card_match:
        widget_tag, card_tag = card_match
        # Map widget_tag -> pattern_key (best effort)
        pattern_key_map = {
            "navigation-carousel": "nav_cards",
            "insurance-product-carousel": "deal_cards",
            "recommended-destinations-carousel": "nav_cards",
            "articles-carousel": "articles",
            "cruises-carousel": "deal_cards",
            "travel-dreams-carousel": "deal_cards",
            "content-feed-widget": "articles",
            "brands-carousel": "trusted_partners",
            "tri-card-content": "value_props",
            # Cleanup-stripped aliases
            "insurance": "deal_cards",
            "articles": "articles",
        }
        return (
            split_card_widget(container, widget_tag, card_tag),
            pattern_key_map.get(widget_tag),
        )
    # 4. External widget (tst-widget etc.) — treat as single block, mark external
    if is_external_widget(container):
        rows = emit_rows_dedup(container)
        return [rows] if rows else [], "widget"
    # 5. Auto-card detection: 3+ direct child <a> elements = card grid
    #    (e.g. /vacation-packages/tours "Recommended for You" — 6 tour <a> cards)
    direct_a_children = [c for c in container.children if isinstance(c, Tag) and c.name.lower() == "a"]
    if len(direct_a_children) >= 3:
        blocks: list[list[tuple[str, str]]] = []
        # Intro: everything before the first <a>
        first_a_idx = next((i for i, c in enumerate(list(container.children)) if isinstance(c, Tag) and c.name.lower() == "a"), -1)
        intro_rows: list[tuple[str, str]] = []
        for i, c in enumerate(container.children):
            if i >= first_a_idx:
                break
            if isinstance(c, Tag):
                intro_rows.extend(emit_rows_dedup(c))
        if intro_rows:
            blocks.append(intro_rows)
        # One block per card <a> -- collapsed to single row each (matches v19 card pattern)
        for card in direct_a_children:
            card_text = element_text(card)
            if card_text and card_text.upper() not in PLACEHOLDER_TEXTS:
                blocks.append([("a", card_text)])
        # Trailing content after the last card (e.g. paginator, "View All" CTA)
        last_a = direct_a_children[-1]
        trailing_rows: list[tuple[str, str]] = []
        seen_last = False
        for c in container.children:
            if seen_last and isinstance(c, Tag):
                trailing_rows.extend(emit_rows_dedup(c))
            if c is last_a:
                seen_last = True
        if trailing_rows:
            blocks.append(trailing_rows)
        return blocks, "deal_cards"
    # 6. Fallback: split by H3/H4 if multiple, else single block
    return split_by_headings(container), None


# ---------------------------------------------------------------------------
# Taxonomy guessing for what / component_type
# ---------------------------------------------------------------------------

def guess_component_type(tag: str, text: str) -> Optional[str]:
    t = tag.lower()
    if t in HEADING_TAGS:
        return "Heading"
    if t == "p":
        return "Copy"
    if t == "li":
        return "Copy"
    if t == "a":
        # If text is short and verb-y, call it Button; else Link
        if len(text) <= 32 and any(text.lower().startswith(v) for v in ("get ", "view ", "find ", "explore ", "match ", "book ", "search ", "learn ", "contact ", "see ", "shop ", "buy ", "talk ", "call ")):
            return "Button"
        return "Button" if len(text) <= 24 else "Copy"
    if t == "button":
        return "Button"
    if t == "span":
        return "Copy"
    if t == "div":
        return "Subheading (div)"
    return None


def guess_what(tag: str, text: str, pattern_key: Optional[str], is_external_widget_container: bool, container_idx: int, block_idx: int) -> Optional[str]:
    t = tag.lower()
    # Hero container (first one) gets H1 / Intro / Widget / Badge style
    if container_idx == 0:
        if t == "h1":
            return "H1"
        if t == "p" and block_idx == 0:
            return "Intro"
        if t == "a" and block_idx == 0:
            return "Widget"
        if t == "span":
            return "Badge"
    if is_external_widget_container:
        return "Widget"
    if pattern_key == "value_props":
        return "Value Prop Bar"
    if pattern_key in ("deal_cards", "nav_cards", "articles"):
        return "Deal Cards"
    if pattern_key == "faqs":
        return "FAQ"
    # Generic section-block / agent-widget = "Image + Value Prop" in v19 nomenclature
    return "Image + Value Prop"


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_rows(slug: str, html: str) -> list[Row]:
    soup = BeautifulSoup(html, "lxml")
    strip_chrome(soup)
    body = soup.body or soup
    # The root is whatever wraps everything inside <app-root>/<app>: tours-landing-page, etc.
    # We just find containers from the body level.
    containers = find_containers(body)

    rows: list[Row] = []
    row_order = 0
    for c_idx, container in enumerate(containers):
        cid = f"c{c_idx}"
        # Use the trimmed clone for all per-container queries so nested
        # containers don't pollute label / widget detection / etc.
        trimmed = trim_nested_containers(container)
        label = container_label_for(trimmed, c_idx)
        blocks, pattern_key = container_blocks(trimmed)
        external = is_external_widget(trimmed)
        # For value-prop blocks, use h-prefix in block_id per v19 convention
        bid_prefix = "h" if pattern_key == "value_props" else "b"
        # The intro block (if present) before VP/cards uses "b" prefix always (it's the heading+intro)
        for b_idx, block in enumerate(blocks):
            # For VPs: block 0 (if it contains the H tag) is intro -> "b0"; the rest are "h0", "h1", ...
            if pattern_key == "value_props":
                if b_idx == 0 and any(t in HEADING_TAGS for t, _ in block):
                    bid = f"{cid}-b0"
                else:
                    h_idx = b_idx - 1 if any(t in HEADING_TAGS for t, _ in blocks[0]) else b_idx
                    bid = f"{cid}-h{h_idx}"
            else:
                bid = f"{cid}-b{b_idx}"
            for tag, text in block:
                if not text:
                    continue
                row_order += 1
                rows.append(Row(
                    slug=slug,
                    row_order=row_order - 1,  # 0-indexed to match v19
                    block_id=bid,
                    container_label=label,
                    tag=tag,
                    text=text,
                    what=guess_what(tag, text, pattern_key, external, c_idx, b_idx),
                    component_type=guess_component_type(tag, text),
                    pattern_key=pattern_key,
                ))
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def fetch_html_via_supabase(slug: str) -> Optional[str]:
    """Read raw_html via PostgREST. Reads SUPABASE_URL + SUPABASE_KEY from env."""
    import requests
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        return None
    r = requests.get(
        f"{url}/rest/v1/crawled_pages",
        params={"slug": f"eq.{slug}", "select": "raw_html"},
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    return data[0]["raw_html"]


def commit_rows_via_supabase(rows: list[Row]) -> None:
    import requests
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL + SUPABASE_KEY required for --commit")
    payload = [r.for_insert() for r in rows]
    r = requests.post(
        f"{url}/rest/v1/lp_blocks",
        json=payload,
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
        timeout=60,
    )
    if not r.ok:
        raise RuntimeError(f"Insert failed: {r.status_code} {r.text[:500]}")


def print_rows_table(rows: list[Row]) -> None:
    print(f"\n{len(rows)} rows for {rows[0].slug if rows else '(none)'}\n")
    print(f"{'#':>3}  {'block_id':<8}  {'tag':<7}  {'container_label':<40}  {'what':<22}  text")
    print("-" * 140)
    for r in rows:
        text_snip = r.text[:70].replace("\n", " ")
        print(f"{r.row_order:>3}  {r.block_id:<8}  {r.tag:<7}  {r.container_label[:40]:<40}  {(r.what or ''):<22}  {text_snip}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--html-file", help="Read HTML from local file instead of Supabase")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Print rows, don't write (default)")
    parser.add_argument("--commit", action="store_true", help="Insert rows into lp_blocks")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of table")
    args = parser.parse_args()

    if args.html_file:
        with open(args.html_file) as f:
            html = f.read()
    else:
        html = fetch_html_via_supabase(args.slug)
        if html is None:
            print(f"ERROR: could not fetch raw_html for {args.slug}. Set SUPABASE_URL + SUPABASE_KEY or pass --html-file.", file=sys.stderr)
            sys.exit(1)

    rows = extract_rows(args.slug, html)

    if args.json:
        print(json.dumps([r.for_insert() for r in rows], indent=2))
    else:
        print_rows_table(rows)

    if args.commit:
        # Safety: refuse to commit if lp_blocks already has rows for this slug
        # (so re-runs don't double-insert silently)
        import requests
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            print("ERROR: --commit requires SUPABASE_URL + SUPABASE_KEY", file=sys.stderr)
            sys.exit(1)
        r = requests.get(
            f"{url}/rest/v1/lp_blocks",
            params={"slug": f"eq.{args.slug}", "select": "id", "limit": "1"},
            headers={"apikey": key, "Authorization": f"Bearer {key}"},
            timeout=20,
        )
        r.raise_for_status()
        if r.json():
            print(f"REFUSING TO COMMIT: lp_blocks already has rows for {args.slug}. Delete them first or use a different slug.", file=sys.stderr)
            sys.exit(2)
        commit_rows_via_supabase(rows)
        print(f"Inserted {len(rows)} rows for {args.slug}.")


if __name__ == "__main__":
    main()
