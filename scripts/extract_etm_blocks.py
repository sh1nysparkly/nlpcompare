#!/usr/bin/env python3
"""
Direct lp_blocks extraction for /travel-insurance/emergency-travel-medical.

The generic extrapolate_lp_blocks.py script can't handle this page's deeply
nested Angular component markup. This script reads the rendered HTML directly
and produces lp_blocks rows at the right granularity, modeling output on the
existing /travel-insurance parent page's curation.

Usage:
    python extract_etm_blocks.py <input.html> [--json]
"""
import json
import re
import sys
from dataclasses import dataclass, asdict
from typing import Optional

from bs4 import BeautifulSoup, NavigableString, Tag


SLUG = "/travel-insurance/emergency-travel-medical"

ICON_NAMES = {
    "flight_takeoff", "verified_user", "support_agent", "family_restroom",
    "check", "remove", "drafts", "call", "comment",
}


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
    curation_version: str = "v1-extrapolated"

    def as_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip() if s else ""


def strip_sups(s: str) -> str:
    return re.sub(r"\s*\d+\s*$", "", s).strip()


def el_text(el) -> str:
    return clean(el.get_text(separator=" ", strip=True))


def prep_soup(html: str) -> BeautifulSoup:
    m = re.search(r"<emergency-travel-medical[^>]*>", html)
    e = re.search(r"</emergency-travel-medical>", html)
    if m and e:
        html = html[m.start():e.end()]
    html = re.sub(r"<sup[^>]*>.*?</sup>", "", html, flags=re.DOTALL)
    html = f"<html><body>{html}</body></html>"
    return BeautifulSoup(html, "html.parser")


class Emitter:
    def __init__(self):
        self.rows: list[Row] = []
        self.order = 0

    def add(self, container_label, block_id, tag, text, what=None,
            component_type=None, pattern_key=None):
        text = clean(text)
        if not text:
            return
        self.rows.append(Row(
            slug=SLUG,
            row_order=self.order,
            block_id=block_id,
            container_label=container_label,
            tag=tag,
            text=text,
            what=what,
            component_type=component_type,
            pattern_key=pattern_key,
        ))
        self.order += 1


def guess_component(tag: str, text: str) -> Optional[str]:
    t = tag.lower()
    if t in ("h1", "h2", "h3", "h4", "h5", "h6"):
        return "Heading"
    if t in ("p", "li", "span"):
        return "Copy"
    if t == "a":
        return "Button" if len(text) <= 24 else "Copy"
    if t == "button":
        return "Button"
    if t == "div":
        return "Subheading (div)"
    return None


def extract_coverage(section, em: Emitter):
    """c0: Hero section with H1, intro, bullet points, CTA."""
    label = "Travel Medical Insurance Plans – Trusted Health Coverage for Canadians"
    cid = "c0"

    h1 = section.find("h1")
    if h1:
        em.add(label, f"{cid}-b0", "h1", el_text(h1), "H1", "Heading")

    ps = section.find_all("p", recursive=True)
    for p in ps:
        text = el_text(p)
        if text and "travel medical insurance" in text.lower() and "considered" in text.lower():
            em.add(label, f"{cid}-b0", "p", text, "Intro", "Copy")
            break

    for li in section.find_all("li"):
        span = li.find("span")
        text = el_text(span) if span else el_text(li)
        text = strip_sups(text)
        if text:
            em.add(label, f"{cid}-b0", "li", text, "Intro", "Copy")

    for a in section.find_all("a"):
        text = el_text(a)
        if text == "Get a Quote":
            em.add(label, f"{cid}-b0", "a", text, "Intro", "Button")
            break

    for span in section.find_all("span"):
        text = el_text(span)
        if "quote in just" in text.lower():
            em.add(label, f"{cid}-b0", "span", text, "Intro", "Copy")
            break


def extract_other_options(section, em: Emitter):
    """c1: Insurance product carousel cards."""
    label = "Browse other travel insurance options"
    cid = "c1"

    h2 = section.find("h2")
    if h2:
        em.add(label, f"{cid}-b0", "h2", el_text(h2), "Deal Cards", "Heading",
               pattern_key="deal_cards")

    cards = section.find_all("insurance-product-card")
    seen_titles = set()
    card_idx = 0
    for card in cards:
        h6 = card.find("h6")
        if not h6:
            continue
        title = el_text(h6)
        if title in seen_titles:
            continue
        seen_titles.add(title)
        card_idx += 1
        bid = f"{cid}-b{card_idx}"
        em.add(label, bid, "h6", title, "Deal Cards", "Heading",
               pattern_key="deal_cards")
        for div in card.find_all("div"):
            cls = div.get("class", [])
            if isinstance(cls, str):
                cls = cls.split()
            if "card-text" in cls:
                em.add(label, bid, "p", el_text(div), "Deal Cards", "Copy",
                       pattern_key="deal_cards")
                break
        links_added = set()
        for a in card.find_all("a"):
            text = el_text(a)
            if text in ("Get a Quote", "Learn More") and text not in links_added:
                em.add(label, bid, "a", text, "Deal Cards", "Button",
                       pattern_key="deal_cards")
                links_added.add(text)


def extract_value_props(section, em: Emitter):
    """c2: Value proposition items."""
    label = "Value props"
    cid = "c2"

    vp_data = [
        ("Single or Multi-Trip", "Choose based on how much you travel"),
        ("Coverage for Pre-existing Conditions", "If eligibility criteria is met"),
        ("24/7 AMA Assistance", "Available by phone, text or online chat"),
        ("No Maximum Age", "For medical only coverage"),
    ]

    for i, (heading, desc) in enumerate(vp_data):
        bid = f"{cid}-h{i}"
        em.add(label, bid, "div", heading, "Value Prop Bar",
               "Subheading (div)", pattern_key="value_props")
        em.add(label, bid, "p", desc, "Value Prop Bar",
               "Copy", pattern_key="value_props")


def extract_find_coverage(section, em: Emitter):
    """c3: Find coverage section with coverage cards."""
    label = "Find Coverage That Fits Your Needs"
    cid = "c3"

    h4 = section.find("h4")
    if h4:
        em.add(label, f"{cid}-b0", "h4", el_text(h4), "Image + Value Prop",
               "Heading")

    for p in section.find_all("p"):
        text = el_text(p)
        if not text:
            continue
        if "emergency medical coverage" in text.lower() and "low as" in text.lower():
            text_clean = text.replace(" * .", ".").replace("*", "").strip()
            text_clean = re.sub(r"\s+", " ", text_clean)
            em.add(label, f"{cid}-b0", "p", text_clean, "Image + Value Prop", "Copy")
        elif "need assistance" in text.lower():
            em.add(label, f"{cid}-b0", "p", text, "Image + Value Prop", "Copy")

    for a in section.find_all("a"):
        text = el_text(a)
        if "contact an agent" in text.lower():
            em.add(label, f"{cid}-b0", "a", text, "Image + Value Prop", "Button")
            break

    # Phone number
    for span in section.find_all("span"):
        text = el_text(span)
        if "1-866-989-6595" in text:
            em.add(label, f"{cid}-b0", "p",
                   "Or, Call us at 1-866-989-6595", "Image + Value Prop", "Copy")
            break

    cards = section.find_all("insurance-coverage-card")
    seen_titles = set()
    for i, card in enumerate(cards):
        h6 = card.find("h6")
        if not h6:
            continue
        title = el_text(h6)
        if title in seen_titles:
            continue
        seen_titles.add(title)
        bid = f"{cid}-b{i + 1}"
        em.add(label, bid, "h6", title, "Deal Cards", "Heading",
               pattern_key="deal_cards")
        for div in card.find_all("div"):
            text = el_text(div)
            if text and text != title and "get a quote" not in text.lower():
                if len(text) > 20:
                    em.add(label, bid, "p", text, "Deal Cards", "Copy",
                           pattern_key="deal_cards")
                    break
        for a in card.find_all("a"):
            text = el_text(a)
            if text == "Get a Quote":
                em.add(label, bid, "a", text, "Deal Cards", "Button",
                       pattern_key="deal_cards")
                break


def extract_trusted(section, em: Emitter):
    """c4: Trusted section with checklist + headline."""
    label = "Trusted for Over 60 Years in Alberta for Travel Insurance Health Coverage"
    cid = "c4"

    h5 = section.find("h5")
    if h5:
        em.add(label, f"{cid}-b1", "h5", el_text(h5), "Image + Value Prop", "Heading")

    for li in section.find_all("li"):
        text = strip_sups(el_text(li))
        # Skip icon text
        if text.startswith("check "):
            text = text[6:]
        if text in ICON_NAMES or not text:
            continue
        em.add(label, f"{cid}-b1", "li", text, "Image + Value Prop", "Copy")

    h3 = section.find("h3")
    if h3:
        em.add(label, f"{cid}-b0", "h3", el_text(h3), "Image + Value Prop", "Heading")

    for p in section.find_all("p"):
        text = el_text(p)
        if text and "peace of mind" in text.lower():
            em.add(label, f"{cid}-b0", "p", text, "Image + Value Prop", "Copy")
            break


def extract_comparison(section, em: Emitter):
    """c5: Comparison table."""
    label = "Emergency Travel Insurance Coverage Comparison"
    cid = "c5"

    comparison_rows = [
        ("Family Coverage", "Yes", "Yes"),
        ("Maximum Age at Application", "84", "All Ages"),
        ("Pre-existing Conditions", "Yes", "Yes"),
        ("Maximum Trip Days", "To Age 59: 365 Days / Ages 60-84: 30 Days", "183 Days"),
        ("Up to $5 million Emergency Medical Coverage", "Yes", "No"),
        ("Trip Cancellation & Interruption", "Yes", "No"),
        ("Travel Accident", "Up to $150,000", "No"),
        ("Baggage Loss, Damage, and Delay Coverage", "Yes", "No"),
        ("Cancel for Any Reason*", "Yes", "No"),
        ("Multi-Trip Option Available", "Yes", "Yes"),
    ]

    em.add(label, f"{cid}-b0", "h3", label, "Image + Value Prop", "Heading")
    em.add(label, f"{cid}-b0", "p",
           "Emergency Medical + Trip | Emergency Medical Only",
           "Image + Value Prop", "Copy")

    for feature, med_trip, med_only in comparison_rows:
        em.add(label, f"{cid}-b0", "p",
               f"{feature} | {med_trip} | {med_only}",
               "Image + Value Prop", "Copy")


def extract_why_insurance(section, em: Emitter):
    """c6: Why you should have travel insurance."""
    label = "Why You Should Have Medical Travel Insurance Inside and Outside of Canada"
    cid = "c6"

    h3 = section.find("h3")
    if h3:
        em.add(label, f"{cid}-b0", "h3", el_text(h3), "Image + Value Prop", "Heading")

    for p in section.find_all("p"):
        text = el_text(p)
        if text and len(text) > 40:
            em.add(label, f"{cid}-b0", "p", text, "Image + Value Prop", "Copy")
            break

    for a in section.find_all("a"):
        text = el_text(a)
        if text == "Learn More":
            em.add(label, f"{cid}-b0", "a", text, "Image + Value Prop", "Button")
            break


def extract_faq(section, em: Emitter):
    """c7: FAQ accordion."""
    label = "Travel Insurance FAQs"
    cid = "c7"
    pattern = "faqs"

    h2 = section.find("h2")
    if h2:
        em.add(label, f"{cid}-b0", "h2", el_text(h2), "FAQ", "Heading",
               pattern_key=pattern)

    intro_span = section.find("span")
    if intro_span:
        text = el_text(intro_span)
        if "need help" in text.lower():
            em.add(label, f"{cid}-b0", "p",
                   "Need help navigating Travel Insurance? We're here to help. "
                   "Speak with our Travel Insurance experts, explore our resources, "
                   "or call 1-866-989-6595",
                   "FAQ", "Copy", pattern_key=pattern)

    buttons = section.find_all("button")
    faq_idx = 1
    seen_questions = set()
    for btn in buttons:
        text = el_text(btn)
        if not text or len(text) < 10:
            continue
        if "previous" in text.lower() or "next" in text.lower():
            continue
        # Deduplicate: the page has two FAQ columns with repeated questions
        text_key = text.lower().strip().rstrip("?")
        if text_key in seen_questions:
            continue
        seen_questions.add(text_key)

        bid = f"{cid}-b{faq_idx}"
        em.add(label, bid, "button", text, "FAQ", "Button", pattern_key=pattern)

        collapse = btn.find_parent()
        if collapse:
            accordion_item = collapse.find_parent()
            if accordion_item:
                body_div = accordion_item.find(
                    lambda t: t.name == "div" and "accordion-body" in
                    " ".join(t.get("class", []))
                )
                if body_div:
                    for child in body_div.children:
                        if isinstance(child, Tag):
                            if child.name in ("p", "ul"):
                                answer = el_text(child)
                                if answer:
                                    em.add(label, bid, "p", answer, "FAQ",
                                           "Copy", pattern_key=pattern)
        faq_idx += 1


def extract_claims(section, em: Emitter):
    """c8: Claims and emergency support."""
    label = "Claims and Emergency Support"
    cid = "c8"

    h3 = section.find("h3")
    if h3:
        em.add(label, f"{cid}-b0", "h3", el_text(h3), "Image + Value Prop", "Heading")

    for span in section.find_all("span"):
        text = el_text(span)
        if "existing policy" in text.lower():
            em.add(label, f"{cid}-b0", "p", text, "Image + Value Prop", "Copy")
            break

    em.add(label, f"{cid}-b0", "p",
           "Email - For assistance: orionassistance@xodus.ca / For claims: orionclaims@xodus.ca",
           "Image + Value Prop", "Copy")
    em.add(label, f"{cid}-b0", "p",
           "Contact - Toll Free: +1.888.657.7481 / Local: 1.416.966.6206",
           "Image + Value Prop", "Copy")
    em.add(label, f"{cid}-b0", "p",
           "Chat - SMS: 1-450-234-8044 / WhatsApp: 1-888-657-7611 / Webchat: orion.xodus.ca/assistance",
           "Image + Value Prop", "Copy")

    for a in section.find_all("a"):
        text = el_text(a)
        if "submit a claim" in text.lower():
            em.add(label, f"{cid}-b0", "a", "Submit a claim", "Image + Value Prop", "Button")
        elif "contact us" in text.lower():
            em.add(label, f"{cid}-b0", "a", "Contact Us", "Image + Value Prop", "Button")


def extract_articles(section, em: Emitter):
    """c9: Related articles carousel."""
    label = "Related articles"
    cid = "c9"
    pattern = "articles"

    articles = [
        ("Get Pre-Existing Medical Condition Coverage",
         "Our travel medical insurance plans provide coverage for pre-existing medical conditions that are stable for three or six months prior to your trip departure date, depending on your age.",
         "4 min read"),
        ("The Most Frequently Asked Questions About AMA Travel Insurance",
         "We know that travel insurance can feel like a complicated subject. That's why we've collected the most common questions we received during our travel insurance information sessions.",
         "9 min read"),
        ("Package Travel Insurance Plans from AMA Travel",
         "If you're looking for a way to get all the benefits of travel medical insurance, but also get coverage for the money you've invested in your trip, a package travel insurance plan is a great option.",
         "6 min read"),
    ]

    for i, (title, desc, readtime) in enumerate(articles):
        bid = f"{cid}-b{i}"
        em.add(label, bid, "p",
               f"ARTICLE | {title} | {desc} | {readtime}",
               "Deal Cards", "Copy", pattern_key=pattern)


def extract_disclaimer(section, em: Emitter):
    """c10: Legal disclaimer and footnotes."""
    label = "Disclaimer"
    cid = "c10"

    for p in section.find_all("p"):
        text = el_text(p)
        if text and "underwritten" in text.lower():
            em.add(label, f"{cid}-b0", "p", text, "Image + Value Prop", "Copy")
            break

    for li in section.find_all("li"):
        text = el_text(li)
        if text and len(text) > 10:
            text = strip_sups(text)
            em.add(label, f"{cid}-b0", "li", text, "Image + Value Prop", "Copy")


def extract_all(html: str) -> list[Row]:
    soup = prep_soup(html)
    body = soup.body or soup

    sections = body.find_all("page-section")
    section_map = {}
    for s in sections:
        sid = s.get("id", "")
        section_map[sid] = s

    em = Emitter()

    extractors = [
        ("coverage", extract_coverage),
        ("other-options", extract_other_options),
        ("value-prop", extract_value_props),
        ("find-coverage", extract_find_coverage),
        ("trusted", extract_trusted),
        ("comparison", extract_comparison),
        ("why-insurance", extract_why_insurance),
        ("faq", extract_faq),
        ("claims-support", extract_claims),
        ("articles", extract_articles),
        ("disclaimer", extract_disclaimer),
    ]

    for sid, fn in extractors:
        if sid in section_map:
            fn(section_map[sid], em)
        else:
            print(f"WARNING: section '{sid}' not found", file=sys.stderr)

    return em.rows


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.html> [--json]", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        html = f.read()

    rows = extract_all(html)

    if "--json" in sys.argv:
        print(json.dumps([r.as_dict() for r in rows], indent=2))
    else:
        print(f"\n{len(rows)} rows for {SLUG}\n")
        print(f"{'#':>3}  {'block_id':<8}  {'tag':<7}  {'container':<45}  {'what':<22}  text")
        print("-" * 160)
        for r in rows:
            text_snip = r.text[:70].replace("\n", " ")
            print(f"{r.row_order:>3}  {r.block_id:<8}  {r.tag:<7}  {r.container_label[:45]:<45}  {(r.what or ''):<22}  {text_snip}")
        print()


if __name__ == "__main__":
    main()
