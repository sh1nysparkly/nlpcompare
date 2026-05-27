#!/usr/bin/env python3
"""
Clean the Emergency Travel Medical page's rendered Angular HTML into a
structure the generic extrapolate_lp_blocks.py script can handle.

The raw rendered HTML has deeply nested Angular component wrappers, Material
Icons spans, duplicate mobile/desktop viewports, and malformed tag nesting.
This script strips all of that down to the content elements the NLP pipeline
actually needs: headings, paragraphs, list items, buttons, links.

Usage:
    python clean_etm_html.py <input.html> <output.html>
"""
import re
import sys
from bs4 import BeautifulSoup, NavigableString, Tag


ICON_NAMES = {
    "flight_takeoff", "verified_user", "support_agent", "family_restroom",
    "check", "remove", "drafts", "call", "comment",
}

KNOWN_PHONE_NUMBERS = {
    "1-866-989-6595", "+1.888.657.7481", "1.416.966.6206",
    "1-450-234-8044", "1-888-657-7611", "1-844-771-1522",
}


def clean_etm(html: str) -> str:
    # Extract just the page content between the emergency-travel-medical tags
    m_start = re.search(r'<emergency-travel-medical[^>]*>', html)
    m_end = re.search(r'</emergency-travel-medical>', html)
    if m_start and m_end:
        html = html[m_start.start():m_end.end()]

    # Wrap in minimal HTML shell so parsers are happy
    html = f"<html><body>{html}</body></html>"

    # Remove Material Icons spans at string level (before parse, to avoid
    # lxml nesting artifacts)
    html = re.sub(
        r'<span[^>]*class="material-symbols-outlined"[^>]*>[^<]*</span>',
        '', html
    )

    # Remove all <img> tags
    html = re.sub(r'<img\b[^>]*/?\s*>', '', html)

    # Remove carousel nav buttons (Previous/Next)
    html = re.sub(
        r'<button[^>]*aria-label="(?:Previous|Next)"[^>]*>.*?</button>',
        '', html, flags=re.DOTALL
    )

    # Remove <sup> footnote references (e.g., <sup><a href="#footnote-2">2</a></sup>)
    html = re.sub(r'<sup[^>]*>.*?</sup>', '', html, flags=re.DOTALL)

    # Parse with html.parser (less aggressive error correction than lxml)
    soup = BeautifulSoup(html, 'html.parser')
    body = soup.body or soup

    # Remove duplicate mobile-only content (the page has both d-none d-md-grid
    # and d-md-none variants for coverage cards and carousels)
    for el in body.find_all(True):
        if el.attrs is None:
            continue
        classes = el.get('class', [])
        if isinstance(classes, str):
            classes = classes.split()
        if 'd-md-none' in classes:
            el.decompose()

    # Remove phone <a> tags that are mobile-only (d-md-none), keep desktop span
    for el in body.find_all('a'):
        if el.attrs is None:
            continue
        classes = el.get('class', [])
        if isinstance(classes, str):
            classes = classes.split()
        if 'd-md-none' in classes:
            el.decompose()

    # Remove desktop-only phone spans (d-none d-md-inline) -- we'll keep the
    # phone number text but not as a separate element
    for el in body.find_all('span'):
        classes = el.get('class', [])
        if isinstance(classes, str):
            classes = classes.split()
        if 'd-none' in classes and 'd-md-inline' in classes:
            el.replace_with(el.get_text())

    # Strip ALL attributes except href on <a> tags
    for el in body.find_all(True):
        if el.name == 'a':
            href = el.get('href')
            el.attrs = {'href': href} if href else {}
        elif el.name == 'page-section':
            eid = el.get('id')
            el.attrs = {'id': eid} if eid else {}
        else:
            el.attrs = {}

    # Remove Angular component tags that are just structural wrappers,
    # but keep their children. We keep page-section (containers).
    angular_wrappers = [
        'emergency-travel-medical', 'generic-section-block',
        'insurance-product-carousel', 'insurance-product-card',
        'insurance-coverage-group', 'insurance-coverage-card',
        'comparison-table', 'product-banner-cta',
        'articles-carousel', 'card-list',
        'carousel', 'info-card', 'value-prop-widget',
        'faqs-widget', 'phone', 'ama-email',
        'app-root', 'app-toasts', 'app-content',
    ]
    for tag_name in angular_wrappers:
        for el in body.find_all(tag_name):
            el.unwrap()

    # Remove purely structural divs: those with layout classes but no direct
    # text content (they just wrap other elements). We identify these as divs
    # whose only content is other block-level elements.
    structural_div_classes = {
        'container', 'section-content', 'section-wrapper', 'row',
        'd-flex', 'col-12', 'col-md', 'col-md-6', 'col-md-8', 'col-xl-6',
        'col-xl-8', 'col-md-7', 'col-xl-4', 'col-sm-6',
        'banner-image', 'widget', 'panel-wrapper',
        'page-container', 'scroll-viewport', 'carousel-container',
        'scroll-container', 'table-wrapper', 'table-grid',
        'card-banner', 'image-container',
        'navigate-btn-container', 'carousel-bottom-nav',
        'mt-auto', 'card-body', 'flex-grow-1', 'position-relative',
        'recommended-badge',
    }

    def is_structural_div(el):
        if el.name != 'div':
            return False
        classes = el.get('class', [])
        if isinstance(classes, str):
            classes = classes.split()
        return bool(set(classes) & structural_div_classes)

    # Multiple passes since unwrapping can reveal more structural divs
    for _ in range(5):
        found = False
        for el in body.find_all('div'):
            if is_structural_div(el):
                el.unwrap()
                found = True
        if not found:
            break

    # Now strip all remaining attributes
    for el in body.find_all(True):
        if el.name == 'a':
            href = el.get('href')
            el.attrs = {'href': href} if href else {}
        elif el.name == 'page-section':
            eid = el.get('id')
            el.attrs = {'id': eid} if eid else {}
        else:
            el.attrs = {}

    # Remove empty elements
    for _ in range(3):
        for el in body.find_all(True):
            if el.name in ('page-section',):
                continue
            text = el.get_text(strip=True)
            if not text and not el.find_all(True):
                el.decompose()

    # Remove leftover icon name text nodes
    for el in body.find_all(text=True):
        if isinstance(el, NavigableString):
            text = str(el).strip()
            if text in ICON_NAMES:
                el.replace_with('')

    # Clean up whitespace
    result = str(body)
    result = re.sub(r'\n\s*\n+', '\n', result)
    result = re.sub(r'>\s+<', '>\n<', result)

    return result


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.html> <output.html>")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        html = f.read()
    cleaned = clean_etm(html)
    with open(sys.argv[2], 'w') as f:
        f.write(cleaned)
    print(f"Cleaned HTML written to {sys.argv[2]}")
