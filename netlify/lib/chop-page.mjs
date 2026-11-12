// Chops a live, fully scrolled page into Page Analysis rows: one row per run of
// text, grouped into blocks, grouped into sections. Rows use the same shape as
// the hand curation in lp_blocks (block_id, container_label, tag, text,
// component_type) so the Lab turns them into blocks with the same code.
//
// Both functions run inside the page via page.evaluate, so each must stay
// fully self-contained: no imports, no references to anything outside it.
//
// markHiddenPanels must run BEFORE the fetcher opens collapsed accordions; it
// tags what was hidden at load so the rows can say "hidden on page" later.
export function markHiddenPanels() {
  for (const el of document.querySelectorAll("body *")) {
    if (el.hidden || el.getAttribute("aria-hidden") === "true" || getComputedStyle(el).display === "none") {
      el.setAttribute("data-sc-hidden", "1");
    }
  }
}

export function chopPage() {
  const HEADINGS = new Set(["h1", "h2", "h3", "h4", "h5", "h6"]);
  const ROW_TAGS = new Set(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "a", "button", "div"]);
  // Tags that never make a card on their own: lists of these are text, not cards.
  const TEXTY = new Set(["p", "li", "span", "button", "label", "dt", "dd", "td", "th", "strong", "em", "b", "i", "small",
    "h1", "h2", "h3", "h4", "h5", "h6", "option", "img", "picture"]);
  const SKIP = new Set(["script", "style", "noscript", "template", "svg", "iframe", "select", "option", "input", "textarea"]);
  const IGNORED_CLASS = /^(ng-|cdk-|mat-|active|selected|show|open|collapsed|is-|swiper-slide-(active|next|prev|visible))/;

  const norm = (s) => String(s || "").replace(/\s+/g, " ").trim();
  const words = (s) => norm(s).split(" ").filter(Boolean).length;

  function isIcon(el) {
    const tag = el.tagName.toLowerCase();
    if (tag === "mat-icon" || /(^|-)icon$/.test(tag)) return true;
    const cls = typeof el.className === "string" ? el.className : "";
    if (/\bmaterial-(icons|symbols)/i.test(cls)) return true;
    const font = getComputedStyle(el).fontFamily || "";
    return /material (icons|symbols)|tst-icons|icomoon|fontawesome|font awesome|glyphicons/i.test(font);
  }

  // The element that "owns" a text node: climb out of inline wrappers until a
  // block-ish element. Buttons and links stay owners so their text is a row.
  function ownerOf(el, stop) {
    let cur = el;
    while (cur && cur !== stop) {
      const tag = cur.tagName.toLowerCase();
      if (HEADINGS.has(tag) || tag === "p" || tag === "li" || tag === "button" || tag === "blockquote") return cur;
      const disp = getComputedStyle(cur).display;
      if (!disp.startsWith("inline") || disp === "inline-block" || disp === "inline-flex" || disp === "inline-grid") return cur;
      cur = cur.parentElement;
    }
    return stop;
  }

  function signature(el) {
    const tag = el.tagName.toLowerCase();
    if (tag.includes("-")) return tag;
    const cls = typeof el.className === "string" ? el.className.split(/\s+/).filter((c) => c && !IGNORED_CLASS.test(c)).sort().join(".") : "";
    return tag + "." + cls;
  }

  // Sections: top-level page-section elements (amatravel.ca's CMS blocks).
  // Other builds fall back to top-level <section> tags, then main's children.
  let sections = [...document.querySelectorAll("page-section")].filter((s) => !s.parentElement.closest("page-section"));
  let seam = "page-section";
  if (!sections.length) {
    sections = [...document.querySelectorAll("section")].filter((s) => !s.parentElement.closest("section") && !s.closest("header, footer, nav"));
    seam = "section";
  }
  if (!sections.length) {
    const main = document.querySelector("main") || document.body;
    sections = [...main.children].filter((c) => !c.matches("header, footer, nav"));
    seam = "main-children";
  }

  const rows = [];
  let rowOrder = 0;

  sections.forEach((section, cIdx) => {
    // 1. Collect text runs, grouped by owner element, in document order.
    const owners = new Map();
    const walker = document.createTreeWalker(section, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const node = walker.currentNode;
      if (!norm(node.nodeValue)) continue;
      const parent = node.parentElement;
      if (!parent) continue;
      let skip = false;
      for (let a = parent; a && a !== section.parentElement; a = a.parentElement) {
        if (SKIP.has(a.tagName.toLowerCase()) || isIcon(a)) { skip = true; break; }
      }
      if (skip) continue;
      // Not rendered at all (display:none chain, e.g. mobile-only duplicates).
      if (parent.checkVisibility && !parent.checkVisibility()) continue;
      const owner = ownerOf(parent, section);
      if (!owners.has(owner)) owners.set(owner, []);
      owners.get(owner).push(node.nodeValue);
    }

    // 2. Find card roots: the outermost repeated sibling (same tag/class) that
    // isn't a texty tag, sits outside any tab strip, and holds text.
    const cardRootOf = (el) => {
      let found = null;
      for (let cur = el; cur && cur !== section; cur = cur.parentElement) {
        const tag = cur.tagName.toLowerCase();
        if (TEXTY.has(tag)) continue;
        if (cur.closest('[role="tablist"]')) continue;
        if (/tab|chip/i.test(tag)) continue;
        // Layout grid columns (Bootstrap col-*, row) are not cards.
        const cls = typeof cur.className === "string" ? cur.className : "";
        if (/(^|\s)(col(-[\w]+)*|row|container(-fluid)?)(\s|$)/.test(cls)) continue;
        const par = cur.parentElement;
        if (!par) continue;
        const sig = signature(cur);
        const twins = [...par.children].filter((c) => signature(c) === sig && norm(c.innerText || c.textContent));
        if (twins.length >= 2) found = cur;
      }
      return found;
    };

    // 3. Turn owners into rows, assigning blocks.
    const sectionRows = [];
    const cardIds = new Map();
    let flowBlock = null;
    let flowCount = 0;
    for (const [owner, texts] of owners) {
      const text = norm(texts.join(" "));
      if (!text) continue;
      let tag = owner.tagName.toLowerCase();
      const role = owner.getAttribute("role");
      if (role === "heading" && owner.getAttribute("aria-level")) tag = "h" + owner.getAttribute("aria-level");
      // A div whose only text is one link reads as that link (e.g. "Advanced Search").
      if (!ROW_TAGS.has(tag) || tag === "div") {
        const links = owner.querySelectorAll("a");
        tag = links.length === 1 && norm(links[0].innerText || links[0].textContent) === text ? "a" : "div";
      }

      const rect = owner.getBoundingClientRect();
      const offscreen = rect.right <= 0 || rect.left >= innerWidth;
      const invisible = owner.checkVisibility && !owner.checkVisibility({ visibilityProperty: true, opacityProperty: true });
      const hidden = !!owner.closest("[data-sc-hidden]") || offscreen || invisible;

      // The TST booking widget is one locked block, as in the hand curation.
      const widget = owner.closest("tst-widget");
      const card = widget ? null : cardRootOf(owner);
      let blockId;
      if (widget) {
        blockId = "c" + cIdx + "-w";
        flowBlock = null;
      } else if (card) {
        if (!cardIds.has(card)) cardIds.set(card, "c" + cIdx + "-k" + cardIds.size);
        blockId = cardIds.get(card);
        flowBlock = null; // text after a card list starts a fresh block
      } else {
        if (!flowBlock || HEADINGS.has(tag)) {
          flowBlock = "c" + cIdx + "-f" + flowCount++;
        }
        blockId = flowBlock;
      }
      const comp = (card ? card.tagName.toLowerCase() : null)
        || [...(function* () { for (let a = owner; a && a !== section; a = a.parentElement) yield a; })()]
          .map((a) => a.tagName.toLowerCase()).find((t) => t.includes("-")) || null;
      sectionRows.push({
        block_id: blockId, tag, text, hidden,
        component_type: widget ? "tst-widget" : comp,
        what: widget ? "Widget" : null,
      });
    }
    if (!sectionRows.length) return;

    // 4. Section label: first heading, else the first component tag.
    const firstHeading = sectionRows.find((r) => HEADINGS.has(r.tag));
    const firstComp = [...section.querySelectorAll("*")].map((e) => e.tagName.toLowerCase()).find((t) => t.includes("-") && t !== "page-section");
    const label = firstHeading ? firstHeading.text : (firstComp || "Section " + (cIdx + 1));

    for (const r of sectionRows) {
      rows.push({ row_order: rowOrder++, container_label: label, ...r });
    }
  });

  return { seam, sections: sections.length, rows };
}
