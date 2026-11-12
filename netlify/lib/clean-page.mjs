// The cleaner from Anna's "Clean HTML for NLP" Chrome extension (popup.js, v1.0),
// copied unchanged so a cleaned fetch matches what the extension produces.
// It runs inside the page via page.evaluate, so it must stay fully
// self-contained: no imports, no references to anything outside the function.
export function cleanPage() {
  const INLINE = new Set([
    "a", "sup", "sub", "strong", "em", "b", "i", "small", "abbr",
    "cite", "code", "time", "mark", "q", "s", "u", "br", "wbr"
  ]);
  const DROP = new Set([
    "script", "style", "link", "noscript", "template", "meta", "title",
    "svg", "iframe", "img", "picture", "source", "video", "audio",
    "canvas", "object", "embed", "map", "area", "base", "slot", "track"
  ]);
  const VOID = new Set(["input", "br", "hr"]);
  // Containers that get pruned if they end up with no content
  const PRUNE_IF_EMPTY = new Set([
    "div", "section", "article", "header", "main", "aside", "p", "ul",
    "ol", "li", "form", "table", "thead", "tbody", "tfoot", "tr",
    "blockquote", "figure", "figcaption", "dl", "nav", "label",
    "fieldset", "details", "span", "footer"
  ]);

  const esc = (s) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const escAttr = (s) => esc(s).replace(/"/g, "&quot;");

  function isHidden(el) {
    if (el.hidden) return true;
    if (el.getAttribute("aria-hidden") === "true") return true;
    const cs = window.getComputedStyle(el);
    return cs.display === "none" || cs.visibility === "hidden";
  }

  // Icon elements render ligature text ("keyboard_arrow_down", "search")
  // that pollutes NLP input. Detect by tag name, class, or icon font.
  function isIcon(el) {
    const tag = el.tagName.toLowerCase();
    if (/(^|-)icon$/.test(tag)) return true;
    const cls = typeof el.className === "string" ? el.className : "";
    if (/\bmaterial-(icons|symbols)/i.test(cls)) return true;
    const font = window.getComputedStyle(el).fontFamily || "";
    return /material (icons|symbols)|tst-icons|icomoon|fontawesome|font awesome|glyphicons/i.test(font);
  }

  function isBreadcrumb(el) {
    const probe = [
      el.getAttribute("aria-label") || "",
      el.id || "",
      typeof el.className === "string" ? el.className : ""
    ].join(" ");
    return /breadcrumb/i.test(probe);
  }

  // Whitelisted attributes per tag. Real hrefs are kept (javascript: junk
  // is neutralized to "#").
  function attrsFor(el, tag) {
    const out = [];
    if (tag === "a") {
      const href = el.getAttribute("href") || "#";
      out.push(["href", /^\s*javascript:/i.test(href) ? "#" : href]);
    }
    if (tag === "nav" && isBreadcrumb(el)) out.push(["aria-label", "Breadcrumb"]);
    if (tag === "input") {
      const type = el.getAttribute("type") || "text";
      out.push(["type", type]);
      if (el.value) out.push(["value", el.value]);
      else if (el.getAttribute("placeholder"))
        out.push(["placeholder", el.getAttribute("placeholder")]);
    }
    if (tag === "textarea" && el.getAttribute("placeholder"))
      out.push(["placeholder", el.getAttribute("placeholder")]);
    if (tag === "button")
      out.push(["type", el.getAttribute("type") || "submit"]);
    if (tag === "option" && el.selected) out.push(["selected", null]);
    if (tag === "th" && el.getAttribute("scope"))
      out.push(["scope", el.getAttribute("scope")]);
    return out
      .map(([k, v]) => (v === null ? ` ${k}` : ` ${k}="${escAttr(v)}"`))
      .join("");
  }

  // Serialize an element's children into "runs": inline content is merged
  // into text lines; block children become their own indented lines.
  function renderChildren(el, depth) {
    if (isIcon(el)) return "";
    const lines = [];
    let inlineBuf = "";

    const flush = () => {
      const t = inlineBuf.replace(/\s+/g, " ").trim();
      if (t) lines.push("  ".repeat(depth) + t);
      inlineBuf = "";
    };

    for (const node of el.childNodes) {
      if (node.nodeType === Node.TEXT_NODE) {
        inlineBuf += esc(node.nodeValue);
      } else if (node.nodeType === Node.ELEMENT_NODE) {
        const tag = node.tagName.toLowerCase();
        if (DROP.has(tag)) continue;
        if (isHidden(node)) continue;
        if (isIcon(node)) continue;
        if (tag === "footer") continue;
        if (tag === "nav" && !isBreadcrumb(node)) continue;
        if (["dialog", "[object]"].includes(tag)) continue;
        if (
          node.getAttribute &&
          ["banner", "cookie", "consent"].some((w) =>
            ((node.id || "") + " " + (node.getAttribute("class") || ""))
              .toLowerCase()
              .includes(w)
          ) &&
          tag !== "header"
        )
          continue;

        if (tag === "span") {
          // unwrap spans: inline-serialize their contents
          inlineBuf += renderInline(node);
          continue;
        }

        if (INLINE.has(tag)) {
          inlineBuf += renderInlineEl(node, tag);
        } else {
          flush();
          const block = renderBlock(node, tag, depth);
          if (block) lines.push(block);
        }
      }
    }
    flush();
    return lines.join("\n");
  }

  function renderInline(el) {
    let s = "";
    for (const node of el.childNodes) {
      if (node.nodeType === Node.TEXT_NODE) s += esc(node.nodeValue);
      else if (node.nodeType === Node.ELEMENT_NODE) {
        const tag = node.tagName.toLowerCase();
        if (DROP.has(tag) || isHidden(node) || isIcon(node)) continue;
        if (tag === "span") s += renderInline(node);
        else if (INLINE.has(tag)) s += renderInlineEl(node, tag);
        else s += " " + renderInline(node) + " "; // block inside inline: flatten
      }
    }
    return s;
  }

  function renderInlineEl(el, tag) {
    if (tag === "br") return "<br/>";
    if (tag === "wbr") return "";
    const inner = renderInline(el).replace(/\s+/g, " ").trim();
    if (!inner && tag !== "br") return "";
    return `<${tag}${attrsFor(el, tag)}>${inner}</${tag}>`;
  }

  function renderBlock(el, tag, depth) {
    const pad = "  ".repeat(depth);
    if (VOID.has(tag)) return `${pad}<${tag}${attrsFor(el, tag)} />`;
    if (tag === "textarea")
      return `${pad}<textarea${attrsFor(el, tag)}>${esc(el.value || "")}</textarea>`;

    const inner = renderChildren(el, depth + 1);
    const hasText = inner.replace(/<[^>]+>/g, "").trim().length > 0;
    const hasControls = /<(input|button|select|textarea)\b/.test(inner);
    if (!hasText && !hasControls && PRUNE_IF_EMPTY.has(tag)) return "";

    // Single short line of content -> render on one line
    if (!inner.includes("\n") && inner.trim().length <= 100) {
      return `${pad}<${tag}${attrsFor(el, tag)}>${inner.trim()}</${tag}>`;
    }
    return `${pad}<${tag}${attrsFor(el, tag)}>\n${inner}\n${pad}</${tag}>`;
  }

  const title = document.title || "";
  const descEl = document.querySelector('meta[name="description"]');
  const desc = descEl ? descEl.getAttribute("content") || "" : "";

  const bodyHtml = renderChildren(document.body, 0);

  const head = [
    "<!DOCTYPE html>",
    "<html>",
    "<head>",
    `  <title>${esc(title)}</title>`,
    desc ? `  <meta name="description" content="${escAttr(desc)}"/>` : null,
    "</head>",
    "<body>"
  ]
    .filter(Boolean)
    .join("\n");

  const html = head + "\n" + bodyHtml + "\n</body>\n</html>\n";

  return {
    html,
    cleanedChars: html.length,
    rawChars: document.documentElement.outerHTML.length
  };
}
