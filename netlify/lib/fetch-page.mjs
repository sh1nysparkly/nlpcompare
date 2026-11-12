// Opens a live page in headless Chrome, scrolls it the way a person would, and
// returns the built HTML. The real scrolling matters: amatravel.ca only builds
// its first two sections until it sees genuine wheel input (script scrolls and
// synthetic events are ignored), so page.mouse.wheel is used on purpose.
import chromium from "@sparticuz/chromium-min";
import puppeteer from "puppeteer-core";
import { cleanPage } from "./clean-page.mjs";

// Must match the @sparticuz/chromium-min version in package.json.
const CHROMIUM_PACK =
  "https://github.com/Sparticuz/chromium/releases/download/v153.0.0/chromium-v153.0.0-pack.x64.tar";
const USER_AGENT =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36";
const SETTLE_MAX_MS = 12000;
const SCROLL_MAX_MS = 90000;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Waits until the page's visible text stops growing, or maxMs passes. Pages that
// never stop changing (rotating banners, live counters) just hit the cap.
async function settle(page, maxMs) {
  const start = Date.now();
  let last = -1;
  let stable = 0;
  while (stable < 3 && Date.now() - start < maxMs) {
    await sleep(500);
    const len = await page.evaluate(() => (document.body.innerText || "").length);
    if (len === last) stable++;
    else { stable = 0; last = len; }
  }
}

async function scrollToBottom(page) {
  const start = Date.now();
  await page.mouse.move(640, 450);
  let atBottom = 0;
  while (atBottom < 3 && Date.now() - start < SCROLL_MAX_MS) {
    await page.mouse.wheel({ deltaY: 700 });
    await sleep(350);
    const done = await page.evaluate(
      () => Math.ceil(scrollY + innerHeight) >= document.documentElement.scrollHeight - 4,
    );
    if (done) { atBottom++; await sleep(1200); }
    else atBottom = 0;
  }
  return atBottom >= 3;
}

// The three helpers below run inside the page (page.evaluate), so each must be
// self-contained.

function visibleWordCount() {
  return (document.body.innerText || "").trim().split(/\s+/).filter(Boolean).length;
}

// The default, uncleaned snapshot: strips what carries no visible text (scripts,
// styles, icons, comments, inline image data) and leaves everything else as built.
function strippedHtml() {
  const doc = document.documentElement.cloneNode(true);
  doc.querySelectorAll("script, style, svg, noscript, template, iframe, link").forEach((el) => el.remove());
  doc.querySelectorAll('[src^="data:"]').forEach((el) => el.removeAttribute("src"));
  const walker = document.createTreeWalker(doc, NodeFilter.SHOW_COMMENT);
  const comments = [];
  while (walker.nextNode()) comments.push(walker.currentNode);
  comments.forEach((c) => c.remove());
  return "<!doctype html>\n" + doc.outerHTML;
}

// Opens collapsed accordion panels (FAQ answers) so the cleaner, which skips
// hidden elements, keeps them. Anna's curation counts FAQ answers. Limited to
// accordion-style panels and <details>, so menus and dropdowns stay closed.
function openCollapsedPanels() {
  document.querySelectorAll("details:not([open])").forEach((el) => { el.open = true; });
  document.querySelectorAll('[aria-expanded="false"][aria-controls]').forEach((toggle) => {
    const panel = document.getElementById(toggle.getAttribute("aria-controls"));
    if (!panel) return;
    const cls = typeof panel.className === "string" ? panel.className : "";
    if (panel.getAttribute("role") !== "region" && !/accordion|collapse|faq/i.test(cls)) return;
    panel.hidden = false;
    panel.removeAttribute("aria-hidden");
    panel.classList.add("show");
    panel.style.display = "block";
  });
}

// clean: true returns the output of the "Clean HTML for NLP" cleaner (with FAQ
// answers opened first) instead of the default stripped snapshot.
export async function fetchPage(url, { clean = false } = {}) {
  const browser = await puppeteer.launch({
    args: await puppeteer.defaultArgs({ args: chromium.args, headless: "shell" }),
    defaultViewport: { width: 1280, height: 900, deviceScaleFactor: 1 },
    executablePath: await chromium.executablePath(CHROMIUM_PACK),
    headless: "shell",
  });
  try {
    const page = await browser.newPage();
    await page.setUserAgent(USER_AGENT);
    // Generous on purpose: ama.ab.ca needs ~23s to reach DOMContentLoaded in this
    // browser (amatravel.ca needs ~2s), so a 30s limit failed about half the time.
    const response = await page.goto(url, { waitUntil: "domcontentloaded", timeout: 90000 });
    await settle(page, SETTLE_MAX_MS);
    const reachedBottom = await scrollToBottom(page);
    await settle(page, 3000);
    let html, words;
    if (clean) {
      await page.evaluate(openCollapsedPanels);
      html = (await page.evaluate(cleanPage)).html;
      const text = html.replace(/<head>[\s\S]*?<\/head>/, " ").replace(/<[^>]+>/g, " ");
      words = text.trim().split(/\s+/).filter(Boolean).length;
    } else {
      html = await page.evaluate(strippedHtml);
      words = await page.evaluate(visibleWordCount);
    }
    return {
      url,
      fetchedAt: new Date().toISOString(),
      httpStatus: response ? response.status() : null,
      reachedBottom,
      clean,
      words,
      html,
    };
  } finally {
    await browser.close();
  }
}
