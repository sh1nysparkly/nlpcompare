// Opens a live page in headless Chrome, scrolls it the way a person would, and
// returns the built HTML. The real scrolling matters: amatravel.ca only builds
// its first two sections until it sees genuine wheel input (script scrolls and
// synthetic events are ignored), so page.mouse.wheel is used on purpose.
import chromium from "@sparticuz/chromium-min";
import puppeteer from "puppeteer-core";

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

// Runs inside the page. Strips what carries no visible text (scripts, styles,
// icons, comments, inline image data) and leaves everything else as built.
function snapshot() {
  const words = (document.body.innerText || "").trim().split(/\s+/).filter(Boolean).length;
  const doc = document.documentElement.cloneNode(true);
  doc.querySelectorAll("script, style, svg, noscript, template, iframe, link").forEach((el) => el.remove());
  doc.querySelectorAll('[src^="data:"]').forEach((el) => el.removeAttribute("src"));
  const walker = document.createTreeWalker(doc, NodeFilter.SHOW_COMMENT);
  const comments = [];
  while (walker.nextNode()) comments.push(walker.currentNode);
  comments.forEach((c) => c.remove());
  return { title: document.title, words, html: "<!doctype html>\n" + doc.outerHTML };
}

export async function fetchPage(url) {
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
    const snap = await page.evaluate(snapshot);
    return {
      url,
      fetchedAt: new Date().toISOString(),
      httpStatus: response ? response.status() : null,
      reachedBottom,
      ...snap,
    };
  } finally {
    await browser.close();
  }
}
