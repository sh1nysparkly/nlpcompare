// Background job: fetch a live page for the Quick Scorer, or (chop: true) chop
// it into sections and blocks for Page Analysis. The caller gets an
// immediate 202 and collects the outcome from /api/fetch-result?id=...
import type { Config } from "@netlify/functions";
import { getStore } from "@netlify/blobs";
import { checkUrl } from "../lib/check-url.mjs";
import { fetchPage } from "../lib/fetch-page.mjs";

export default async (req: Request) => {
  const body = await req.json().catch(() => ({}));
  const id = String(body.id ?? "").replace(/[^a-zA-Z0-9-]/g, "").slice(0, 64);
  if (!id) return;
  const store = getStore({ name: "fetched-pages", consistency: "strong" });
  const url = checkUrl(body.url);
  if (!url) {
    await store.setJSON(id, { status: "error", message: "That isn't a public https address." });
    return;
  }
  await store.setJSON(id, { status: "running" });
  try {
    await store.setJSON(id, { status: "done", ...(await fetchPage(url, { clean: body.clean === true, chop: body.chop === true })) });
  } catch (err) {
    // Caught on purpose: an uncaught error would make Netlify retry the fetch.
    console.error("fetch-page failed", url, err);
    const reason = String((err as Error)?.message || err).split("\n")[0].slice(0, 200);
    await store.setJSON(id, { status: "error", message: "The page couldn't be loaded (" + reason + ")." });
  }
};

export const config: Config = {
  background: true,
  path: "/api/fetch-page",
  method: "POST",
  rateLimit: { windowLimit: 6, windowSize: 60, aggregateBy: ["ip", "domain"] },
};
