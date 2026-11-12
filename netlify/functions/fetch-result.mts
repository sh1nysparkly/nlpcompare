// Hands a finished fetch back to the Lab, then forgets it. Nothing is kept.
import type { Config } from "@netlify/functions";
import { getStore } from "@netlify/blobs";

export default async (req: Request) => {
  const id = (new URL(req.url).searchParams.get("id") ?? "").replace(/[^a-zA-Z0-9-]/g, "").slice(0, 64);
  const store = getStore({ name: "fetched-pages", consistency: "strong" });
  const entry = id ? await store.get(id, { type: "json" }) : null;
  if (entry && entry.status !== "running") await store.delete(id);
  return Response.json(entry ?? { status: "pending" }, { headers: { "cache-control": "no-store" } });
};

export const config: Config = { path: "/api/fetch-result", method: "GET" };
