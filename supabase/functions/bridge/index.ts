import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { createClient } from "npm:@supabase/supabase-js@2";

const corsHeaders: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

// =============================================================================
// Google service account auth (cached access token)
// =============================================================================

let cachedToken: { token: string; expiresAt: number } | null = null;

function b64urlEncode(bytes: Uint8Array | string): string {
  const b64 = typeof bytes === "string"
    ? btoa(bytes)
    : btoa(String.fromCharCode(...bytes));
  return b64.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

async function getGoogleAccessToken(): Promise<string> {
  if (cachedToken && cachedToken.expiresAt > Date.now() + 60_000) {
    return cachedToken.token;
  }

  const credsJson = Deno.env.get("GOOGLE_NLP_CREDENTIALS");
  if (!credsJson) {
    throw new Error("GOOGLE_NLP_CREDENTIALS env var not set");
  }
  const sa = JSON.parse(credsJson);

  const now = Math.floor(Date.now() / 1000);
  const header = { alg: "RS256", typ: "JWT" };
  const payload = {
    iss: sa.client_email,
    scope: "https://www.googleapis.com/auth/cloud-language",
    aud: "https://oauth2.googleapis.com/token",
    iat: now,
    exp: now + 3600,
  };

  const headerB64 = b64urlEncode(JSON.stringify(header));
  const payloadB64 = b64urlEncode(JSON.stringify(payload));
  const signingInput = `${headerB64}.${payloadB64}`;

  const pemContents = sa.private_key
    .replace(/-----BEGIN PRIVATE KEY-----/, "")
    .replace(/-----END PRIVATE KEY-----/, "")
    .replace(/\s/g, "");
  const binaryDer = Uint8Array.from(atob(pemContents), (c: string) => c.charCodeAt(0));
  const privateKey = await crypto.subtle.importKey(
    "pkcs8",
    binaryDer.buffer,
    { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" },
    false,
    ["sign"],
  );

  const encoder = new TextEncoder();
  const signature = await crypto.subtle.sign(
    "RSASSA-PKCS1-v1_5",
    privateKey,
    encoder.encode(signingInput),
  );
  const sigB64 = b64urlEncode(new Uint8Array(signature));

  const jwt = `${signingInput}.${sigB64}`;

  const tokenRes = await fetch("https://oauth2.googleapis.com/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: `grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer&assertion=${jwt}`,
  });

  if (!tokenRes.ok) {
    throw new Error(`Token exchange failed: ${tokenRes.status} ${await tokenRes.text()}`);
  }

  const tokenData = await tokenRes.json();
  cachedToken = {
    token: tokenData.access_token,
    expiresAt: Date.now() + (tokenData.expires_in - 60) * 1000,
  };

  return cachedToken.token;
}

// Central IT API key (GOOGLE_NLP_API_KEY) wins when set; otherwise fall back to the
// service account in GOOGLE_NLP_CREDENTIALS. Deleting the key secret = rollback.
async function googleAuthHeaders(): Promise<Record<string, string>> {
  const apiKey = Deno.env.get("GOOGLE_NLP_API_KEY");
  if (apiKey) return { "X-Goog-Api-Key": apiKey };
  const token = await getGoogleAccessToken();
  return { "Authorization": `Bearer ${token}` };
}

// =============================================================================
// Cache helpers
// =============================================================================

async function sha256Hex(text: string): Promise<string> {
  const buf = new TextEncoder().encode(text);
  const hash = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(hash))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function getSupabaseClient() {
  const url = Deno.env.get("SUPABASE_URL");
  const key = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
  if (!url || !key) throw new Error("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY");
  return createClient(url, key);
}

async function cacheLookup(
  contentHash: string,
  contentMode: string,
  nlpMethod: string,
): Promise<any | null> {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from("nlp_cache")
    .select("response")
    .eq("content_hash", contentHash)
    .eq("content_mode", contentMode)
    .eq("nlp_method", nlpMethod)
    .limit(1)
    .maybeSingle();
  if (error) {
    console.error("cacheLookup error", error);
    return null;
  }
  return data ? data.response : null;
}

async function cacheStore(
  contentHash: string,
  contentMode: string,
  nlpMethod: string,
  response: any,
  metadata: { slug?: string; crawl_date?: string; source?: string; label?: string },
): Promise<void> {
  const supabase = getSupabaseClient();
  const row: any = {
    content_hash: contentHash,
    content_mode: contentMode,
    nlp_method: nlpMethod,
    response,
  };
  if (metadata.slug) row.slug = metadata.slug;
  if (metadata.crawl_date) row.crawl_date = metadata.crawl_date;
  if (metadata.source) row.source = metadata.source;
  if (metadata.label) row.label = metadata.label;
  const { error } = await supabase
    .from("nlp_cache")
    .upsert(row, { onConflict: "content_hash,content_mode,nlp_method", ignoreDuplicates: true });
  if (error) console.error("cacheStore error", error);
}

// =============================================================================
// NLP API wrappers (cache-wrapped)
// =============================================================================

function wordCount(text: string): number {
  return text.split(/\s+/).filter(Boolean).length;
}

function modeFromContentType(contentType: string): string {
  return contentType === "HTML" ? "html" : "text";
}

async function callClassify(text: string, contentType: string, authHeaders: Record<string, string>) {
  const res = await fetch("https://language.googleapis.com/v1/documents:classifyText", {
    method: "POST",
    headers: {
      ...authHeaders,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      document: { type: contentType, content: text },
      classificationModelOptions: {
        v2Model: { contentCategoriesVersion: "V2" },
      },
    }),
  });
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`classifyText failed: ${res.status} ${errText}`);
  }
  return await res.json();
}

async function callAnalyzeEntities(text: string, contentType: string, authHeaders: Record<string, string>) {
  const res = await fetch("https://language.googleapis.com/v1/documents:analyzeEntities", {
    method: "POST",
    headers: {
      ...authHeaders,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      document: { type: contentType, content: text },
    }),
  });
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`analyzeEntities failed: ${res.status} ${errText}`);
  }
  return await res.json();
}

// Cached classify: returns the post-processed shape, hits cache when available.
async function cachedClassify(
  text: string,
  contentType: string,
  cacheMeta: any,
  forceRefresh: boolean,
): Promise<{ result: any; cached: boolean }> {
  const mode = modeFromContentType(contentType);
  const hash = await sha256Hex(text);

  if (!forceRefresh) {
    const hit = await cacheLookup(hash, mode, "classify");
    if (hit) return { result: hit, cached: true };
  }

  const auth = await googleAuthHeaders();
  const data = await callClassify(text, contentType, auth);
  const wc = wordCount(text);
  const categories = (data.categories || [])
    .map((c: any) => ({ name: c.name, confidence: Math.round(c.confidence * 10000) / 10000 }))
    .sort((a: any, b: any) => b.confidence - a.confidence);

  const result = {
    categories,
    word_count: wc,
    content_type: contentType,
    primary_category: categories[0]?.name || "Unclassified",
    primary_confidence: categories[0]?.confidence || 0,
  };

  await cacheStore(hash, mode, "classify", result, cacheMeta || {});
  return { result, cached: false };
}

// Cached analyze_entities: returns the post-processed shape, hits cache when available.
async function cachedAnalyzeEntities(
  text: string,
  contentType: string,
  cacheMeta: any,
  forceRefresh: boolean,
): Promise<{ result: any; cached: boolean }> {
  const mode = modeFromContentType(contentType);
  const hash = await sha256Hex(text);

  if (!forceRefresh) {
    const hit = await cacheLookup(hash, mode, "analyze");
    if (hit) return { result: hit, cached: true };
  }

  const auth = await googleAuthHeaders();
  const data = await callAnalyzeEntities(text, contentType, auth);
  const entities = (data.entities || []).map((e: any) => ({
    name: e.name,
    type: e.type,
    salience: Math.round(e.salience * 10000) / 10000,
    mentions: (e.mentions || []).length,
  }));

  const result = { entities };
  await cacheStore(hash, mode, "analyze", result, cacheMeta || {});
  return { result, cached: false };
}

async function nlpClassify(params: any) {
  const text = params?.text || "";
  const contentType = (params?.content_type || "PLAIN_TEXT").toUpperCase();
  const cacheMeta = params?.cache || {};
  const forceRefresh = !!params?.force_refresh;

  const wc = wordCount(text);
  if (wc < 20) {
    return { error: "Text too short for classification. Need at least 20 words.", word_count: wc };
  }

  const { result, cached } = await cachedClassify(text, contentType, cacheMeta, forceRefresh);
  return { ...result, cached };
}

async function nlpAnalyze(params: any) {
  const text = params?.text || "";
  const contentType = (params?.content_type || "PLAIN_TEXT").toUpperCase();
  const includeEntities = params?.include_entities !== false;
  const includeCategories = params?.include_categories !== false;
  const cacheMeta = params?.cache || {};
  const forceRefresh = !!params?.force_refresh;

  const wc = wordCount(text);
  const result: any = {
    text_length: text.length,
    word_count: wc,
    content_type: contentType,
  };
  let anyCached = true;
  let anyCalled = false;

  if (includeEntities) {
    const { result: r, cached } = await cachedAnalyzeEntities(text, contentType, cacheMeta, forceRefresh);
    result.entities = r.entities;
    if (!cached) anyCalled = true;
    if (!cached) anyCached = false;
  }

  if (includeCategories) {
    if (wc >= 20) {
      const { result: r, cached } = await cachedClassify(text, contentType, cacheMeta, forceRefresh);
      result.categories = r.categories;
      if (!cached) anyCalled = true;
      if (!cached) anyCached = false;
    } else {
      result.categories = [];
      result.classification_note = "Text too short for classification (need 20+ words)";
    }
  }

  result.cached = anyCached && !anyCalled;
  return result;
}

// =============================================================================
// SQL bridge via exec_sql RPC
// =============================================================================

async function execSQL(query: string): Promise<any[]> {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase.rpc("exec_sql", { query });
  if (error) throw new Error(`exec_sql failed: ${error.message}`);
  return Array.isArray(data) ? data : [];
}

// =============================================================================
// HTTP entry
// =============================================================================

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  if (req.method !== "POST") {
    return new Response(JSON.stringify({ error: "POST only" }), {
      status: 405,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }

  try {
    const body = await req.json();
    const { tool } = body;
    const params = body.params || {};
    const query = body.query || params.query;

    let result: any;

    if (tool === "nlp_classify") {
      result = await nlpClassify(params);
    } else if (tool === "nlp_analyze") {
      result = await nlpAnalyze(params);
    } else if (tool === "sql") {
      if (!query) {
        result = { error: "Missing 'query' field" };
      } else {
        result = await execSQL(query);
      }
    } else {
      result = { error: `Unknown tool: ${tool}` };
    }

    return new Response(JSON.stringify(result), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return new Response(JSON.stringify({ error: msg }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
