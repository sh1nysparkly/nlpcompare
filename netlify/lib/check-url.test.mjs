import test from "node:test";
import assert from "node:assert/strict";
import { checkUrl } from "./check-url.mjs";

test("accepts public https pages", () => {
  assert.equal(checkUrl("https://www.amatravel.ca/cruises"), "https://www.amatravel.ca/cruises");
  assert.equal(checkUrl("  https://ama.ab.ca/  "), "https://ama.ab.ca/");
  assert.equal(checkUrl("https://example.com/a?b=1#frag"), "https://example.com/a?b=1");
});

test("rejects anything that is not a public https hostname", () => {
  for (const bad of [
    "", null, undefined, "not a url", "amatravel.ca/cruises",
    "http://www.amatravel.ca/", "ftp://example.com/", "file:///etc/passwd",
    "https://localhost/", "https://printer.local/", "https://db.internal/",
    "https://127.0.0.1/", "https://169.254.169.254/latest", "https://[::1]/",
    "https://example.com:8443/", "https://user:pw@example.com/", "https://intranet/",
  ]) {
    assert.equal(checkUrl(bad), null, String(bad));
  }
});
