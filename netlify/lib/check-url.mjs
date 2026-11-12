// Returns a cleaned https URL the fetcher is willing to open, or null.
// Public https hostnames only: no IP addresses, ports, logins or local names.
export function checkUrl(raw) {
  let u;
  try {
    u = new URL(String(raw ?? "").trim());
  } catch {
    return null;
  }
  if (u.protocol !== "https:" || u.port || u.username || u.password) return null;
  const host = u.hostname.toLowerCase();
  if (!host.includes(".") || host.includes(":") || /^[\d.]+$/.test(host)) return null;
  if (/(^|\.)(localhost|local|internal|lan|home|corp)$/.test(host)) return null;
  u.hash = "";
  return u.href;
}
