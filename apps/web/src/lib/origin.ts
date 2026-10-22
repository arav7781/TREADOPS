/**
 * Same-origin policy for BFF mutations (CSRF defence in depth on top of SameSite cookies).
 * Pure function so it can be unit-tested without the Next.js runtime.
 *
 * - A browser always sends Origin on cross-site POSTs, so a present Origin must match an allowed origin
 *   (APP_ORIGIN, comma-separated, or the request's own host).
 * - No Origin: allowed only when Sec-Fetch-Site is absent (non-browser client) or same-origin/none.
 */
export interface OriginInput {
  origin: string | null;
  host: string | null;
  forwardedHost?: string | null;
  forwardedProto?: string | null;
  appOrigin?: string | null;
  secFetchSite?: string | null;
}

function normalise(o: string): string | null {
  try {
    const u = new URL(o);
    return `${u.protocol}//${u.host}`.toLowerCase();
  } catch {
    return null;
  }
}

export function allowedOrigins(input: OriginInput): Set<string> {
  const out = new Set<string>();
  for (const o of (input.appOrigin ?? "").split(",").map((s) => s.trim()).filter(Boolean)) {
    const n = normalise(o);
    if (n) out.add(n);
  }
  const host = input.forwardedHost ?? input.host;
  if (host) {
    const proto = (input.forwardedProto ?? "http").split(",")[0].trim();
    out.add(`${proto}://${host}`.toLowerCase());
    // Behind a TLS-terminating proxy the browser sees https while the app sees http on the same host.
    if (proto === "http") out.add(`https://${host}`.toLowerCase());
  }
  return out;
}

export function isSameOrigin(input: OriginInput): boolean {
  if (!input.origin) {
    const site = input.secFetchSite?.toLowerCase();
    return !site || site === "same-origin" || site === "none";
  }
  const o = normalise(input.origin);
  return o !== null && allowedOrigins(input).has(o);
}
