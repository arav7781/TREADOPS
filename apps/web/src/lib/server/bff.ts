import "server-only";

import { isSameOrigin } from "@/lib/origin";

export const SESSION_COOKIE = "treadops_session";
const FORWARD_REQUEST_HEADERS = ["content-type", "idempotency-key", "x-correlation-id", "x-request-id", "accept"];
const FORWARD_RESPONSE_HEADERS = ["content-type", "x-request-id", "location", "retry-after", "cache-control",
  "idempotent-replay", "x-accel-buffering", "x-overlay", "x-frame-at"];
const HEADERS_TIMEOUT_MS = 15_000;

export function apiBase(): string {
  return (process.env.API_INTERNAL_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
}

export function errorResponse(status: number, code: string, message: string, requestId?: string | null,
  retryable = false): Response {
  return Response.json({ error: { code, message, request_id: requestId ?? null, retryable, details: {} } },
    { status, headers: retryable ? { "Retry-After": "5" } : undefined });
}

/** Returns an error Response when a mutation fails the same-origin check, else null. */
export function assertSameOrigin(request: Request): Response | null {
  const h = request.headers;
  const ok = isSameOrigin({
    origin: h.get("origin"),
    host: h.get("host"),
    forwardedHost: h.get("x-forwarded-host"),
    forwardedProto: h.get("x-forwarded-proto"),
    appOrigin: process.env.APP_ORIGIN ?? null,
    secFetchSite: h.get("sec-fetch-site"),
  });
  return ok ? null : errorResponse(403, "cross_origin_rejected", "Cross-origin request rejected.");
}

export function sessionCookieOptions(expiresAt?: string) {
  return {
    httpOnly: true,
    sameSite: "lax" as const,
    secure: process.env.SESSION_COOKIE_SECURE === "true",
    path: "/",
    ...(expiresAt ? { expires: new Date(expiresAt) } : {}),
  };
}

export function readCookie(request: Request, name: string): string | null {
  const raw = request.headers.get("cookie");
  if (!raw) return null;
  for (const part of raw.split(";")) {
    const [k, ...v] = part.trim().split("=");
    if (k === name) return decodeURIComponent(v.join("="));
  }
  return null;
}

const SESSION_INVALID = new Set(["session_expired", "session_revoked", "invalid_token", "user_inactive", "missing_token"]);

/**
 * Forward a request to the internal API. Multipart bodies are streamed through untouched: the incoming
 * Content-Type (with its boundary) is forwarded, never re-generated. Timeouts and connection failures become
 * typed 504/502 errors in the shared error contract.
 */
export async function forward(request: Request, upstreamPath: string, token: string | null): Promise<Response> {
  const url = `${apiBase()}${upstreamPath}`;
  const headers = new Headers();
  for (const name of FORWARD_REQUEST_HEADERS) {
    const v = request.headers.get(name);
    if (v) headers.set(name, v);
  }
  if (token) headers.set("authorization", `Bearer ${token}`);
  const method = request.method.toUpperCase();
  const hasBody = method !== "GET" && method !== "HEAD";
  let upstream: Response;
  // The timeout covers the wait for response headers only, so long bodies (server-sent events, reports) keep
  // streaming; the upstream request is cancelled as soon as the browser goes away.
  const headersTimeout = new AbortController();
  const timer = setTimeout(() => headersTimeout.abort(new DOMException("upstream headers timeout", "TimeoutError")),
    HEADERS_TIMEOUT_MS);
  try {
    upstream = await fetch(url, {
      method,
      headers,
      body: hasBody ? request.body : undefined,
      // Required by undici for streaming request bodies.
      ...(hasBody ? { duplex: "half" } : {}),
      cache: "no-store",
      redirect: "manual",
      signal: AbortSignal.any([headersTimeout.signal, request.signal]),
    } as RequestInit);
  } catch (err) {
    const name = (err as { name?: string })?.name;
    if (name === "TimeoutError" || name === "AbortError") {
      return errorResponse(504, "upstream_timeout", "The TreadOps API did not answer within 15 seconds.",
        headers.get("x-request-id"), true);
    }
    return errorResponse(502, "api_unavailable", "The TreadOps API is unreachable.", headers.get("x-request-id"), true);
  } finally {
    clearTimeout(timer);
  }
  const out = new Headers();
  for (const name of FORWARD_RESPONSE_HEADERS) {
    const v = upstream.headers.get(name);
    if (v) out.set(name, name === "location" ? v.replace(/^\/v1\//, "/api/v1/") : v);
  }
  if (!out.has("cache-control")) out.set("cache-control", "no-store");
  const response = new Response(upstream.body, { status: upstream.status, headers: out });
  if (upstream.status === 401) {
    // Clear a dead session so the proxy redirects the next navigation to /login.
    const body = await upstream.clone().json().catch(() => null) as { error?: { code?: string } } | null;
    if (body?.error?.code && SESSION_INVALID.has(body.error.code)) {
      response.headers.append("set-cookie", `${SESSION_COOKIE}=; Path=/; Max-Age=0; HttpOnly; SameSite=Lax`);
    }
  }
  return response;
}
