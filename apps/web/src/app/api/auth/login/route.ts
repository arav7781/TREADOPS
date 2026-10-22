import { NextResponse } from "next/server";

import { apiBase, assertSameOrigin, errorResponse, SESSION_COOKIE, sessionCookieOptions } from "@/lib/server/bff";

interface LoginBody {
  email?: unknown;
  password?: unknown;
}

export async function POST(request: Request) {
  const rejected = assertSameOrigin(request);
  if (rejected) return rejected;
  const body = (await request.json().catch(() => null)) as LoginBody | null;
  if (!body || typeof body.email !== "string" || typeof body.password !== "string") {
    return errorResponse(422, "validation_error", "Email and password are required.");
  }
  let upstream: Response;
  try {
    upstream = await fetch(`${apiBase()}/v1/auth/login`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ email: body.email, password: body.password }),
      cache: "no-store",
      signal: AbortSignal.timeout(15_000),
    });
  } catch {
    return errorResponse(502, "api_unavailable", "The TreadOps API is unreachable.", null, true);
  }
  const data = (await upstream.json().catch(() => null)) as
    | { access_token?: string; expires_at?: string; user?: unknown; error?: unknown }
    | null;
  if (!upstream.ok || !data?.access_token) {
    return NextResponse.json(data ?? { error: { code: "login_failed", message: "Sign-in failed." } },
      { status: upstream.ok ? 502 : upstream.status });
  }
  // The token stays in an HttpOnly cookie; the browser only learns who signed in.
  const res = NextResponse.json({ user: data.user, expires_at: data.expires_at });
  res.cookies.set(SESSION_COOKIE, data.access_token, sessionCookieOptions(data.expires_at));
  return res;
}
