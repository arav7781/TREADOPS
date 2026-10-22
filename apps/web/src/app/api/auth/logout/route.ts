import { NextResponse } from "next/server";

import { apiBase, assertSameOrigin, readCookie, SESSION_COOKIE } from "@/lib/server/bff";

export async function POST(request: Request) {
  const rejected = assertSameOrigin(request);
  if (rejected) return rejected;
  const token = readCookie(request, SESSION_COOKIE);
  if (token) {
    // Revoke server-side; a failure here must not keep the browser signed in.
    await fetch(`${apiBase()}/v1/auth/logout`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}` },
      cache: "no-store",
      signal: AbortSignal.timeout(5_000),
    }).catch(() => null);
  }
  const res = NextResponse.json({ status: "signed_out" });
  res.cookies.set(SESSION_COOKIE, "", { httpOnly: true, sameSite: "lax", path: "/", maxAge: 0 });
  return res;
}
