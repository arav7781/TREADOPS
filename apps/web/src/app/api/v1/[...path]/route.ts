import { assertSameOrigin, errorResponse, forward, readCookie, SESSION_COOKIE } from "@/lib/server/bff";

/**
 * Same-origin BFF for the /v1 API. The browser never learns API_INTERNAL_URL or the bearer token.
 * Media links (/v1/media/<signed token>) are capability URLs and are forwarded without a session.
 */
function upstreamPath(request: Request, path: string[]): string | null {
  if (path.some((p) => p === ".." || p === "." || p.includes("\\"))) return null;
  const search = new URL(request.url).search;
  return `/v1/${path.map(encodeURIComponent).join("/")}${search}`;
}

async function handle(request: Request, ctx: RouteContext<"/api/v1/[...path]">): Promise<Response> {
  const { path } = await ctx.params;
  const target = upstreamPath(request, path);
  if (!target) return errorResponse(400, "invalid_path", "Invalid API path.");
  if (request.method !== "GET" && request.method !== "HEAD") {
    const rejected = assertSameOrigin(request);
    if (rejected) return rejected;
  }
  const token = readCookie(request, SESSION_COOKIE);
  const isMedia = path[0] === "media";
  if (!token && !isMedia) {
    return errorResponse(401, "missing_token", "Authentication required.");
  }
  return forward(request, target, isMedia ? null : token);
}

export const GET = handle;
export const POST = handle;
export const PATCH = handle;
export const DELETE = handle;
