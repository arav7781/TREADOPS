import { describe, expect, it, vi } from "vitest";

import { api, ApiError, mediaUrl } from "@/lib/api";

describe("mediaUrl", () => {
  it("routes API media paths through the same-origin BFF", () => {
    expect(mediaUrl("/v1/media/abc.def")).toBe("/api/v1/media/abc.def");
  });
  it("keeps BFF paths and drops anything else", () => {
    expect(mediaUrl("/api/v1/media/x")).toBe("/api/v1/media/x");
    expect(mediaUrl("https://storage.example/bucket/key")).toBeNull();
    expect(mediaUrl(null)).toBeNull();
  });
});

describe("api()", () => {
  it("parses the shared error contract into ApiError", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({
      error: { code: "stale_revision", message: "Another reviewer changed this label", request_id: "req-1", retryable: false, details: { current_revision: 2 } },
    }), { status: 409 }));
    const err = (await api("/v1/inspections/x/reviews", { method: "POST", json: {} }).catch((e: unknown) => e)) as ApiError;
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(409);
    expect(err.code).toBe("stale_revision");
    expect(err.requestId).toBe("req-1");
    expect(err.details.current_revision).toBe(2);
  });

  it("sends JSON, idempotency and correlation headers to the /api prefix", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("{}", { status: 200 }));
    await api("/v1/replays", { method: "POST", json: { scenario: "baseline" }, idempotencyKey: "k-1" });
    const [url, init] = spy.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/v1/replays");
    const h = new Headers(init.headers);
    expect(h.get("content-type")).toBe("application/json");
    expect(h.get("Idempotency-Key")).toBe("k-1");
    expect(h.get("X-Correlation-ID")).toBeTruthy();
  });

  it("maps network failures to a retryable network_error", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("fetch failed"));
    const err = (await api("/v1/me").catch((e: unknown) => e)) as ApiError;
    expect(err.code).toBe("network_error");
    expect(err.retryable).toBe(true);
  });
});
