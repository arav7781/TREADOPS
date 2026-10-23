import { describe, expect, it } from "vitest";

import { isSameOrigin } from "@/lib/origin";

describe("isSameOrigin (BFF CSRF check)", () => {
  const base = { host: "localhost:3100", appOrigin: null as string | null };

  it("accepts the request's own origin", () => {
    expect(isSameOrigin({ ...base, origin: "http://localhost:3100" })).toBe(true);
  });

  it("rejects a foreign origin", () => {
    expect(isSameOrigin({ ...base, origin: "https://evil.example" })).toBe(false);
  });

  it("accepts a configured APP_ORIGIN even when the host differs (reverse proxy)", () => {
    expect(isSameOrigin({ origin: "https://treadops.example", host: "web:3000", appOrigin: "https://treadops.example" })).toBe(true);
  });

  it("uses forwarded host and proto when present", () => {
    expect(isSameOrigin({ origin: "https://ops.example", host: "web:3000", forwardedHost: "ops.example", forwardedProto: "https" })).toBe(true);
  });

  it("treats a missing Origin as non-browser unless Sec-Fetch-Site says cross-site", () => {
    expect(isSameOrigin({ ...base, origin: null })).toBe(true);
    expect(isSameOrigin({ ...base, origin: null, secFetchSite: "same-origin" })).toBe(true);
    expect(isSameOrigin({ ...base, origin: null, secFetchSite: "cross-site" })).toBe(false);
  });

  it("rejects malformed origins", () => {
    expect(isSameOrigin({ ...base, origin: "not a url" })).toBe(false);
  });
});
