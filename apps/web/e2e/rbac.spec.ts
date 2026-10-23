import { expect, test } from "@playwright/test";

import { authFile } from "./helpers";

test.describe("backend enforces roles even when the UI hides actions", () => {
  test.use({ storageState: authFile("viewer") });

  test("viewer cannot start a replay or promote through the BFF", async ({ page, baseURL }) => {
    await page.goto("/dashboard");
    const headers = { origin: baseURL!, "content-type": "application/json" };
    const replay = await page.request.post("/api/v1/replays", { headers, data: { scenario: "baseline", count: 1 } });
    expect(replay.status()).toBe(403);
    expect((await replay.json()).error.code).toBe("forbidden");
    const promote = await page.request.post("/api/v1/deployments/promote", {
      headers, data: { model_version_id: "00000000-0000-0000-0000-000000000000", expected_generation: 0, reason: "not allowed for viewers" },
    });
    expect(promote.status()).toBe(403);
    const audit = await page.request.get("/api/v1/audit-events");
    expect(audit.status()).toBe(403);
  });

  test("viewer sees an access-denied state on the audit page", async ({ page }) => {
    await page.goto("/audit");
    await expect(page.getByText("Access denied")).toBeVisible();
  });

  test("cross-origin mutations are rejected by the BFF", async ({ page }) => {
    await page.goto("/dashboard");
    const res = await page.request.post("/api/v1/monitoring/windows", { headers: { origin: "https://evil.example" } });
    expect(res.status()).toBe(403);
    expect((await res.json()).error.code).toBe("cross_origin_rejected");
  });
});

test("API calls without a session get a typed 401", async ({ request }) => {
  const res = await request.get("/api/v1/me");
  expect(res.status()).toBe(401);
  expect((await res.json()).error.code).toBe("missing_token");
});
