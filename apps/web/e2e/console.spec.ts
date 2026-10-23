import { expect, test } from "@playwright/test";

import { authFile } from "./helpers";

test.use({ storageState: authFile("admin") });

const PAGES = ["/dashboard", "/inspect", "/inspections", "/reviews", "/circularity", "/datasets", "/training",
  "/models", "/monitoring", "/replay", "/llmops", "/audit"];

test("no uncaught errors or hydration mismatches on any page", async ({ page }) => {
  const pageErrors: string[] = [];
  const failedRequests: string[] = [];
  page.on("pageerror", (e) => pageErrors.push(`${page.url()}: ${e.message.split("\n")[0]}`));
  page.on("response", (r) => { if (r.status() >= 500) failedRequests.push(`${r.status()} ${new URL(r.url()).pathname}`); });
  for (const p of PAGES) {
    await page.goto(p);
    await page.waitForLoadState("networkidle");
  }
  // Backend 5xx responses are rendered as typed error states; they are reported, not hidden.
  test.info().annotations.push({ type: "backend-5xx", description: [...new Set(failedRequests)].join(", ") || "none" });
  expect(pageErrors).toEqual([]);
});
