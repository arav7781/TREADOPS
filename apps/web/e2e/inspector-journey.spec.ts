import { expect, test } from "@playwright/test";

import { authFile, modelState, realTyreImage } from "./helpers";

test.use({ storageState: authFile("inspector") });

test("upload a real tyre photo, watch the stages, review it and see it persist", async ({ page }) => {
  await page.goto("/dashboard");
  const models = await modelState(page.request);
  test.skip(models.state !== "ready", "No trained model is active yet (/api/v1/models reports untrained)");

  await page.goto("/inspect");
  const input = page.locator('input[type="file"]').first();
  await expect(input).toBeEnabled(); // enabled once the role and model state have loaded
  await input.setInputFiles(realTyreImage());
  await expect(page.getByAltText("Decoded preview of the selected tyre photo")).toBeVisible();
  await page.getByRole("button", { name: "Inspect tyre" }).click();
  await expect(page).toHaveURL(/\/inspect\?id=/);

  // All eight recorded stages arrive (awaited, retrying assertions - no fixed sleeps).
  for (const stage of ["ingest", "canonical", "letterbox", "tensor", "attribution", "classified"]) {
    await expect(page.locator(`[data-stage="${stage}"] img`)).toBeVisible({ timeout: 90_000 });
  }
  await expect(page.getByRole("heading", { name: "Decision" })).toBeVisible();
  await expect(page.getByText(/Calibrated p\(defective\)/)).toBeVisible();

  const id = new URL(page.url()).searchParams.get("id")!;
  await page.goto(`/inspections/${id}`);
  await page.getByText("Defective", { exact: true }).first().click();
  await page.getByLabel("Reason (required)").fill("e2e: sidewall cracking visible near the shoulder");
  await page.getByRole("button", { name: "Save review" }).click();
  await expect(page.getByText(/Saved revision r1/)).toBeVisible();

  await page.reload();
  await expect(page.getByText("Current revision r1 · Defective")).toBeVisible();
  await expect(page.getByRole("listitem").filter({ hasText: /^r1 · Defective/ })).toBeVisible();
  await expect(page.getByText("“e2e: sidewall cracking visible near the shoulder”")).toBeVisible();
});
