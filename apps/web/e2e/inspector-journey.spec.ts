import { expect, test } from "@playwright/test";

import { authFile, modelState, realTyreImage } from "./helpers";

test.use({ storageState: authFile("inspector") });

test("send a real tyre photo to Bib, watch the stages in the chat, review it and see it persist", async ({ page }) => {
  await page.goto("/dashboard");
  const models = await modelState(page.request);
  test.skip(models.state !== "ready", "No trained model is active yet (/api/v1/models reports untrained)");

  await page.goto("/inspect");
  const input = page.locator("#chat-photo");
  await expect(input).toBeEnabled(); // enabled once the role and model state have loaded
  await input.setInputFiles(realTyreImage());
  await expect(page.getByAltText("Selected tyre photo preview")).toBeVisible();
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page).toHaveURL(/\/inspect\?session=/);

  // The agent's YOLO tool creates the inspection; all recorded stages arrive inside the chat (no fixed sleeps).
  for (const stage of ["ingest", "canonical", "letterbox", "tensor", "attribution", "classified"]) {
    await expect(page.locator(`[data-stage="${stage}"] img`)).toBeVisible({ timeout: 90_000 });
  }
  const result = page.getByTestId("chat-result");
  await expect(result).toBeVisible({ timeout: 90_000 });
  await expect(result.getByText(/Calibrated p\(defective\)/)).toBeVisible();

  const href = await result.getByRole("link", { name: /Full inspection/ }).getAttribute("href");
  const id = href!.split("/").pop()!;
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
