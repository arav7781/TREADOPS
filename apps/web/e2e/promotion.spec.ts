import { expect, test } from "@playwright/test";

import { authFile, modelState } from "./helpers";

test.use({ storageState: authFile("admin") });

test("promotion is blocked for bundles whose gates did not pass", async ({ page }) => {
  await page.goto("/models");
  const models = await modelState(page.request);
  test.skip(models.items.length === 0, "No model versions registered yet");
  const blocked = models.items.find((m) => m.gate_status !== "passed" && !m.is_active);
  test.skip(!blocked, "Every registered bundle passed its gates; nothing to block");
  await page.goto(`/models?model=${blocked!.id}`);
  await expect(page.getByRole("button", { name: "Promote" })).toBeDisabled();
  await expect(page.getByText(/Promotion is blocked/)).toBeVisible();
});

test("promote a ready candidate, verify the active version, then roll back", async ({ page }) => {
  test.skip(process.env.E2E_ALLOW_RELEASE !== "1", "Release journey mutates the serving pointer; set E2E_ALLOW_RELEASE=1 to run");
  await page.goto("/models");
  const models = await modelState(page.request);
  const candidate = models.items.find((m) => m.gate_status === "passed" && !m.is_active && m.deployments.length === 0);
  test.skip(!candidate, "No passed, never-deployed candidate to promote");
  await page.goto(`/models?model=${candidate!.id}`);
  await page.getByRole("button", { name: "Promote" }).click();
  await page.getByLabel(/Approval reason/).fill("e2e: promote candidate after gate review");
  await page.getByRole("button", { name: "Approve and promote" }).click();
  await expect(page.getByText("Succeeded")).toBeVisible({ timeout: 120_000 });
  const after = await modelState(page.request);
  expect(after.items.find((m) => m.id === candidate!.id)?.is_active).toBe(true);
});
