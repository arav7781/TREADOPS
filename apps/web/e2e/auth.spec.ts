import { expect, test } from "@playwright/test";

import { demoPassword } from "./global-setup";
import { authFile } from "./helpers";

test("unauthenticated pages redirect to login with a return path", async ({ page }) => {
  await page.goto("/models");
  await expect(page).toHaveURL(/\/login\?next=%2Fmodels/);
  await expect(page.getByRole("heading", { name: /Every casing/ })).toBeVisible();
  await expect(page.getByRole("form", { name: "Sign in" })).toBeVisible();
});

test("wrong password shows a generic error", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel("Email").fill("inspector@treadops.local");
  await page.getByLabel("Password").fill("definitely-wrong");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByText("Email or password is incorrect.")).toBeVisible();
});

test("inspector signs in through the form and lands on the dashboard", async ({ page }) => {
  await page.goto("/login?next=/dashboard");
  await page.getByLabel("Email").fill("inspector@treadops.local");
  await page.getByLabel("Password").fill(demoPassword());
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(/\/dashboard/);
  await expect(page.getByRole("heading", { name: "Overview" })).toBeVisible();
  const nav = page.getByRole("navigation", { name: "Primary" }).first();
  await expect(nav.getByRole("link", { name: "Bib · Inspect & ask", exact: true })).toBeVisible();
  await expect(nav.getByRole("link", { name: "Audit", exact: true })).toHaveCount(0);
});

test.describe("navigation follows roles", () => {
  test.use({ storageState: authFile("viewer") });
  test("viewer sees read-only navigation", async ({ page }) => {
    await page.goto("/dashboard");
    const nav = page.getByRole("navigation", { name: "Primary" }).first();
    await expect(nav.getByRole("link", { name: "Bib · Inspect & ask", exact: true })).toBeVisible(); // chat yes, photos no
    await nav.getByRole("button", { name: "Operate" }).click();
    await expect(nav.getByRole("link", { name: "Overview", exact: true })).toBeVisible();
    await nav.getByRole("button", { name: "MLOps" }).click();
    await expect(nav.getByRole("link", { name: "Models", exact: true })).toBeVisible();
    for (const hidden of ["Training", "Replay"]) {
      await expect(nav.getByRole("link", { name: hidden, exact: true })).toHaveCount(0);
    }
    await expect(nav.getByRole("button", { name: "Governance" })).toHaveCount(0); // no LLMOps or Audit for viewers
  });
});

test.describe("admin navigation", () => {
  test.use({ storageState: authFile("admin") });
  test("admin sees audit and LLMOps", async ({ page }) => {
    await page.goto("/dashboard");
    const nav = page.getByRole("navigation", { name: "Primary" }).first();
    await nav.getByRole("button", { name: "Governance" }).click();
    await expect(nav.getByRole("link", { name: "Audit", exact: true })).toBeVisible();
    await expect(nav.getByRole("link", { name: "LLMOps", exact: true })).toBeVisible();
  });
});
