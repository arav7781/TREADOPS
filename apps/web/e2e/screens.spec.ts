import { expect, test } from "@playwright/test";

import { authFile } from "./helpers";

test.use({ storageState: authFile("admin") });

const PAGES = ["/dashboard", "/inspect", "/models", "/datasets", "/monitoring", "/copilot", "/circularity", "/llmops"];
const WIDTHS = [390, 768, 1440];

for (const width of WIDTHS) {
  test(`key pages render without horizontal overflow at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    for (const p of PAGES) {
      await page.goto(p);
      await page.waitForLoadState("networkidle");
      await expect(page.locator("main h1").first()).toBeVisible();
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
      expect(overflow, `${p} overflows at ${width}px`).toBeLessThanOrEqual(1);
      await page.screenshot({ path: `test-results/screens/${width}${p.replace(/\//g, "_")}.png`, fullPage: true });
    }
  });
}
