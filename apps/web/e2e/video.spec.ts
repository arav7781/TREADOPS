import { existsSync } from "node:fs";
import path from "node:path";

import { expect, test } from "@playwright/test";

import { authFile, expectTyreVerdict, modelState } from "./helpers";

// Built by `python scripts/sim_cameras.py clip`: one tyre photo swept like a tyre turning in front of a camera.
const CLIP = path.resolve(__dirname, "../../../runs/sim-cameras/turning-tyre.mp4");

test.use({ storageState: authFile("inspector") });

test("a video clip is sampled in the browser and inspected frame by frame", async ({ page }) => {
  test.skip(!existsSync(CLIP), "run `python scripts/sim_cameras.py clip` first");
  await page.goto("/inspect");
  test.skip((await modelState(page.request)).state !== "ready", "No trained model is active yet");
  await expect(page.locator("#chat-video")).toBeEnabled();
  await page.locator("#chat-video").setInputFiles(CLIP);
  const strip = page.getByRole("list", { name: "Frames to send" });
  await expect(strip).toBeVisible({ timeout: 60_000 });
  const sent = await strip.getByRole("listitem").count();
  expect(sent).toBeGreaterThanOrEqual(2);
  await page.getByLabel("Message", { exact: true }).fill("Is this tyre damaged anywhere around the tread?");
  await page.getByRole("button", { name: "Send" }).click();
  await expectTyreVerdict(page);
  await expect(page.getByRole("list", { name: "Frames", exact: true }).getByRole("listitem")).toHaveCount(sent);
});
