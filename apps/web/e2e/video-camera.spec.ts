import { existsSync } from "node:fs";
import path from "node:path";

import { expect, test } from "@playwright/test";

import { authFile, expectTyreVerdict, modelState } from "./helpers";

// Chromium's fake camera plays this file (built by `python scripts/sim_cameras.py clip`) as the live feed.
const CAMERA_FEED = path.resolve(__dirname, "../../../runs/sim-cameras/turning-tyre.mjpeg");

test.use({
  storageState: authFile("inspector"),
  permissions: ["camera"],
  launchOptions: {
    ...(process.env.PW_EXECUTABLE ? { executablePath: process.env.PW_EXECUTABLE } : {}),
    args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-video-capture=${CAMERA_FEED}`],
  },
});

test("recording with the live camera sends the sharpest frames of the turn", async ({ page }) => {
  test.skip(!existsSync(CAMERA_FEED), "run `python scripts/sim_cameras.py clip` first");
  await page.goto("/inspect");
  test.skip((await modelState(page.request)).state !== "ready", "No trained model is active yet");
  await page.getByRole("button", { name: "Attach a tyre video" }).click();
  await page.getByRole("menuitem", { name: "Record with camera" }).click();
  await page.getByRole("button", { name: "Start recording" }).click();
  await page.waitForTimeout(6000);
  await page.getByRole("button", { name: /^Stop/ }).click();
  await expect(page.getByRole("list", { name: "Frames to send" })).toBeVisible({ timeout: 20_000 });
  await page.getByRole("button", { name: "Send" }).click();
  await expectTyreVerdict(page);
});
