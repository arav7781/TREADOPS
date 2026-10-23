import { defineConfig } from "@playwright/test";

/**
 * End-to-end tests run against the real stack: Next.js (port 3100) -> BFF -> FastAPI -> workers.
 * Browser: PW_EXECUTABLE (an existing Chromium/headless-shell binary) or PW_CHANNEL (default "chrome"), so no
 * download is required on networks that block the Playwright CDN.
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 180_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"], ["junit", { outputFile: "test-results/junit.xml" }]],
  globalSetup: "./e2e/global-setup.ts",
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://localhost:3100",
    ...(process.env.PW_EXECUTABLE
      ? { launchOptions: { executablePath: process.env.PW_EXECUTABLE } }
      : { channel: process.env.PW_CHANNEL ?? "chrome" }),
    headless: true,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
});
