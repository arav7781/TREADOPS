import { expect, type Page } from "@playwright/test";
import type { APIRequestContext } from "@playwright/test";
import { readdirSync } from "node:fs";
import path from "node:path";

export const authFile = (role: string) => path.join(__dirname, ".auth", `${role}.json`);

export async function modelState(api: APIRequestContext): Promise<{ state: string; items: { id: string; gate_status: string; is_active: boolean; deployments: unknown[] }[]; generation: number }> {
  const res = await api.get("/api/v1/models");
  return res.json();
}

/** A real tyre photo from the canonical D1 images (no synthetic fixture for the journey test). */
export function realTyreImage(): string {
  const dir = path.resolve(__dirname, "../../../data/processed/images");
  const file = readdirSync(dir).filter((f) => f.endsWith(".jpg")).sort()[7];
  return path.join(dir, file);
}

/** A video turn reached its tyre-level verdict and the LLM answered with it. */
export async function expectTyreVerdict(page: Page) {
  await expect(page.getByRole("list", { name: "Video frames you sent" })).toBeVisible({ timeout: 30_000 });
  // One inspection per frame, then the tyre-level verdict from the worst frame, then the LLM's answer.
  await expect(page.getByTestId("video-verdict")).toBeVisible({ timeout: 180_000 });
  await expect(page.getByTestId("video-verdict").getByText(/^(Damaged|Not damaged)$/)).toBeVisible();
  await expect(page.locator('[data-role="assistant"]').last().getByText(/\b(Damaged|Not damaged)\b/).last())
    .toBeVisible({ timeout: 120_000 });
}
