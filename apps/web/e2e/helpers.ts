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
