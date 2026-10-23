import { request, type FullConfig } from "@playwright/test";
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import path from "node:path";

export const ROLES = ["viewer", "inspector", "mlengineer", "admin"] as const;

/** DEMO_PASSWORD from the environment or the repository .env (never printed). */
export function demoPassword(): string {
  if (process.env.DEMO_PASSWORD) return process.env.DEMO_PASSWORD;
  const envFile = path.resolve(__dirname, "../../../.env");
  if (existsSync(envFile)) {
    const line = readFileSync(envFile, "utf8").split("\n").find((l) => l.startsWith("DEMO_PASSWORD="));
    if (line) return line.slice("DEMO_PASSWORD=".length).trim();
  }
  throw new Error("DEMO_PASSWORD not set and not found in .env");
}

export default async function globalSetup(config: FullConfig) {
  const baseURL = config.projects[0].use.baseURL ?? "http://localhost:3100";
  const dir = path.join(__dirname, ".auth");
  mkdirSync(dir, { recursive: true });
  const password = demoPassword();
  for (const role of ROLES) {
    const ctx = await request.newContext({ baseURL, extraHTTPHeaders: { origin: baseURL } });
    const res = await ctx.post("/api/auth/login", { data: { email: `${role}@treadops.local`, password } });
    if (!res.ok()) throw new Error(`login for ${role} failed with HTTP ${res.status()}`);
    await ctx.storageState({ path: path.join(dir, `${role}.json`) });
    await ctx.dispose();
  }
}
