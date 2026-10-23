"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { Query } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";

import { ApiError } from "./api";

function shouldRetry(failureCount: number, error: unknown): boolean {
  if (error instanceof ApiError) {
    if (error.status === 401 || error.status === 403 || error.status === 404 || error.status === 422) return false;
    return error.retryable && failureCount < 2;
  }
  return failureCount < 1;
}

export function QueryProvider({ children }: { children: ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            retry: shouldRetry,
            retryDelay: (n) => Math.min(8000, 1000 * 2 ** n),
            refetchOnWindowFocus: false,
            // Background polling pauses while the tab is hidden.
            refetchIntervalInBackground: false,
            staleTime: 5_000,
          },
          mutations: { retry: false },
        },
      }),
  );
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

/**
 * Poll every ~1.5 s while `isActive(data)` holds; back off exponentially on errors; stop at terminal states.
 */
export function pollWhile<T>(isActive: (data: T | undefined) => boolean, intervalMs = 1500) {
  return (query: Query<T, Error, T, readonly unknown[]>) => {
    if (query.state.status === "error") {
      return Math.min(30_000, intervalMs * 2 ** Math.min(5, query.state.fetchFailureCount));
    }
    return isActive(query.state.data) ? intervalMs : false;
  };
}

export const qk = {
  me: ["me"] as const,
  models: ["models"] as const,
  evidence: (id: string) => ["models", id, "evidence"] as const,
  deployments: ["deployments"] as const,
  overview: (filters: Record<string, string>) => ["metrics", "overview", filters] as const,
  inspections: (filters: Record<string, string>) => ["inspections", filters] as const,
  inspection: (id: string) => ["inspection", id] as const,
  reviewQueue: (sort: string, triage: string) => ["reviews", "queue", sort, triage] as const,
  datasets: ["datasets"] as const,
  manifest: (id: string, role: string, offset: number) => ["datasets", id, "manifest", role, offset] as const,
  recipes: ["training", "recipes"] as const,
  trainingRuns: ["training", "runs"] as const,
  trainingRun: (id: string) => ["training", "runs", id] as const,
  job: (id: string) => ["job", id] as const,
  windows: ["monitoring", "windows"] as const,
  alerts: ["monitoring", "alerts"] as const,
  scenarios: ["replay", "scenarios"] as const,
  replays: ["replay", "list"] as const,
  audit: (filters: Record<string, string>) => ["audit", filters] as const,
  ledger: ["circularity", "ledger"] as const,
  passport: (id: string) => ["circularity", "passport", id] as const,
  copilotStatus: ["copilot", "status"] as const,
  cameras: ["cameras"] as const,
  cameraSnapshot: (id: string) => ["cameras", id, "snapshot"] as const,
  copilotSessions: ["copilot", "sessions"] as const,
  copilotSession: (id: string) => ["copilot", "session", id] as const,
  llmopsOverview: ["llmops", "overview"] as const,
  traces: (filters: Record<string, string>) => ["llmops", "traces", filters] as const,
  trace: (id: string) => ["llmops", "trace", id] as const,
  evals: ["llmops", "evals"] as const,
};
