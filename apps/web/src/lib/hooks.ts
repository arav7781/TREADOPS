"use client";

import { useQuery } from "@tanstack/react-query";
import { useSyncExternalStore } from "react";

import { api } from "./api";
import { TERMINAL_INSPECTION, isActiveJob } from "./format";
import { pollWhile, qk } from "./query";
import type { InspectionDetail, JobView, Me, ModelsResponse } from "./types";

const noopSubscribe = () => () => {};

/** False during SSR and hydration, true afterwards (no setState-in-effect, no mismatch). */
export function useHydrated(): boolean {
  return useSyncExternalStore(noopSubscribe, () => true, () => false);
}

/**
 * Queries shared with the app shell are often already cached when a page's Suspense boundary hydrates.
 * Withholding their data until hydration finishes keeps the first client render identical to the server HTML.
 */
function hydrationSafe<T extends { data: unknown }>(q: T, hydrated: boolean): T {
  return hydrated ? q : { ...q, data: undefined };
}

export function useMe() {
  const q = useQuery({ queryKey: qk.me, queryFn: () => api<Me>("/v1/me"), staleTime: 60_000 });
  return hydrationSafe(q, useHydrated());
}

export function useModels(refetchInterval: number | false = false) {
  const q = useQuery({ queryKey: qk.models, queryFn: () => api<ModelsResponse>("/v1/models"), staleTime: 15_000, refetchInterval });
  return hydrationSafe(q, useHydrated());
}

/** Live inspection: polls ~1.5 s until a terminal status, with error backoff and hidden-tab pause. */
export function useInspection(id: string | null | undefined) {
  return useQuery({
    queryKey: qk.inspection(id ?? "none"),
    queryFn: () => api<InspectionDetail>(`/v1/inspections/${id}`),
    enabled: Boolean(id),
    refetchInterval: pollWhile<InspectionDetail>((d) => !d || !TERMINAL_INSPECTION.has(d.status)),
  });
}

export function useJob(id: string | null | undefined, interval = 1500) {
  return useQuery({
    queryKey: qk.job(id ?? "none"),
    queryFn: () => api<JobView>(`/v1/jobs/${id}`),
    enabled: Boolean(id),
    refetchInterval: pollWhile<JobView>((d) => !d || isActiveJob(d.status), interval),
  });
}
