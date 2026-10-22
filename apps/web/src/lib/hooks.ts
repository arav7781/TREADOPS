"use client";

import { useQuery } from "@tanstack/react-query";

import { api } from "./api";
import { TERMINAL_INSPECTION, isActiveJob } from "./format";
import { pollWhile, qk } from "./query";
import type { InspectionDetail, JobView } from "./types";

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
