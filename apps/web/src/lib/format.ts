import type { GateState, JobStatus, Triage } from "./types";

export const TRIAGE_LABEL: Record<Triage, string> = {
  good_candidate: "Retread candidate",
  manual_review: "Expert review",
  defect_flagged: "Defect flagged",
};

export type Verdict = "damaged" | "not_damaged" | "uncertain";
export const VERDICT_LABEL: Record<Verdict, string> = { damaged: "Damaged", not_damaged: "Not damaged", uncertain: "Uncertain" };

/** Mirrors treadops_api.verdicts.verdict_of: "not damaged" only for a clean auto-clear; anything else is uncertain. */
export function verdictOf(damaged: boolean, triage?: Triage | null, quality?: string | null): Verdict {
  if (damaged) return "damaged";
  return triage === "good_candidate" && quality !== "low_quality" ? "not_damaged" : "uncertain";
}

export const TERMINAL_JOB: ReadonlySet<string> = new Set(["succeeded", "failed", "cancelled"]);
export const TERMINAL_INSPECTION: ReadonlySet<string> = new Set(["succeeded", "failed", "cancelled", "rejected"]);

export function isActiveJob(status: JobStatus | string | null | undefined): boolean {
  return Boolean(status) && !TERMINAL_JOB.has(String(status));
}

export function pct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "n/a";
  return `${(v * 100).toFixed(digits)}%`;
}

export function num(v: number | null | undefined, digits = 0): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "n/a";
  return v.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

export function ms(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "n/a";
  if (v >= 1000) return `${(v / 1000).toFixed(2)} s`;
  return `${v.toFixed(v < 10 ? 1 : 0)} ms`;
}

export function shortSha(s: string | null | undefined, n = 12): string {
  return s ? s.slice(0, n) : "n/a";
}

export function dateTime(iso: string | null | undefined): string {
  if (!iso) return "n/a";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export function timeOnly(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function ago(iso: string | null | undefined, now: number = Date.now()): string {
  if (!iso) return "n/a";
  const s = Math.max(0, (now - new Date(iso).getTime()) / 1000);
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "n/a";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`;
}

export function gateLabel(s: GateState | string | null | undefined): string {
  switch (s) {
    case "passed":
      return "Passed";
    case "failed":
      return "Failed";
    case "insufficient_evidence":
      return "Insufficient evidence";
    case "not_applicable":
      return "Not applicable";
    default:
      return s ? String(s) : "Unknown";
  }
}

export function humanize(s: string | null | undefined): string {
  if (!s) return "";
  return s.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}
