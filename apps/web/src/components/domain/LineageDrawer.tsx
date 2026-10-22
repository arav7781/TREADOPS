"use client";

import { GitBranch } from "lucide-react";

import { shortSha } from "@/lib/format";
import type { ModelTuple } from "@/lib/types";

function fmt(v: unknown): string {
  if (v === null || v === undefined || v === "") return "n/a";
  if (typeof v === "string" && /^[0-9a-f]{40,64}$/.test(v)) return shortSha(v, 16);
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

const ORDER = [
  "model_version", "bundle_sha256", "deployment_generation", "dataset_snapshot_id", "split_sha256", "preprocess_version",
  "preprocess_sha256", "calibration_version", "policy_version", "git_sha", "training_run", "mlflow_run_id",
  "image_sha256", "image_id", "worker",
];

/** Full lineage from inspection to bundle, run, dataset, code, transform, calibrator and policy. */
export function LineageDrawer({ lineage, deployment, defaultOpen = false }: {
  lineage: Record<string, unknown> | null | undefined;
  deployment?: ModelTuple;
  defaultOpen?: boolean;
}) {
  const merged: Record<string, unknown> = { ...(deployment ?? {}), ...(lineage ?? {}) };
  delete merged.bundle_uri;
  const keys = [...ORDER.filter((k) => k in merged), ...Object.keys(merged).filter((k) => !ORDER.includes(k)).sort()];
  if (!keys.length) return <p className="text-muted">Lineage appears once the prediction is stored.</p>;
  return (
    <details open={defaultOpen} className="group rounded-2xl border border-line bg-surface-muted">
      <summary className="flex cursor-pointer list-none items-center gap-2 px-4 py-3 font-bold text-ink">
        <GitBranch aria-hidden className="h-4 w-4 text-primary" /> Lineage
        <span className="text-sm font-normal text-muted">({keys.length} fields)</span>
      </summary>
      <dl className="grid grid-cols-1 gap-x-6 gap-y-1.5 px-4 pb-4 sm:grid-cols-2">
        {keys.map((k) => (
          <div key={k} className="flex min-w-0 justify-between gap-3 border-b border-line/60 py-1">
            <dt className="text-sm text-muted">{k}</dt>
            <dd className="mono truncate text-sm text-ink" title={typeof merged[k] === "string" ? String(merged[k]) : undefined}>
              {fmt(merged[k])}
            </dd>
          </div>
        ))}
      </dl>
    </details>
  );
}
