"use client";

import { useQuery } from "@tanstack/react-query";
import { Activity, BookCheck, Boxes, Gauge, Hourglass, Radar } from "lucide-react";
import Link from "next/link";

import { TriageBars, TriageTimeline } from "@/components/domain/charts";
import { GateStatus, StatusBadge } from "@/components/domain/badges";
import { InspectionTable } from "@/components/domain/InspectionTable";
import { Badge, Card, EmptyState, ErrorState, inputClass, KeyValue, PageHeader, SkeletonRows, Stat } from "@/components/ui";
import { api } from "@/lib/api";
import { ago, duration, humanize, ms, num, shortSha } from "@/lib/format";
import { qk } from "@/lib/query";
import type { InspectionList, MetricsOverview } from "@/lib/types";
import { toQuery, useUrlState } from "@/lib/useUrlState";

const WINDOWS: Record<string, { label: string; hours: number; bucket: string }> = {
  "24h": { label: "Last 24 hours", hours: 24, bucket: "hour" },
  "7d": { label: "Last 7 days", hours: 168, bucket: "hour" },
  "30d": { label: "Last 30 days", hours: 720, bucket: "day" },
};

export function Dashboard() {
  const [f, setF] = useUrlState(["window", "source", "line_id", "camera_id", "metadata_origin"] as const, { window: "7d" });
  const w = WINDOWS[f.window] ?? WINDOWS["7d"];
  const filters = { bucket: w.bucket, source: f.source, line_id: f.line_id, camera_id: f.camera_id, metadata_origin: f.metadata_origin };
  const overview = useQuery({
    queryKey: qk.overview({ ...filters, window: f.window }),
    // The window start is computed when the request is made so each refresh slides the window forward.
    queryFn: () => api<MetricsOverview>(`/v1/metrics/overview${toQuery({ ...filters,
      since: new Date(Date.now() - w.hours * 3600_000).toISOString() })}`),
    refetchInterval: 10_000,
  });
  const recent = useQuery({
    queryKey: qk.inspections({ limit: "8", source: f.source, line_id: f.line_id, camera_id: f.camera_id, metadata_origin: f.metadata_origin }),
    queryFn: () => api<InspectionList>(`/v1/inspections${toQuery({ limit: 8, source: f.source, line_id: f.line_id, camera_id: f.camera_id, metadata_origin: f.metadata_origin })}`),
    refetchInterval: 10_000,
  });
  const o = overview.data;

  return (
    <>
      <PageHeader eyebrow="Operations" title="Overview"
        subtitle="Live aggregates computed from stored inspections and jobs. Every figure carries its cohort size and observation time." />
      <form className="card mb-6 grid gap-3 p-4 sm:grid-cols-2 lg:grid-cols-5" aria-label="Dashboard filters" onSubmit={(e) => e.preventDefault()}>
        <FilterSelect id="f-window" label="Time window" value={f.window} onChange={(v) => setF({ window: v })}
          options={Object.entries(WINDOWS).map(([k, v]) => [k, v.label])} />
        <FilterSelect id="f-source" label="Source" value={f.source} onChange={(v) => setF({ source: v })}
          options={[["", "All sources"], ["upload", "Upload"], ["replay", "Replay"], ["copilot", "Copilot"], ["api", "API"]]} />
        <FilterSelect id="f-origin" label="Metadata origin" value={f.metadata_origin} onChange={(v) => setF({ metadata_origin: v })}
          options={[["", "All"], ["user_provided", "User provided"], ["simulated", "Simulated"], ["observed", "Observed"]]} />
        <FilterText id="f-line" label="Line" value={f.line_id} placeholder="e.g. SIM-L02" onChange={(v) => setF({ line_id: v })} />
        <FilterText id="f-camera" label="Camera" value={f.camera_id} placeholder="e.g. SIM-C04" onChange={(v) => setF({ camera_id: v })} />
      </form>

      {overview.isError ? (
        <ErrorState error={overview.error} onRetry={() => overview.refetch()} />
      ) : !o ? (
        <SkeletonRows rows={5} />
      ) : (
        <>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <Stat label="Inspections" icon={<Gauge className="h-5 w-5" />} value={num(o.cohort_size)}
              hint={`${num(o.inspections.by_status.succeeded ?? 0)} completed · ${num(o.inspections.by_status.failed ?? 0)} failed`} />
            <Stat label="Review required" icon={<BookCheck className="h-5 w-5" />} tone="warn" value={num(o.review.awaiting)}
              hint={o.review.oldest_waiting_at ? `oldest waiting ${ago(o.review.oldest_waiting_at)}` : "queue empty"} />
            <Stat label="Inference queue" icon={<Hourglass className="h-5 w-5" />} tone="info"
              value={num(Object.values(o.queue.by_status).reduce((a, b) => a + b, 0))}
              hint={o.queue.oldest_queued_age_s !== null ? `oldest queued ${duration(o.queue.oldest_queued_age_s)}` : "nothing waiting"} />
            <Stat label="Drift" icon={<Radar className="h-5 w-5" />} tone={o.drift.open_alerts ? "bad" : "accent"}
              value={o.drift.latest_window ? humanize(o.drift.latest_window.status) : "No window yet"}
              hint={o.drift.latest_window ? `${o.drift.open_alerts} open alert(s) · ${o.drift.latest_window.distinct_parents} parents · ${ago(o.drift.latest_window.created_at)}` : "computed every 200 completions"} />
          </div>

          <div className="mt-6 grid gap-6 xl:grid-cols-3">
            <Card className="xl:col-span-2" title="Prediction distribution and latency" subtitle={`${w.label} · observed ${ago(o.observed_at)}`}>
              {o.timeline.length ? <TriageTimeline data={o.timeline} /> : <EmptyState title="No completed inspections in this window" />}
            </Card>
            <Card title="Active model" actions={<Link href="/models" className="text-sm font-semibold text-primary-ink hover:underline">Models →</Link>}>
              {o.active_model.state === "untrained" ? (
                <EmptyState title="No trained model available" icon={<Boxes className="h-7 w-7" />}>
                  Promote a validated bundle from the Models page before inspections can be accepted.
                </EmptyState>
              ) : (
                <KeyValue items={[
                  ["Version", <strong key="v">v{o.active_model.registry_version ?? "?"}</strong>],
                  ["Bundle", <span key="b" className="mono">{shortSha(o.active_model.bundle_sha256)}</span>],
                  ["Generation", o.active_model.generation],
                  ["Policy", o.active_model.policy_version ?? "n/a"],
                  ["Data snapshot", <span key="s" className="mono">{o.active_model.snapshot ? shortSha(o.active_model.snapshot, 8) : "DVC baseline split"}</span>],
                  ["Last gate", <GateStatus key="g" status={o.active_model.gate_status} />],
                ]} />
              )}
            </Card>
          </div>

          <div className="mt-6 grid gap-6 xl:grid-cols-3">
            <Card title="Triage distribution" subtitle={`n = ${num(Object.values(o.inspections.by_triage).reduce((a, b) => a + b, 0))} predictions`}>
              <TriageBars counts={o.inspections.by_triage} />
            </Card>
            <Card title="Latency" subtitle={`measured over ${num(o.latency_ms.n)} predictions`}>
              <KeyValue items={[
                ["Worker total p50 / p95", `${ms(o.latency_ms.worker_total_p50)} / ${ms(o.latency_ms.worker_total_p95)}`],
                ["Model inference p50 / p95", `${ms(o.latency_ms.inference_p50)} / ${ms(o.latency_ms.inference_p95)}`],
                ["Queue wait p95", ms(o.latency_ms.queue_p95)],
                ["Quality", Object.entries(o.inspections.by_quality).map(([k, v]) => `${humanize(k)} ${v}`).join(" · ") || "n/a"],
                ["Origins", Object.entries(o.inspections.by_metadata_origin).map(([k, v]) => `${humanize(k)} ${v}`).join(" · ") || "n/a"],
              ]} />
            </Card>
            <Card title="Workers" subtitle="heartbeats from worker processes">
              {o.workers.length === 0 ? <EmptyState title="No worker heartbeats" icon={<Activity className="h-7 w-7" />} /> : (
                <ul className="space-y-2">
                  {o.workers.map((wk) => (
                    <li key={wk.name} className="flex items-center justify-between gap-2 rounded-xl bg-surface-muted px-3 py-2">
                      <span className="min-w-0">
                        <span className="block truncate font-semibold text-ink">{wk.queue}</span>
                        <span className="mono block truncate text-xs text-muted">{wk.name}</span>
                      </span>
                      <Badge tone={wk.alive ? (wk.paused ? "warn" : "good") : "bad"}>{wk.alive ? (wk.paused ? "paused" : "alive") : `stale ${ago(wk.last_seen_at)}`}</Badge>
                    </li>
                  ))}
                </ul>
              )}
            </Card>
          </div>
        </>
      )}

      <Card className="mt-6" title="Recent inspections" actions={<Link href="/inspections" className="text-sm font-semibold text-primary-ink hover:underline">All inspections →</Link>}>
        {recent.isError ? <ErrorState error={recent.error} onRetry={() => recent.refetch()} />
          : !recent.data ? <SkeletonRows />
            : recent.data.items.length === 0 ? <EmptyState title="No inspections yet">Upload a tyre photo on the Inspect page or start a replay.</EmptyState>
              : <InspectionTable items={recent.data.items} compact />}
        {recent.data?.items.some((i) => i.status !== "succeeded" && i.status !== "failed") && (
          <p className="mt-2 text-sm text-muted"><StatusBadge status="running" /> Some inspections are still processing.</p>
        )}
      </Card>
    </>
  );
}

function FilterSelect({ id, label, value, onChange, options }: {
  id: string; label: string; value: string; onChange: (v: string) => void; options: [string, string][];
}) {
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={id} className="text-sm font-semibold text-ink-soft">{label}</label>
      <select id={id} className={inputClass} value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
      </select>
    </div>
  );
}

function FilterText({ id, label, value, onChange, placeholder }: {
  id: string; label: string; value: string; onChange: (v: string) => void; placeholder?: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={id} className="text-sm font-semibold text-ink-soft">{label}</label>
      <input id={id} className={inputClass} defaultValue={value} placeholder={placeholder}
        onBlur={(e) => onChange(e.target.value.trim())}
        onKeyDown={(e) => { if (e.key === "Enter") onChange((e.target as HTMLInputElement).value.trim()); }} />
    </div>
  );
}
