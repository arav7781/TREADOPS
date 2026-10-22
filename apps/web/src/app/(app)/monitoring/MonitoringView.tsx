"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import Link from "next/link";
import { useState } from "react";

import { DriftWindowCard } from "@/components/domain/DriftWindowCard";
import { JobProgress } from "@/components/domain/JobProgress";
import { useMe } from "@/components/AppShell";
import { Badge, Button, Card, EmptyState, ErrorState, inputClass, PageHeader, SkeletonRows } from "@/components/ui";
import { api } from "@/lib/api";
import { ago, dateTime, humanize } from "@/lib/format";
import { qk } from "@/lib/query";
import { hasRole } from "@/lib/roles";
import type { DriftAlert, WindowsResponse } from "@/lib/types";

const NEXT: Record<string, string[]> = { open: ["acknowledged", "investigating", "resolved"], acknowledged: ["investigating", "resolved"], investigating: ["resolved"], resolved: [] };

function AlertReview({ alert, canReview }: { alert: DriftAlert; canReview: boolean }) {
  const qc = useQueryClient();
  const [state, setState] = useState(NEXT[alert.state][0] ?? "");
  const [reason, setReason] = useState("");
  const m = useMutation({
    mutationFn: () => api(`/v1/drift-alerts/${alert.id}/review`, { method: "POST", json: { state, reason } }),
    onSuccess: () => { setReason(""); qc.invalidateQueries({ queryKey: qk.alerts }); },
  });
  return (
    <li className="rounded-2xl border border-line p-4">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={alert.state === "resolved" ? "good" : alert.severity === "high" ? "bad" : "warn"}>{humanize(alert.state)}</Badge>
        <span className="font-bold text-ink">{alert.features.join(", ")}</span>
        <Badge>{alert.severity}</Badge>
        <span className="text-sm text-muted">opened {ago(alert.created_at)} · {alert.window_ids.length} windows</span>
      </div>
      <p className="mt-2 text-ink-soft">
        Value {alert.details.value?.toFixed(3) ?? "n/a"} vs threshold {alert.details.threshold ?? "n/a"}. {alert.recommendation}
      </p>
      {alert.details.sample_inspection_ids && (
        <p className="mt-1 text-sm">Samples: {alert.details.sample_inspection_ids.map((id) => (
          <Link key={id} href={`/inspections/${id}`} className="mono mr-2 text-primary-ink hover:underline">{id.slice(0, 8)}</Link>
        ))}</p>
      )}
      {alert.reviews.length > 0 && (
        <ol className="mt-2 space-y-1 text-sm text-ink-soft">
          {alert.reviews.map((r, i) => <li key={i}>{dateTime(r.at)} · {r.by}: {humanize(r.from)} → {humanize(r.to)} — “{r.reason}”</li>)}
        </ol>
      )}
      {canReview && NEXT[alert.state].length > 0 && (
        <form className="mt-3 grid gap-2 sm:grid-cols-[auto_1fr_auto]" onSubmit={(e) => { e.preventDefault(); m.mutate(); }}>
          <label className="sr-only" htmlFor={`as-${alert.id}`}>New state</label>
          <select id={`as-${alert.id}`} className={inputClass} value={state} onChange={(e) => setState(e.target.value)}>
            {NEXT[alert.state].map((s) => <option key={s} value={s}>{humanize(s)}</option>)}
          </select>
          <label className="sr-only" htmlFor={`ar-${alert.id}`}>Reason</label>
          <input id={`ar-${alert.id}`} className={inputClass} required minLength={3} value={reason} onChange={(e) => setReason(e.target.value)} placeholder="What you checked / decided" />
          <Button type="submit" busy={m.isPending} disabled={reason.trim().length < 3}>Record</Button>
          {m.error ? <div className="sm:col-span-3"><ErrorState error={m.error} title="Review not saved" /></div> : null}
        </form>
      )}
    </li>
  );
}

export function MonitoringView() {
  const me = useMe();
  const qc = useQueryClient();
  const windows = useQuery({ queryKey: qk.windows, queryFn: () => api<WindowsResponse>("/v1/monitoring/windows?limit=20"), refetchInterval: 20_000 });
  const alerts = useQuery({ queryKey: qk.alerts, queryFn: () => api<{ items: DriftAlert[] }>("/v1/drift-alerts"), refetchInterval: 20_000 });
  const [sel, setSel] = useState<string | null>(null);
  const [job, setJob] = useState<string | null>(null);
  const run = useMutation({
    mutationFn: () => api<{ job_id: string }>("/v1/monitoring/windows", { method: "POST" }),
    onSuccess: (r) => setJob(r.job_id),
  });
  const isMl = hasRole(me.data?.roles, "ml_engineer");
  const items = windows.data?.items ?? [];
  const current = items.find((w) => w.id === sel) ?? items[0];
  return (
    <>
      <PageHeader eyebrow="Observability" title="Monitoring"
        subtitle="Latest 500 completed inspections vs the active bundle's frozen reference, every 200 completions. Alerts need two consecutive windows and only open a review — they never retrain or promote."
        actions={isMl && <Button onClick={() => run.mutate()} busy={run.isPending}>Compute window now</Button>} />
      {run.error ? <div className="mb-4"><ErrorState error={run.error} title="Window not started" /></div> : null}
      {job && <div className="mb-6"><JobProgress jobId={job} label="Monitoring window" onDone={() => { qc.invalidateQueries({ queryKey: qk.windows }); qc.invalidateQueries({ queryKey: qk.alerts }); }} /></div>}
      <div className="grid gap-6 xl:grid-cols-[280px_1fr]">
        <Card title="Windows">
          {windows.isError ? <ErrorState error={windows.error} onRetry={() => windows.refetch()} /> : !windows.data ? <SkeletonRows rows={4} />
            : items.length === 0 ? <EmptyState title="No windows yet">{windows.data.message}</EmptyState> : (
              <ul className="space-y-1.5">
                {items.map((w) => (
                  <li key={w.id}>
                    <button type="button" onClick={() => setSel(w.id)} aria-pressed={current?.id === w.id}
                      className={clsx("w-full rounded-xl border px-3 py-2 text-left", current?.id === w.id ? "border-primary bg-primary-soft" : "border-transparent hover:bg-surface-muted")}>
                      <span className="flex items-center justify-between gap-2"><span className="font-semibold">{ago(w.created_at)}</span>
                        <Badge tone={w.status === "ok" ? "good" : w.status === "alert_candidate" ? "bad" : "warn"}>{humanize(w.status)}</Badge></span>
                      <span className="text-sm text-muted">{w.n_inspections} insp · {w.n_distinct_parents} parents</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
        </Card>
        <div className="min-w-0">{current ? <DriftWindowCard w={current} /> : !windows.isLoading && <EmptyState title="Select a window" />}</div>
      </div>
      <Card className="mt-6" title="Drift alerts" subtitle="Review states: open → acknowledged → investigating → resolved">
        {alerts.isError ? <ErrorState error={alerts.error} onRetry={() => alerts.refetch()} /> : !alerts.data ? <SkeletonRows rows={2} />
          : alerts.data.items.length === 0 ? <EmptyState title="No drift alerts" /> : (
            <ul className="space-y-3">{alerts.data.items.map((a) => <AlertReview key={a.id} alert={a} canReview={isMl} />)}</ul>
          )}
      </Card>
    </>
  );
}
