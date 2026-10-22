"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { FlaskConical } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

import { JobProgress } from "@/components/domain/JobProgress";
import { SourceBadge, StatusBadge } from "@/components/domain/badges";
import { useMe } from "@/components/AppShell";
import { Badge, Button, Card, EmptyState, ErrorState, inputClass, PageHeader, ProgressBar, SkeletonRows } from "@/components/ui";
import { api } from "@/lib/api";
import { ago, humanize } from "@/lib/format";
import { pollWhile, qk } from "@/lib/query";
import { hasRole } from "@/lib/roles";
import type { ReplayItem, ReplayScenarios } from "@/lib/types";

const DESCRIBE: Record<string, string> = {
  baseline: "Clean pool, stable line/camera mix — the reference behaviour.",
  low_light: "After the start point, a share of inputs is darkened (brightness factor).",
  camera_blur: "A recorded subset gets Gaussian blur; labels and parent IDs are preserved.",
  class_prior_shift: "The schedule samples more defective casings — a prior shift, not automatically a failure.",
  burst_and_outage: "Burst traffic plus duplicate messages; pause/restart workers during the run as the operator drill.",
};

export function ReplayView() {
  const me = useMe();
  const qc = useQueryClient();
  const scenarios = useQuery({ queryKey: qk.scenarios, queryFn: () => api<ReplayScenarios>("/v1/replays/scenarios") });
  const list = useQuery({
    queryKey: qk.replays,
    queryFn: () => api<{ items: ReplayItem[] }>("/v1/replays"),
    refetchInterval: pollWhile<{ items: ReplayItem[] }>((d) => Boolean(d?.items.some((r) => ["queued", "running"].includes(r.status))), 2000),
  });
  const [scenario, setScenario] = useState("low_light");
  const [seed, setSeed] = useState<number | "">("");
  const [rate, setRate] = useState<number | "">("");
  const [count, setCount] = useState<number | "">("");
  const [job, setJob] = useState<string | null>(null);
  const defaults = scenarios.data?.defaults;
  const start = useMutation({
    mutationFn: () => api<{ job_id: string }>("/v1/replays", {
      method: "POST",
      json: { scenario, seed: seed === "" ? defaults?.seed : seed, rate: rate === "" ? defaults?.rate : rate, count: count === "" ? defaults?.count : count },
    }),
    onSuccess: (r) => { setJob(r.job_id); qc.invalidateQueries({ queryKey: qk.replays }); },
  });
  const canRun = hasRole(me.data?.roles, "ml_engineer");
  return (
    <>
      <PageHeader eyebrow="Simulation" title="Operational replay"
        subtitle="Factory-like traffic built from real public photos and simulated line/camera/timestamp metadata. An integration and robustness tool — not evidence of an industrial deployment." />
      <div role="note" className="mb-6 flex items-start gap-3 rounded-2xl border-2 border-sim/40 bg-sim-soft p-4">
        <FlaskConical aria-hidden className="mt-0.5 h-6 w-6 text-sim" />
        <div>
          <p className="font-black tracking-wide text-sim">SIMULATED METADATA</p>
          <p className="text-ink-soft">{scenarios.data?.disclosure ?? "Replays use real public photographs with simulated operational metadata. They are never training-eligible."}</p>
        </div>
      </div>
      {scenarios.isError ? <ErrorState error={scenarios.error} onRetry={() => scenarios.refetch()} /> : !scenarios.data ? <SkeletonRows rows={3} /> : (
        <Card title="Start a replay" subtitle={`Defaults: ${defaults?.count} events at ${defaults?.rate}/s, seed ${defaults?.seed}. Throughput is measured, not assumed.`}>
          <fieldset>
            <legend className="mb-2 text-sm font-semibold text-ink-soft">Scenario</legend>
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
              {scenarios.data.items.map((s) => {
                const params = Object.entries(s).filter(([k]) => !["id", "version"].includes(k));
                return (
                  <label key={s.id} className={clsx("cursor-pointer rounded-2xl border-2 p-4 transition",
                    scenario === s.id ? "border-primary bg-primary-soft" : "border-line hover:border-line-strong")}>
                    <input type="radio" name="scenario" value={s.id} checked={scenario === s.id} onChange={() => setScenario(s.id)} className="sr-only" />
                    <span className="block font-bold text-ink">{humanize(s.id)} <span className="text-sm font-normal text-muted">{s.version}</span></span>
                    <span className="mt-1 block text-sm text-ink-soft">{DESCRIBE[s.id] ?? ""}</span>
                    {params.length > 0 && <span className="mono mt-2 block text-xs text-muted">{params.map(([k, v]) => `${k}=${String(v)}`).join(" · ")}</span>}
                  </label>
                );
              })}
            </div>
          </fieldset>
          <form className="mt-4 grid gap-3 sm:grid-cols-4" onSubmit={(e) => { e.preventDefault(); start.mutate(); }}>
            {([["Seed", seed, setSeed, defaults?.seed, 0, 2 ** 31], ["Rate (events/s)", rate, setRate, defaults?.rate, 0.1, 20], ["Count", count, setCount, defaults?.count, 1, 5000]] as const).map(([label, v, set, def, min, max]) => (
              <div key={label} className="flex flex-col gap-1">
                <label htmlFor={`rp-${label}`} className="text-sm font-semibold text-ink-soft">{label}</label>
                <input id={`rp-${label}`} type="number" step="any" min={min} max={max} className={inputClass} placeholder={String(def ?? "")}
                  value={v} onChange={(e) => (set as (x: number | "") => void)(e.target.value === "" ? "" : Number(e.target.value))} />
              </div>
            ))}
            <Button type="submit" busy={start.isPending} disabled={!canRun} className="self-end">Start replay</Button>
            {!canRun && <p className="text-sm text-muted sm:col-span-4">Starting replays requires the ML engineer role.</p>}
            {start.error ? <div className="sm:col-span-4"><ErrorState error={start.error} title="Replay not started" /></div> : null}
          </form>
          {job && <div className="mt-4"><JobProgress jobId={job} label={`Replay ${humanize(scenario)}`} cancellable onDone={() => qc.invalidateQueries({ queryKey: qk.replays })} /></div>}
        </Card>
      )}
      <Card className="mt-6" title="Replay history">
        {list.isError ? <ErrorState error={list.error} onRetry={() => list.refetch()} /> : !list.data ? <SkeletonRows rows={3} />
          : list.data.items.length === 0 ? <EmptyState title="No replays yet" /> : (
            <ul className="space-y-3">
              {list.data.items.map((r) => {
                const res = (r.result ?? {}) as Record<string, number | string | null>;
                return (
                  <li key={r.job_id} className="rounded-2xl border border-line p-4">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-bold text-ink">{humanize(r.params.scenario)}</span>
                      <StatusBadge status={r.status} /><SourceBadge origin="simulated" />
                      <span className="text-sm text-muted">seed {r.params.seed} · {r.params.rate}/s · {r.params.count} events · {ago(r.created_at)}</span>
                      <Link href={`/inspections?source=replay`} className="ml-auto text-sm font-semibold text-primary-ink hover:underline">Inspections →</Link>
                    </div>
                    {["queued", "running"].includes(r.status) && <div className="mt-2"><ProgressBar value={r.progress} label="Replay progress" /><p className="mt-1 text-sm text-muted">{r.progress_message}</p></div>}
                    <div className="mt-2 flex flex-wrap gap-2 text-sm">
                      <Badge>{r.inspections} inspections</Badge><Badge tone="good">{r.completed} completed</Badge>
                      {res.accepted !== undefined && <Badge tone="primary">{String(res.accepted)} accepted</Badge>}
                      {res.deduplicated !== undefined && <Badge tone="info">{String(res.deduplicated)} deduplicated</Badge>}
                      {res.rejected !== undefined && <Badge tone={Number(res.rejected) ? "bad" : "neutral"}>{String(res.rejected)} rejected</Badge>}
                      {res.backpressure_waits !== undefined && <Badge tone="warn">{String(res.backpressure_waits)} backpressure waits</Badge>}
                      {res.measured_rate_per_s !== undefined && <Badge>measured {String(res.measured_rate_per_s)}/s over {String(res.elapsed_s)} s</Badge>}
                    </div>
                    {r.error && <p className="mt-2 text-bad">{r.error}</p>}
                  </li>
                );
              })}
            </ul>
          )}
      </Card>
    </>
  );
}
