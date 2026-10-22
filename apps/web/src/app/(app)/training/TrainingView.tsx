"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import Link from "next/link";
import { useState } from "react";

import { EpochChart } from "@/components/domain/charts";
import { GateStatus, StatusBadge } from "@/components/domain/badges";
import { JobProgress } from "@/components/domain/JobProgress";
import { useMe } from "@/components/AppShell";
import { Badge, Button, Card, EmptyState, ErrorState, inputClass, KeyValue, PageHeader, SkeletonRows } from "@/components/ui";
import { api } from "@/lib/api";
import { dateTime, duration, pct, shortSha } from "@/lib/format";
import { pollWhile, qk } from "@/lib/query";
import { hasRole } from "@/lib/roles";
import type { DatasetsResponse, Recipe, TrainingRunView } from "@/lib/types";
import { useUrlState } from "@/lib/useUrlState";

const ACTIVE = new Set(["queued", "running"]);

function RunDetail({ id }: { id: string }) {
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: qk.trainingRun(id),
    queryFn: () => api<TrainingRunView>(`/v1/training-runs/${id}`),
    refetchInterval: pollWhile<TrainingRunView>((d) => !d || ACTIVE.has(d.status), 2000),
  });
  if (q.isError) return <ErrorState error={q.error} onRetry={() => q.refetch()} />;
  const r = q.data;
  if (!r) return <SkeletonRows rows={5} />;
  const m = (r.metrics ?? {}) as { summary?: Record<string, number | null>; gate_status?: string; duration_s?: number; epochs_completed?: number; registry?: { version?: string; status?: string } };
  return (
    <div className="space-y-5">
      {r.job && ACTIVE.has(r.status) && (
        <JobProgress jobId={r.job.job_id} label={`Training ${r.config_id}`} cancellable
          onDone={() => qc.invalidateQueries({ queryKey: qk.trainingRuns })} />
      )}
      <EpochChart epochs={r.epoch_metrics} />
      <KeyValue items={[
        ["Status", <StatusBadge key="s" status={r.status} />],
        ["Recipe", <span key="r">{r.config_id} <span className="mono text-muted">{shortSha(r.config_sha256, 8)}</span></span>],
        ["Snapshot", <span key="sn" className="mono">{shortSha(r.snapshot_id, 8)}</span>],
        ["Epochs", m.epochs_completed ?? r.epoch_metrics.length],
        ["Duration", m.duration_s ? duration(m.duration_s) : "n/a"],
        ["MLflow run", r.mlflow_run_id ? <span key="ml" className="mono">{r.mlflow_run_id}</span> : "not tracked"],
        ["Checkpoint", r.checkpoint_sha256 ? <span key="c" className="mono">{shortSha(r.checkpoint_sha256)}</span> : "n/a"],
        ["Gate (development)", m.summary ? `recall ${pct(m.summary.gate_defect_recall ?? null)} · specificity ${pct(m.summary.gate_specificity ?? null)}` : "n/a"],
        ["Locked test", m.summary ? `recall ${pct(m.summary.test_defect_recall ?? null)} · specificity ${pct(m.summary.test_specificity ?? null)}` : "n/a"],
        ["Started / finished", `${dateTime(r.started_at)} → ${dateTime(r.finished_at)}`],
      ]} />
      {r.model_version && (
        <div className="flex flex-wrap items-center gap-2 rounded-2xl bg-surface-muted p-3">
          <span className="font-semibold text-ink">Registered bundle</span>
          <span className="mono text-sm">{shortSha(r.model_version.bundle_sha256)}</span>
          <GateStatus status={r.model_version.gate_status} />
          <Badge>{r.model_version.stage}</Badge>
          <Link href={`/models?model=${r.model_version.id}`} className="ml-auto font-semibold text-primary-ink hover:underline">Evidence →</Link>
        </div>
      )}
      {r.error && <ErrorState error={new Error(r.error)} title={`Run ${r.status}`} />}
    </div>
  );
}

export function TrainingView() {
  const me = useMe();
  const qc = useQueryClient();
  const [{ run }, setUrl] = useUrlState(["run"] as const);
  const recipes = useQuery({ queryKey: qk.recipes, queryFn: () => api<{ items: Recipe[] }>("/v1/training/recipes") });
  const datasets = useQuery({ queryKey: qk.datasets, queryFn: () => api<DatasetsResponse>("/v1/datasets") });
  const runs = useQuery({
    queryKey: qk.trainingRuns,
    queryFn: () => api<{ items: TrainingRunView[] }>("/v1/training-runs"),
    refetchInterval: pollWhile<{ items: TrainingRunView[] }>((d) => Boolean(d?.items.some((r) => ACTIVE.has(r.status))), 3000),
  });
  const [recipe, setRecipe] = useState("");
  const [snapshot, setSnapshot] = useState("");
  const start = useMutation({
    mutationFn: () => api<{ training_run_id: string }>("/v1/training-runs", { method: "POST", json: { snapshot_id: snapshot, config_id: recipe } }),
    onSuccess: (r) => {
      setUrl({ run: r.training_run_id });
      qc.invalidateQueries({ queryKey: qk.trainingRuns });
    },
  });
  const canTrain = hasRole(me.data?.roles, "ml_engineer");
  const snaps = datasets.data?.snapshots ?? [];
  const selectedRun = run || runs.data?.items[0]?.id || "";

  return (
    <>
      <PageHeader eyebrow="Model training" title="Training"
        subtitle="Allowlisted recipes only — the API never accepts commands or paths. One training slot: a second request is rejected while a run is active." />
      <div className="grid gap-6 xl:grid-cols-[minmax(0,420px)_1fr]">
        <div className="space-y-6">
          <Card title="Start a run">
            {!canTrain ? <p className="text-muted">Starting training requires the ML engineer role.</p> : (
              <form className="space-y-3" onSubmit={(e) => { e.preventDefault(); start.mutate(); }}>
                <div className="flex flex-col gap-1">
                  <label htmlFor="t-snap" className="text-sm font-semibold text-ink-soft">Dataset snapshot</label>
                  <select id="t-snap" required className={inputClass} value={snapshot} onChange={(e) => setSnapshot(e.target.value)}>
                    <option value="">Select a snapshot…</option>
                    {snaps.map((s) => <option key={s.id} value={s.id}>{s.description || "snapshot"} · {s.counts.included} images · {shortSha(s.manifest_sha256, 8)}</option>)}
                  </select>
                </div>
                <div className="flex flex-col gap-1">
                  <label htmlFor="t-recipe" className="text-sm font-semibold text-ink-soft">Recipe</label>
                  <select id="t-recipe" required className={inputClass} value={recipe} onChange={(e) => setRecipe(e.target.value)}>
                    <option value="">Select a recipe…</option>
                    {recipes.data?.items.map((r) => <option key={r.id} value={r.id}>{r.id} · {r.model} · {r.epochs} epochs</option>)}
                  </select>
                </div>
                {start.error ? <ErrorState error={start.error} title="Run not started" /> : null}
                <Button type="submit" busy={start.isPending} disabled={!snapshot || !recipe}>Start training</Button>
              </form>
            )}
          </Card>
          <Card title="Recipes" subtitle="configs/recipes/*.yaml">
            {recipes.isError ? <ErrorState error={recipes.error} /> : !recipes.data ? <SkeletonRows rows={3} /> : (
              <ul className="space-y-2">
                {recipes.data.items.map((r) => (
                  <li key={r.id} className="rounded-xl bg-surface-muted px-3 py-2">
                    <p className="font-bold text-ink">{r.id} <span className="mono text-sm font-normal text-muted">{shortSha(r.sha256, 8)}</span></p>
                    <p className="text-sm text-ink-soft">{r.model} · {r.imgsz}px · batch {r.batch} · {r.epochs} epochs · {r.optimizer} lr {r.lr0} · patience {r.patience} · seed {r.seed}{r.fraction ? ` · fraction ${r.fraction}` : ""}</p>
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </div>
        <div className="space-y-6">
          <Card title="Runs">
            {runs.isError ? <ErrorState error={runs.error} onRetry={() => runs.refetch()} /> : !runs.data ? <SkeletonRows rows={3} />
              : runs.data.items.length === 0 ? <EmptyState title="No training runs yet">The baseline model was trained by the DVC pipeline; challengers trained here appear below.</EmptyState> : (
                <ul className="flex flex-wrap gap-2">
                  {runs.data.items.map((r) => (
                    <li key={r.id}>
                      <button type="button" onClick={() => setUrl({ run: r.id })} aria-pressed={selectedRun === r.id}
                        className={clsx("flex items-center gap-2 rounded-xl border px-3 py-2 text-left", selectedRun === r.id ? "border-primary bg-primary-soft" : "border-line hover:bg-surface-muted")}>
                        <span className="font-semibold">{r.config_id}</span><StatusBadge status={r.status} />
                        <span className="text-sm text-muted">{dateTime(r.created_at)}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
          </Card>
          {selectedRun && <Card title="Run detail"><RunDetail key={selectedRun} id={selectedRun} /></Card>}
        </div>
      </div>
    </>
  );
}
