"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Rocket, Undo2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { CalibrationChart, ConfusionMatrix } from "@/components/domain/charts";
import { GateStatus, StatusBadge } from "@/components/domain/badges";
import { JobProgress } from "@/components/domain/JobProgress";
import { ModelComparisonTable } from "@/components/domain/ModelComparisonTable";
import { useMe } from "@/components/AppShell";
import { Badge, Button, Card, EmptyState, ErrorState, inputClass, KeyValue, PageHeader, SkeletonRows } from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import { dateTime, gateLabel, humanize, ms, num, pct, shortSha } from "@/lib/format";
import { qk } from "@/lib/query";
import { hasRole } from "@/lib/roles";
import type { DeploymentsResponse, Evidence, ModelsResponse, ModelView, PairedDiff, Ratio, ReleaseRequested } from "@/lib/types";
import { useUrlState } from "@/lib/useUrlState";

function RatioCell({ r }: { r: Ratio | undefined }) {
  if (!r) return <>n/a</>;
  if (r.value === null) return <span className="text-warn">insufficient ({r.denominator})</span>;
  return <span className="mono">{pct(r.value)} <span className="text-muted">({r.numerator}/{r.denominator}{r.ci95 ? `, CI ${pct(r.ci95[0])}–${pct(r.ci95[1])}` : ""})</span></span>;
}

function Diff({ d }: { d: PairedDiff | undefined }) {
  if (!d || d.status !== "ok" || d.difference === undefined) return <span className="text-muted">n/a</span>;
  return <span className="mono">{d.difference >= 0 ? "+" : ""}{(d.difference * 100).toFixed(1)} pp{d.ci95 ? ` [${(d.ci95[0] * 100).toFixed(1)}, ${(d.ci95[1] * 100).toFixed(1)}]` : ""}</span>;
}

function ReleaseDialog({ model, action, generation, open, onClose }: {
  model: ModelView;
  action: "promote" | "rollback";
  generation: number;
  open: boolean;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const qc = useQueryClient();
  const [reason, setReason] = useState("");
  const [job, setJob] = useState<string | null>(null);
  useEffect(() => {
    if (open && !ref.current?.open) ref.current?.showModal();
    if (!open && ref.current?.open) ref.current.close();
  }, [open]);
  const m = useMutation({
    mutationFn: () => api<ReleaseRequested>(`/v1/deployments/${action}`, {
      method: "POST", json: { model_version_id: model.id, expected_generation: generation, reason },
    }),
    onSuccess: (r) => setJob(r.job_id),
  });
  const blocked = model.gate_status !== "passed";
  return (
    <dialog ref={ref} onClose={onClose} aria-labelledby="release-title"
      className="m-auto w-[min(560px,92vw)] rounded-2xl border border-line p-0 shadow-2xl backdrop:bg-ink/40">
      <div className="space-y-4 p-6">
        <h2 id="release-title" className="text-xl font-bold text-ink">{action === "promote" ? "Promote" : "Roll back to"} {model.model_family} v{model.registry_version ?? "?"}</h2>
        <KeyValue items={[
          ["Bundle", <span key="b" className="mono">{shortSha(model.bundle_sha256, 16)}</span>],
          ["Gates", <GateStatus key="g" status={model.gate_status} />],
          ["Expected generation", <span key="e" className="mono">{generation} → {generation + 1}</span>],
        ]} />
        {blocked ? (
          <p role="alert" className="rounded-xl bg-bad-soft p-3 text-ink">
            Blocked: gates are <strong>{gateLabel(model.gate_status)}</strong>. A failing or insufficient bundle can never be
            {action === "promote" ? " promoted" : " restored"}.
          </p>
        ) : job ? (
          <JobProgress jobId={job} label={`${humanize(action)} (stage → warm → swap)`} onDone={() => {
            qc.invalidateQueries({ queryKey: qk.models });
            qc.invalidateQueries({ queryKey: qk.deployments });
          }} />
        ) : (
          <form id="release-form" onSubmit={(e) => { e.preventDefault(); if (reason.trim().length >= 8) m.mutate(); }} className="space-y-2">
            <label htmlFor="release-reason" className="text-sm font-semibold text-ink-soft">Approval reason (min. 8 characters, recorded in the audit log)</label>
            <textarea id="release-reason" required minLength={8} maxLength={1000} rows={3} className={inputClass}
              value={reason} onChange={(e) => setReason(e.target.value)} />
            {m.error ? <ErrorState error={m.error} title={m.error instanceof ApiError && m.error.code === "generation_conflict" ? "Serving deployment changed — reload and retry" : "Release not started"} /> : null}
          </form>
        )}
        <div className="flex justify-end gap-2">
          <Button variant="secondary" onClick={onClose}>{job ? "Close" : "Cancel"}</Button>
          {!blocked && !job && (
            <Button type="submit" form="release-form" busy={m.isPending} disabled={reason.trim().length < 8}
              variant={action === "rollback" ? "danger" : "primary"}>
              {action === "promote" ? "Approve and promote" : "Approve rollback"}
            </Button>
          )}
        </div>
      </div>
    </dialog>
  );
}

function ShadowPanel({ model, canRun, evidenceShadow }: { model: ModelView; canRun: boolean; evidenceShadow: Evidence["shadow"] }) {
  const qc = useQueryClient();
  const [job, setJob] = useState<string | null>(null);
  const m = useMutation({
    mutationFn: () => api<{ job_id: string }>(`/v1/models/${model.id}/shadow`, { method: "POST", json: { max_images: 120 } }),
    onSuccess: (r) => setJob(r.job_id),
  });
  const report = evidenceShadow?.report?.scenarios;
  return (
    <div className="space-y-3">
      {canRun && !model.is_active && (
        <div className="flex flex-wrap items-center gap-2">
          <Button variant="secondary" onClick={() => m.mutate()} busy={m.isPending}>Run paired shadow evaluation</Button>
          <span className="text-sm text-muted">Champion and challenger see identical tensors; nothing becomes user-visible.</span>
        </div>
      )}
      {m.error ? <ErrorState error={m.error} title="Shadow not started" /> : null}
      {job && <JobProgress jobId={job} label="Shadow evaluation" onDone={() => qc.invalidateQueries({ queryKey: qk.evidence(model.id) })} />}
      {report ? (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[720px] text-left text-sm">
            <caption className="mb-1 text-left text-muted">{evidenceShadow?.report?.cohort} · {evidenceShadow?.rows} paired rows · job <StatusBadge status={evidenceShadow?.status} /></caption>
            <thead><tr className="border-b border-line text-muted">
              {["Scenario", "n", "Triage agreement", "Mean |Δp|", "Champion recall", "Challenger recall", "Δ recall (95% CI)", "Δ specificity"].map((h) => <th key={h} scope="col" className="py-1.5 pr-3">{h}</th>)}
            </tr></thead>
            <tbody>{Object.entries(report).map(([k, s]) => (
              <tr key={k} className="border-b border-line/60">
                <td className="py-1.5 pr-3 font-semibold">{k}</td><td className="mono">{s.n}</td><td className="mono">{pct(s.triage_agreement)}</td>
                <td className="mono">{s.mean_abs_p_delta.toFixed(4)}</td><td className="mono">{pct(s.champion.defect_recall.value)}</td>
                <td className="mono">{pct(s.challenger.defect_recall.value)}</td><td><Diff d={s.recall_difference} /></td><td><Diff d={s.specificity_difference} /></td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      ) : !job && <p className="text-muted">{model.is_active ? "The active model is the champion; shadow a challenger against it." : "No shadow report yet."}</p>}
    </div>
  );
}

function EvidencePanel({ model, generation, isAdmin, isMl }: { model: ModelView; generation: number; isAdmin: boolean; isMl: boolean }) {
  const q = useQuery({ queryKey: qk.evidence(model.id), queryFn: () => api<Evidence>(`/v1/models/${model.id}/evidence`) });
  const [dialog, setDialog] = useState<"promote" | "rollback" | null>(null);
  if (q.isError) return <ErrorState error={q.error} onRetry={() => q.refetch()} />;
  const e = q.data;
  if (!e) return <SkeletonRows rows={8} />;
  const g = e.gate_report;
  const previouslyServed = model.deployments.length > 0;
  return (
    <div className="space-y-6">
      <Card title={`Evidence · ${model.model_family} v${model.registry_version ?? "?"}`}
        subtitle={g.evidence_label}
        actions={<>
          <GateStatus status={g.status} label="Overall" />
          {isAdmin && !model.is_active && (previouslyServed
            ? <Button variant="danger" onClick={() => setDialog("rollback")}><Undo2 aria-hidden className="h-4 w-4" /> Roll back to this</Button>
            : <Button onClick={() => setDialog("promote")} disabled={g.status !== "passed"} title={g.status !== "passed" ? `Blocked: gates ${gateLabel(g.status)}` : undefined}>
              <Rocket aria-hidden className="h-4 w-4" /> Promote</Button>)}
        </>}>
        {isAdmin && !model.is_active && g.status !== "passed" && (
          <p role="note" className="mb-4 rounded-xl bg-warn-soft p-3 text-ink">Promotion is blocked: overall gate status is <strong>{gateLabel(g.status)}</strong>.</p>
        )}
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="min-w-0">
            <h3 className="mb-2 font-bold text-ink">Hard gates</h3>
            <ul className="space-y-2">
              {g.hard_gates.map((h) => (
                <li key={h.id} className="flex min-w-0 flex-wrap items-center justify-between gap-2 rounded-xl bg-surface-muted px-3 py-2">
                  <span className="font-semibold">{humanize(h.id)}</span><GateStatus status={h.status} />
                  {h.detail !== undefined && <span className="mono block w-full min-w-0 max-w-full truncate text-xs text-muted" title={JSON.stringify(h.detail)}>{typeof h.detail === "string" ? h.detail : JSON.stringify(h.detail)}</span>}
                </li>
              ))}
            </ul>
          </div>
          <div className="min-w-0">
            <h3 className="mb-2 font-bold text-ink">Quality targets</h3>
            <ul className="space-y-2">
              {g.quality_targets.map((t) => (
                <li key={t.id} className="rounded-xl bg-surface-muted px-3 py-2">
                  <div className="flex items-center justify-between gap-2"><span className="font-semibold">{humanize(t.id)} ≥ {pct(t.target, 0)}</span><GateStatus status={t.status} /></div>
                  <p className="mono text-sm text-ink-soft">{pct(t.value)} ({t.numerator ?? "?"}/{t.support}){t.ci95 ? ` · 95% CI ${pct(t.ci95[0])}–${pct(t.ci95[1])}` : ""}</p>
                </li>
              ))}
              <li className="rounded-xl bg-surface-muted px-3 py-2">
                <div className="flex items-center justify-between gap-2"><span className="font-semibold">Comparative bound vs champion</span><GateStatus status={g.comparative?.status} /></div>
                {g.comparative?.reason ? <p className="text-sm text-muted">{String(g.comparative.reason)}</p> : null}
              </li>
            </ul>
          </div>
        </div>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Gate cohort (development)" subtitle={e.gate_cohort ? `${e.gate_cohort.binary.support.total} images · ${e.gate_cohort.parents} parents` : undefined}>
          {e.gate_cohort ? (<>
            <ConfusionMatrix m={e.gate_cohort.binary} />
            <KeyValue className="mt-3" items={[["Defect recall", <RatioCell key="r" r={e.gate_cohort.binary.defect_recall} />], ["Specificity", <RatioCell key="s" r={e.gate_cohort.binary.specificity} />],
              ["Average precision", num(e.gate_cohort.probabilistic.average_precision, 3)], ["ECE", num(e.gate_cohort.probabilistic.ece, 3)]]} />
          </>) : <EmptyState title="No gate cohort metrics" />}
        </Card>
        <Card title="Locked final test" subtitle={e.locked_test?.evidence_label}>
          {e.locked_test?.binary ? (<>
            <ConfusionMatrix m={e.locked_test.binary} />
            <KeyValue className="mt-3" items={[["Defect recall", <RatioCell key="r" r={e.locked_test.binary.defect_recall} />], ["Specificity", <RatioCell key="s" r={e.locked_test.binary.specificity} />],
              ["Precision", <RatioCell key="p" r={e.locked_test.binary.precision} />], ["AP / Brier / NLL", `${num(e.locked_test.probabilistic.average_precision, 3)} / ${num(e.locked_test.probabilistic.brier, 4)} / ${num(e.locked_test.probabilistic.nll, 4)}`],
              ["False negatives → retread candidate", e.locked_test.triage?.false_negatives_routed_to_good_candidate ?? "n/a"]]} />
          </>) : <EmptyState title="Locked test not run for this bundle" />}
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Calibration" subtitle={e.calibration ? `temperature T=${e.calibration.temperature.toFixed(3)} fitted on ${e.calibration.n_fitted} ${e.calibration.fitted_on_role}-role images` : undefined}>
          <CalibrationChart bins={e.locked_test?.probabilistic?.reliability ?? e.gate_cohort?.probabilistic?.reliability} title={e.locked_test?.probabilistic?.reliability ? "Locked-test reliability" : "Gate-cohort reliability"} />
          {e.calibration && <p className="mt-2 text-sm text-muted">Calibration-role NLL {num(e.calibration.nll_before, 4)} → {num(e.calibration.nll_after, 4)}</p>}
        </Card>
        <Card title="Policy" subtitle={e.policy ? `policy ${e.policy.version}` : undefined}>
          {e.policy ? (
            <KeyValue items={[
              ["Binary evaluation threshold", <span key="b" className="mono">{e.policy.binary_threshold.toFixed(4)}</span>],
              ["Selection", e.policy.selection ? `${String(e.policy.selection.rule ?? "")} on ${String(e.policy.selection.selected_on_role ?? "policy")} role` : "n/a"],
              ["Triage", <span key="t" className="mono">good &lt; {e.policy.triage_low} ≤ review &lt; {e.policy.triage_high} ≤ defect</span>],
              ["Final clearance", "human inspector"],
            ]} />
          ) : <p className="text-muted">No policy file.</p>}
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Stress robustness (paired test parents)" subtitle={e.stress?.pairing}>
          {e.stress?.scenarios ? (
            <div className="overflow-x-auto"><table className="w-full text-left text-sm">
              <thead><tr className="border-b border-line text-muted"><th scope="col" className="py-1.5">Scenario</th><th scope="col">Recall</th><th scope="col">Specificity</th><th scope="col">Δ recall vs clean</th><th scope="col">Low-quality share</th></tr></thead>
              <tbody>
                {e.stress.clean && <tr className="border-b border-line/60"><td className="py-1.5 font-semibold">clean</td><td className="mono">{pct(e.stress.clean.defect_recall.value)}</td><td className="mono">{pct(e.stress.clean.specificity.value)}</td><td>—</td><td>—</td></tr>}
                {Object.entries(e.stress.scenarios).map(([k, s]) => (
                  <tr key={k} className="border-b border-line/60"><td className="py-1.5 font-semibold">{k}</td><td className="mono">{pct(s.binary.defect_recall.value)}</td>
                    <td className="mono">{pct(s.binary.specificity.value)}</td><td><Diff d={s.recall_vs_clean} /></td><td className="mono">{pct(s.quality_low_fraction)}</td></tr>
                ))}
              </tbody>
            </table></div>
          ) : <EmptyState title="No stress report" />}
        </Card>
        <Card title="Serving checks">
          <KeyValue items={[
            ["ONNX parity", e.parity ? <span key="p"><GateStatus status={e.parity.status === "passed" ? "passed" : "failed"} /> <span className="mono text-sm">mean {e.parity.mean_abs_probability_error.toExponential(2)} · max {e.parity.max_abs_probability_error.toExponential(2)} · triage agreement {pct(e.parity.triage_agreement)} (n={e.parity.n_images})</span></span> : "not exported"],
            ["CPU latency (model only)", e.latency?.model_only ? `p50 ${ms(e.latency.model_only.p50_ms)} · p95 ${ms(e.latency.model_only.p95_ms)} (n=${e.latency.model_only.n}, warm, batch 1)` : "n/a"],
            ["Preprocess", e.latency?.preprocess ? `p95 ${ms(e.latency.preprocess.p95_ms)}` : "n/a"],
            ["Hardware", e.latency?.hardware ? String((e.latency.hardware as Record<string, unknown>).cpu_brand ?? (e.latency.hardware as Record<string, unknown>).machine ?? "") : "n/a"],
            ["Training", e.training ? `${String(e.training.epochs_completed ?? "?")} epochs on ${String(e.training.device ?? "?")} · ${String(e.training.fitted_image_count ?? "?")} fitted images` : "n/a"],
          ]} />
        </Card>
      </div>

      <Card title="Shadow evaluation"><ShadowPanel model={model} canRun={isMl} evidenceShadow={e.shadow} /></Card>
      <Card title="Model card">
        {e.model_card ? <pre className="max-h-[480px] overflow-auto whitespace-pre-wrap rounded-xl bg-surface-muted p-4 text-sm leading-relaxed text-ink-soft">{e.model_card}</pre> : <p className="text-muted">No model card.</p>}
      </Card>
      {dialog && <ReleaseDialog model={model} action={dialog} generation={generation} open onClose={() => setDialog(null)} />}
    </div>
  );
}

export function ModelsView() {
  const me = useMe();
  const [{ model }, setUrl] = useUrlState(["model"] as const);
  const q = useQuery({ queryKey: qk.models, queryFn: () => api<ModelsResponse>("/v1/models"), refetchInterval: 15_000 });
  const deps = useQuery({ queryKey: qk.deployments, queryFn: () => api<DeploymentsResponse>("/v1/deployments"), refetchInterval: 15_000 });
  if (q.isError) return <><PageHeader eyebrow="Model lifecycle" title="Models" /><ErrorState error={q.error} onRetry={() => q.refetch()} /></>;
  const d = q.data;
  if (!d) return <SkeletonRows rows={8} />;
  const selected = d.items.find((m) => m.id === model) ?? d.challenger ?? d.active ?? d.items[0];
  const isAdmin = hasRole(me.data?.roles, "admin");
  return (
    <>
      <PageHeader eyebrow="Model lifecycle" title="Models"
        subtitle="Candidate → validated → shadow → approved → active. Promotion requires passed gates, an admin reason, a successful warm-up and the expected generation."
        actions={<Badge tone="primary">serving generation {d.generation}</Badge>} />
      {d.items.length === 0 ? (
        <EmptyState title="No model versions registered">{d.message ?? "No trained model available."} Run the DVC pipeline and import the bundle, or start a training run.</EmptyState>
      ) : (
        <div className="space-y-6">
          {!d.active && <p role="status" className="rounded-2xl bg-warn-soft p-4 font-semibold text-ink">No trained model available for serving yet — promote a validated bundle below.</p>}
          <Card title="Champion and challengers"><ModelComparisonTable items={d.items} selectedId={selected?.id} onSelect={(id) => setUrl({ model: id })} /></Card>
          {selected && <EvidencePanel key={selected.id} model={selected} generation={d.generation} isAdmin={isAdmin} isMl={hasRole(me.data?.roles, "ml_engineer")} />}
        </div>
      )}
      <Card className="mt-6" title="Deployment history" subtitle="The application deployment table is the authoritative serving pointer">
        {deps.isError ? <ErrorState error={deps.error} onRetry={() => deps.refetch()} /> : !deps.data ? <SkeletonRows rows={3} />
          : deps.data.items.length === 0 ? <EmptyState title="Nothing deployed yet" /> : (
            <ol className="space-y-2">
              {deps.data.items.map((x) => (
                <li key={x.deployment_id} className="flex flex-wrap items-center gap-2 rounded-xl bg-surface-muted px-3 py-2">
                  <Badge tone={x.state === "active" ? "good" : "neutral"}>gen {x.generation}</Badge>
                  <span className="font-semibold">{humanize(x.action)}</span>
                  <span>v{x.registry_version ?? "?"} <span className="mono text-sm text-muted">{shortSha(x.bundle_sha256)}</span></span>
                  <span className="text-ink-soft">“{x.reason}”</span>
                  <span className="ml-auto text-sm text-muted">{x.approved_by ?? "system"} · {dateTime(x.created_at)}</span>
                </li>
              ))}
            </ol>
          )}
      </Card>
    </>
  );
}
