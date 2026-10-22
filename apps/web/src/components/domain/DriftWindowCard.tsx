"use client";

import { PsiChart } from "@/components/domain/charts";
import { Badge, Card, KeyValue, type Tone } from "@/components/ui";
import { dateTime, humanize, ms, num, pct, shortSha } from "@/lib/format";
import type { MonitoringWindow } from "@/lib/types";

const TONE: Record<string, Tone> = { ok: "good", alert_candidate: "bad", insufficient_data: "warn" };

export function DriftWindowCard({ w, psiThreshold = 0.2 }: { w: MonitoringWindow; psiThreshold?: number }) {
  const lq = w.labelled_quality ?? {};
  const emb = w.embedding_drift ?? {};
  return (
    <Card title={`Window · ${dateTime(w.created_at)}`}
      subtitle={`${w.n_inspections} inspections · ${w.n_distinct_parents} distinct parents · reference ${shortSha(w.reference_version, 10)} · cohort ${shortSha(w.cohort_sha256, 10)}`}
      actions={<Badge tone={TONE[w.status] ?? "neutral"}>{humanize(w.status)}</Badge>}>
      {w.status === "insufficient_data" && (
        <p role="status" className="mb-4 rounded-xl bg-warn-soft p-3 text-ink">
          Insufficient data: fewer than the required distinct parent images, so this window cannot be called healthy.
          Metrics below are shown for investigation only.
        </p>
      )}
      <div className="grid gap-6 lg:grid-cols-2">
        <div>
          <h3 className="mb-2 font-bold text-ink">Input shift: descriptor PSI</h3>
          <PsiChart psi={w.descriptor_psi} threshold={psiThreshold} />
        </div>
        <div className="space-y-4">
          <div>
            <h3 className="mb-2 font-bold text-ink">Embedding drift (fixed reference encoder)</h3>
            <KeyValue items={[
              ["Domain-classifier AUC", emb.auc === null || emb.auc === undefined ? humanize(emb.status ?? "n/a") : <span key="a" className="mono">{emb.auc.toFixed(3)}</span>],
              ["Mean PC PSI", emb.pc_psi_mean === undefined ? "n/a" : emb.pc_psi_mean.toFixed(3)],
              ["Note", emb.note ?? "0.5 = indistinguishable from reference"],
            ]} />
          </div>
          <div>
            <h3 className="mb-2 font-bold text-ink">Output shift</h3>
            <KeyValue items={[
              ["Current triage mix", Object.entries(w.prediction_shift.current_triage_fraction ?? {}).map(([k, v]) => `${humanize(k)} ${pct(v)}`).join(" · ") || "n/a"],
              ["Reference triage mix", Object.entries(w.prediction_shift.reference_triage_fraction ?? {}).map(([k, v]) => `${humanize(k)} ${pct(v ?? null)}`).join(" · ") || "n/a"],
              ["Mean p(defective)", num(w.prediction_shift.current_mean_p_defective, 3)],
            ]} />
            {w.prediction_shift.note && <p className="mt-1 text-sm text-muted">{w.prediction_shift.note}.</p>}
          </div>
        </div>
      </div>
      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        <div>
          <h3 className="mb-2 font-bold text-ink">Labelled quality</h3>
          <KeyValue items={[
            ["Status", humanize(lq.status ?? "n/a")],
            ["Coverage", lq.coverage === undefined ? "n/a" : `${pct(lq.coverage)} (${lq.labelled ?? 0} labels)`],
            ["Recall", lq.defect_recall?.value === undefined || lq.defect_recall?.value === null ? "n/a" : `${pct(lq.defect_recall.value)} (${lq.defect_recall.numerator}/${lq.defect_recall.denominator})`],
            ["Specificity", lq.specificity?.value === undefined || lq.specificity?.value === null ? "n/a" : `${pct(lq.specificity.value)} (${lq.specificity.numerator}/${lq.specificity.denominator})`],
            ["Median label age", lq.median_label_age_h === undefined ? "n/a" : `${lq.median_label_age_h.toFixed(1)} h`],
            ["Disagreement", lq.disagreement_rate === undefined ? "n/a" : pct(lq.disagreement_rate)],
          ]} />
        </div>
        <div>
          <h3 className="mb-2 font-bold text-ink">System health</h3>
          <KeyValue items={[
            ["Queue p95", ms(w.system_health.queue_ms?.p95_ms)],
            ["Worker total p95", ms(w.system_health.worker_total_ms?.p95_ms)],
            ["Failed jobs", w.system_health.failed_jobs ?? "n/a"],
            ["Retries", w.system_health.retries ?? "n/a"],
          ]} />
        </div>
        <div>
          <h3 className="mb-2 font-bold text-ink">Evidently (version-pinned)</h3>
          {w.evidently?.status === "ok" ? (
            <KeyValue items={[
              ["Drifted columns", w.evidently.drifted_columns ? `${w.evidently.drifted_columns.count} (${pct(w.evidently.drifted_columns.share)})` : "n/a"],
              ...Object.entries(w.evidently.per_column_drift_score ?? {}).map(([k, v]) => [`${k} K-S p`, v.toExponential(2)] as [string, string]),
            ]} />
          ) : <p className="text-muted">{humanize(w.evidently?.status ?? "not run")}{w.evidently?.error ? `: ${w.evidently.error}` : ""}{w.evidently?.reason ? `: ${w.evidently.reason}` : ""}</p>}
        </div>
      </div>
      {(w.breakdown.by_camera?.length ?? 0) > 0 && (
        <div className="mt-6 overflow-x-auto">
          <table className="w-full min-w-[520px] text-left text-sm">
            <caption className="mb-1 text-left font-bold text-ink">By camera</caption>
            <thead><tr className="border-b border-line text-muted"><th scope="col" className="py-1">Camera</th><th scope="col">n</th><th scope="col">Mean brightness</th><th scope="col">Mean p(defective)</th></tr></thead>
            <tbody>{w.breakdown.by_camera!.map((r) => (
              <tr key={String(r.camera_id)} className="border-b border-line/60"><td className="py-1">{String(r.camera_id)}</td><td className="mono">{String(r.n)}</td>
                <td className="mono">{Number(r.mean_brightness).toFixed(3)}</td><td className="mono">{Number(r.mean_p).toFixed(3)}</td></tr>
            ))}</tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
