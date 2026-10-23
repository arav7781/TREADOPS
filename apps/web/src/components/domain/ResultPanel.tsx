"use client";

import { useQuery } from "@tanstack/react-query";
import { Leaf, Recycle, Search } from "lucide-react";
import Link from "next/link";

import { QualityBadge, SourceBadge, TriageBadge } from "@/components/domain/badges";
import { LineageDrawer } from "@/components/domain/LineageDrawer";
import { ProbabilityPanel } from "@/components/domain/ProbabilityPanel";
import { SecondOpinionPanel } from "@/components/domain/SecondOpinion";
import { Badge, Card, KeyValue } from "@/components/ui";
import { api } from "@/lib/api";
import { ms, shortSha } from "@/lib/format";
import { qk } from "@/lib/query";
import type { Evidence, InspectionDetail } from "@/lib/types";

const PATHWAY_ICON = { retread_intake: Recycle, expert_inspection: Search, material_recovery: Leaf } as const;

/** Decision + evidence for one inspection: triage, probabilities, quality, circular pathway and lineage. */
export function ResultPanel({ d }: { d: InspectionDetail }) {
  const p = d.prediction;
  const mvId = d.deployment.model_version_id;
  // Thresholds come from the pinned model version's policy, never from constants in the UI.
  const evidence = useQuery({
    queryKey: qk.evidence(mvId ?? "none"),
    queryFn: () => api<Evidence>(`/v1/models/${mvId}/evidence`),
    enabled: Boolean(mvId && p),
    staleTime: 300_000,
  });
  if (!p) return null;
  const policy = evidence.data?.policy;
  const inference = d.stages.find((s) => s.name === "inference")?.params as Record<string, unknown> | undefined;
  const circ = p.circularity;
  const Icon = circ ? PATHWAY_ICON[circ.pathway] ?? Recycle : Recycle;
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card title="Decision" subtitle={p.triage_reason}
        actions={<><TriageBadge triage={p.triage} large /><SourceBadge origin={d.metadata_origin} /></>}>
        <ProbabilityPanel raw={p.raw_probs} calibrated={p.calibrated_probs} pDefective={p.p_defective}
          temperature={typeof inference?.temperature === "number" ? inference.temperature : null}
          thresholds={policy ? { low: policy.triage_low, high: policy.triage_high, binary: policy.binary_threshold } : undefined} />
        {p.second_opinion && <SecondOpinionPanel s={p.second_opinion} className="mt-4" />}
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <QualityBadge status={p.quality_status} reasons={p.quality.reasons} />
          <Badge tone="neutral">binary @ {policy ? `threshold ${policy.binary_threshold.toFixed(3)}` : "policy threshold"}: {p.binary_defective ? "defective" : "good"}</Badge>
        </div>
        <KeyValue className="mt-4" items={[
          ["Queue wait", ms(p.timing.queue_ms)],
          ["Preprocess", ms(p.timing.preprocess_ms)],
          ["Model inference", ms(p.timing.inference_ms)],
          ["Attribution", ms(p.timing.attribution_ms)],
          ["Worker total", ms(p.timing.worker_total_ms)],
        ]} />
      </Card>
      <div className="flex flex-col gap-6">
        {circ && (
          <Card title="Circular pathway" subtitle="Model recommendation — an inspector decides the disposition">
            <div className="flex items-start gap-3">
              <span aria-hidden className="rounded-xl bg-accent-soft p-2.5 text-accent-ink"><Icon className="h-6 w-6" /></span>
              <div>
                <p className="text-lg font-bold text-ink">{circ.label}</p>
                <p className="text-ink-soft">{circ.action}</p>
                {circ.note && <p className="mt-1 font-semibold text-warn">{circ.note}</p>}
              </div>
            </div>
            {circ.estimate_if_verified && (
              <div className="mt-4 rounded-2xl bg-good-soft p-4">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge tone="good">ESTIMATE</Badge>
                  <span className="text-sm text-ink-soft">if an inspector verifies a retread</span>
                </div>
                <p className="mt-2 text-ink">
                  <strong className="text-2xl">{circ.estimate_if_verified.raw_material_saved_kg} kg</strong> raw material ·{" "}
                  <strong className="text-2xl">{circ.estimate_if_verified.co2e_avoided_kg} kg</strong> CO₂e avoided
                </p>
                <p className="mt-1 text-sm text-muted">{circ.scope}; factors {circ.factors_version}. Counted in the ledger only after verification.</p>
              </div>
            )}
            <Link href={`/circularity/passport/${d.inspection_id}`} className="mt-4 inline-block font-semibold text-primary-ink hover:underline">
              Open casing passport →
            </Link>
          </Card>
        )}
        <Card title="Lineage" subtitle={`bundle ${shortSha(d.deployment.bundle_sha256)} · generation ${d.deployment.generation ?? "?"}`}>
          <LineageDrawer lineage={p.lineage} deployment={d.deployment} />
          {p.attribution?.note && <p className="mt-3 text-sm text-muted">Heatmap: {p.attribution.method} — {p.attribution.note}.</p>}
        </Card>
      </div>
    </div>
  );
}
