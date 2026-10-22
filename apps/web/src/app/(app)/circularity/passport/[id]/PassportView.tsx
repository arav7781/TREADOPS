"use client";

import { useQuery } from "@tanstack/react-query";
import { Printer } from "lucide-react";
import Link from "next/link";
import { use } from "react";

import { LineageDrawer } from "@/components/domain/LineageDrawer";
import { SourceBadge, TriageBadge } from "@/components/domain/badges";
import { Badge, Button, Card, ErrorState, KeyValue, SkeletonRows } from "@/components/ui";
import { api, mediaUrl } from "@/lib/api";
import { dateTime, humanize, pct, shortSha } from "@/lib/format";
import { qk } from "@/lib/query";
import type { Passport } from "@/lib/types";

export function PassportView({ idPromise }: { idPromise: Promise<string> }) {
  const id = use(idPromise);
  const q = useQuery({ queryKey: qk.passport(id), queryFn: () => api<Passport>(`/v1/circularity/passport/${id}`) });
  if (q.isError) return <ErrorState error={q.error} onRetry={() => q.refetch()} />;
  const p = q.data;
  if (!p) return <SkeletonRows rows={8} />;
  const img = mediaUrl(p.model_assessment.classified_image_url) ?? mediaUrl(p.casing.image_url);
  return (
    <article className="mx-auto max-w-4xl space-y-6" aria-label="Casing passport">
      <div className="no-print flex flex-wrap items-center gap-2">
        <Link href={`/inspections/${p.inspection_id}`} className="font-semibold text-primary-ink hover:underline">← Inspection</Link>
        <Button variant="secondary" className="ml-auto" onClick={() => window.print()}><Printer aria-hidden className="h-4 w-4" /> Print</Button>
      </div>
      <Card>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-sm font-bold uppercase tracking-wider text-accent-ink">Casing passport</p>
            <h1 className="mono text-3xl font-extrabold text-ink">{p.passport_id}</h1>
            <p className="text-ink-soft">Issued {dateTime(p.issued_at)} · inspection {p.inspection_id.slice(0, 8)}</p>
          </div>
          <div className="flex flex-col items-end gap-2">
            <Badge tone={p.status === "verified" ? "good" : "warn"} className="px-3 py-1 text-base">{p.status === "verified" ? "Inspector verified" : "Awaiting inspector"}</Badge>
            <SourceBadge origin={p.casing.metadata_origin} />
          </div>
        </div>
        <div className="mt-6 grid gap-6 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          {img && <img src={img} alt="Classified casing photo" className="w-full rounded-2xl border border-line object-contain" />}
          <div className="space-y-5">
            <section>
              <h2 className="mb-2 font-bold text-ink">Model assessment</h2>
              <div className="mb-2"><TriageBadge triage={p.model_assessment.triage} large /></div>
              <KeyValue items={[
                ["p(defective)", pct(p.model_assessment.p_defective, 2)],
                ["Image quality", humanize(p.model_assessment.quality_status) || "n/a"],
                ["Recommended pathway", p.model_assessment.recommended_pathway ?? "n/a"],
                ["Tyre category", humanize(p.casing.tyre_category) || "n/a"],
                ["Image SHA-256", <span key="s" className="mono">{shortSha(p.casing.image_sha256, 16)}</span>],
              ]} />
            </section>
            <section>
              <h2 className="mb-2 font-bold text-ink">Inspector verification</h2>
              <KeyValue items={[
                ["Label", humanize(p.inspector_verification.label) || "not yet labelled"],
                ["Disposition", humanize(p.inspector_verification.disposition) || "not decided"],
                ["Revision", `r${p.inspector_verification.revision}`],
              ]} />
            </section>
            {p.impact_estimate && (
              <section className="rounded-2xl bg-good-soft p-4">
                <Badge tone="good">ESTIMATE</Badge>
                <p className="mt-2 text-ink"><strong>{p.impact_estimate.raw_material_saved_kg} kg</strong> raw material · <strong>{p.impact_estimate.co2e_avoided_kg} kg</strong> CO₂e avoided</p>
              </section>
            )}
          </div>
        </div>
      </Card>
      <Card title="Lineage"><LineageDrawer lineage={p.lineage} defaultOpen /></Card>
      <p className="rounded-2xl bg-warn-soft p-4 text-ink">{p.disclaimer}</p>
    </article>
  );
}
