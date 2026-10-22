"use client";

import { useQuery } from "@tanstack/react-query";
import { ExternalLink, Factory, Leaf, Recycle, Search } from "lucide-react";

import { SimpleBars } from "@/components/domain/charts";
import { Badge, Card, EmptyState, ErrorState, KeyValue, PageHeader, SkeletonRows, Stat } from "@/components/ui";
import { api } from "@/lib/api";
import { humanize, num, shortSha, TRIAGE_LABEL } from "@/lib/format";
import { qk } from "@/lib/query";
import type { Ledger, Triage } from "@/lib/types";

const PATHWAYS = [
  { key: "retread_intake", label: "Retread intake", icon: Recycle, text: "Sound casing → second life via retreading (full casing NDT still required)." },
  { key: "expert_inspection", label: "Expert inspection", icon: Search, text: "Uncertain or low-quality evidence → a qualified inspector decides." },
  { key: "material_recovery", label: "Material recovery", icon: Factory, text: "Unsafe casing → removed from service, certified end-of-life recovery." },
];

export function CircularityView() {
  const q = useQuery({ queryKey: qk.ledger, queryFn: () => api<Ledger>("/v1/circularity/ledger"), refetchInterval: 30_000 });
  if (q.isError) return <ErrorState error={q.error} onRetry={() => q.refetch()} />;
  const l = q.data;
  if (!l) return <SkeletonRows rows={8} />;
  const verifiedTotal = Object.values(l.verified).reduce((a, b) => a + b, 0);
  return (
    <>
      <PageHeader eyebrow="Sustainability" title="Casing circularity ledger"
        subtitle="Every inspected casing is routed toward reuse, expert inspection or material recovery. Impact counts only inspector-verified retreads, using Michelin's published truck/bus factors." />
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="Verified retreads" icon={<Recycle className="h-5 w-5" />} tone="good" value={num(l.verified.retread ?? 0)} hint={`${num(verifiedTotal)} verified dispositions in total`} />
        <Stat label="Raw material saved" icon={<Leaf className="h-5 w-5" />} tone="accent" value={`${num(l.estimates.raw_material_saved_kg)} kg`} hint={<Badge tone="good">ESTIMATE</Badge>} />
        <Stat label="CO₂e avoided" icon={<Leaf className="h-5 w-5" />} tone="accent" value={`${num(l.estimates.co2e_avoided_kg, 1)} kg`} hint={<Badge tone="good">ESTIMATE</Badge>} />
        <Stat label="New-tyre material equivalents" icon={<Factory className="h-5 w-5" />} value={num(l.estimates.new_tyre_equivalents_avoided_material, 2)} hint="saved kg ÷ new-tyre kg" />
      </div>
      <p role="note" className="mt-4 rounded-2xl bg-warn-soft p-4 text-ink">
        <strong>{l.label}.</strong> {l.basis}. Scope: {l.scope}.
      </p>

      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        {PATHWAYS.map((p) => {
          const Icon = p.icon;
          return (
            <Card key={p.key} title={<span className="flex items-center gap-2"><Icon aria-hidden className="h-5 w-5 text-accent" />{p.label}</span>}>
              <p className="text-ink-soft">{p.text}</p>
              <p className="mt-3 text-3xl font-extrabold text-ink">{num(l.recommended_by_model[p.key] ?? 0)}</p>
              <p className="text-sm text-muted">model recommendations (not counted in impact)</p>
            </Card>
          );
        })}
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card title="Verified dispositions" subtitle="Recorded by inspectors in reviews">
          <SimpleBars label="Verified dispositions" color="#0f9f8f" data={Object.entries(l.verified).map(([k, v]) => ({ name: humanize(k), value: v }))} />
          <KeyValue className="mt-3" items={Object.entries(l.verified_retreads_by_metadata_origin).map(([k, v]) => [`Verified retreads · ${humanize(k)}`, v])} />
        </Card>
        <Card title="Model vs inspector" subtitle="How model triage compared with the verified disposition">
          {l.model_vs_inspector.length === 0 ? <EmptyState title="No verified dispositions yet">Record dispositions in the review queue.</EmptyState> : (
            <table className="w-full text-left">
              <thead><tr className="border-b border-line text-sm text-muted"><th scope="col" className="py-2">Model triage</th><th scope="col">Inspector disposition</th><th scope="col">Count</th></tr></thead>
              <tbody>{l.model_vs_inspector.map((r, i) => (
                <tr key={i} className="border-b border-line/60"><td className="py-2">{TRIAGE_LABEL[r.model_triage as Triage] ?? r.model_triage}</td><td>{humanize(r.inspector_disposition)}</td><td className="mono">{r.count}</td></tr>
              ))}</tbody>
            </table>
          )}
        </Card>
      </div>

      <Card className="mt-6" title="Factors and sources" subtitle={`factors ${l.factors_version} · sha ${shortSha(l.factors_sha256)}`}>
        <KeyValue items={Object.entries(l.factors).map(([k, v]) => [humanize(k), String(v)])} />
        <ul className="mt-4 space-y-2">
          {l.sources.map((s) => (
            <li key={s.id} className="rounded-xl bg-surface-muted p-3">
              <a href={s.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 font-semibold text-primary-ink hover:underline">{s.title} <ExternalLink aria-hidden className="h-3.5 w-3.5" /></a>
              <p className="text-sm text-ink-soft">Used for: {s.used_for.map(humanize).join(", ")}</p>
              {s.note && <p className="text-sm text-muted">{s.note}</p>}
            </li>
          ))}
        </ul>
      </Card>
    </>
  );
}
