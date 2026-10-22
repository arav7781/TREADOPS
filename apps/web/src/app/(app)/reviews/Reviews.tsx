"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { Keyboard } from "lucide-react";
import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";

import { ProbabilityPanel } from "@/components/domain/ProbabilityPanel";
import { ReviewForm } from "@/components/domain/ReviewForm";
import { QualityBadge, SourceBadge, TriageBadge } from "@/components/domain/badges";
import { useMe } from "@/components/AppShell";
import { Card, EmptyState, ErrorState, inputClass, PageHeader, SkeletonRows } from "@/components/ui";
import { api, mediaUrl } from "@/lib/api";
import { ago, pct } from "@/lib/format";
import { useInspection } from "@/lib/hooks";
import { qk } from "@/lib/query";
import { hasRole } from "@/lib/roles";
import type { ReviewQueue } from "@/lib/types";
import { toQuery, useUrlState } from "@/lib/useUrlState";

const SUGGESTED: Record<string, string> = { good_candidate: "retread", manual_review: "reinspect", defect_flagged: "material_recovery" };

function Detail({ id, canReview, onSaved }: { id: string; canReview: boolean; onSaved: () => void }) {
  const q = useInspection(id);
  if (q.isError) return <ErrorState error={q.error} onRetry={() => q.refetch()} />;
  const d = q.data;
  if (!d) return <SkeletonRows rows={5} />;
  const img = mediaUrl(d.image.url);
  const heat = mediaUrl(d.prediction?.heatmap_url);
  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2">
        {img && <figure>{/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={img} alt="Tyre photo under review" className="aspect-square w-full rounded-xl border border-line bg-surface-muted object-contain" />
          <figcaption className="mt-1 text-sm text-muted">Original</figcaption></figure>}
        {heat && <figure>{/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={heat} alt="Occlusion attribution heatmap" className="aspect-square w-full rounded-xl border border-line bg-surface-muted object-contain" />
          <figcaption className="mt-1 text-sm text-muted">Attribution (not a defect location)</figcaption></figure>}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <TriageBadge triage={d.prediction?.triage} large />
        <QualityBadge status={d.prediction?.quality_status} reasons={d.prediction?.quality.reasons} />
        <SourceBadge origin={d.metadata_origin} />
        <Link href={`/inspections/${d.inspection_id}`} className="ml-auto text-sm font-semibold text-primary-ink hover:underline">Full detail →</Link>
      </div>
      {d.prediction && <ProbabilityPanel raw={d.prediction.raw_probs} calibrated={d.prediction.calibrated_probs} pDefective={d.prediction.p_defective} />}
      {canReview ? (
        <ReviewForm key={`${d.inspection_id}-${d.review.revision}`} inspectionId={d.inspection_id} expectedRevision={d.review.revision}
          current={{ label: d.review.label, disposition: d.review.disposition }}
          suggestedDisposition={d.prediction ? SUGGESTED[d.prediction.triage] : null} onSaved={onSaved} />
      ) : <p className="text-muted">Viewer access: labelling requires the inspector role.</p>}
    </div>
  );
}

export function Reviews() {
  const [f, setF] = useUrlState(["sort", "triage"] as const, { sort: "p_defective_desc" });
  const me = useMe();
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: qk.reviewQueue(f.sort, f.triage),
    queryFn: () => api<ReviewQueue>(`/v1/reviews/queue${toQuery({ sort: f.sort, triage: f.triage, limit: 50 })}`),
    refetchInterval: 15_000,
  });
  const items = useMemo(() => q.data?.items ?? [], [q.data]);
  const [selected, setSelected] = useState<string | null>(null);
  const current = selected && items.some((i) => i.inspection_id === selected) ? selected : items[0]?.inspection_id ?? null;
  const detailRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest("input, textarea, select, [contenteditable=true]")) return;
      const idx = items.findIndex((i) => i.inspection_id === current);
      if (e.key === "j" || e.key === "ArrowDown") { e.preventDefault(); setSelected(items[Math.min(items.length - 1, idx + 1)]?.inspection_id ?? null); }
      if (e.key === "k" || e.key === "ArrowUp") { e.preventDefault(); setSelected(items[Math.max(0, idx - 1)]?.inspection_id ?? null); }
      if (e.key === "Enter") { e.preventDefault(); detailRef.current?.querySelector<HTMLInputElement>("input[type=radio]")?.focus(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [items, current]);

  const saved = () => {
    const idx = items.findIndex((i) => i.inspection_id === current);
    setSelected(items[idx + 1]?.inspection_id ?? null);
    qc.invalidateQueries({ queryKey: ["reviews"] });
  };

  return (
    <>
      <PageHeader eyebrow="Human in the loop" title="Review queue"
        subtitle="Completed inspections without a verified label. Your label and disposition feed monitoring, the circularity ledger and eligible training snapshots."
        actions={<span className="flex items-center gap-2 text-sm text-muted"><Keyboard aria-hidden className="h-4 w-4" /> j/k or ↑/↓ to move · Enter to label</span>} />
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
        <Card title={q.data ? `${q.data.total} awaiting review` : "Queue"}
          subtitle={q.data?.oldest_waiting_at ? `oldest waiting ${ago(q.data.oldest_waiting_at)}` : undefined}
          actions={
            <>
              <label htmlFor="r-sort" className="sr-only">Sort</label>
              <select id="r-sort" className={inputClass} value={f.sort} onChange={(e) => setF({ sort: e.target.value })}>
                <option value="p_defective_desc">Highest p(defective)</option>
                <option value="p_defective_asc">Lowest p(defective)</option>
                <option value="oldest">Oldest first</option>
                <option value="newest">Newest first</option>
              </select>
              <label htmlFor="r-triage" className="sr-only">Triage</label>
              <select id="r-triage" className={inputClass} value={f.triage} onChange={(e) => setF({ triage: e.target.value })}>
                <option value="">All triage</option>
                <option value="defect_flagged">Defect flagged</option>
                <option value="manual_review">Expert review</option>
                <option value="good_candidate">Retread candidate</option>
              </select>
            </>}>
          {q.isError ? <ErrorState error={q.error} onRetry={() => q.refetch()} />
            : !q.data ? <SkeletonRows rows={6} />
              : items.length === 0 ? <EmptyState title="Queue is empty">Every completed inspection has a verified label.</EmptyState>
                : (
                  <ul role="listbox" aria-label="Inspections awaiting review" className="max-h-[70vh] space-y-1.5 overflow-auto pr-1">
                    {items.map((i) => {
                      const thumb = mediaUrl(i.thumbnail_url);
                      const active = i.inspection_id === current;
                      return (
                        <li key={i.inspection_id} role="option" aria-selected={active}>
                          <button type="button" onClick={() => setSelected(i.inspection_id)}
                            className={clsx("flex w-full items-center gap-3 rounded-xl border px-3 py-2 text-left transition",
                              active ? "border-primary bg-primary-soft" : "border-transparent hover:bg-surface-muted")}>
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            {thumb && <img src={thumb} alt="" className="h-12 w-12 rounded-lg object-cover" />}
                            <span className="min-w-0 flex-1">
                              <span className="flex flex-wrap items-center gap-2"><TriageBadge triage={i.triage} /><SourceBadge origin={i.metadata_origin} /></span>
                              <span className="mono mt-0.5 block text-sm text-muted">{i.inspection_id.slice(0, 8)} · p={pct(i.p_defective)} · {ago(i.created_at)}</span>
                            </span>
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                )}
        </Card>
        <div ref={detailRef}>
          <Card title="Selected inspection">
            {current ? <Detail id={current} canReview={hasRole(me.data?.roles, "inspector")} onSaved={saved} />
              : <EmptyState title="Nothing selected" />}
          </Card>
        </div>
      </div>
    </>
  );
}
