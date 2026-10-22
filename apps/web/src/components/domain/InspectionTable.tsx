"use client";

import Link from "next/link";

import { SourceBadge, StatusBadge, TriageBadge } from "@/components/domain/badges";
import { mediaUrl } from "@/lib/api";
import { ago, humanize, pct } from "@/lib/format";
import type { InspectionSummary } from "@/lib/types";

export function InspectionTable({ items, compact, selectedId, onSelect }: {
  items: InspectionSummary[];
  compact?: boolean;
  selectedId?: string | null;
  onSelect?: (id: string) => void;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[720px] text-left">
        <thead>
          <tr className="border-b border-line text-sm text-muted">
            <th scope="col" className="py-2 pr-3 font-semibold">Image</th>
            <th scope="col" className="py-2 pr-3 font-semibold">Triage</th>
            <th scope="col" className="py-2 pr-3 font-semibold">p(defective)</th>
            <th scope="col" className="py-2 pr-3 font-semibold">Human label</th>
            <th scope="col" className="py-2 pr-3 font-semibold">Source</th>
            {!compact && <th scope="col" className="py-2 pr-3 font-semibold">Line / camera</th>}
            <th scope="col" className="py-2 pr-3 font-semibold">Model</th>
            <th scope="col" className="py-2 pr-3 font-semibold">Status</th>
          </tr>
        </thead>
        <tbody>
          {items.map((i) => {
            const thumb = mediaUrl(i.thumbnail_url);
            return (
              <tr key={i.inspection_id} aria-selected={selectedId === i.inspection_id || undefined}
                onClick={onSelect ? () => onSelect(i.inspection_id) : undefined}
                className={`border-b border-line/70 align-middle transition hover:bg-surface-muted ${selectedId === i.inspection_id ? "bg-primary-soft" : ""}`}>
                <td className="py-2 pr-3">
                  <Link href={`/inspections/${i.inspection_id}`} className="flex items-center gap-3 font-semibold text-primary-ink hover:underline">
                    {thumb ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={thumb} alt="" loading="lazy" className="h-12 w-12 rounded-lg border border-line object-cover" />
                    ) : <span className="h-12 w-12 rounded-lg bg-line" />}
                    <span className="mono text-sm">{i.inspection_id.slice(0, 8)}</span>
                  </Link>
                  <span className="block pl-[60px] text-sm text-muted">{ago(i.created_at)}</span>
                </td>
                <td className="py-2 pr-3"><TriageBadge triage={i.triage} /></td>
                <td className="mono py-2 pr-3">{i.p_defective === null ? "n/a" : pct(i.p_defective, 1)}</td>
                <td className="py-2 pr-3">{i.human_label ? `${humanize(i.human_label)}${i.disposition ? ` · ${humanize(i.disposition)}` : ""}` : <span className="text-muted">awaiting</span>}</td>
                <td className="py-2 pr-3"><div className="flex flex-col items-start gap-1"><span className="text-sm capitalize">{i.source}</span><SourceBadge origin={i.metadata_origin} /></div></td>
                {!compact && <td className="py-2 pr-3 text-sm">{i.line_id ?? "—"} / {i.camera_id ?? "—"}{i.scenario_id ? <span className="block text-muted">{i.scenario_id}</span> : null}</td>}
                <td className="py-2 pr-3 text-sm">v{i.model_version ?? "?"} <span className="text-muted">gen {i.deployment_generation}</span></td>
                <td className="py-2 pr-3"><StatusBadge status={i.status} /></td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
