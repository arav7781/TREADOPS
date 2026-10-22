"use client";

import clsx from "clsx";

import { ms } from "@/lib/format";
import type { TraceSpan } from "@/lib/types";

const KIND: Record<string, string> = { llm: "bg-primary", tool: "bg-accent", guard: "bg-warn", node: "bg-muted" };

/** Span waterfall: one row per span, offset by start time, indented by parent depth. */
export function TraceWaterfall({ spans, total }: { spans: TraceSpan[]; total: number }) {
  if (!spans.length) return <p className="text-muted">No spans recorded.</p>;
  const byId = new Map(spans.map((s) => [s.id, s]));
  const depth = (s: TraceSpan): number => {
    let d = 0;
    let p = s.parent ? byId.get(s.parent) : undefined;
    while (p && d < 10) {
      d += 1;
      p = p.parent ? byId.get(p.parent) : undefined;
    }
    return d;
  };
  const span = Math.max(total, ...spans.map((s) => s.start_ms + s.duration_ms), 1);
  return (
    <div role="table" aria-label="Trace spans" className="space-y-1">
      <div role="row" className="flex text-xs text-muted">
        <span role="columnheader" className="w-56 shrink-0">span</span>
        <span role="columnheader" className="flex-1">timeline (0 – {ms(span)})</span>
        <span role="columnheader" className="w-20 shrink-0 text-right">duration</span>
      </div>
      {spans.map((s) => (
        <div role="row" key={s.id} className="flex items-center gap-2 text-sm">
          <span role="cell" className="mono w-56 shrink-0 truncate" style={{ paddingLeft: depth(s) * 12 }} title={JSON.stringify(s.attributes)}>
            <span className={clsx("mr-1.5 inline-block h-2 w-2 rounded-full", KIND[s.kind] ?? "bg-muted")} />{s.name}
          </span>
          <span role="cell" className="relative h-5 flex-1 rounded bg-surface-muted">
            <span className={clsx("absolute top-0.5 h-4 rounded", KIND[s.kind] ?? "bg-muted", s.status === "error" && "ring-2 ring-bad")}
              style={{ left: `${(s.start_ms / span) * 100}%`, width: `${Math.max(0.5, (s.duration_ms / span) * 100)}%` }} />
          </span>
          <span role="cell" className="mono w-20 shrink-0 text-right">{ms(s.duration_ms)}</span>
        </div>
      ))}
      <p className="pt-1 text-xs text-muted">● llm · <span className="text-accent">●</span> tool · <span className="text-warn">●</span> guardrail · <span className="text-muted">●</span> graph node. Hover a span for its attributes.</p>
    </div>
  );
}
