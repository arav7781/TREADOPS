"use client";

import clsx from "clsx";
import { ScanSearch } from "lucide-react";

import { Badge } from "@/components/ui";
import type { DamageLocator } from "@/lib/types";

const name = (c: string) => c.replace("_", " ");

/** What the damage locator found: boxes are drawn on the classified image; a confident one blocks an auto-clear. */
export function DamagePanel({ d, className }: { d: DamageLocator; className?: string }) {
  const found = d.boxes.filter((b) => b.confidence >= 0.25);
  return (
    <section aria-label="Damage locator" data-testid="damage-locator"
      className={clsx("rounded-xl border p-3", d.escalate ? "border-warn/40 bg-warn-soft" : "border-line bg-surface-muted", className)}>
      <div className="flex flex-wrap items-center gap-2">
        <ScanSearch aria-hidden className="h-4 w-4 text-primary" />
        <span className="font-bold text-ink">Damage locator</span>
        <Badge tone={d.escalate ? "warn" : found.length ? "neutral" : "good"}>
          {d.escalate ? "Damage found" : found.length ? "Weak signs only" : "No damage boxes"}
        </Badge>
      </div>
      {found.length > 0 ? (
        <ul className="mt-1 flex flex-wrap gap-1.5">
          {found.slice(0, 6).map((b, i) => (
            <li key={i} className={clsx("rounded-full px-2.5 py-0.5 text-sm", b.escalates ? "bg-brand-yellow text-black" : "bg-surface text-ink")}>
              {name(b.class)} <span className="mono">{b.confidence.toFixed(2)}</span>
            </li>
          ))}
        </ul>
      ) : <p className="mt-1 text-sm text-ink-soft">No cut, chunk, crack, bulge or foreign object above the display threshold.</p>}
      <p className="mono mt-1 text-xs text-muted">locator {d.version} · escalates at {d.threshold.toFixed(2)} · boxes are drawn on the classified image</p>
    </section>
  );
}
