"use client";

import clsx from "clsx";

import { GateStatus } from "@/components/domain/badges";
import { Badge } from "@/components/ui";
import { dateTime, ms, pct, shortSha } from "@/lib/format";
import type { ModelView } from "@/lib/types";

export function ModelComparisonTable({ items, selectedId, onSelect }: {
  items: ModelView[];
  selectedId?: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[900px] text-left">
        <caption className="mb-2 text-left text-sm text-muted">
          Gate metrics are development evidence (outer validation); locked-test metrics are the final benchmark.
        </caption>
        <thead>
          <tr className="border-b border-line text-sm text-muted">
            {["Version", "Stage", "Gates", "Gate recall", "Gate specificity", "Test recall", "Test specificity", "CPU p95", "Runtimes", "Created", ""].map((h) => (
              <th key={h} scope="col" className="py-2 pr-3 font-semibold">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {items.map((m) => (
            <tr key={m.id} className={clsx("border-b border-line/60", selectedId === m.id && "bg-primary-soft")}>
              <td className="py-2 pr-3">
                <span className="font-bold text-ink">{m.model_family} v{m.registry_version ?? "?"}</span>
                <span className="mono block text-xs text-muted">{shortSha(m.bundle_sha256)}</span>
              </td>
              <td className="py-2 pr-3">{m.is_active ? <Badge tone="good">champion · active</Badge> : <Badge>{m.stage}</Badge>}</td>
              <td className="py-2 pr-3"><GateStatus status={m.gate_status} /></td>
              <td className="mono py-2 pr-3">{pct(m.headline.gate_recall)}</td>
              <td className="mono py-2 pr-3">{pct(m.headline.gate_specificity)}</td>
              <td className="mono py-2 pr-3">{pct(m.headline.test_recall)}</td>
              <td className="mono py-2 pr-3">{pct(m.headline.test_specificity)}</td>
              <td className="mono py-2 pr-3">{ms(m.headline.model_p95_ms)}</td>
              <td className="py-2 pr-3 text-sm">{m.runtimes?.join(", ") ?? "n/a"}</td>
              <td className="py-2 pr-3 text-sm">{dateTime(m.created_at)}</td>
              <td className="py-2 pr-3">
                <button type="button" onClick={() => onSelect(m.id)} aria-pressed={selectedId === m.id}
                  className="rounded-lg px-2 py-1 font-semibold text-primary-ink hover:bg-primary-soft">Evidence</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
