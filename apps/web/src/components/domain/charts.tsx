"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ReactNode } from "react";

import { pct, timeOnly } from "@/lib/format";
import type { BinaryMetrics, PsiEntry, ReliabilityBin } from "@/lib/types";

const C = {
  primary: "#6f5ff0",
  accent: "#0f9f8f",
  good: "#0f8f75",
  warn: "#d38b12",
  bad: "#c92a44",
  muted: "#8a90a6",
  grid: "#e4e5ef",
};

function Figure({ summary, children, height = 240 }: { summary: string; children: ReactNode; height?: number }) {
  return (
    <figure className="w-full">
      <div style={{ height }} className="w-full" aria-hidden>
        {children}
      </div>
      <figcaption className="mt-2 text-sm text-muted">{summary}</figcaption>
    </figure>
  );
}

export function TriageTimeline({ data }: {
  data: { t: string; good_candidate: number; manual_review: number; defect_flagged: number; avg_total_ms: number | null }[];
}) {
  const totals = data.reduce((a, d) => ({ g: a.g + d.good_candidate, m: a.m + d.manual_review, d: a.d + d.defect_flagged }),
    { g: 0, m: 0, d: 0 });
  const rows = data.map((d) => ({ ...d, label: new Date(d.t).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit" }) }));
  return (
    <Figure summary={`Over ${data.length} time buckets: ${totals.g} retread candidates, ${totals.m} expert reviews, ${totals.d} defect flags; line = mean worker latency (ms).`}>
      <ResponsiveContainer>
        <ComposedChart data={rows} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
          <CartesianGrid stroke={C.grid} vertical={false} />
          <XAxis dataKey="label" tick={{ fontSize: 12 }} minTickGap={24} />
          <YAxis yAxisId="n" allowDecimals={false} tick={{ fontSize: 12 }} />
          <YAxis yAxisId="ms" orientation="right" tick={{ fontSize: 12 }} />
          <Tooltip />
          <Legend wrapperStyle={{ fontSize: 13 }} />
          <Bar yAxisId="n" dataKey="good_candidate" name="Retread candidate" stackId="t" fill={C.good} />
          <Bar yAxisId="n" dataKey="manual_review" name="Expert review" stackId="t" fill={C.warn} />
          <Bar yAxisId="n" dataKey="defect_flagged" name="Defect flagged" stackId="t" fill={C.bad} radius={[4, 4, 0, 0]} />
          <Line yAxisId="ms" dataKey="avg_total_ms" name="Mean latency (ms)" stroke={C.primary} dot={false} strokeWidth={2} />
        </ComposedChart>
      </ResponsiveContainer>
    </Figure>
  );
}

export function TriageBars({ counts }: { counts: Record<string, number> }) {
  const rows = [
    { name: "Retread candidate", key: "good_candidate", fill: C.good },
    { name: "Expert review", key: "manual_review", fill: C.warn },
    { name: "Defect flagged", key: "defect_flagged", fill: C.bad },
  ].map((r) => ({ ...r, value: counts[r.key] ?? 0 }));
  const total = rows.reduce((a, r) => a + r.value, 0);
  return (
    <Figure height={180} summary={total ? rows.map((r) => `${r.name} ${r.value} (${pct(r.value / total)})`).join("; ") : "No completed predictions in this window."}>
      <ResponsiveContainer>
        <BarChart data={rows} layout="vertical" margin={{ left: 24, right: 16 }}>
          <XAxis type="number" allowDecimals={false} tick={{ fontSize: 12 }} />
          <YAxis type="category" dataKey="name" width={130} tick={{ fontSize: 13 }} />
          <Tooltip />
          <Bar dataKey="value" name="Inspections" radius={[0, 6, 6, 0]} fill={C.primary} />
        </BarChart>
      </ResponsiveContainer>
    </Figure>
  );
}

export function CalibrationChart({ bins, title }: { bins: ReliabilityBin[] | undefined; title?: string }) {
  if (!bins?.length) return <p className="text-muted">No reliability data in this report.</p>;
  const pts = bins.map((b) => ({ x: b.mean_predicted, y: b.observed_defective_rate, n: b.count }));
  const worst = pts.reduce((a, p) => Math.max(a, Math.abs(p.x - p.y)), 0);
  return (
    <Figure summary={`${title ?? "Reliability"}: ${pts.length} bins; largest gap between predicted and observed defective rate ${pct(worst)}. Diagonal = perfect calibration.`}>
      <ResponsiveContainer>
        <ComposedChart data={pts} margin={{ top: 8, right: 16, left: -8, bottom: 8 }}>
          <CartesianGrid stroke={C.grid} />
          <XAxis type="number" dataKey="x" domain={[0, 1]} tick={{ fontSize: 12 }} label={{ value: "mean predicted", position: "insideBottom", offset: -4, fontSize: 12 }} />
          <YAxis type="number" domain={[0, 1]} tick={{ fontSize: 12 }} />
          <Tooltip formatter={(v) => (typeof v === "number" ? v.toFixed(3) : String(v))} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={C.muted} strokeDasharray="4 4" />
          <Line dataKey="y" stroke={C.primary} strokeWidth={2} dot={false} name="observed" />
          <Scatter dataKey="y" fill={C.primary} name="bins" />
        </ComposedChart>
      </ResponsiveContainer>
    </Figure>
  );
}

export function ConfusionMatrix({ m, caption }: { m: BinaryMetrics | undefined | null; caption?: string }) {
  if (!m) return <p className="text-muted">No confusion matrix.</p>;
  const c = m.confusion;
  const cell = (v: number, good: boolean) => (
    <td className={`mono rounded-xl p-3 text-center text-xl font-bold ${good ? "bg-good-soft text-good" : "bg-bad-soft text-bad"}`}>{v}</td>
  );
  return (
    <table className="w-full border-separate border-spacing-1 text-sm">
      <caption className="mb-2 text-left text-muted">{caption ?? `Threshold ${m.threshold.toFixed(3)}; defective is the positive class.`}</caption>
      <thead>
        <tr><th /><th scope="col" className="text-muted">Pred. defective</th><th scope="col" className="text-muted">Pred. good</th></tr>
      </thead>
      <tbody>
        <tr><th scope="row" className="text-right text-muted">Actual defective</th>{cell(c.tp, true)}{cell(c.fn, false)}</tr>
        <tr><th scope="row" className="text-right text-muted">Actual good</th>{cell(c.fp, false)}{cell(c.tn, true)}</tr>
      </tbody>
    </table>
  );
}

export function PsiChart({ psi, threshold = 0.2 }: { psi: Record<string, PsiEntry> | undefined; threshold?: number }) {
  const rows = Object.entries(psi ?? {}).map(([k, v]) => ({ name: k, psi: Number(v.psi.toFixed(4)) }));
  if (!rows.length) return <p className="text-muted">No descriptor PSI computed.</p>;
  const over = rows.filter((r) => r.psi > threshold).map((r) => r.name);
  return (
    <Figure height={220} summary={`PSI per descriptor vs the frozen reference: ${rows.map((r) => `${r.name} ${r.psi}`).join(", ")}. ${over.length ? `Above ${threshold}: ${over.join(", ")}.` : `None above ${threshold}.`}`}>
      <ResponsiveContainer>
        <BarChart data={rows} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
          <CartesianGrid stroke={C.grid} vertical={false} />
          <XAxis dataKey="name" tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip />
          <ReferenceLine y={threshold} stroke={C.bad} strokeDasharray="5 4" label={{ value: `alert ${threshold}`, fontSize: 12, fill: C.bad, position: "insideTopRight" }} />
          <Bar dataKey="psi" name="PSI" fill={C.primary} radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </Figure>
  );
}

export function EpochChart({ epochs }: { epochs: ({ epoch: number } & Record<string, number>)[] }) {
  if (!epochs.length) return <p className="text-muted">No epochs completed yet.</p>;
  const rows = epochs.map((e) => ({
    epoch: e.epoch,
    acc: e["metrics/accuracy_top1"] ?? null,
    loss: e.train_loss ?? e["train_loss"] ?? null,
    val: e["val/loss"] ?? null,
  }));
  const last = rows[rows.length - 1];
  return (
    <Figure summary={`${rows.length} epochs; latest top-1 accuracy on dev_val ${last.acc === null ? "n/a" : pct(last.acc)}, train loss ${last.loss === null ? "n/a" : last.loss.toFixed(4)}.`}>
      <ResponsiveContainer>
        <LineChart data={rows} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
          <CartesianGrid stroke={C.grid} />
          <XAxis dataKey="epoch" tick={{ fontSize: 12 }} />
          <YAxis yAxisId="acc" domain={[0, 1]} tick={{ fontSize: 12 }} />
          <YAxis yAxisId="loss" orientation="right" tick={{ fontSize: 12 }} />
          <Tooltip />
          <Legend wrapperStyle={{ fontSize: 13 }} />
          <Line yAxisId="acc" dataKey="acc" name="dev_val top-1" stroke={C.accent} strokeWidth={2} dot={false} />
          <Line yAxisId="loss" dataKey="loss" name="train loss" stroke={C.primary} strokeWidth={2} dot={false} />
          <Line yAxisId="loss" dataKey="val" name="val loss" stroke={C.warn} strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </Figure>
  );
}

export function SimpleBars({ data, label, color = C.primary, height = 200 }: {
  data: { name: string; value: number }[];
  label: string;
  color?: string;
  height?: number;
}) {
  if (!data.length) return <p className="text-muted">No data yet.</p>;
  return (
    <Figure height={height} summary={`${label}: ${data.map((d) => `${d.name} ${d.value}`).join(", ")}.`}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
          <CartesianGrid stroke={C.grid} vertical={false} />
          <XAxis dataKey="name" tick={{ fontSize: 12 }} interval={0} angle={data.length > 5 ? -20 : 0} textAnchor={data.length > 5 ? "end" : "middle"} height={data.length > 5 ? 60 : 30} />
          <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
          <Tooltip />
          <Bar dataKey="value" name={label} fill={color} radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </Figure>
  );
}

export function LatencyLine({ data }: { data: { t: string; p95_ms: number | null; turns: number }[] }) {
  if (!data.length) return <p className="text-muted">No turns yet.</p>;
  const rows = data.map((d) => ({ ...d, label: timeOnly(d.t) || d.t }));
  return (
    <Figure summary={`${data.length} hourly buckets; p95 latency per bucket shown as a line, turns as bars.`}>
      <ResponsiveContainer>
        <ComposedChart data={rows} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
          <CartesianGrid stroke={C.grid} vertical={false} />
          <XAxis dataKey="label" tick={{ fontSize: 12 }} />
          <YAxis yAxisId="n" allowDecimals={false} tick={{ fontSize: 12 }} />
          <YAxis yAxisId="ms" orientation="right" tick={{ fontSize: 12 }} />
          <Tooltip />
          <Bar yAxisId="n" dataKey="turns" fill={C.accent} name="turns" radius={[4, 4, 0, 0]} />
          <Line yAxisId="ms" dataKey="p95_ms" stroke={C.primary} name="p95 ms" strokeWidth={2} dot={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </Figure>
  );
}
