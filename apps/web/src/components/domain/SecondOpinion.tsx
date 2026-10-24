"use client";

import clsx from "clsx";
import { Eye } from "lucide-react";

import { Badge, type Tone } from "@/components/ui";
import { ms, TRIAGE_LABEL } from "@/lib/format";
import type { SecondOpinion } from "@/lib/types";

const DECISION: Record<SecondOpinion["decision"], { label: string; tone: Tone }> = {
  agrees: { label: "Agrees", tone: "good" },
  disagrees: { label: "Sees damage", tone: "warn" },
  unavailable: { label: "No usable answer", tone: "warn" },
};

function summary(s: SecondOpinion): string {
  const yolo = TRIAGE_LABEL[s.policy_triage] ?? s.policy_triage;
  if (s.decision === "disagrees") {
    return `The classifier alone would have auto-cleared this tyre (${yolo}). A vision model sees damage with ${s.confidence} confidence`
      + `${s.location ? ` (${s.location})` : ""}, so the models disagree and an expert must check it.`;
  }
  if (s.decision === "unavailable") {
    return `The vision model gave no usable answer${s.error ? ` (${s.error.replaceAll("_", " ")})` : ""}, so the tyre is not auto-cleared.`;
  }
  return s.damage_visible
    ? `The vision model noted only low-confidence cues, so the classifier's auto-clear (${yolo}) stands.`
    : `A vision model re-checked the classifier's auto-clear (${yolo}) and sees no damage.`;
}

/** The vision LLM's re-check of an auto-clear: what it saw, and whether it kept the tyre away from auto-clearance. */
export function SecondOpinionPanel({ s, className }: { s: SecondOpinion; className?: string }) {
  const d = DECISION[s.decision] ?? DECISION.unavailable;
  const findings = s.findings ?? [];
  return (
    <section aria-label="Vision second opinion"
      className={clsx("rounded-xl border p-3", s.decision === "agrees" ? "border-line bg-surface-muted" : "border-warn/40 bg-warn-soft", className)}>
      <div className="flex flex-wrap items-center gap-2">
        <Eye aria-hidden className="h-4 w-4 text-primary" />
        <span className="font-bold text-ink">Vision second opinion</span>
        <Badge tone={d.tone}>{d.label}</Badge>
      </div>
      <p className="mt-1 text-sm text-ink-soft">{summary(s)}</p>
      {findings.length > 0 && (
        <ul className="ml-5 mt-1 list-disc text-sm text-ink">
          {findings.map((f) => <li key={f}>{f}</li>)}
        </ul>
      )}
      <p className="mono mt-1 text-xs text-muted">{s.model} · prompt {s.prompt_version} · {ms(s.latency_ms)} · a second opinion, not a clearance</p>
    </section>
  );
}
