"use client";

import { CheckCircle2, CircleDashed, CircleSlash, FlaskConical, ShieldAlert, XCircle } from "lucide-react";

import { Badge, type Tone } from "@/components/ui";
import { gateLabel, humanize, TRIAGE_LABEL } from "@/lib/format";
import type { GateState, Triage } from "@/lib/types";

/** Makes data provenance visible: simulated operational metadata must never pass as real plant data. */
export function SourceBadge({ origin, className }: { origin: string | null | undefined; className?: string }) {
  if (origin === "simulated") {
    return (
      <Badge tone="sim" className={className} title="Real public photo with simulated line/camera/timestamp metadata">
        <FlaskConical aria-hidden className="h-3.5 w-3.5" /> SIMULATED METADATA
      </Badge>
    );
  }
  if (origin === "user_provided") return <Badge tone="info" className={className}>User upload</Badge>;
  if (origin === "observed") return <Badge tone="neutral" className={className}>Public dataset</Badge>;
  return null;
}

const TRIAGE_TONE: Record<Triage, Tone> = { good_candidate: "good", manual_review: "warn", defect_flagged: "bad" };

export function TriageBadge({ triage, large }: { triage: Triage | null | undefined; large?: boolean }) {
  if (!triage) return <Badge>Pending</Badge>;
  return (
    <Badge tone={TRIAGE_TONE[triage]} className={large ? "px-3.5 py-1 text-base" : undefined}>
      {TRIAGE_LABEL[triage]}
    </Badge>
  );
}

/**
 * The quality-check verdict: the policy's binary decision (damaged / not damaged), plus "expert check" when the
 * triage is uncertain or the image quality is low.
 */
export function VerdictBadge({ damaged, triage, quality, large }: {
  damaged: boolean | null | undefined;
  triage?: Triage | null;
  quality?: string | null;
  large?: boolean;
}) {
  if (damaged === null || damaged === undefined) return <Badge>Pending</Badge>;
  const expert = triage === "manual_review" || quality === "low_quality";
  return (
    <span className="inline-flex flex-wrap items-center gap-1.5">
      <Badge tone={damaged ? "bad" : "good"} className={large ? "px-3.5 py-1 text-base font-bold" : "font-bold"}>
        {damaged ? "Damaged" : "Not damaged"}
      </Badge>
      {expert && <Badge tone="warn">Expert check</Badge>}
    </span>
  );
}

const GATE_TONE: Record<string, Tone> = { passed: "good", failed: "bad", insufficient_evidence: "warn", not_applicable: "neutral" };

export function GateStatus({ status, label }: { status: GateState | string | null | undefined; label?: string }) {
  const s = status ?? "unknown";
  const Icon = s === "passed" ? CheckCircle2 : s === "failed" ? XCircle : s === "insufficient_evidence" ? CircleDashed : CircleSlash;
  return (
    <Badge tone={GATE_TONE[s] ?? "neutral"} title={label}>
      <Icon aria-hidden className="h-3.5 w-3.5" />
      {label ? `${label}: ` : ""}
      {gateLabel(s)}
    </Badge>
  );
}

const JOB_TONE: Record<string, Tone> = {
  queued: "info", running: "primary", succeeded: "good", failed: "bad", retry_wait: "warn",
  cancellation_requested: "warn", cancelled: "neutral",
};

export function StatusBadge({ status }: { status: string | null | undefined }) {
  if (!status) return null;
  const active = status === "running" || status === "queued" || status === "retry_wait";
  return (
    <Badge tone={JOB_TONE[status] ?? "neutral"}>
      {active && <span aria-hidden className="pulse-dot h-2 w-2 rounded-full bg-current" />}
      {humanize(status)}
    </Badge>
  );
}

export function QualityBadge({ status, reasons }: { status: string | null | undefined; reasons?: string[] }) {
  if (!status) return null;
  if (status === "accepted") return <Badge tone="good">Quality accepted</Badge>;
  return (
    <Badge tone="warn" title={reasons?.join(", ")}>
      <ShieldAlert aria-hidden className="h-3.5 w-3.5" /> Low quality{reasons?.length ? `: ${reasons.join(", ")}` : ""}
    </Badge>
  );
}
