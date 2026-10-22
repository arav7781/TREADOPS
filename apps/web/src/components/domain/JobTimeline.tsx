"use client";

import clsx from "clsx";

import { humanize, timeOnly } from "@/lib/format";
import type { TimelineEvent } from "@/lib/types";

const DOT: Record<string, string> = {
  queued: "bg-info", running: "bg-primary", succeeded: "bg-good", failed: "bg-bad", retry_wait: "bg-warn",
  cancellation_requested: "bg-warn", cancelled: "bg-muted",
};

export function JobTimeline({ events, attempt, maxAttempts, nextRetryAt }: {
  events: TimelineEvent[] | undefined;
  attempt?: number;
  maxAttempts?: number;
  nextRetryAt?: string | null;
}) {
  if (!events?.length) return <p className="text-muted">No job transitions recorded yet.</p>;
  return (
    <div>
      {attempt !== undefined && (
        <p className="mb-3 text-sm text-muted">
          Attempt {attempt} of {maxAttempts ?? "?"}
          {nextRetryAt ? ` · next retry ${timeOnly(nextRetryAt)}` : ""}
        </p>
      )}
      <ol className="relative space-y-3 border-l-2 border-line pl-5">
        {events.map((e, i) => (
          <li key={`${e.at}-${i}`} className="relative">
            <span aria-hidden className={clsx("absolute -left-[27px] top-1.5 h-3 w-3 rounded-full ring-4 ring-surface", DOT[e.to] ?? "bg-muted")} />
            <p className="font-semibold text-ink">
              {humanize(e.to)} <span className="mono text-sm font-normal text-muted">{timeOnly(e.at)} · attempt {e.attempt}</span>
            </p>
            {e.reason && <p className="text-sm text-ink-soft">{e.reason}</p>}
          </li>
        ))}
      </ol>
    </div>
  );
}
