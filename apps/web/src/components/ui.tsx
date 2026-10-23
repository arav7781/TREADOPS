"use client";

import clsx from "clsx";
import { AlertTriangle, Inbox, Lock, RefreshCw } from "lucide-react";
import type { ButtonHTMLAttributes, ReactNode } from "react";

import { ApiError } from "@/lib/api";

export function Card({ children, className, title, subtitle, actions, as: As = "section" }: {
  children: ReactNode;
  className?: string;
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  as?: "section" | "div" | "article";
}) {
  return (
    <As className={clsx("card min-w-0 p-5", className)}>
      {(title || actions) && (
        <header className="mb-4 flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            {title && <h2 className="text-lg font-bold text-ink">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-sm text-muted">{subtitle}</p>}
          </div>
          {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
        </header>
      )}
      {children}
    </As>
  );
}

export type Tone = "neutral" | "primary" | "accent" | "good" | "warn" | "bad" | "info" | "sim";

const TONE: Record<Tone, string> = {
  neutral: "bg-surface-muted text-ink-soft ring-line",
  primary: "bg-primary-soft text-primary-ink ring-primary/25",
  accent: "bg-accent-soft text-accent-ink ring-accent/25",
  good: "bg-good-soft text-good ring-good/25",
  warn: "bg-warn-soft text-warn ring-warn/25",
  bad: "bg-bad-soft text-bad ring-bad/25",
  info: "bg-info-soft text-info ring-info/25",
  sim: "bg-sim-soft text-sim ring-sim/30",
};

export function Badge({ tone = "neutral", children, className, title }: {
  tone?: Tone;
  children: ReactNode;
  className?: string;
  title?: string;
}) {
  return (
    <span title={title} className={clsx(
      "inline-flex items-center gap-1 whitespace-nowrap rounded-full px-2.5 py-0.5 text-sm font-semibold ring-1 ring-inset",
      TONE[tone], className)}>
      {children}
    </span>
  );
}

type Variant = "primary" | "secondary" | "ghost" | "danger";
const VARIANT: Record<Variant, string> = {
  primary: "bg-primary text-white hover:bg-primary-strong disabled:bg-primary/40",
  secondary: "bg-surface text-ink ring-1 ring-inset ring-line-strong hover:bg-surface-muted disabled:text-muted",
  ghost: "text-ink-soft hover:bg-surface-muted disabled:text-muted",
  danger: "bg-bad text-white hover:brightness-95 disabled:bg-bad/40",
};

export function Button({ variant = "primary", className, children, busy, ...rest }:
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; busy?: boolean }) {
  return (
    <button
      {...rest}
      disabled={rest.disabled || busy}
      aria-busy={busy || undefined}
      className={clsx(
        "inline-flex min-h-10 items-center justify-center gap-2 rounded-xl px-4 py-2 text-base font-semibold transition",
        "disabled:cursor-not-allowed", VARIANT[variant], className)}
    >
      {busy && <RefreshCw aria-hidden className="h-4 w-4 animate-spin" />}
      {children}
    </button>
  );
}

export function Skeleton({ className, label = "Loading" }: { className?: string; label?: string }) {
  return <div role="status" aria-label={label} className={clsx("animate-pulse rounded-xl bg-line/70", className)} />;
}

export function SkeletonRows({ rows = 4 }: { rows?: number }) {
  return (
    <div className="space-y-2" role="status" aria-label="Loading">
      {Array.from({ length: rows }, (_, i) => <div key={i} className="h-10 animate-pulse rounded-xl bg-line/60" />)}
    </div>
  );
}

export function EmptyState({ title, children, icon }: { title: string; children?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-line-strong bg-surface-muted px-6 py-10 text-center">
      <span aria-hidden className="text-muted">{icon ?? <Inbox className="h-7 w-7" />}</span>
      <p className="font-bold text-ink">{title}</p>
      {children && <div className="max-w-prose text-muted">{children}</div>}
    </div>
  );
}

/** Renders the shared error contract: code, message and request id, plus a retry path. */
export function ErrorState({ error, onRetry, title }: { error: unknown; onRetry?: () => void; title?: string }) {
  const e = error instanceof ApiError ? error : null;
  if (e?.status === 403) {
    return (
      <div role="alert" className="flex items-start gap-3 rounded-2xl border border-warn/30 bg-warn-soft p-4">
        <Lock aria-hidden className="mt-0.5 h-5 w-5 text-warn" />
        <div>
          <p className="font-bold text-ink">Access denied</p>
          <p className="text-ink-soft">{e.message}</p>
          {e.requestId && <p className="mono mt-1 text-sm text-muted">request {e.requestId}</p>}
        </div>
      </div>
    );
  }
  const message = e?.message ?? (error instanceof Error ? error.message : "Something went wrong.");
  return (
    <div role="alert" className="flex flex-wrap items-start gap-3 rounded-2xl border border-bad/30 bg-bad-soft p-4">
      <AlertTriangle aria-hidden className="mt-0.5 h-5 w-5 text-bad" />
      <div className="min-w-0 flex-1">
        <p className="font-bold text-ink">{title ?? "Could not load this data"}</p>
        <p className="text-ink-soft">{message}</p>
        <p className="mono mt-1 text-sm text-muted">
          {e ? `${e.code}${e.status ? ` · HTTP ${e.status}` : ""}` : "client_error"}
          {e?.requestId ? ` · request ${e.requestId}` : ""}
        </p>
      </div>
      {onRetry && (
        <Button variant="secondary" onClick={onRetry}>
          <RefreshCw aria-hidden className="h-4 w-4" /> Retry
        </Button>
      )}
    </div>
  );
}

const TONE_TEXT: Record<Tone, string> = {
  neutral: "text-muted", primary: "text-primary", accent: "text-accent", good: "text-good", warn: "text-warn",
  bad: "text-bad", info: "text-info", sim: "text-sim",
};

export function Stat({ label, value, hint, tone, icon }: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: Tone;
  icon?: ReactNode;
}) {
  return (
    <div className="card flex flex-col gap-1 p-4">
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-semibold uppercase tracking-wide text-muted">{label}</span>
        {icon && <span aria-hidden className={TONE_TEXT[tone ?? "primary"]}>{icon}</span>}
      </div>
      <span className="text-2xl font-extrabold text-ink">{value}</span>
      {hint && <span className="text-sm text-muted">{hint}</span>}
    </div>
  );
}

export function Field({ label, htmlFor, children, hint }: { label: string; htmlFor: string; children: ReactNode; hint?: ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={htmlFor} className="text-sm font-semibold text-ink-soft">{label}</label>
      {children}
      {hint && <span className="text-sm text-muted">{hint}</span>}
    </div>
  );
}

export const inputClass =
  "min-h-10 w-full rounded-xl border border-line-strong bg-surface px-3 py-2 text-base text-ink placeholder:text-muted focus:border-primary";

export function PageHeader({ title, subtitle, actions, eyebrow }: {
  title: string;
  subtitle?: ReactNode;
  actions?: ReactNode;
  eyebrow?: string;
}) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0">
        {eyebrow && <p className="text-sm font-bold uppercase tracking-wider text-accent-ink">{eyebrow}</p>}
        <h1 className="text-3xl font-extrabold tracking-tight text-ink">{title}</h1>
        {subtitle && <p className="mt-1 max-w-3xl text-ink-soft">{subtitle}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}

export function KeyValue({ items, className }: { items: [ReactNode, ReactNode][]; className?: string }) {
  return (
    <dl className={clsx("grid grid-cols-[minmax(0,auto)_1fr] gap-x-4 gap-y-1.5 text-base", className)}>
      {items.map(([k, v], i) => (
        <div key={i} className="contents">
          <dt className="text-muted">{k}</dt>
          <dd className="min-w-0 break-words text-ink">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function ProgressBar({ value, label }: { value: number; label: string }) {
  const v = Math.max(0, Math.min(1, value));
  return (
    <div className="w-full">
      <div className="h-2.5 w-full overflow-hidden rounded-full bg-line" role="progressbar" aria-label={label}
        aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(v * 100)}>
        <div className="h-full rounded-full bg-gradient-to-r from-primary to-accent transition-all" style={{ width: `${v * 100}%` }} />
      </div>
    </div>
  );
}
