"use client";

import clsx from "clsx";
import {
  Activity,
  BookCheck,
  Cctv,
  Boxes,
  Brain,
  ChevronDown,
  ClipboardList,
  Database,
  FlaskConical,
  Gauge,
  LogOut,
  Menu,
  Recycle,
  ScanLine,
  ScrollText,
  Sparkles,
  X,
} from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Suspense, useEffect, useRef, useState, type ReactNode } from "react";

import { Logo } from "@/components/BrandMark";
import { useMe } from "@/lib/hooks";
import { hasRole } from "@/lib/roles";
import type { Role } from "@/lib/types";

interface NavItem {
  href: string;
  label: string;
  icon: typeof Gauge;
  role: Role;
  hint?: string;
}

/** Top navigation in the michelin.in pattern: a few dropdown groups, a plain link and one yellow call to action. */
const GROUPS: { label: string; items: NavItem[] }[] = [
  {
    label: "Operate",
    items: [
      { href: "/dashboard", label: "Overview", icon: Gauge, role: "viewer", hint: "Throughput, queue and drift at a glance" },
      { href: "/inspections", label: "Inspections", icon: ClipboardList, role: "viewer", hint: "Every inspection, pinned to its model" },
      { href: "/cameras", label: "Cameras", icon: Cctv, role: "viewer", hint: "Live line cameras and their quality checks" },
      { href: "/reviews", label: "Review queue", icon: BookCheck, role: "viewer", hint: "Inspector labels and dispositions" },
    ],
  },
  {
    label: "MLOps",
    items: [
      { href: "/datasets", label: "Datasets", icon: Database, role: "viewer", hint: "Snapshots, splits and leakage checks" },
      { href: "/training", label: "Training", icon: Brain, role: "ml_engineer", hint: "Allowlisted recipes on one training slot" },
      { href: "/models", label: "Models", icon: Boxes, role: "viewer", hint: "Gates, evidence, promotion and rollback" },
      { href: "/monitoring", label: "Monitoring", icon: Activity, role: "viewer", hint: "Drift windows and alerts" },
      { href: "/replay", label: "Replay", icon: FlaskConical, role: "ml_engineer", hint: "Deterministic stress scenarios" },
    ],
  },
  {
    label: "Governance",
    items: [
      { href: "/llmops", label: "LLMOps", icon: Sparkles, role: "ml_engineer", hint: "Copilot traces, guardrails, evals" },
      { href: "/audit", label: "Audit", icon: ScrollText, role: "admin", hint: "Who changed what, and why" },
    ],
  },
];
const LINKS: NavItem[] = [{ href: "/circularity", label: "Circularity", icon: Recycle, role: "viewer" }];
const CTA: NavItem = { href: "/inspect", label: "Bib · Inspect & ask", icon: ScanLine, role: "viewer" };

const isActive = (pathname: string, href: string) => pathname === href || pathname.startsWith(`${href}/`);
const visible = (roles: string[] | undefined, items: NavItem[]) => items.filter((i) => roles === undefined || hasRole(roles, i.role));

function DesktopNav({ roles, pathname }: { roles: string[] | undefined; pathname: string }) {
  const [open, setOpen] = useState<string | null>(null);
  const ref = useRef<HTMLElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(null); };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(null); };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("mousedown", onDown); document.removeEventListener("keydown", onKey); };
  }, [open]);
  return (
    <nav ref={ref} aria-label="Primary" className="hidden h-full min-w-0 flex-1 items-center gap-6 lg:flex xl:gap-8">
      {GROUPS.map((g) => {
        const items = visible(roles, g.items);
        if (!items.length) return null;
        const active = items.some((i) => isActive(pathname, i.href));
        const isOpen = open === g.label;
        return (
          <div key={g.label} className="relative h-full">
            <button type="button" aria-expanded={isOpen} aria-haspopup="true" onClick={() => setOpen(isOpen ? null : g.label)}
              className={clsx("flex h-full items-center gap-1.5 border-b-[3px] pt-[3px] text-[15px] font-medium transition",
                active ? "border-brand-yellow text-white" : "border-transparent text-white/85 hover:text-white")}>
              {g.label}<ChevronDown aria-hidden className={clsx("h-4 w-4 transition", isOpen && "rotate-180")} />
            </button>
            {isOpen && (
              <div className="stage-in absolute left-0 top-full z-50 w-80 rounded-b-2xl rounded-tr-2xl bg-white p-2 text-ink shadow-2xl">
                <ul>
                  {items.map((i) => {
                    const Icon = i.icon;
                    const on = isActive(pathname, i.href);
                    return (
                      <li key={i.href}>
                        <Link href={i.href} onClick={() => setOpen(null)} aria-current={on ? "page" : undefined}
                          aria-label={i.label} aria-describedby={i.hint ? `hint-${i.href.slice(1)}` : undefined}
                          className={clsx("flex items-start gap-3 rounded-xl px-3 py-2.5 transition",
                            on ? "bg-accent-soft shadow-[inset_3px_0_0_var(--brand-yellow)]" : "hover:bg-surface-muted")}>
                          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary-soft text-primary"><Icon aria-hidden className="h-5 w-5" /></span>
                          <span className="min-w-0">
                            <span className="block font-semibold">{i.label}</span>
                            {i.hint && <span id={`hint-${i.href.slice(1)}`} className="block text-sm text-muted">{i.hint}</span>}
                          </span>
                        </Link>
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}
          </div>
        );
      })}
      {visible(roles, LINKS).map((l) => (
        <Link key={l.href} href={l.href} aria-current={isActive(pathname, l.href) ? "page" : undefined}
          className={clsx("flex h-full items-center border-b-[3px] pt-[3px] text-[15px] font-medium transition",
            isActive(pathname, l.href) ? "border-brand-yellow text-white" : "border-transparent text-white/85 hover:text-white")}>
          {l.label}
        </Link>
      ))}
      <Link href={CTA.href} aria-label={CTA.label} aria-current={isActive(pathname, CTA.href) ? "page" : undefined}
        className="ml-auto flex h-11 items-center gap-2 rounded-lg bg-brand-yellow px-4 font-semibold text-black transition hover:bg-brand-yellow-strong">
        <ScanLine aria-hidden className="h-5 w-5" /><span>Bib<span className="hidden xl:inline"> · Inspect &amp; ask</span></span>
      </Link>
    </nav>
  );
}

function DesktopNavWithPath({ roles }: { roles: string[] | undefined }) {
  return <DesktopNav roles={roles} pathname={usePathname()} />;
}

function MobileNav({ roles, onNavigate }: { roles: string[] | undefined; onNavigate: () => void }) {
  return (
    <nav aria-label="Primary" className="space-y-5 border-t border-white/10 bg-black px-4 pb-6 pt-4 text-white lg:hidden">
      <Link href={CTA.href} onClick={onNavigate} className="flex h-12 items-center justify-center gap-2 rounded-lg bg-brand-yellow font-semibold text-black">
        <ScanLine aria-hidden className="h-5 w-5" />{CTA.label}
      </Link>
      {[...GROUPS, { label: "More", items: LINKS }].map((g) => {
        const items = visible(roles, g.items);
        if (!items.length) return null;
        return (
          <div key={g.label}>
            <p className="mb-1 text-xs font-bold uppercase tracking-[0.16em] text-brand-yellow">{g.label}</p>
            <ul className="grid grid-cols-2 gap-1">
              {items.map((i) => {
                const Icon = i.icon;
                return (
                  <li key={i.href}>
                    <Link href={i.href} onClick={onNavigate} className="flex items-center gap-2 rounded-lg px-2 py-2 text-white/90 hover:bg-white/10">
                      <Icon aria-hidden className="h-4 w-4 text-white/60" />{i.label}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        );
      })}
    </nav>
  );
}

/** Watches for 401s from any API call and sends the user to /login with a return path. */
function AuthWatcher() {
  const router = useRouter();
  const pathname = usePathname();
  useEffect(() => {
    const onUnauthorized = () => router.replace(`/login?expired=1&next=${encodeURIComponent(pathname)}`);
    window.addEventListener("treadops:unauthorized", onUnauthorized);
    return () => window.removeEventListener("treadops:unauthorized", onUnauthorized);
  }, [router, pathname]);
  return null;
}

export function AppShell({ children }: { children: ReactNode }) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const me = useMe();

  const signOut = async () => {
    await fetch("/api/auth/logout", { method: "POST" }).catch(() => null);
    router.replace("/login");
  };

  const roles = me.data?.roles;
  return (
    <div className="flex min-h-screen flex-col">
      <Suspense fallback={null}><AuthWatcher /></Suspense>
      <a href="#main" className="sr-only-focusable absolute left-2 top-2 z-50 rounded-lg bg-brand-yellow px-3 py-2 text-black">
        Skip to content
      </a>
      <header className="sticky top-0 z-40 bg-black text-white">
        <div className="mx-auto flex h-16 max-w-[1400px] items-center gap-3 px-4 lg:px-8">
          <Link href="/dashboard" className="mr-4 flex shrink-0 items-center xl:mr-8" aria-label="TreadOps home">
            <Logo onDark priority className="h-8 w-auto" />
          </Link>
          <Suspense fallback={<DesktopNav roles={roles} pathname="" />}>
            <DesktopNavWithPath roles={roles} />
          </Suspense>
          <div className="ml-auto flex items-center gap-2 lg:ml-0">
            <button type="button" onClick={signOut}
              className="flex h-11 items-center gap-2 rounded-lg px-3 text-sm font-semibold text-white ring-1 ring-inset ring-white/80 transition hover:bg-white/10 sm:px-4">
              <LogOut aria-hidden className="h-4 w-4" /><span className="hidden sm:inline">Sign out</span><span className="sr-only sm:hidden">Sign out</span>
            </button>
            <button type="button" className="rounded-lg p-2.5 text-white hover:bg-white/10 lg:hidden"
              aria-label={open ? "Close menu" : "Open menu"} aria-expanded={open} onClick={() => setOpen((o) => !o)}>
              {open ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </button>
          </div>
        </div>
        {open && <MobileNav roles={roles} onNavigate={() => setOpen(false)} />}
      </header>

      <main id="main" className="mx-auto w-full max-w-[1400px] flex-1 px-4 py-6 lg:px-8 lg:py-8">{children}</main>

      <footer className="site-footer bg-navy text-white">
        <div className="mx-auto flex max-w-[1400px] flex-col gap-3 px-4 py-8 lg:flex-row lg:items-center lg:gap-10 lg:px-8">
          <Logo onDark className="h-8 w-auto shrink-0" />
          <p className="text-sm text-white/70">
            Triage aid only: final clearance requires a qualified inspector. Public photographs; operational metadata in
            replays is simulated. Independent prototype, not affiliated with or endorsed by Michelin.
          </p>
        </div>
      </footer>
    </div>
  );
}

export { useMe };
