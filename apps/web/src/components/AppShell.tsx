"use client";

import clsx from "clsx";
import {
  Activity,
  BookCheck,
  Bot,
  Boxes,
  Brain,
  ClipboardList,
  Database,
  FlaskConical,
  Gauge,
  LogOut,
  Menu,
  Recycle,
  ScrollText,
  Sparkles,
  X,
} from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Suspense, useEffect, useState, type ReactNode } from "react";

import { shortSha } from "@/lib/format";
import { useMe, useModels } from "@/lib/hooks";
import { hasRole, primaryRole } from "@/lib/roles";
import type { Role } from "@/lib/types";

interface NavItem {
  href: string;
  label: string;
  icon: typeof Gauge;
  role: Role;
}

const NAV: { section: string; items: NavItem[] }[] = [
  {
    section: "Operate",
    items: [
      { href: "/dashboard", label: "Overview", icon: Gauge, role: "viewer" },
      { href: "/inspect", label: "Bib · Inspect & ask", icon: Bot, role: "viewer" },
      { href: "/inspections", label: "Inspections", icon: ClipboardList, role: "viewer" },
      { href: "/reviews", label: "Review queue", icon: BookCheck, role: "viewer" },
      { href: "/circularity", label: "Circularity", icon: Recycle, role: "viewer" },
    ],
  },
  {
    section: "MLOps",
    items: [
      { href: "/datasets", label: "Datasets", icon: Database, role: "viewer" },
      { href: "/training", label: "Training", icon: Brain, role: "ml_engineer" },
      { href: "/models", label: "Models", icon: Boxes, role: "viewer" },
      { href: "/monitoring", label: "Monitoring", icon: Activity, role: "viewer" },
      { href: "/replay", label: "Replay", icon: FlaskConical, role: "ml_engineer" },
      { href: "/llmops", label: "LLMOps", icon: Sparkles, role: "ml_engineer" },
      { href: "/audit", label: "Audit", icon: ScrollText, role: "admin" },
    ],
  },
];

function Nav({ roles, onNavigate }: { roles: string[] | undefined; onNavigate?: () => void }) {
  return (
    <Suspense fallback={<NavList roles={roles} pathname="" />}>
      <NavWithPath roles={roles} onNavigate={onNavigate} />
    </Suspense>
  );
}

function NavWithPath({ roles, onNavigate }: { roles: string[] | undefined; onNavigate?: () => void }) {
  return <NavList roles={roles} pathname={usePathname()} onNavigate={onNavigate} />;
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

function NavList({ roles, pathname, onNavigate }: { roles: string[] | undefined; pathname: string; onNavigate?: () => void }) {
  return (
    <nav aria-label="Primary" className="flex flex-col gap-5">
      {NAV.map((group) => {
        const items = group.items.filter((i) => roles === undefined || hasRole(roles, i.role));
        if (!items.length) return null;
        return (
          <div key={group.section}>
            <p className="mb-1.5 px-3 text-xs font-bold uppercase tracking-[0.14em] text-white/45">{group.section}</p>
            <ul className="flex flex-col gap-0.5">
              {items.map((item) => {
                const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
                const Icon = item.icon;
                return (
                  <li key={item.href}>
                    <Link href={item.href} onClick={onNavigate} aria-current={active ? "page" : undefined}
                      className={clsx("flex items-center gap-3 rounded-lg px-3 py-2 font-semibold transition",
                        active ? "bg-white/12 text-white shadow-[inset_3px_0_0_var(--brand-yellow)]" : "text-white/75 hover:bg-white/8 hover:text-white")}>
                      <Icon aria-hidden className={clsx("h-5 w-5", active ? "text-brand-yellow" : "text-white/55")} />
                      {item.label}
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

export function AppShell({ children }: { children: ReactNode }) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const me = useMe();
  const models = useModels();

  const signOut = async () => {
    await fetch("/api/auth/logout", { method: "POST" }).catch(() => null);
    router.replace("/login");
  };

  const active = models.data?.active;
  const roles = me.data?.roles;
  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[256px_1fr]">
      <Suspense fallback={null}><AuthWatcher /></Suspense>
      <a href="#main" className="sr-only-focusable absolute left-2 top-2 z-50 rounded-lg bg-primary px-3 py-2 text-white">
        Skip to content
      </a>
      <aside className="hidden bg-navy px-4 py-5 lg:sticky lg:top-0 lg:flex lg:h-dvh lg:flex-col lg:overflow-y-auto">
        <Link href="/dashboard" className="mb-7 flex items-center gap-2.5 px-2">
          <BrandMark />
          <span className="font-display text-2xl font-extrabold italic tracking-tight text-white">TREADOPS</span>
        </Link>
        <Nav roles={roles} />
        <p className="mt-auto px-3 pt-8 text-xs leading-relaxed text-white/50">
          Triage aid only. Final clearance requires a qualified inspector. Public photos; operational metadata in
          replays is simulated.
        </p>
      </aside>

      <div className="flex min-w-0 flex-col has-[[data-fullbleed]]:h-dvh">
        <header className="sticky top-0 z-30 flex flex-wrap items-center gap-3 border-b border-line bg-surface/90 px-4 py-3 backdrop-blur lg:px-8">
          <button type="button" className="rounded-lg p-2 text-ink-soft hover:bg-surface-muted lg:hidden"
            aria-label={open ? "Close menu" : "Open menu"} aria-expanded={open} onClick={() => setOpen((o) => !o)}>
            {open ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
          </button>
          <span className="font-display font-extrabold italic tracking-tight text-navy lg:hidden">TREADOPS</span>
          <span className="rounded-full bg-sim-soft px-2.5 py-0.5 text-sm font-bold text-sim">LOCAL · DEMO DATA</span>
          <span className="hidden items-center gap-1.5 text-sm text-ink-soft sm:flex" aria-live="polite">
            <span aria-hidden className={clsx("h-2.5 w-2.5 rounded-full", active ? "bg-good" : "bg-warn")} />
            {models.isLoading ? "Loading model…" : models.isError ? "Model status unavailable" : active
              ? <>Model <strong className="text-ink">{active.model_family} v{active.registry_version ?? "?"}</strong>
                <span className="mono hidden text-muted md:inline">({shortSha(active.bundle_sha256, 8)} · gen {models.data?.generation})</span></>
              : "No trained model available"}
          </span>
          <div className="ml-auto flex items-center gap-3">
            {me.data && (
              <span className="hidden text-right text-sm leading-tight sm:block">
                <span className="block font-semibold text-ink">{me.data.display_name}</span>
                <span className="block capitalize text-muted">{primaryRole(me.data.roles)}</span>
              </span>
            )}
            <button type="button" onClick={signOut} className="flex items-center gap-1.5 rounded-xl px-3 py-2 text-sm font-semibold text-ink-soft hover:bg-surface-muted">
              <LogOut aria-hidden className="h-4 w-4" /> <span className="hidden sm:inline">Sign out</span><span className="sr-only sm:hidden">Sign out</span>
            </button>
          </div>
        </header>
        {open && (
          <div className="border-b border-line bg-navy px-4 py-4 lg:hidden">
            <Nav roles={roles} onNavigate={() => setOpen(false)} />
          </div>
        )}
        {/* A page that marks itself data-fullbleed (the Bib chat) gets the whole viewport below the header. */}
        <main id="main" className="mx-auto w-full max-w-[1400px] flex-1 px-4 py-6 lg:px-8 lg:py-8 has-[[data-fullbleed]]:flex has-[[data-fullbleed]]:min-h-0 has-[[data-fullbleed]]:max-w-none has-[[data-fullbleed]]:p-0 lg:has-[[data-fullbleed]]:p-0">{children}</main>
      </div>
    </div>
  );
}

/** TreadOps' own mark: a casing in profile with tread blocks (not a Michelin logo). */
function BrandMark() {
  return (
    <svg aria-hidden viewBox="0 0 36 36" className="h-9 w-9 shrink-0">
      <circle cx="18" cy="18" r="17" fill="var(--brand-yellow)" />
      <circle cx="18" cy="18" r="11" fill="none" stroke="var(--navy)" strokeWidth="4" strokeDasharray="4.2 2.6" />
      <circle cx="18" cy="18" r="4.5" fill="var(--navy)" />
    </svg>
  );
}

export { useMe };
