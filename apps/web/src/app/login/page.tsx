import type { Metadata } from "next";
import { Suspense } from "react";

import { BrandMark, Wordmark } from "@/components/BrandMark";
import { TreadArc } from "@/components/ui";

import { LoginForm } from "./LoginForm";

export const metadata: Metadata = { title: "Sign in" };

export default function LoginPage() {
  return (
    <main className="relative flex min-h-screen flex-col overflow-hidden bg-black text-white">
      <TreadArc className="-right-40 top-24 h-[42rem] w-[42rem] text-white/[0.05]" />
      <header className="relative flex h-16 items-center gap-2.5 px-4 lg:px-10">
        <BrandMark />
        <Wordmark />
      </header>
      <div className="relative mx-auto grid w-full max-w-6xl flex-1 items-center gap-10 px-4 py-10 lg:grid-cols-[1.1fr_1fr] lg:px-8">
        <div>
          <p className="text-sm font-bold uppercase tracking-[0.16em] text-brand-yellow">Casing intake for retreading</p>
          <h1 className="headline mt-3 text-4xl sm:text-6xl">Every casing,<br /><span className="headline-light">decided with evidence</span></h1>
          <p className="mt-4 max-w-lg text-white/75">
            YOLO triage of a phone photo, a governed MLOps release loop and Bib, a guarded copilot, so more casings
            get a second life and every claim is traceable.
          </p>
        </div>
        <div className="rounded-3xl bg-navy p-3 shadow-2xl sm:p-5">
          <p className="mb-3 inline-flex border-b-2 border-white px-1 pb-2 font-semibold">Sign in</p>
          <div className="text-ink">
            <Suspense fallback={<div className="card h-80 animate-pulse" />}>
              <LoginForm />
            </Suspense>
          </div>
        </div>
      </div>
      <p className="relative px-4 pb-6 text-center text-sm text-white/55">
        Local prototype with seeded fixture accounts. Public photographs (Mendeley D1, CC BY 4.0); replay metadata is
        simulated. Not Michelin production data; not affiliated with or endorsed by Michelin.
      </p>
    </main>
  );
}
