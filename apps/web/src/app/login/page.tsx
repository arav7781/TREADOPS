import type { Metadata } from "next";
import { Suspense } from "react";

import { LoginForm } from "./LoginForm";

export const metadata: Metadata = { title: "Sign in" };

export default function LoginPage() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-gradient-to-br from-primary-soft via-canvas to-accent-soft px-4 py-10">
      <div className="w-full max-w-md">
        <div className="mb-6 text-center">
          <span aria-hidden className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-primary to-accent text-2xl font-black text-white shadow-lg">T</span>
          <h1 className="text-3xl font-black tracking-tight text-ink">TREADOPS</h1>
          <p className="mt-1 text-ink-soft">Casing triage · MLOps release loop · circularity copilot</p>
        </div>
        <Suspense fallback={<div className="card h-80 animate-pulse" />}>
          <LoginForm />
        </Suspense>
        <p className="mt-6 text-center text-sm text-muted">
          Local prototype with seeded fixture accounts. Public photographs (Mendeley D1, CC BY 4.0); replay metadata is
          simulated. Not Michelin production data.
        </p>
      </div>
    </main>
  );
}
