"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";

import { Button, inputClass } from "@/components/ui";

export function LoginForm() {
  const router = useRouter();
  const params = useSearchParams();
  const expired = params.get("expired") === "1";
  const next = params.get("next");
  const safeNext = next && next.startsWith("/") && !next.startsWith("//") ? next : "/dashboard";
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ email: email.trim().toLowerCase(), password }),
      });
      if (!res.ok) {
        const body = (await res.json().catch(() => null)) as { error?: { code?: string } } | null;
        // Generic message: never reveal whether the account exists.
        setError(body?.error?.code === "api_unavailable" ? "The TreadOps API is unavailable. Try again shortly."
          : "Email or password is incorrect.");
        return;
      }
      router.replace(safeNext);
      router.refresh();
    } catch {
      setError("Could not reach the server.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <form onSubmit={submit} className="card space-y-4 p-6" aria-label="Sign in">
      {expired && (
        <p role="status" className="rounded-xl bg-warn-soft px-3 py-2 text-ink">Your session expired. Sign in again to continue.</p>
      )}
      <div className="flex flex-col gap-1">
        <label htmlFor="email" className="text-sm font-semibold text-ink-soft">Email</label>
        <input id="email" type="email" autoComplete="username" required className={inputClass}
          value={email} onChange={(e) => setEmail(e.target.value)} placeholder="inspector@treadops.local" />
      </div>
      <div className="flex flex-col gap-1">
        <label htmlFor="password" className="text-sm font-semibold text-ink-soft">Password</label>
        <input id="password" type="password" autoComplete="current-password" required className={inputClass}
          value={password} onChange={(e) => setPassword(e.target.value)} />
      </div>
      {error && <p role="alert" className="text-bad">{error}</p>}
      <Button type="submit" busy={busy} className="w-full">Sign in</Button>
      <p className="text-sm text-muted">Demo roles: viewer, inspector, mlengineer, admin @treadops.local (password from <code>.env</code> DEMO_PASSWORD).</p>
    </form>
  );
}
