import type { Metadata } from "next";
import { Suspense } from "react";

import { SkeletonRows } from "@/components/ui";

import { Dashboard } from "./Dashboard";

export const metadata: Metadata = { title: "Overview" };

export default function DashboardPage() {
  return (
    <Suspense fallback={<SkeletonRows rows={6} />}>
      <Dashboard />
    </Suspense>
  );
}
