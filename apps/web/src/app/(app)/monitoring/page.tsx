import type { Metadata } from "next";
import { Suspense } from "react";

import { SkeletonRows } from "@/components/ui";

import { MonitoringView } from "./MonitoringView";

export const metadata: Metadata = { title: "Monitoring" };

export default function MonitoringPage() {
  return (
    <Suspense fallback={<SkeletonRows rows={8} />}>
      <MonitoringView />
    </Suspense>
  );
}
