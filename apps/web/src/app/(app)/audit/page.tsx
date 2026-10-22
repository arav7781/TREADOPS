import type { Metadata } from "next";
import { Suspense } from "react";

import { SkeletonRows } from "@/components/ui";

import { AuditView } from "./AuditView";

export const metadata: Metadata = { title: "Audit" };

export default function AuditPage() {
  return (
    <Suspense fallback={<SkeletonRows rows={8} />}>
      <AuditView />
    </Suspense>
  );
}
