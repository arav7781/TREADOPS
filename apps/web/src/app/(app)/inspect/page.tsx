import type { Metadata } from "next";
import { Suspense } from "react";

import { SkeletonRows } from "@/components/ui";

import { Inspect } from "./Inspect";

export const metadata: Metadata = { title: "Inspect" };

export default function InspectPage() {
  return (
    <Suspense fallback={<SkeletonRows rows={6} />}>
      <Inspect />
    </Suspense>
  );
}
