import type { Metadata } from "next";
import { Suspense } from "react";

import { SkeletonRows } from "@/components/ui";

import { CircularityView } from "./CircularityView";

export const metadata: Metadata = { title: "Circularity" };

export default function CircularityPage() {
  return (
    <Suspense fallback={<SkeletonRows rows={8} />}>
      <CircularityView />
    </Suspense>
  );
}
