import type { Metadata } from "next";
import { Suspense } from "react";

import { SkeletonRows } from "@/components/ui";

import { TrainingView } from "./TrainingView";

export const metadata: Metadata = { title: "Training" };

export default function TrainingPage() {
  return (
    <Suspense fallback={<SkeletonRows rows={8} />}>
      <TrainingView />
    </Suspense>
  );
}
