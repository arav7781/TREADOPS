import type { Metadata } from "next";
import { Suspense } from "react";

import { SkeletonRows } from "@/components/ui";

import { ModelsView } from "./ModelsView";

export const metadata: Metadata = { title: "Models" };

export default function ModelsPage() {
  return (
    <Suspense fallback={<SkeletonRows rows={8} />}>
      <ModelsView />
    </Suspense>
  );
}
