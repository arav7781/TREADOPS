import type { Metadata } from "next";
import { Suspense } from "react";

import { SkeletonRows } from "@/components/ui";

import { DatasetsView } from "./DatasetsView";

export const metadata: Metadata = { title: "Datasets" };

export default function DatasetsPage() {
  return (
    <Suspense fallback={<SkeletonRows rows={8} />}>
      <DatasetsView />
    </Suspense>
  );
}
