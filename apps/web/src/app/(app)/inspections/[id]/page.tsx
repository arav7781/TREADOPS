import type { Metadata } from "next";
import { Suspense } from "react";

import { SkeletonRows } from "@/components/ui";

import { InspectionView } from "./InspectionView";

export const metadata: Metadata = { title: "Inspection" };

export default function InspectionPage({ params }: PageProps<"/inspections/[id]">) {
  return (
    <Suspense fallback={<SkeletonRows rows={8} />}>
      <InspectionView idPromise={params.then((p) => p.id)} />
    </Suspense>
  );
}
