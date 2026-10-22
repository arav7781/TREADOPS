import type { Metadata } from "next";
import { Suspense } from "react";

import { SkeletonRows } from "@/components/ui";

import { Inspections } from "./Inspections";

export const metadata: Metadata = { title: "Inspections" };

export default function InspectionsPage() {
  return (
    <Suspense fallback={<SkeletonRows rows={8} />}>
      <Inspections />
    </Suspense>
  );
}
