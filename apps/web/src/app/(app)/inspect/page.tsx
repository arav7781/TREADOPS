import type { Metadata } from "next";
import { Suspense } from "react";

import { SkeletonRows } from "@/components/ui";

import { BibView } from "./BibView";

export const metadata: Metadata = { title: "Bib · Inspect & ask" };

export default function BibPage() {
  return (
    <Suspense fallback={<SkeletonRows rows={8} />}>
      <BibView />
    </Suspense>
  );
}
