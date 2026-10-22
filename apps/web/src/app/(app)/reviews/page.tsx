import type { Metadata } from "next";
import { Suspense } from "react";

import { SkeletonRows } from "@/components/ui";

import { Reviews } from "./Reviews";

export const metadata: Metadata = { title: "Review queue" };

export default function ReviewsPage() {
  return (
    <Suspense fallback={<SkeletonRows rows={8} />}>
      <Reviews />
    </Suspense>
  );
}
