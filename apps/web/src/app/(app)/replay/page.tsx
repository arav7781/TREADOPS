import type { Metadata } from "next";
import { Suspense } from "react";

import { SkeletonRows } from "@/components/ui";

import { ReplayView } from "./ReplayView";

export const metadata: Metadata = { title: "Replay" };

export default function ReplayPage() {
  return (
    <Suspense fallback={<SkeletonRows rows={8} />}>
      <ReplayView />
    </Suspense>
  );
}
