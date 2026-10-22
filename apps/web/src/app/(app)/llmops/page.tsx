import type { Metadata } from "next";
import { Suspense } from "react";

import { SkeletonRows } from "@/components/ui";

import { LLMOpsView } from "./LLMOpsView";

export const metadata: Metadata = { title: "LLMOps" };

export default function LLMOpsPage() {
  return (
    <Suspense fallback={<SkeletonRows rows={8} />}>
      <LLMOpsView />
    </Suspense>
  );
}
