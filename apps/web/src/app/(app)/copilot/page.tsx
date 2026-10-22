import type { Metadata } from "next";
import { Suspense } from "react";

import { SkeletonRows } from "@/components/ui";

import { CopilotView } from "./CopilotView";

export const metadata: Metadata = { title: "Copilot" };

export default function CopilotPage() {
  return (
    <Suspense fallback={<SkeletonRows rows={8} />}>
      <CopilotView />
    </Suspense>
  );
}
