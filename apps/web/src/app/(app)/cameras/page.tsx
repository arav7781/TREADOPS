import type { Metadata } from "next";
import { Suspense } from "react";

import { SkeletonRows } from "@/components/ui";

import { CamerasView } from "./CamerasView";

export const metadata: Metadata = { title: "Cameras" };

export default function CamerasPage() {
  return (
    <Suspense fallback={<SkeletonRows rows={8} />}>
      <CamerasView />
    </Suspense>
  );
}
