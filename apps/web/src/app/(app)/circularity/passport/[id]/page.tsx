import type { Metadata } from "next";
import { Suspense } from "react";

import { SkeletonRows } from "@/components/ui";

import { PassportView } from "./PassportView";

export const metadata: Metadata = { title: "Casing passport" };

export default function PassportPage({ params }: PageProps<"/circularity/passport/[id]">) {
  return (
    <Suspense fallback={<SkeletonRows rows={8} />}>
      <PassportView idPromise={params.then((p) => p.id)} />
    </Suspense>
  );
}
