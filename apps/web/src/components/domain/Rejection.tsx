"use client";

import clsx from "clsx";
import { ImageOff } from "lucide-react";

import type { Rejection } from "@/lib/types";

/** An image the subject gate refused: why, with the measured score and the vision model's description. */
export function RejectionPanel({ r, className }: { r: Rejection; className?: string }) {
  const s = r.subject ?? {};
  return (
    <section aria-label="Image rejected" data-testid="rejected" className={clsx("rounded-xl border border-warn/40 bg-warn-soft p-3", className)}>
      <div className="flex flex-wrap items-center gap-2">
        <ImageOff aria-hidden className="h-4 w-4 text-warn" />
        <span className="font-bold text-ink">Not a tyre photo · no verdict</span>
      </div>
      <p className="mt-1 text-sm text-ink-soft">{r.message}</p>
      <ul className="ml-5 mt-1 list-disc text-sm text-ink">
        {typeof s.score === "number" && (
          <li>Unusual-image score {s.score.toFixed(3)} (unusual above {s.unusual_above}): unlike the tyre photos the model was trained on</li>
        )}
        {s.vision_tyre_visible === false && (
          <li>Vision model sees no tyre{s.vision_subject ? `: “${s.vision_subject}”` : ""}</li>
        )}
      </ul>
      <p className="mono mt-1 text-xs text-muted">
        subject gate {s.version ?? "?"}{r.vision?.model ? ` · ${r.vision.model} · prompt ${r.vision.prompt_version}` : ""} · both checks must agree to reject
      </p>
    </section>
  );
}
