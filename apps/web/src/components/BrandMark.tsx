import Image from "next/image";

import logo from "../../public/brand/treadops-logo.png";
import logoOnDark from "../../public/brand/treadops-logo-on-dark.png";
import mark from "../../public/brand/treadops-mark.png";
import markOnDark from "../../public/brand/treadops-mark-on-dark.png";

/**
 * The TreadOps logo (supplied by the authors). `onDark` swaps the blue wordmark for white so it stays legible on
 * the black header, navy footer and sign-in page; the yellow stays.
 */
export function Logo({ onDark = false, className = "h-9 w-auto", priority = false }: {
  onDark?: boolean;
  className?: string;
  priority?: boolean;
}) {
  return <Image src={onDark ? logoOnDark : logo} alt="TreadOps" className={className} priority={priority} />;
}

/** Just the tyre "T" mark, for tight spaces. */
export function LogoMark({ onDark = false, className = "h-9 w-9" }: { onDark?: boolean; className?: string }) {
  return <Image src={onDark ? markOnDark : mark} alt="" aria-hidden className={className} />;
}
