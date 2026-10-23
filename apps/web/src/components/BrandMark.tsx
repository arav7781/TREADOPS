/** TreadOps' own mark: a casing in profile with tread blocks. Deliberately not a Michelin logo. */
export function BrandMark({ className = "h-9 w-9" }: { className?: string }) {
  return (
    <svg aria-hidden viewBox="0 0 36 36" className={`shrink-0 ${className}`}>
      <circle cx="18" cy="18" r="17" fill="var(--brand-yellow)" />
      <circle cx="18" cy="18" r="11" fill="none" stroke="#000" strokeWidth="4" strokeDasharray="4.2 2.6" />
      <circle cx="18" cy="18" r="4.5" fill="#000" />
    </svg>
  );
}

export function Wordmark({ className = "text-2xl" }: { className?: string }) {
  return <span className={`headline text-white ${className}`}>TreadOps</span>;
}
