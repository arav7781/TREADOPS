"use client";

import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * Copilot answers as GitHub-flavoured markdown (headings, lists, tables, rules, code, links) plus [KB:id] citation
 * chips. Raw HTML is never rendered, and images are dropped so an answer cannot make the browser fetch a URL.
 */
const KB = /\[KB:([^\]\s]+)\]/g;

const components: Components = {
  h1: ({ children }) => <h3 className="pt-1 text-lg font-bold text-ink">{children}</h3>,
  h2: ({ children }) => <h3 className="pt-1 text-lg font-bold text-ink">{children}</h3>,
  h3: ({ children }) => <h4 className="pt-1 font-bold text-ink">{children}</h4>,
  h4: ({ children }) => <h5 className="font-semibold text-ink">{children}</h5>,
  h5: ({ children }) => <h6 className="font-semibold text-ink">{children}</h6>,
  h6: ({ children }) => <h6 className="font-semibold text-ink">{children}</h6>,
  p: ({ children }) => <p className="leading-relaxed">{children}</p>,
  ul: ({ children }) => <ul className="ml-5 list-disc space-y-1 marker:text-muted">{children}</ul>,
  ol: ({ children }) => <ol className="ml-5 list-decimal space-y-1 marker:font-semibold marker:text-muted">{children}</ol>,
  li: ({ children }) => <li className="pl-1 leading-relaxed [&>ol]:mt-1 [&>ul]:mt-1">{children}</li>,
  hr: () => <hr className="border-line" />,
  strong: ({ children }) => <strong className="font-bold text-ink">{children}</strong>,
  em: ({ children }) => <em className="italic">{children}</em>,
  blockquote: ({ children }) => (
    <blockquote className="rounded-r-xl border-l-4 border-warn bg-warn-soft px-3 py-2 text-ink-soft [&>*+*]:mt-2">{children}</blockquote>
  ),
  code: ({ children }) => <code className="mono rounded bg-surface-muted px-1 text-[0.9em] text-ink">{children}</code>,
  pre: ({ children }) => (
    <pre className="mono overflow-x-auto rounded-xl bg-surface-muted p-3 text-[0.85em] text-ink [&_code]:bg-transparent [&_code]:p-0">{children}</pre>
  ),
  table: ({ children }) => (
    <div className="overflow-x-auto rounded-xl border border-line">
      <table className="w-full border-collapse text-sm">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-surface-muted text-left text-ink">{children}</thead>,
  th: ({ children, style }) => <th style={style} className="border-b border-line px-3 py-1.5 font-semibold">{children}</th>,
  td: ({ children, style }) => <td style={style} className="border-t border-line px-3 py-1.5 align-top">{children}</td>,
  a: ({ href, children }) => href?.startsWith("#kb:") ? (
    <span className="mx-0.5 inline-flex rounded-full bg-accent-soft px-2 py-0.5 text-xs font-bold text-accent-ink" title="Knowledge-base citation">
      {children}
    </span>
  ) : (
    <a href={href} target="_blank" rel="noopener noreferrer nofollow" className="font-semibold text-primary underline underline-offset-2">
      {children}
    </a>
  ),
};

export function Markdown({ text }: { text: string }) {
  return (
    <div className="space-y-3 text-ink-soft">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components} disallowedElements={["img"]}>
        {text.replace(KB, "[KB:$1](#kb:$1)")}
      </ReactMarkdown>
    </div>
  );
}
