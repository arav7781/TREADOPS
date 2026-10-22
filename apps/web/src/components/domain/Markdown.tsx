"use client";

import { Fragment, type ReactNode } from "react";

/**
 * Minimal, safe markdown for copilot answers: paragraphs, **bold**, `code`, "- " lists, "> " quotes, and
 * [KB:id] citation chips. Builds React nodes only - never injects HTML.
 */
function inline(text: string, keyBase: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\[KB:[^\]]+\]|_[^_]+_)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const tok = m[0];
    const k = `${keyBase}-${i++}`;
    if (tok.startsWith("**")) out.push(<strong key={k} className="font-bold text-ink">{tok.slice(2, -2)}</strong>);
    else if (tok.startsWith("`")) out.push(<code key={k} className="mono rounded bg-surface-muted px-1 text-[0.9em]">{tok.slice(1, -1)}</code>);
    else if (tok.startsWith("[KB:")) out.push(
      <span key={k} className="mx-0.5 inline-flex rounded-full bg-accent-soft px-2 py-0.5 text-xs font-bold text-accent-ink" title="Knowledge-base citation">
        {tok.slice(1, -1)}
      </span>);
    else out.push(<em key={k} className="text-muted">{tok.slice(1, -1)}</em>);
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export function Markdown({ text }: { text: string }) {
  const blocks: ReactNode[] = [];
  let list: string[] = [];
  let quote: string[] = [];
  let para: string[] = [];
  const flushPara = () => {
    if (!para.length) return;
    const k = `p${blocks.length}`;
    blocks.push(<p key={k} className="leading-relaxed">{inline(para.join(" "), k)}</p>);
    para = [];
  };
  const flushList = () => {
    if (!list.length) return;
    const k = `l${blocks.length}`;
    blocks.push(<ul key={k} className="ml-5 list-disc space-y-1">{list.map((li, i) => <li key={i}>{inline(li, `${k}-${i}`)}</li>)}</ul>);
    list = [];
  };
  const flushQuote = () => {
    if (!quote.length) return;
    const k = `q${blocks.length}`;
    blocks.push(<blockquote key={k} className="rounded-r-xl border-l-4 border-warn bg-warn-soft px-3 py-2 text-ink-soft">{inline(quote.join(" "), k)}</blockquote>);
    quote = [];
  };
  for (const raw of text.replace(/\r/g, "").split("\n")) {
    const line = raw.trimEnd();
    if (!line.trim()) {
      flushPara(); flushList(); flushQuote();
    } else if (/^\s*[-*]\s+/.test(line) || /^\d+\.\s/.test(line)) {
      flushPara(); flushQuote();
      list.push(line.replace(/^\s*[-*]\s+/, "").replace(/^\d+\.\s/, ""));
    } else if (line.startsWith(">")) {
      flushPara(); flushList();
      quote.push(line.replace(/^>\s?/, ""));
    } else {
      flushList(); flushQuote();
      para.push(line);
    }
  }
  flushPara(); flushList(); flushQuote();
  return <div className="space-y-3 text-ink-soft">{blocks.map((b, i) => <Fragment key={i}>{b}</Fragment>)}</div>;
}
