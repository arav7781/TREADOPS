"""Tiny BM25 retriever over docs/kb/*.md, chunked by `##` sections. Deterministic and dependency-free."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from treadops_ml.common import repo_path

TOKEN = re.compile(r"[a-z0-9]+")
STOP = set(["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "were", "will", "with", "what", "which", "who", "how", "why", "when", "can", "does", "do", "i", "my", "me", "you", "your", "our", "we"])


@dataclass(frozen=True)
class Passage:
    id: str
    doc: str
    heading: str
    text: str


def _tokens(s: str) -> list[str]:
    return [t for t in TOKEN.findall(s.lower()) if t not in STOP]


@lru_cache
def passages() -> tuple[Passage, ...]:
    out = []
    for path in sorted(repo_path("docs/kb").glob("*.md")):
        doc_id = "-".join(path.stem.split("-")[:2])
        title = path.read_text().splitlines()[0].lstrip("# ").strip()
        for i, section in enumerate(re.split(r"\n## ", path.read_text())[1:], start=1):
            heading, _, body = section.partition("\n")
            out.append(Passage(f"{doc_id}#{i}", title, heading.strip(), " ".join(body.split())))
    return tuple(out)


@lru_cache
def _index():
    ps = passages()
    docs = [_tokens(p.heading + " " + p.text) for p in ps]
    df = Counter(t for d in docs for t in set(d))
    avg = sum(len(d) for d in docs) / max(len(docs), 1)
    return ps, docs, df, avg


def search(query: str, k: int = 3) -> list[dict]:
    ps, docs, df, avg = _index()
    q = _tokens(query)
    n = len(ps)
    scores = []
    for p, d in zip(ps, docs, strict=True):
        tf = Counter(d)
        s = 0.0
        for t in q:
            if t not in tf:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(d) / avg))
        scores.append((s, p))
    ranked = [(s, p) for s, p in sorted(scores, key=lambda x: -x[0]) if s > 0][:k]
    return [{"id": p.id, "document": p.doc, "section": p.heading, "text": p.text, "score": round(s, 3)}
            for s, p in ranked]
