# ADR 0003 - Reviewed duplicate grouping for leakage-safe splits

Status: accepted (2026-10-22)

## Context
D1 has no tyre identity. The archive holds 1856 image members, 28 of them byte-identical copies; the remaining
1828 unique photos include many burst shots of the same tyre with consecutive file numbers.

## Evidence from review
Contact sheets of candidate pairs (pHash Hamming distance and cosine similarity of the pinned YOLO26n-cls
ImageNet embeddings) showed:
- pHash-only matches (distance <= 6) with low embedding cosine (~0.38) joined *different* tyres, including a
  good and a defective tyre.
- Pairs at cosine 0.90-0.98 were the same physical tyre photographed again.
- Pairs at cosine 0.86-0.90 were the same tyre only when their capture numbers were adjacent.

## Decision
Group when cosine >= 0.90, or pHash <= 6 with cosine >= 0.85, or same folder with capture-index distance <= 3
and cosine >= 0.80. This yields 948 groups, all label-pure (largest 47 images). Every candidate pair and
decision is listed in `data/manifests/duplicate_review.csv`.

## Consequences
Over-grouping only coarsens the split (actual outer fractions 69.8 / 15.0 / 15.2); under-grouping would inflate
test metrics. Residual leakage between visually different photos of the same tyre is still possible and is
listed as a limitation in the model card.
