"""The three-way tyre verdict shown everywhere (Bib, results, camera overlay), from a stored prediction."""

from __future__ import annotations


def verdict_of(binary_defective: bool, triage: str, quality_status: str) -> str:
    """Damaged if the policy says so; 'not damaged' only for a clean auto-clear; anything else is uncertain."""
    if binary_defective:
        return "damaged"
    return "not_damaged" if triage == "good_candidate" and quality_status != "low_quality" else "uncertain"
