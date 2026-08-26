"""Mapping scoring: exact_match against expert-rated options + relevant /
banned-substring diagnostics.

Each per-item scorer takes a predicted reference-product name (the
single-shot or agentic ``submit_mapping`` argument) plus the dataset
item's expected-output dict (release schema: ``options: list[str]`` +
optional ``relevant_substring_1`` / ``relevant_substring_2`` /
``banned_substring``). Returns ``None`` when the prediction is missing
or the score is not applicable.
"""

from __future__ import annotations

from typing import Any


def normalize_reference_product(s: str | None) -> str:
    """Case- and whitespace-insensitive key for a reference-product name.

    Shared with ``mapping_ef`` so EF-table keys and exact-match
    comparisons agree on what counts as the same activity.
    """
    return (s or "").lower().strip()


def score_exact_match(predicted: str | None, expected: dict[str, Any]) -> bool | None:
    """True iff ``predicted`` matches one of the canonical reference
    products in ``expected['options']`` (case- and whitespace-insensitive)."""
    if predicted is None:
        return None
    pred = normalize_reference_product(predicted)
    if not pred:
        return None
    for opt in expected.get("options") or []:
        if pred == normalize_reference_product(opt):
            return True
    return False


def score_relevant_substring(
    predicted: str | None, expected: dict[str, Any]
) -> bool | None:
    """Diagnostic: ``relevant_substring_1`` (case-insensitive) appears in
    predicted. ``None`` if the dataset item didn't define one."""
    rs = expected.get("relevant_substring_1")
    if rs is None or predicted is None:
        return None
    pred = normalize_reference_product(predicted)
    if not pred:
        return None
    return rs.lower() in pred


def score_banned_substring_absent(
    predicted: str | None, expected: dict[str, Any]
) -> bool | None:
    """Diagnostic: ``banned_substring`` is absent from predicted (good).
    ``None`` if the dataset item didn't define one."""
    banned = expected.get("banned_substring")
    if banned is None or predicted is None:
        return None
    pred = normalize_reference_product(predicted)
    if not pred:
        return None
    return banned.lower() not in pred


def run_mean_bool(values: list[bool | None]) -> float:
    """Mean over the not-None entries; 0.0 if none."""
    vals = [v for v in values if v is not None]
    if not vals:
        return 0.0
    return sum(1.0 if v else 0.0 for v in vals) / len(vals)
