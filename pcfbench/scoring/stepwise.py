"""Violation-only scorer for the compositional Task 7 pipeline.

Scores the structural invariants that a bottom-up decompose-triage-map-
rate-sum pipeline can violate regardless of the user's emission-factor
lookup: mass-balance, ghost components (rate=0 entries), recursion
depth, and per-stage success rates. The shipping default
``ef_resolver`` (registered in ``evals.runner``) returns ``None`` for
every material, so ``StepwiseResult.kgco2e_predicted`` reflects energy
contributions only and is not a PCF estimate. We deliberately do NOT
score it here. Reviewers with an ecoinvent license can plug in their
own resolver and grade the kgCO2e column themselves against the EPD
ground truth.
"""

from __future__ import annotations

import statistics
from typing import Any

from pcfbench.agents.stepwise import (
    StepwiseResult,
)

# Mass-fraction thresholds. Sums slightly over 1.0 are normal in LCA
# (process losses, recycled-content double-counting); the over-thresholds
# flag the egregious tail. Sums *below* 1.0 violate conservation of mass
# — process losses push the sum up, never down.
_MASS_OVER_THRESHOLDS = (1.0, 1.1, 1.2, 1.5, 2.0, 5.0)
_MASS_UNDER_THRESHOLDS = (1.0, 0.95, 0.8, 0.5, 0.1)

_MASS_PER_MASS_UNITS = frozenset(
    {"kg/kg", "kg per kg", "kg/kg product", "kg per kg product"}
)


def _is_mass_per_mass(unit: str) -> bool:
    return (unit or "").strip().lower() in _MASS_PER_MASS_UNITS


def score_stepwise_violations(
    out: StepwiseResult | None, expected: dict[str, Any]
) -> dict[str, Any]:
    """Per-item structural-invariant score for one ``StepwiseResult``.

    ``out`` is a ``StepwiseResult`` (or ``None`` on pipeline failure).
    ``expected`` is the EPD ground-truth dict; we capture truth kgCO2e
    for downstream license-holder grading but do NOT derive any score
    from it here.
    """
    if out is None:
        return {
            "stepwise_success": False,
            "mass_fraction_sum": None,
            "ghost_components": None,
            "n_components_total": 0,
            "n_components_material": 0,
            "n_components_energy": 0,
            "max_depth_reached": None,
            "kgco2e_truth": expected.get("kgco2e"),
        }

    breakdown = out.breakdown or []
    materials = [c for c in breakdown if c.kind == "material"]
    energies = [c for c in breakdown if c.kind == "energy"]

    mass_per_mass_components = [c for c in materials if _is_mass_per_mass(c.rate_unit)]
    mass_fraction_sum = sum(c.rate_value for c in mass_per_mass_components)

    ghost_total = sum(1 for c in breakdown if c.rate_value == 0.0)

    stages = out.stages or []
    stage_counts: dict[str, int] = {}
    stage_failures: dict[str, int] = {}
    for s in stages:
        prefix = (s.stage.split(":", 1)[0]) if s.stage else "unknown"
        stage_counts[prefix] = stage_counts.get(prefix, 0) + 1
        if not s.success:
            stage_failures[prefix] = stage_failures.get(prefix, 0) + 1

    item: dict[str, Any] = {
        "stepwise_success": bool(out.success),
        "mass_fraction_sum": mass_fraction_sum,
        "ghost_components": ghost_total,
        "n_components_material": len(materials),
        "n_components_energy": len(energies),
        "n_components_total": len(breakdown),
        "max_depth_reached": out.max_depth_reached,
        "kgco2e_truth": expected.get("kgco2e"),
        "n_decompose_calls": stage_counts.get("decompose", 0),
        "n_triage_calls": stage_counts.get("triage", 0),
        "n_mapping_calls": stage_counts.get("map", 0),
        "n_rate_calls": stage_counts.get("rate", 0),
        "n_decompose_failures": stage_failures.get("decompose", 0),
        "n_triage_failures": stage_failures.get("triage", 0),
        "n_mapping_failures": stage_failures.get("map", 0),
        "n_rate_failures": stage_failures.get("rate", 0),
    }
    for t in _MASS_OVER_THRESHOLDS:
        item[f"mass_over_{t}"] = mass_fraction_sum > t
    for t in _MASS_UNDER_THRESHOLDS:
        item[f"mass_under_{t}"] = mass_fraction_sum < t
    return item


def _mean(values: list[float | int | None]) -> float | None:
    vs = [v for v in values if v is not None]
    if not vs:
        return None
    return float(statistics.mean(vs))


def _median(values: list[float | int | None]) -> float | None:
    vs = [v for v in values if v is not None]
    if not vs:
        return None
    return float(statistics.median(vs))


def _rate_true(values: list[bool | None]) -> float | None:
    vs = [v for v in values if v is not None]
    if not vs:
        return None
    return sum(1.0 for v in vs if v) / len(vs)


def summarize_stepwise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Run-level aggregates over the per-item scores. Designed to be
    called from ``evals.runner._summarize`` for the stepwise eval."""

    def _col(name: str) -> list[Any]:
        return [r["scores"].get(name) for r in rows]

    summary: dict[str, Any] = {
        "success_rate": _rate_true(_col("stepwise_success")),
        "mean_mass_fraction_sum": _mean(_col("mass_fraction_sum")),
        "median_mass_fraction_sum": _median(_col("mass_fraction_sum")),
        "mean_ghost_components": _mean(_col("ghost_components")),
        "fraction_with_any_ghost": _rate_true(
            [(g is not None and g > 0) for g in _col("ghost_components")]
        ),
        "mean_max_depth_reached": _mean(_col("max_depth_reached")),
        "mean_n_components_material": _mean(_col("n_components_material")),
        "mean_n_components_energy": _mean(_col("n_components_energy")),
    }
    for t in _MASS_OVER_THRESHOLDS:
        summary[f"fraction_mass_over_{t}"] = _rate_true(_col(f"mass_over_{t}"))
    for t in _MASS_UNDER_THRESHOLDS:
        summary[f"fraction_mass_under_{t}"] = _rate_true(_col(f"mass_under_{t}"))
    return summary
