"""EF-weighted mapping scoring: grade a Task 3 prediction by the kgCO2e
error the mapping would introduce, not by whether the name string matched.

``score_exact_match`` treats every miss identically, but ecoinvent
emission factors span roughly five orders of magnitude: mapping an
organic chemical to rhodium overstates the result by ~10^4x, while
confusing two grades of kraft paper is nearly free. This module resolves
the predicted and the labelled reference products to kgCO2e per unit and
scores the ratio between them.

Error is measured in log10 space because emission factors are
log-distributed and the quantity a practitioner cares about is the
*factor* the footprint is off by. Absolute kgCO2e error is not available
at this stage: the mapping task never sees a mass.

The single headline metric is ``ef_error_factor`` — the geometric-mean
factor by which a mapping misstates carbon intensity, i.e.
``10 ** mean(|log10 ratio|)``. It is deliberately continuous: the
reviewer's objection to exact match is that a 10,000x miss "formally
counts as a single error", and any threshold-based rate keeps that flaw
(within-2x and a >10x rate both score an 82,430x miss the same as an 11x
miss). Averaging |log10| instead weights each error by its magnitude —
in the 109-item Task 3 set the rhodium-for-organics item carries ~16x
the weight of a 2x miss, where exact match gives both 1/109. Averaging
absolute rather than squared log error keeps one catastrophic item from
dominating the run, which is why ``rmse_log10`` is reported but not led
with.

Task 3 option sets are unordered — every option is a defensible mapping —
so the reference EF is the option closest to the prediction in log space.
That makes an exact match score exactly 0.0 and never penalises a model
for picking a different accepted answer than the one listed first.

The EF table is injected rather than shipped: ecoinvent EF values are
licensed. The authors build one with
``analysis.baselines.build_mapping_ef_table``; anyone with a licence can
supply their own ``Mapping[str, EmissionFactor]``.
"""

from __future__ import annotations

import dataclasses
import enum
import math
import statistics
from collections.abc import Iterable, Mapping
from typing import Any

from pcfbench.scoring.mapping import (
    normalize_reference_product,
    score_exact_match,
)

# Tolerance bands, expressed as the factor the EF may be off by. 2x is
# the band Task 7 already reports; 10x is the "wrong material family"
# threshold used for the gross-error rate.
_TOLERANCE_FACTORS: tuple[float, ...] = (2.0, 5.0, 10.0)
_GROSS_ERROR_FACTOR = 10.0


@dataclasses.dataclass(frozen=True)
class EmissionFactor:
    """kgCO2e per one ``unit`` of the reference product.

    ``unit`` is the picklist activity's own unit — "kg" for all but the
    three energy-carrier markets admitted by ``ENERGY_ALLOWLIST_UUIDS``
    ("kWh", "m3", "MJ"). Ratios across different units are meaningless,
    so the scorer refuses them rather than reporting a number.
    """

    value: float
    unit: str


class EfSkipReason(enum.Enum):
    """Why an item carries no ``log10_ratio``."""

    NO_PREDICTION = "no_prediction"
    PREDICTION_UNRESOLVED = "prediction_unresolved"
    LABEL_UNRESOLVED = "label_unresolved"
    UNIT_MISMATCH = "unit_mismatch"
    NON_POSITIVE_EF = "non_positive_ef"


@dataclasses.dataclass(frozen=True)
class MappingEfScore:
    """Per-item EF-weighted mapping score.

    ``log10_ratio`` is ``log10(ef_predicted / ef_expected)``: +1.0 means
    the mapping overstates the material's carbon intensity by 10x, -1.0
    that it understates it by 10x, 0.0 that it is exactly right (which an
    exact match always is).
    """

    exact_match: bool | None
    log10_ratio: float | None
    ef_predicted: float | None
    ef_expected: float | None
    expected_option: str | None
    # log10 spread of the label's own option set (max EF / min EF). The
    # noise floor of this metric: label ambiguity a model cannot be
    # blamed for. None when the item has a single resolvable option.
    label_log10_spread: float | None
    skip_reason: EfSkipReason | None

    @property
    def factor_error(self) -> float | None:
        """``log10_ratio`` re-expressed as a multiplicative factor >= 1."""
        if self.log10_ratio is None:
            return None
        return 10.0 ** abs(self.log10_ratio)


def score_ef_log_ratio(
    predicted: str | None,
    expected: dict[str, Any],
    ef_table: Mapping[str, EmissionFactor],
) -> MappingEfScore:
    """Score one mapping by the log10 EF ratio against the nearest
    accepted option.

    ``ef_table`` is keyed by ``normalize_reference_product(name)``.
    Every early return records a ``skip_reason`` so the aggregate can
    report *why* coverage is short of 100% instead of silently dropping
    items.
    """
    exact = score_exact_match(predicted, expected)

    pred_ef = _lookup(predicted, ef_table)
    if pred_ef is None:
        reason = (
            EfSkipReason.NO_PREDICTION
            if not normalize_reference_product(predicted)
            else EfSkipReason.PREDICTION_UNRESOLVED
        )
        return _skipped(exact, reason)

    options = _resolve_options(expected.get("options") or [], ef_table)
    if not options:
        return _skipped(
            exact, EfSkipReason.LABEL_UNRESOLVED, ef_predicted=pred_ef.value
        )

    spread = _label_log10_spread(options)

    same_unit = [(name, ef) for name, ef in options if ef.unit == pred_ef.unit]
    if not same_unit:
        return _skipped(
            exact,
            EfSkipReason.UNIT_MISMATCH,
            ef_predicted=pred_ef.value,
            label_log10_spread=spread,
        )

    positive = [(name, ef) for name, ef in same_unit if ef.value > 0.0]
    if pred_ef.value <= 0.0 or not positive:
        # The picklist deliberately keeps zero-EF activities (the
        # ``is_zero_ef`` filter is skipped), so this is a real branch,
        # not a data error. A log ratio is undefined either way.
        return _skipped(
            exact,
            EfSkipReason.NON_POSITIVE_EF,
            ef_predicted=pred_ef.value,
            label_log10_spread=spread,
        )

    best_name, best_ef = min(
        positive, key=lambda pair: abs(math.log10(pred_ef.value / pair[1].value))
    )
    return MappingEfScore(
        exact_match=exact,
        log10_ratio=math.log10(pred_ef.value / best_ef.value),
        ef_predicted=pred_ef.value,
        ef_expected=best_ef.value,
        expected_option=best_name,
        label_log10_spread=spread,
        skip_reason=None,
    )


def _lookup(
    name: str | None, ef_table: Mapping[str, EmissionFactor]
) -> EmissionFactor | None:
    key = normalize_reference_product(name)
    if not key:
        return None
    return ef_table.get(key)


def _resolve_options(
    options: Iterable[str], ef_table: Mapping[str, EmissionFactor]
) -> list[tuple[str, EmissionFactor]]:
    """Resolve an option list to (name, EF) pairs, dropping names absent
    from the EF table."""
    resolved: list[tuple[str, EmissionFactor]] = []
    for name in options:
        ef = _lookup(name, ef_table)
        if ef is not None:
            resolved.append((name, ef))
    return resolved


def _label_log10_spread(options: list[tuple[str, EmissionFactor]]) -> float | None:
    values = [ef.value for _, ef in options if ef.value > 0.0]
    if len(values) < 2:
        return None
    return math.log10(max(values) / min(values))


def _skipped(
    exact: bool | None,
    reason: EfSkipReason,
    *,
    ef_predicted: float | None = None,
    label_log10_spread: float | None = None,
) -> MappingEfScore:
    return MappingEfScore(
        exact_match=exact,
        log10_ratio=None,
        ef_predicted=ef_predicted,
        ef_expected=None,
        expected_option=None,
        label_log10_spread=label_log10_spread,
        skip_reason=reason,
    )


@dataclasses.dataclass(frozen=True)
class MappingEfMetrics:
    """Run-level EF-weighted mapping metrics.

    Two denominators are in play and the distinction matters:

    * ``n_attempted`` — items where the model returned a name. The
      bounded rates (``within_*x``, ``gross_error_rate``) use this, so
      they are directly comparable to ``exact_match``: a name that
      resolves to no ecoinvent activity counts as a failure, exactly as
      it does under exact match.
    * ``n_scored`` — items that additionally yielded a defined log
      ratio. The unbounded statistics (``rmse_log10``, ``bias_log10``,
      ``r2_log10``) use this, because an undefined ratio has no
      magnitude to average. ``scored_rate`` reports the shortfall and
      ``skip_counts`` says what caused it.
    """

    n_total: int
    n_attempted: int
    n_scored: int
    scored_rate: float
    exact_match: float
    # HEADLINE. Mean |log10 ratio|, and the same figure as a factor:
    # ``ef_error_factor`` is the geometric-mean factor by which a mapping
    # misstates the material's carbon intensity. 1.00x is a perfect run.
    #
    # This is the metric that answers reviewer yQUd W3/Q3, because it is
    # the only one here that is *continuous*: every threshold rate below
    # (including within_2x and the >10x rate) still scores an 82,430x
    # miss identically to an 11x miss, which is the reviewer's original
    # objection to exact match relocated to a new cutoff. Averaging
    # |log10| is linear in orders of magnitude, so it weights an error by
    # its size without letting the single worst item dominate the way
    # ``rmse_log10`` does.
    mean_abs_log10: float | None
    ef_error_factor: float | None
    # Share of attempted items whose EF lands within 2x / 5x / 10x, and
    # the complement of the 10x band. Bounded, denominated like
    # ``exact_match``, and useful for reporting the shape of the error
    # distribution -- but secondary, per the note above.
    within_2x: float
    within_5x: float
    within_10x: float
    gross_error_rate: float
    # Root-mean-square log10 error. 0.30 == a typical 2x miss, 1.00 == a
    # typical 10x miss. Report alongside the bounded rates, never alone:
    # squaring in log space still lets one catastrophic item dominate a
    # whole run, so its cross-model ranking is unstable (a run whose
    # worst miss is 8e4x scores worse than one with more but milder
    # errors). ``p90_factor`` is the robust read of the same tail.
    rmse_log10: float | None
    typical_factor: float | None
    # 90th percentile of |log10 ratio|: the tail a practitioner would
    # actually hit, without the max-item sensitivity of RMSE. Preferred
    # over a median, which sits at exactly 0 for any run whose exact
    # match exceeds 50%.
    p90_abs_log10: float | None
    p90_factor: float | None
    median_abs_log10: float | None
    median_factor: float | None
    # Mean signed log10 error: >0 means the run systematically maps to
    # higher-carbon activities than the label.
    bias_log10: float | None
    geometric_bias: float | None
    # Variance of log10 EF explained across items. Distinguishes a run
    # that tracks the carbon-intensity spread from one that regresses
    # every material to a mid-range factor.
    r2_log10: float | None
    # Median label-ambiguity spread, in log10 EF, over multi-option
    # items: the metric's own noise floor.
    median_label_log10_spread: float | None
    skip_counts: Mapping[str, int]


def aggregate_ef_metrics(scores: Iterable[MappingEfScore]) -> MappingEfMetrics:
    """Aggregate per-item scores into run-level EF-weighted metrics."""
    items = list(scores)
    attempted = [s for s in items if s.skip_reason is not EfSkipReason.NO_PREDICTION]
    scored = [s for s in attempted if s.log10_ratio is not None]
    errors = [s.log10_ratio for s in scored if s.log10_ratio is not None]

    n_attempted = len(attempted)
    tolerances = {
        factor: _rate(
            sum(1 for e in errors if abs(e) <= math.log10(factor)), n_attempted
        )
        for factor in _TOLERANCE_FACTORS
    }
    # An item with no defined ratio cannot be shown to be within
    # tolerance, so it counts against every band and as a gross error.
    n_within_gross = sum(1 for e in errors if abs(e) <= math.log10(_GROSS_ERROR_FACTOR))

    spreads = [s.label_log10_spread for s in items if s.label_log10_spread is not None]
    abs_errors = [abs(e) for e in errors]
    rmse = _rmse(errors)
    p90 = _quantile(abs_errors, 0.9)
    median_abs = statistics.median(abs_errors) if abs_errors else None
    mean_abs = statistics.fmean(abs_errors) if abs_errors else None

    return MappingEfMetrics(
        n_total=len(items),
        n_attempted=n_attempted,
        n_scored=len(scored),
        scored_rate=_rate(len(scored), n_attempted),
        exact_match=_rate(sum(1 for s in items if s.exact_match), n_attempted),
        mean_abs_log10=mean_abs,
        ef_error_factor=_as_factor(mean_abs),
        within_2x=tolerances[2.0],
        within_5x=tolerances[5.0],
        within_10x=tolerances[10.0],
        gross_error_rate=_rate(n_attempted - n_within_gross, n_attempted),
        rmse_log10=rmse,
        typical_factor=_as_factor(rmse),
        p90_abs_log10=p90,
        p90_factor=_as_factor(p90),
        median_abs_log10=median_abs,
        median_factor=_as_factor(median_abs),
        bias_log10=statistics.fmean(errors) if errors else None,
        geometric_bias=(10.0 ** statistics.fmean(errors)) if errors else None,
        r2_log10=_r2_log10(scored),
        median_label_log10_spread=statistics.median(spreads) if spreads else None,
        skip_counts=_skip_counts(items),
    )


def _quantile(values: list[float], q: float) -> float | None:
    """Linear-interpolated quantile. ``statistics.quantiles`` needs at
    least two points and returns cut points rather than a single q."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def _r2_log10(scored: list[MappingEfScore]) -> float | None:
    """Fraction of the cross-item variance in log10 expected EF that the
    predicted EFs explain. ``None`` when the labels carry no spread to
    explain (fewer than two items, or all the same EF)."""
    truth = [
        math.log10(s.ef_expected)
        for s in scored
        if s.ef_expected is not None and s.ef_expected > 0.0
    ]
    residuals = [s.log10_ratio for s in scored if s.log10_ratio is not None]
    if len(truth) < 2 or len(truth) != len(residuals):
        return None
    mean_truth = statistics.fmean(truth)
    ss_tot = sum((t - mean_truth) ** 2 for t in truth)
    if ss_tot == 0.0:
        return None
    ss_res = sum(r**2 for r in residuals)
    return 1.0 - ss_res / ss_tot


def _skip_counts(items: list[MappingEfScore]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in items:
        if s.skip_reason is not None:
            counts[s.skip_reason.value] = counts.get(s.skip_reason.value, 0) + 1
    return counts


def _rmse(errors: list[float]) -> float | None:
    if not errors:
        return None
    return math.sqrt(statistics.fmean([e**2 for e in errors]))


def _as_factor(log10_value: float | None) -> float | None:
    return None if log10_value is None else 10.0**log10_value


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator
