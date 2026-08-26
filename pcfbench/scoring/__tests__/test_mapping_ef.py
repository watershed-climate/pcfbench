"""Tests for the EF-weighted mapping metric."""

from __future__ import annotations

import math

import pytest

from pcfbench.scoring.mapping_ef import (
    EfSkipReason,
    EmissionFactor,
    aggregate_ef_metrics,
    score_ef_log_ratio,
)

EF_TABLE = {
    "kraft paper": EmissionFactor(value=1.0, unit="kg"),
    "kraft paper, bleached": EmissionFactor(value=2.0, unit="kg"),
    "chemical, organic": EmissionFactor(value=2.0, unit="kg"),
    "rhodium": EmissionFactor(value=20_000.0, unit="kg"),
    "mid-range ef": EmissionFactor(value=40.0, unit="kg"),
    "zero ef thing": EmissionFactor(value=0.0, unit="kg"),
    "electricity, low voltage": EmissionFactor(value=0.485, unit="kWh"),
}


def _expected(*options: str) -> dict[str, list[str]]:
    return {"options": list(options)}


def test_exact_match_scores_zero_error() -> None:
    score = score_ef_log_ratio("kraft paper", _expected("kraft paper"), EF_TABLE)
    assert score.exact_match is True
    assert score.log10_ratio == 0.0
    assert score.factor_error == 1.0
    assert score.skip_reason is None


def test_case_and_whitespace_insensitive() -> None:
    score = score_ef_log_ratio("  Kraft Paper ", _expected("kraft paper"), EF_TABLE)
    assert score.log10_ratio == 0.0


def test_rhodium_for_organics_is_a_four_order_error() -> None:
    """The reviewer's motivating case: one string miss, ~10^4x impact."""
    score = score_ef_log_ratio("rhodium", _expected("chemical, organic"), EF_TABLE)
    assert score.exact_match is False
    assert score.log10_ratio == pytest.approx(4.0)
    assert score.factor_error == pytest.approx(10_000.0)


def test_near_miss_is_cheap() -> None:
    """Same string-match verdict as the rhodium case, 2x the impact."""
    score = score_ef_log_ratio(
        "kraft paper, bleached", _expected("kraft paper"), EF_TABLE
    )
    assert score.exact_match is False
    assert score.log10_ratio == pytest.approx(math.log10(2.0))


def test_understatement_is_signed_negative() -> None:
    score = score_ef_log_ratio("chemical, organic", _expected("rhodium"), EF_TABLE)
    assert score.log10_ratio == pytest.approx(-4.0)


def test_credits_nearest_option_since_option_sets_are_unordered() -> None:
    """Options are all defensible, so the label EF is the closest one --
    a prediction is never penalised for the option ordering."""
    score = score_ef_log_ratio(
        "kraft paper, bleached",
        _expected("rhodium", "kraft paper, bleached"),
        EF_TABLE,
    )
    assert score.log10_ratio == 0.0
    assert score.expected_option == "kraft paper, bleached"
    # Label ambiguity itself spans 4 orders of magnitude here.
    assert score.label_log10_spread == pytest.approx(4.0)


def test_missing_prediction_is_not_an_error_magnitude() -> None:
    score = score_ef_log_ratio(None, _expected("kraft paper"), EF_TABLE)
    assert score.exact_match is None
    assert score.skip_reason is EfSkipReason.NO_PREDICTION


def test_hallucinated_name_is_flagged_unresolved() -> None:
    score = score_ef_log_ratio(
        "unobtainium, market for", _expected("kraft paper"), EF_TABLE
    )
    assert score.exact_match is False
    assert score.skip_reason is EfSkipReason.PREDICTION_UNRESOLVED
    assert score.log10_ratio is None


def test_zero_ef_activity_yields_no_ratio() -> None:
    score = score_ef_log_ratio("zero ef thing", _expected("kraft paper"), EF_TABLE)
    assert score.skip_reason is EfSkipReason.NON_POSITIVE_EF


def test_cross_unit_comparison_is_refused() -> None:
    score = score_ef_log_ratio(
        "electricity, low voltage", _expected("kraft paper"), EF_TABLE
    )
    assert score.skip_reason is EfSkipReason.UNIT_MISMATCH
    assert score.log10_ratio is None


def test_unresolvable_label_is_distinguished_from_bad_prediction() -> None:
    score = score_ef_log_ratio(
        "kraft paper", _expected("trifluoroacetic acidc"), EF_TABLE
    )
    assert score.skip_reason is EfSkipReason.LABEL_UNRESOLVED


# --- aggregation -----------------------------------------------------------


def test_aggregate_rmse_and_tolerance_bands() -> None:
    scores = [
        score_ef_log_ratio("kraft paper", _expected("kraft paper"), EF_TABLE),
        score_ef_log_ratio("kraft paper, bleached", _expected("kraft paper"), EF_TABLE),
        score_ef_log_ratio("rhodium", _expected("chemical, organic"), EF_TABLE),
    ]
    m = aggregate_ef_metrics(scores)

    assert m.n_total == 3
    assert m.n_attempted == 3
    assert m.n_scored == 3
    assert m.exact_match == pytest.approx(1 / 3)
    # errors: 0.0, log10(2), 4.0
    expected_rmse = math.sqrt((0.0 + math.log10(2.0) ** 2 + 16.0) / 3)
    assert m.rmse_log10 == pytest.approx(expected_rmse)
    assert m.typical_factor == pytest.approx(10.0**expected_rmse)
    # The 2x band is inclusive, so the exact match and the 2x miss pass.
    assert m.within_2x == pytest.approx(2 / 3)
    assert m.within_10x == pytest.approx(2 / 3)
    assert m.gross_error_rate == pytest.approx(1 / 3)
    assert m.bias_log10 == pytest.approx((math.log10(2.0) + 4.0) / 3)


def test_unresolved_prediction_counts_against_tolerance_but_not_rmse() -> None:
    """Same denominator as exact_match, so the two are comparable."""
    scores = [
        score_ef_log_ratio("kraft paper", _expected("kraft paper"), EF_TABLE),
        score_ef_log_ratio("unobtainium", _expected("kraft paper"), EF_TABLE),
    ]
    m = aggregate_ef_metrics(scores)

    assert m.n_attempted == 2
    assert m.n_scored == 1
    assert m.scored_rate == pytest.approx(0.5)
    assert m.exact_match == pytest.approx(0.5)
    assert m.within_2x == pytest.approx(0.5)
    assert m.gross_error_rate == pytest.approx(0.5)
    # RMSE sees only the one defined ratio.
    assert m.rmse_log10 == pytest.approx(0.0)
    assert m.skip_counts == {"prediction_unresolved": 1}


def test_missing_prediction_leaves_both_denominators() -> None:
    scores = [
        score_ef_log_ratio("kraft paper", _expected("kraft paper"), EF_TABLE),
        score_ef_log_ratio(None, _expected("kraft paper"), EF_TABLE),
    ]
    m = aggregate_ef_metrics(scores)
    assert m.n_total == 2
    assert m.n_attempted == 1
    assert m.exact_match == pytest.approx(1.0)
    assert m.within_2x == pytest.approx(1.0)


def test_ef_error_factor_is_one_for_a_perfect_run() -> None:
    scores = [score_ef_log_ratio("kraft paper", _expected("kraft paper"), EF_TABLE)] * 5
    m = aggregate_ef_metrics(scores)
    assert m.mean_abs_log10 == pytest.approx(0.0)
    assert m.ef_error_factor == pytest.approx(1.0)


def test_ef_error_factor_is_the_uniform_error_when_every_item_misses_alike() -> None:
    """Ten items each 2x off must report exactly 2.00x, not something
    attenuated -- the geometric mean of a constant is that constant."""
    scores = [
        score_ef_log_ratio("kraft paper, bleached", _expected("kraft paper"), EF_TABLE)
    ] * 10
    assert aggregate_ef_metrics(scores).ef_error_factor == pytest.approx(2.0)


def test_ef_error_factor_weights_a_catastrophe_above_a_lesser_gross_error() -> None:
    """The reviewer's objection, encoded. Two runs each have nine exact
    matches and one error; one errs by 20x, the other by 10,000x. Exact
    match and *both* threshold rates score them identically -- which is
    the same "formally counts as a single error" flaw at a new cutoff.
    Only the continuous metric separates them."""
    base = [score_ef_log_ratio("kraft paper", _expected("kraft paper"), EF_TABLE)] * 9
    # 2.0 predicted against a 40.0 label: 20x off, so past the 10x band.
    mild = aggregate_ef_metrics(
        base
        + [
            score_ef_log_ratio(
                "kraft paper, bleached", _expected("mid-range ef"), EF_TABLE
            )
        ]
    )
    severe = aggregate_ef_metrics(
        base + [score_ef_log_ratio("rhodium", _expected("chemical, organic"), EF_TABLE)]
    )

    # Exact match cannot tell the two runs apart.
    assert mild.exact_match == severe.exact_match == pytest.approx(0.9)
    # Nor can within-2x.
    assert mild.within_2x == severe.within_2x == pytest.approx(0.9)
    # Nor can a >10x rate: both runs have exactly one item past 10x.
    assert mild.gross_error_rate == severe.gross_error_rate == pytest.approx(0.1)
    # The headline metric does, in proportion to the log magnitudes.
    assert mild.mean_abs_log10 is not None and severe.mean_abs_log10 is not None
    assert mild.mean_abs_log10 == pytest.approx(math.log10(20.0) / 10)
    assert severe.mean_abs_log10 == pytest.approx(4.0 / 10)
    assert severe.mean_abs_log10 / mild.mean_abs_log10 == pytest.approx(
        4.0 / math.log10(20.0)
    )


def test_p90_tracks_the_tail_where_the_median_saturates_at_zero() -> None:
    """Any run above 50% exact match has median error 0, which is why the
    tail quantile is reported instead."""
    scores = [
        score_ef_log_ratio("kraft paper", _expected("kraft paper"), EF_TABLE)
    ] * 9 + [score_ef_log_ratio("rhodium", _expected("chemical, organic"), EF_TABLE)]
    m = aggregate_ef_metrics(scores)

    assert m.median_abs_log10 == pytest.approx(0.0)
    assert m.median_factor == pytest.approx(1.0)
    # Nine zeros then 4.0: position 0.9*9 = 8.1 interpolates one tenth of
    # the way from ordered[8]=0.0 to ordered[9]=4.0.
    assert m.p90_abs_log10 == pytest.approx(0.4)
    assert m.p90_factor == pytest.approx(10.0**0.4)


def test_p90_is_defined_for_a_single_scored_item() -> None:
    scores = [score_ef_log_ratio("rhodium", _expected("chemical, organic"), EF_TABLE)]
    m = aggregate_ef_metrics(scores)
    assert m.p90_abs_log10 == pytest.approx(4.0)


def test_rmse_is_max_dominated_where_the_gross_error_rate_is_not() -> None:
    """The reason the bounded rates lead the table: one catastrophic item
    outweighs many mild ones under RMSE, and the two metrics disagree."""
    # Run A: four 20x misses (2.0 predicted against a 40.0 label).
    run_a = [
        score_ef_log_ratio("kraft paper, bleached", _expected("mid-range ef"), EF_TABLE)
        for _ in range(4)
    ]
    # Run B: one 10,000x miss plus three exact matches.
    run_b = [
        score_ef_log_ratio("rhodium", _expected("chemical, organic"), EF_TABLE)
    ] + [score_ef_log_ratio("kraft paper", _expected("kraft paper"), EF_TABLE)] * 3

    a, b = aggregate_ef_metrics(run_a), aggregate_ef_metrics(run_b)
    assert a.scored_rate == 1.0 and b.scored_rate == 1.0

    # Every one of A's items is outside 10x; only one of B's four is.
    assert a.gross_error_rate == pytest.approx(1.0)
    assert b.gross_error_rate == pytest.approx(0.25)
    # RMSE reverses that ordering, because B's single item is enormous.
    assert a.rmse_log10 is not None and b.rmse_log10 is not None
    assert a.rmse_log10 == pytest.approx(math.log10(20.0))
    assert b.rmse_log10 == pytest.approx(2.0)
    assert b.rmse_log10 > a.rmse_log10


def test_r2_is_one_when_every_ef_is_recovered() -> None:
    scores = [
        score_ef_log_ratio("kraft paper", _expected("kraft paper"), EF_TABLE),
        score_ef_log_ratio("rhodium", _expected("rhodium"), EF_TABLE),
    ]
    assert aggregate_ef_metrics(scores).r2_log10 == pytest.approx(1.0)


def test_r2_goes_negative_when_predictions_are_worse_than_the_mean() -> None:
    scores = [
        score_ef_log_ratio("rhodium", _expected("kraft paper"), EF_TABLE),
        score_ef_log_ratio("kraft paper", _expected("rhodium"), EF_TABLE),
    ]
    r2 = aggregate_ef_metrics(scores).r2_log10
    assert r2 is not None and r2 < 0.0


def test_r2_is_none_without_label_spread() -> None:
    scores = [
        score_ef_log_ratio("kraft paper", _expected("kraft paper"), EF_TABLE),
        score_ef_log_ratio("kraft paper", _expected("kraft paper"), EF_TABLE),
    ]
    assert aggregate_ef_metrics(scores).r2_log10 is None


def test_empty_run_aggregates_without_dividing_by_zero() -> None:
    m = aggregate_ef_metrics([])
    assert m.n_total == 0
    assert m.exact_match == 0.0
    assert m.rmse_log10 is None
    assert m.r2_log10 is None
