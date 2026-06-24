"""Unit tests for :mod:`symmetric_concordance.concordance` (pure, no lifelines)."""

import numpy as np
import pytest

from symmetric_concordance import (
    KaplanMeierCensoring,
    symmetric_concordance_index,
    symmetric_concordance_ipcw,
)


def test_perfect_agreement_all_observed() -> None:
    # Same ordering in both series; every event observed.
    r = symmetric_concordance_index([10, 20, 30, 40, 50], [11, 19, 31, 39, 52])
    assert r.concordance == 1.0
    assert r.n_usable == 10  # all pairs orderable


def test_reversed_ordering() -> None:
    r = symmetric_concordance_index([10, 20, 30, 40], [40, 30, 20, 10])
    assert r.concordance == 0.0


def test_partial_concordance_is_fraction_of_pairs() -> None:
    # pairs (0,1) and (0,2) agree, (1,2) disagrees -> 2/3
    r = symmetric_concordance_index([1, 2, 3], [1, 3, 2])
    assert r.concordance == pytest.approx(2 / 3)
    assert r.n_usable == 3


def test_tied_times_are_not_orderable() -> None:
    # the (0,1) pair ties in gold time and is dropped
    r = symmetric_concordance_index([5, 5, 10], [1, 2, 3])
    assert r.n_usable == 2
    assert r.n_pairs == 3


def test_resolution_times_is_binding_max() -> None:
    # one usable pair; binding time is max(gold det 20, pred det 12) = 20
    r = symmetric_concordance_index([20, 90], [12, 80], [1, 0], [1, 0])
    assert np.allclose(r.resolution_times, [20.0])


def test_censored_prediction_is_not_an_event_at_last_date() -> None:
    """The leak the metric is meant to fix.

    Predictions make NO calls -> every predicted row is censored at the gold
    date. The symmetric metric sees zero predicted events, hence no orderable
    pairs, and correctly reports "no information" (NaN).
    """
    times = [10, 20, 30, 40, 50]
    r = symmetric_concordance_index(times, times, [1, 1, 1, 1, 1], [0, 0, 0, 0, 0])
    assert r.n_usable == 0
    assert np.isnan(r.concordance)
    assert np.isnan(symmetric_concordance_ipcw(times, times, [1] * 5, [0] * 5).concordance)


def test_censored_larger_member_still_usable() -> None:
    """A pair with one event and a later censor IS orderable (event is smaller)."""
    # p0 event at 10, p1 censored at 90 (gold); event at 12, censored at 80 (pred)
    r = symmetric_concordance_index([10, 90], [12, 80], [1, 0], [1, 0])
    assert r.n_usable == 1
    assert r.concordance == 1.0  # both say p0 before p1


def test_fewer_than_two_subjects_is_nan() -> None:
    r = symmetric_concordance_index([10.0], [12.0])
    assert r.n_pairs == 0
    assert np.isnan(r.concordance)


def test_float_dunder_returns_concordance() -> None:
    r = symmetric_concordance_index([1, 2, 3], [1, 2, 3])
    assert float(r) == r.concordance == 1.0


def test_ipcw_equals_unweighted_without_censoring() -> None:
    rng = np.random.default_rng(0)
    t_gold = rng.uniform(5, 60, 40)
    t_pred = t_gold + rng.normal(0, 5, 40)
    plain = symmetric_concordance_index(t_gold, t_pred)
    weighted = symmetric_concordance_ipcw(t_gold, t_pred)
    assert np.isclose(plain.concordance, weighted.concordance)


def test_callable_censoring_and_weight_floor() -> None:
    gold, pred = [10, 20, 30], [10, 30, 20]
    hi = symmetric_concordance_ipcw(
        gold,
        pred,
        weight_floor=0.05,
        censoring=lambda t: np.where(np.asarray(t, dtype=float) < 15, 0.5, 0.02),
    )
    lo = symmetric_concordance_ipcw(
        gold,
        pred,
        weight_floor=0.001,
        censoring=lambda t: np.where(np.asarray(t, dtype=float) < 15, 0.5, 0.02),
    )
    assert symmetric_concordance_index(gold, pred).concordance == pytest.approx(2 / 3)
    assert not np.isclose(hi.concordance, lo.concordance)


def test_kmcensoring_instance_matches_default() -> None:
    gold_t, gold_e = [10, 20, 30, 40], [1, 1, 0, 1]
    pred_t, pred_e = [12, 18, 35, 38], [1, 1, 1, 1]
    default = symmetric_concordance_ipcw(gold_t, pred_t, gold_e, pred_e)
    km = KaplanMeierCensoring().fit(gold_t, gold_e)
    explicit = symmetric_concordance_ipcw(gold_t, pred_t, gold_e, pred_e, censoring=km)
    assert np.isclose(default.concordance, explicit.concordance)


def test_bad_censoring_raises() -> None:
    with pytest.raises(TypeError, match="censoring must be"):
        symmetric_concordance_ipcw([1, 2, 3], [1, 2, 3], censoring=42)  # type: ignore[arg-type]


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        symmetric_concordance_index([1, 2, 3], [1, 2])


def test_nan_times_raise() -> None:
    with pytest.raises(ValueError, match="contains NaNs"):
        symmetric_concordance_index([1, 2, np.nan], [1, 2, 3])


def test_nan_observed_raises() -> None:
    with pytest.raises(ValueError, match="contains NaNs"):
        symmetric_concordance_index([1, 2, 3], [1, 2, 3], gold_observed=[1, np.nan, 0])


def test_non_1d_input_raises() -> None:
    with pytest.raises(ValueError, match="1-dimensional"):
        symmetric_concordance_index([[1, 2], [3, 4]], [1, 2, 3, 4])


def test_observed_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length as the times"):
        symmetric_concordance_index([1, 2, 3], [1, 2, 3], gold_observed=[1, 1])
