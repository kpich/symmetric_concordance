"""Unit tests for :mod:`symmetric_concordance.concordance` (pure, no lifelines)."""

import numpy as np
import pytest

from symmetric_concordance import symmetric_concordance_index


def test_perfect_agreement_all_observed() -> None:
    # Same ordering in both series; every event observed.
    r = symmetric_concordance_index([10, 20, 30, 40, 50], [11, 19, 31, 39, 52])
    assert r.concordance == 1.0
    assert r.concordance_ipcw == 1.0
    assert r.n_usable == 10  # all pairs orderable


def test_reversed_ordering() -> None:
    r = symmetric_concordance_index([10, 20, 30, 40], [40, 30, 20, 10])
    assert r.concordance == 0.0


def test_no_censoring_ipcw_equals_unweighted() -> None:
    rng = np.random.default_rng(0)
    t_gold = rng.uniform(5, 60, 40)
    t_pred = t_gold + rng.normal(0, 5, 40)
    r = symmetric_concordance_index(t_gold, t_pred)
    assert np.isclose(r.concordance, r.concordance_ipcw)


def test_censored_prediction_is_not_an_event_at_last_date() -> None:
    """The leak the metric is meant to fix.

    Predictions make NO calls -> every predicted row is censored at the gold
    date. The symmetric metric sees zero predicted events, hence no orderable
    pairs, and correctly reports "no information" (NaN).
    """
    times = [10, 20, 30, 40, 50]
    r = symmetric_concordance_index(
        times, times, gold_observed=[1, 1, 1, 1, 1], pred_observed=[0, 0, 0, 0, 0]
    )
    assert r.n_usable == 0
    assert np.isnan(r.concordance)
    assert np.isnan(r.concordance_ipcw)


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


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        symmetric_concordance_index([1, 2, 3], [1, 2])


def test_nan_times_raise() -> None:
    with pytest.raises(ValueError, match="contains NaNs"):
        symmetric_concordance_index([1, 2, np.nan], [1, 2, 3])
