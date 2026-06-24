"""Unit tests for :mod:`symmetric_concordance.censoring` (pure, no lifelines)."""

import numpy as np
import pytest

from symmetric_concordance.censoring import KaplanMeierCensoring


def test_known_small_example() -> None:
    # Censoring events (observed == 0) at t=2 and t=4.
    #   t=2: at risk 3, 1 drop -> G = 2/3
    #   t=4: at risk 1, 1 drop -> G = 0
    km = KaplanMeierCensoring().fit([1, 2, 3, 4], [1, 0, 1, 0])
    g = km.predict([0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 4.0, 5.0])
    expected = [1.0, 1.0, 1.0, 2 / 3, 2 / 3, 2 / 3, 0.0, 0.0]
    assert np.allclose(g, expected)


def test_all_observed_keeps_g_at_one() -> None:
    # No censoring events -> G(t) == 1 everywhere.
    km = KaplanMeierCensoring().fit([1, 2, 3], [1, 1, 1])
    assert np.allclose(km.predict([0, 1, 2, 3, 4]), 1.0)


def test_one_before_first_step() -> None:
    km = KaplanMeierCensoring().fit([5, 6], [0, 0])
    assert km.predict([0.0, 4.999])[0] == 1.0
    assert km.predict([0.0, 4.999])[1] == 1.0


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        KaplanMeierCensoring().predict([1.0])


def test_fit_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same shape"):
        KaplanMeierCensoring().fit([1, 2, 3], [1, 0])


def test_fit_non_1d_raises() -> None:
    with pytest.raises(ValueError, match="1-dimensional"):
        KaplanMeierCensoring().fit([[1, 2], [3, 4]], [1, 0, 1, 0])
