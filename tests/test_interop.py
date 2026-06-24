"""Integration tests verifying interop with lifelines.

These import lifelines (heavy: ~0.6s) and are skipped if it is not installed,
which is why they live here rather than next to the source as unit tests.
"""

import numpy as np
import pytest

from symmetric_concordance import KaplanMeierCensoring, symmetric_concordance_index


def test_lifelines_spuriously_perfect_on_the_leak_case() -> None:
    """lifelines, fed the predicted time as a marker, reports a spurious 1.0."""
    concordance_index = pytest.importorskip("lifelines.utils").concordance_index
    times = [10, 20, 30, 40, 50]
    observed = [1, 1, 1, 1, 1]
    assert concordance_index(times, times, observed) == 1.0


def test_builtin_km_matches_lifelines_kmf() -> None:
    KaplanMeierFitter = pytest.importorskip("lifelines").KaplanMeierFitter
    rng = np.random.default_rng(1)
    n = 60
    gold_t = rng.uniform(5, 60, n)
    gold_e = rng.integers(0, 2, n).astype(bool)
    pred_t = gold_t + rng.normal(0, 5, n)
    pred_e = rng.integers(0, 2, n).astype(bool)

    kmf = KaplanMeierFitter().fit(gold_t, 1 - gold_e.astype(int))

    r_builtin = symmetric_concordance_index(gold_t, pred_t, gold_e, pred_e)
    r_lifelines = symmetric_concordance_index(gold_t, pred_t, gold_e, pred_e, censoring=kmf)
    assert np.isclose(r_builtin.concordance_ipcw, r_lifelines.concordance_ipcw)


def test_builtin_km_matches_lifelines_predict_values() -> None:
    KaplanMeierFitter = pytest.importorskip("lifelines").KaplanMeierFitter
    rng = np.random.default_rng(2)
    times = rng.uniform(1, 50, 80)
    observed = rng.integers(0, 2, 80).astype(bool)

    ours = KaplanMeierCensoring().fit(times, observed)
    kmf = KaplanMeierFitter().fit(times, 1 - observed.astype(int))

    grid = np.linspace(0, 55, 25)
    assert np.allclose(ours.predict(grid), np.asarray(kmf.predict(grid), dtype=float))
