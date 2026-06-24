"""Concordance between two *right-censored* survival series.

The textbook C-index (Harrell; Uno's IPCW version) assumes a fully-observed
marker scored against a censored outcome. lifelines' ``concordance_index`` is
that: it consumes ``(event_times, predicted_scores, event_observed)`` and only
uses the *outcome's* censoring flag -- the marker is treated as a plain number.

That breaks when the "marker" (e.g. an inferred event time) is itself
right-censored. Feeding the raw inferred time in as a score makes a *censored*
prediction at time ``C`` indistinguishable from an *event* at ``C``: the
prediction's own event flag is silently discarded. When un-resolved predictions
fall back to the gold date, that leaks the answer into the score and inflates
the index.

The fix is to use *both* series' event indicators to decide whether a pair is
orderable. For a pair ``(i, j)`` and one right-censored margin, the order is
known only if the *smaller* time is an event (the standard Harrell comparability
rule). A pair is usable only if it is orderable in *both* margins; it is
concordant if the two margins agree on the direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .censoring import SupportsPredict, resolve_censoring

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike, NDArray


@dataclass
class SymmetricConcordanceResult:
    """All concordance flavours for one gold/predicted comparison.

    Attributes
    ----------
    concordance
        Comparable-pairs concordance using both event flags (0.5 = chance).
    concordance_ipcw
        Same, IPCW-reweighted to undo the short-time bias. ``nan`` when
        ``ipcw=False``.
    n_usable
        Number of pairs orderable in both margins.
    n_pairs
        Total pairs, ``n * (n - 1) / 2``.
    frac_usable
        ``n_usable / n_pairs``.
    resolution_times
        Per usable pair, the binding time ``max(gold_t*, pred_t*)`` at which the
        pair becomes orderable -- the distribution that reveals short-time bias.
    """

    concordance: float
    concordance_ipcw: float
    n_usable: int
    n_pairs: int
    frac_usable: float
    resolution_times: NDArray[np.float64]

    def __float__(self) -> float:
        """Return the unweighted concordance, for ``float(result)`` convenience."""
        return self.concordance


def _check_times(values: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional")
    if np.isnan(arr).any():
        raise ValueError(f"{name} contains NaNs; please drop or correct them")
    return arr


def _check_observed(values: ArrayLike | None, n: int, name: str) -> NDArray[np.bool_]:
    if values is None:
        return np.ones(n, dtype=bool)
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(f"{name} must be 1-dimensional with the same length as the times")
    return arr.astype(bool)


def _empty_result(n_pairs: int, frac_usable: float) -> SymmetricConcordanceResult:
    return SymmetricConcordanceResult(
        concordance=float("nan"),
        concordance_ipcw=float("nan"),
        n_usable=0,
        n_pairs=n_pairs,
        frac_usable=frac_usable,
        resolution_times=np.empty(0, dtype=np.float64),
    )


def symmetric_concordance_index(
    gold_times: ArrayLike,
    pred_times: ArrayLike,
    gold_observed: ArrayLike | None = None,
    pred_observed: ArrayLike | None = None,
    *,
    ipcw: bool = True,
    censoring: SupportsPredict | Callable[..., Any] | None = None,
    weight_floor: float = 0.05,
) -> SymmetricConcordanceResult:
    """Concordance between two right-censored series, aligned position-wise.

    The two series must already be aligned (row ``i`` of ``gold_*`` describes the
    same subject as row ``i`` of ``pred_*``). Times default to all-observed when
    no event flags are given, recovering the ordinary fully-observed case.

    Parameters
    ----------
    gold_times, pred_times
        Length-n observed times for the gold (outcome) and predicted series.
    gold_observed, pred_observed
        Length-n event flags, truthy where an event was observed and falsy where
        the subject was right-censored. ``None`` means all observed.
    ipcw
        If ``True`` (default), also compute the IPCW-reweighted concordance.
    censoring
        Estimate of the censoring survival ``G(t)`` for IPCW. ``None`` fits a
        built-in Kaplan-Meier estimator on the gold series. May also be a
        lifelines ``KaplanMeierFitter`` (or any object with ``.predict``), or a
        plain callable ``G(t)``.
    weight_floor
        ``G`` is clipped from below at this value before forming the weights
        ``1 / G(t)**2``, to keep late, sparsely-observed pairs from exploding.

    Returns
    -------
    SymmetricConcordanceResult
        The concordance flavours and supporting pair counts. When fewer than two
        subjects are supplied, or no pair is orderable in both margins, the
        concordance fields are ``nan``.

    Notes
    -----
    Comparisons use a strict ``<`` on times, so exact time ties are treated as
    not orderable and tied predictions receive no half-credit (unlike lifelines'
    ``concordance_index``).
    """
    gold_t = _check_times(gold_times, "gold_times")
    pred_t = _check_times(pred_times, "pred_times")
    n = gold_t.shape[0]
    if pred_t.shape[0] != n:
        raise ValueError("gold_times and pred_times must have the same length")
    gold_e = _check_observed(gold_observed, n, "gold_observed")
    pred_e = _check_observed(pred_observed, n, "pred_observed")

    if n < 2:
        return _empty_result(n_pairs=0, frac_usable=float("nan"))

    a, b = np.triu_indices(n, k=1)
    n_pairs = int(a.shape[0])

    tga, tgb = gold_t[a], gold_t[b]
    ega, egb = gold_e[a], gold_e[b]
    tfa, tfb = pred_t[a], pred_t[b]
    efa, efb = pred_e[a], pred_e[b]

    # Order known within a margin only if the *smaller* time is an event.
    g_a_first = (tga < tgb) & ega
    g_b_first = (tgb < tga) & egb
    g_known = g_a_first | g_b_first

    f_a_first = (tfa < tfb) & efa
    f_b_first = (tfb < tfa) & efb
    f_known = f_a_first | f_b_first

    usable = g_known & f_known
    n_usable = int(usable.sum())
    if n_usable == 0:
        return _empty_result(n_pairs=n_pairs, frac_usable=0.0)

    # When usable, exactly one direction holds per margin; agreement == same dir.
    agree = (g_a_first == f_a_first)[usable]

    # Binding time per usable pair: max of each margin's determining event time.
    tg_det = np.where(g_a_first, tga, tgb)[usable]
    tf_det = np.where(f_a_first, tfa, tfb)[usable]
    t_det = np.maximum(tg_det, tf_det).astype(np.float64)

    concordance = float(agree.mean())

    concordance_ipcw = float("nan")
    if ipcw:
        g = resolve_censoring(censoring, gold_t, gold_e)
        g_at = np.clip(g(t_det), weight_floor, None)
        weights = 1.0 / g_at**2
        concordance_ipcw = float(np.sum(weights * agree) / np.sum(weights))

    return SymmetricConcordanceResult(
        concordance=concordance,
        concordance_ipcw=concordance_ipcw,
        n_usable=n_usable,
        n_pairs=n_pairs,
        frac_usable=n_usable / n_pairs,
        resolution_times=t_det,
    )
