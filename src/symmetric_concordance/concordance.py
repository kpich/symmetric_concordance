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

:func:`symmetric_concordance_index` is the plain comparable-pairs index;
:func:`symmetric_concordance_ipcw` is the IPCW-reweighted version.
"""

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._fenwick import Fenwick
from .censoring import SupportsPredict, resolve_censoring


@dataclass
class SymmetricConcordanceResult:
    """Concordance and supporting pair counts for one gold/predicted comparison.

    Attributes
    ----------
    concordance
        The concordance (0.5 = chance). Unweighted from
        :func:`symmetric_concordance_index`, IPCW-reweighted from
        :func:`symmetric_concordance_ipcw`. ``nan`` if no pair is usable.
    n_usable
        Number of pairs orderable in both margins.
    n_pairs
        Total pairs, ``n * (n - 1) / 2``.
    frac_usable
        ``n_usable / n_pairs``.
    resolution_times
        Per usable pair, the binding time ``max(gold_t*, pred_t*)`` at which the
        pair becomes orderable -- the distribution that reveals short-time bias.
        Always populated by :func:`symmetric_concordance_ipcw`, which needs it
        anyway; empty from :func:`symmetric_concordance_index` unless asked for
        with ``resolution_times=True``, since it is the one field that costs
        O(n**2) to produce.
    """

    concordance: float
    n_usable: int
    n_pairs: int
    frac_usable: float
    resolution_times: NDArray[np.float64]

    def __float__(self) -> float:
        """Return the concordance, for ``float(result)`` convenience."""
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
    if np.issubdtype(arr.dtype, np.floating) and np.isnan(arr).any():
        raise ValueError(f"{name} contains NaNs; please drop or correct them")
    return arr.astype(bool)


def _validate(
    gold_times: ArrayLike,
    pred_times: ArrayLike,
    gold_observed: ArrayLike | None,
    pred_observed: ArrayLike | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    gold_t = _check_times(gold_times, "gold_times")
    pred_t = _check_times(pred_times, "pred_times")
    n = gold_t.shape[0]
    if pred_t.shape[0] != n:
        raise ValueError("gold_times and pred_times must have the same length")
    gold_e = _check_observed(gold_observed, n, "gold_observed")
    pred_e = _check_observed(pred_observed, n, "pred_observed")
    return gold_t, pred_t, gold_e, pred_e


def _comparable_pairs(
    gold_t: NDArray[np.float64],
    pred_t: NDArray[np.float64],
    gold_e: NDArray[np.bool_],
    pred_e: NDArray[np.bool_],
) -> tuple[NDArray[np.bool_], NDArray[np.float64], int, int]:
    """Return ``(agree, t_det, n_usable, n_pairs)`` over pairs orderable in both margins.

    ``agree`` flags each usable pair as concordant; ``t_det`` is its binding time.
    Both are empty when there are no usable pairs.
    """
    n = gold_t.shape[0]
    empty_b = np.empty(0, dtype=bool)
    empty_f = np.empty(0, dtype=np.float64)
    if n < 2:
        return empty_b, empty_f, 0, 0

    a, b = np.triu_indices(n, k=1)
    n_pairs = int(a.shape[0])
    tga, tgb = gold_t[a], gold_t[b]
    ega, egb = gold_e[a], gold_e[b]
    tfa, tfb = pred_t[a], pred_t[b]
    efa, efb = pred_e[a], pred_e[b]

    # Order known within a margin only if the *smaller* time is an event.
    g_a_first = (tga < tgb) & ega
    g_b_first = (tgb < tga) & egb
    f_a_first = (tfa < tfb) & efa
    f_b_first = (tfb < tfa) & efb

    usable = (g_a_first | g_b_first) & (f_a_first | f_b_first)
    n_usable = int(usable.sum())
    if n_usable == 0:
        return empty_b, empty_f, 0, n_pairs

    # When usable, exactly one direction holds per margin; agreement == same dir.
    agree = (g_a_first == f_a_first)[usable]
    # Binding time per usable pair: max of each margin's determining event time.
    tg_det = np.where(g_a_first, tga, tgb)[usable]
    tf_det = np.where(f_a_first, tfa, tfb)[usable]
    t_det = np.maximum(tg_det, tf_det).astype(np.float64)
    return agree, t_det, n_usable, n_pairs


def _count_pairs_fenwick(
    gold_t: NDArray[np.float64],
    pred_t: NDArray[np.float64],
    gold_e: NDArray[np.bool_],
    pred_e: NDArray[np.bool_],
) -> tuple[int, int]:
    """Return ``(n_concordant, n_discordant)`` in O(n log n) time and O(n) memory.

    Attribute every usable pair to its *gold-first* member ``a`` (the one gold
    puts first, so ``gold_e[a]`` and ``gold_t[a] < gold_t[b]``). Exactly one
    member qualifies per usable pair, so nothing is counted twice:

    * concordant -- the prediction agrees, both flags coming from ``a``:
      ``eg_a & ef_a & tg_a < tg_b & tf_a < tf_b``
    * discordant -- the prediction disagrees, so ``b`` carries the pred flag:
      ``eg_a & ef_b & tg_a < tg_b & tf_b < tf_a``

    Both are 2-D dominance counts, so sweep subjects in *descending* gold time
    -- everything already inserted then satisfies ``tg_b > tg_a`` -- and prefix-
    count over predicted-time rank with two Fenwick trees.

    Ties are the whole difficulty. The metric compares with a strict ``<`` in
    both margins, so equal-``gold_t`` subjects are queried as a group *before*
    any of them is inserted (a tied pair must not see itself), and predicted
    times are ranked with :func:`numpy.unique` so tied times share a rank and a
    prefix query is strict for free.
    """
    n = gold_t.shape[0]  # callers guarantee n >= 2
    uniq_pred, tf_rank = np.unique(pred_t, return_inverse=True)
    ranks: list[int] = tf_rank.ravel().tolist()
    n_ranks = uniq_pred.shape[0]

    order = np.argsort(gold_t, kind="stable")[::-1]  # descending gold time
    tg_sorted = gold_t[order]
    starts = np.flatnonzero(np.concatenate(([True], tg_sorted[1:] != tg_sorted[:-1])))
    bounds: list[int] = np.append(starts, n).tolist()

    # Python lists, not arrays: the sweep is scalar work and numpy element
    # access would cost more per item than the whole Fenwick update.
    sweep: list[int] = order.tolist()
    ge: list[bool] = gold_e.tolist()
    pe: list[bool] = pred_e.tolist()

    fen_all = Fenwick(n_ranks)  # every subject inserted so far
    fen_ef = Fenwick(n_ranks)  # ... restricted to those with a predicted event
    inserted = 0
    concordant = 0
    discordant = 0
    for lo, hi in pairwise(bounds):
        group = sweep[lo:hi]
        for a in group:  # query the whole tied group first
            if not ge[a]:
                continue
            r = ranks[a]
            discordant += fen_ef.pref(r)  # inserted b with ef_b and tf_b < tf_a
            if pe[a]:
                concordant += inserted - fen_all.pref(r + 1)  # ... with tf_b > tf_a
        for a in group:  # then insert it
            r = ranks[a]
            fen_all.add(r)
            if pe[a]:
                fen_ef.add(r)
            inserted += 1
    return concordant, discordant


def _result(
    concordance: float, t_det: NDArray[np.float64], n_usable: int, n_pairs: int
) -> SymmetricConcordanceResult:
    frac_usable = n_usable / n_pairs if n_pairs else float("nan")
    return SymmetricConcordanceResult(concordance, n_usable, n_pairs, frac_usable, t_det)


def symmetric_concordance_index(
    gold_times: ArrayLike,
    pred_times: ArrayLike,
    gold_observed: ArrayLike | None = None,
    pred_observed: ArrayLike | None = None,
    *,
    resolution_times: bool = False,
) -> SymmetricConcordanceResult:
    """Comparable-pairs concordance between two right-censored series.

    The series are aligned position-wise (row ``i`` of ``gold_*`` is the same
    subject as row ``i`` of ``pred_*``). ``pred_times`` only needs to increase
    with survival (bigger = later event); event flags default to all-observed.

    Parameters
    ----------
    gold_times, pred_times
        Length-n times for the gold (outcome) and predicted series.
    gold_observed, pred_observed
        Length-n event flags, truthy where an event was observed and falsy where
        the subject was right-censored. ``None`` means all observed.
    resolution_times
        Also return the per-usable-pair binding times in
        :attr:`SymmetricConcordanceResult.resolution_times`. Off by default:
        that is one value *per pair*, so producing it needs the O(n**2) path
        (~150 ms and ~900 MB at n=4,644, and out of reach by n=50,000), whereas
        the counts alone come from an O(n log n) sweep in O(n) memory. When left
        off, the field is an empty array; every other field is populated either
        way. Both paths return identical counts and concordance.

    Returns
    -------
    SymmetricConcordanceResult
        ``concordance`` is ``nan`` when fewer than two subjects are supplied or
        no pair is orderable in both margins.

    Notes
    -----
    Comparisons use a strict ``<`` on times, so exact ties are not orderable and
    tied predictions get no half-credit (unlike lifelines' ``concordance_index``).
    """
    gold_t, pred_t, gold_e, pred_e = _validate(gold_times, pred_times, gold_observed, pred_observed)
    if resolution_times:
        agree, t_det, n_usable, n_pairs = _comparable_pairs(gold_t, pred_t, gold_e, pred_e)
        concordance = float(agree.mean()) if n_usable else float("nan")
        return _result(concordance, t_det, n_usable, n_pairs)

    n = gold_t.shape[0]
    if n < 2:
        return _result(float("nan"), np.empty(0, dtype=np.float64), 0, 0)
    n_concordant, n_discordant = _count_pairs_fenwick(gold_t, pred_t, gold_e, pred_e)
    n_usable = n_concordant + n_discordant
    concordance = n_concordant / n_usable if n_usable else float("nan")
    return _result(concordance, np.empty(0, dtype=np.float64), n_usable, n * (n - 1) // 2)


def symmetric_concordance_ipcw(
    gold_times: ArrayLike,
    pred_times: ArrayLike,
    gold_observed: ArrayLike | None = None,
    pred_observed: ArrayLike | None = None,
    *,
    censoring: SupportsPredict | Callable[..., Any] | None = None,
    weight_floor: float = 0.05,
) -> SymmetricConcordanceResult:
    """IPCW-reweighted version of :func:`symmetric_concordance_index`.

    Each usable pair is weighted by ``1 / G(t)**2`` at its binding time, undoing
    the short-time bias from informative censoring.

    This enumerates pairs densely, so it is O(n**2) in both time and memory --
    unlike the default path of :func:`symmetric_concordance_index`, which is
    O(n log n). A pair's binding time ``max(gold_t*, pred_t*)`` depends on
    *both* members when the two margins disagree, so the weighted total does not
    reduce to a dominance count the way the unweighted one does. Budget for the
    dense cost (~150 ms and ~900 MB at n=4,644) or subsample.

    Parameters
    ----------
    gold_times, pred_times, gold_observed, pred_observed
        As in :func:`symmetric_concordance_index`.
    censoring
        Estimate of the censoring survival ``G(t)``. ``None`` fits a built-in
        Kaplan-Meier estimator on the gold series. May also be a lifelines
        ``KaplanMeierFitter`` (or any object with ``.predict``), or a callable
        ``G(t)``.
    weight_floor
        ``G`` is clipped from below at this value before forming the weights, to
        keep late, sparsely-observed pairs from exploding.

    Returns
    -------
    SymmetricConcordanceResult
        ``concordance`` holds the IPCW-reweighted value (``nan`` if no pair is
        usable).

    References
    ----------
    .. [1] Uno H, Cai T, Pencina MJ, D'Agostino RB, Wei LJ. "On the C-statistics
       for evaluating overall adequacy of risk prediction procedures with
       censored survival data." Statistics in Medicine 30(10):1105-1117, 2011.
    """
    gold_t, pred_t, gold_e, pred_e = _validate(gold_times, pred_times, gold_observed, pred_observed)
    agree, t_det, n_usable, n_pairs = _comparable_pairs(gold_t, pred_t, gold_e, pred_e)
    if n_usable == 0:
        return _result(float("nan"), t_det, n_usable, n_pairs)

    g = resolve_censoring(censoring, gold_t, gold_e)
    g_at = np.clip(g(t_det), weight_floor, None)
    weights = 1.0 / g_at**2
    concordance = float(np.sum(weights * agree) / np.sum(weights))
    return _result(concordance, t_det, n_usable, n_pairs)
