"""Censoring-distribution estimation for the IPCW correction.

The inverse-probability-of-censoring weighting (IPCW) used by
:func:`symmetric_concordance.concordance.symmetric_concordance_index` needs an
estimate of the *censoring* survival ``G(t) = P(C > t)`` -- the probability that
a subject has not yet been censored by time ``t``.

This module ships a small, dependency-free product-limit (Kaplan-Meier)
estimator of ``G``. It is interchangeable with lifelines'
``KaplanMeierFitter``: anything exposing a ``predict`` method (or any plain
callable ``G(t)``) can be handed to the metric instead, so the package stays
numpy-only while interoperating seamlessly with lifelines.
"""

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray


@runtime_checkable
class SupportsPredict(Protocol):
    """Anything that can evaluate a survival curve at given times.

    lifelines' ``KaplanMeierFitter`` satisfies this, as does
    :class:`KaplanMeierCensoring`.
    """

    def predict(self, times: Any) -> Any:  # noqa: D102
        ...


class KaplanMeierCensoring:
    """Product-limit estimate of the censoring survival ``G(t) = P(C > t)``.

    The censoring distribution is estimated with the standard Kaplan-Meier
    estimator after *flipping* the observed flag: a subject that was censored in
    the original data counts as a "censoring event" here. This mirrors the
    convention ``kmf.fit(times, 1 - observed)`` used with lifelines.
    """

    def __init__(self) -> None:
        self._times: NDArray[np.float64] | None = None
        self._surv: NDArray[np.float64] | None = None

    def fit(self, times: ArrayLike, observed: ArrayLike) -> "KaplanMeierCensoring":
        """Fit ``G`` from follow-up ``times`` and event flags ``observed``.

        Parameters
        ----------
        times
            Length-n observed times (event or censoring).
        observed
            Length-n flags, truthy where the event of interest was observed and
            falsy where the subject was censored.

        Returns
        -------
        KaplanMeierCensoring
            ``self``, fitted in place.
        """
        t = np.asarray(times, dtype=float)
        o = np.asarray(observed)
        if t.ndim != 1:
            raise ValueError("times must be 1-dimensional")
        if t.shape != o.shape:
            raise ValueError("times and observed must have the same shape")

        censor_event = ~o.astype(bool)  # censoring is the "event" for G
        order = np.argsort(t, kind="stable")
        t_sorted = t[order]
        e_sorted = censor_event[order]

        uniq = np.unique(t_sorted)
        surv = np.empty(uniq.shape[0], dtype=np.float64)
        running = 1.0
        for k, ut in enumerate(uniq):
            at_risk = int(np.count_nonzero(t_sorted >= ut))
            failures = int(np.count_nonzero((t_sorted == ut) & e_sorted))
            if at_risk > 0:
                running *= 1.0 - failures / at_risk
            surv[k] = running

        self._times = uniq
        self._surv = surv
        return self

    def predict(self, times: ArrayLike) -> NDArray[np.float64]:
        """Evaluate ``G`` at ``times`` using right-continuous step lookup.

        ``G`` is ``1.0`` before the first step and carries the last value
        forward after the final step.
        """
        if self._times is None or self._surv is None:
            raise RuntimeError("call fit() before predict()")
        query = np.atleast_1d(np.asarray(times, dtype=float))
        idx = np.searchsorted(self._times, query, side="right") - 1
        clipped = np.clip(idx, 0, self._surv.shape[0] - 1)
        out = np.where(idx < 0, 1.0, self._surv[clipped])
        return out.astype(np.float64)


def resolve_censoring(
    censoring: SupportsPredict | Callable[..., Any] | None,
    gold_times: ArrayLike,
    gold_observed: ArrayLike,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Normalize ``censoring`` into a ``G(t)`` callable returning a float array.

    ``None`` fits a :class:`KaplanMeierCensoring` on the gold series; an object
    with ``.predict`` (e.g. a lifelines ``KaplanMeierFitter``) is used directly;
    a plain callable is assumed to already be ``G``.
    """
    if censoring is None:
        predict: Callable[..., Any] = KaplanMeierCensoring().fit(gold_times, gold_observed).predict
    elif isinstance(censoring, SupportsPredict):
        predict = censoring.predict
    elif callable(censoring):
        predict = censoring
    else:
        raise TypeError("censoring must be None, an object with .predict, or a callable G(t)")

    def g(times: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.atleast_1d(np.asarray(predict(times), dtype=float)).astype(np.float64)

    return g
