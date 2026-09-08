"""Time the O(n log n) counting path against the dense pair enumeration.

Not a test and not run by CI: timings are machine- and load-dependent, so
asserting on them buys flakiness rather than coverage. This exists so the
numbers quoted in release notes can be re-measured after a change.

    uv run python bench/bench_concordance.py [n ...]
"""

import argparse
import time
import tracemalloc
from collections.abc import Callable
from functools import partial

import numpy as np

from symmetric_concordance import symmetric_concordance_index

DEFAULT_SIZES = (1_000, 4_644, 20_000, 50_000)
# Beyond this the dense path allocates tens of GB; skip it rather than swap.
DENSE_LIMIT = 20_000
REPEATS = 3


def make_cohort(n: int, grid: float | None, seed: int = 0) -> tuple[np.ndarray, ...]:
    """Build a cohort with realistic censoring in both margins.

    ``grid`` rounds times onto a coarse scale, the way real follow-up in months
    off a day-resolution date lands on a few thousand distinct values. That is
    the favourable case for the sweep -- fewer distinct predicted times means a
    shallower Fenwick -- so ``None`` (continuous, every time distinct) is worth
    measuring as the other bound.
    """
    rng = np.random.default_rng(seed)
    gold_t = rng.exponential(24.0, n)  # months
    pred_t = np.abs(gold_t + rng.normal(0, 6.0, n))
    if grid is not None:
        gold_t, pred_t = np.round(gold_t / grid) * grid, np.round(pred_t / grid) * grid
    gold_e = rng.random(n) < 0.6
    pred_e = rng.random(n) < 0.5
    return gold_t, pred_t, gold_e, pred_e


def measure(fn: Callable[[], object]) -> tuple[float, float, object]:
    """Return ``(best seconds, peak MiB, result)`` for ``fn``.

    Time and memory are measured in separate passes: tracemalloc hooks every
    allocation, which taxes the Python-level sweep far more than the numpy path
    and would make the comparison read backwards.
    """
    best = float("inf")
    result: object = None
    for _ in range(REPEATS):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)

    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return best, peak / 2**20, result


def main() -> None:
    """Print one table per tie regime over the requested cohort sizes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sizes", nargs="*", type=int, default=list(DEFAULT_SIZES))
    args = parser.parse_args()

    for label, grid in (("times on a 0.1-month grid", 0.1), ("continuous times", None)):
        print(f"\n**{label}**\n")
        run_table(args.sizes, grid)


def run_table(sizes: list[int], grid: float | None) -> None:
    """Print one markdown table of dense-vs-fast over ``sizes``."""
    print("| n | dense | fast | speedup | dense peak | fast peak |")
    print("|---|---|---|---|---|---|")
    for n in sizes:
        cohort = make_cohort(n, grid)
        fast_s, fast_mib, fast_r = measure(partial(symmetric_concordance_index, *cohort))
        if n > DENSE_LIMIT:
            print(
                f"| {n:,} | not runnable | {fast_s * 1e3:.0f} ms | -- | "
                f"~{n * (n - 1) / 2 * 24 / 2**30:.0f} GB | {fast_mib:.0f} MiB |"
            )
            continue
        dense_s, dense_mib, dense_r = measure(
            partial(symmetric_concordance_index, *cohort, resolution_times=True)
        )
        assert fast_r.concordance == dense_r.concordance, n
        print(
            f"| {n:,} | {dense_s * 1e3:.0f} ms | {fast_s * 1e3:.0f} ms | "
            f"{dense_s / fast_s:.0f}x | {dense_mib:.0f} MiB | {fast_mib:.0f} MiB |"
        )


if __name__ == "__main__":
    main()
