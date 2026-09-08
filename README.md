# symmetric-concordance

Harrell's concordance index (C-index), extended to right-censored predictions.

The concordance index measures how well a model ranks right-censored survival outcomes: like
AUROC, it is a pairwise ranking score (0.5 is chance, 1 is perfect ordering) restricted to
the pairs whose ordering is actually known. lifelines' `concordance_index` computes Harrell's
version, but only the *outcome* may be censored; the prediction is treated as a fully
observed number.

When the prediction is itself a right-censored time (say an inferred time-to-event that for
some subjects is only a lower bound), the natural extension applies the same comparability
rule on both sides: a pair counts only when its order is known in the gold series and in the
predicted series (the smaller time is an event). This keeps the same ranking-loss
interpretation while letting both series be censored.

## Install

```bash
pip install symmetric-concordance
```

## Usage

```python
from symmetric_concordance import symmetric_concordance_index

gold_times    = [10, 20, 30, 40, 50]
gold_observed = [1, 1, 1, 1, 1]
pred_times    = [12, 33, 25, 44, 55]
pred_observed = [1, 1, 1, 0, 1]   # the 4th prediction is censored

r = symmetric_concordance_index(gold_times, pred_times, gold_observed, pred_observed)
r.concordance         # 0.888...  (8 of 9 usable pairs agree)
r.n_usable, r.n_pairs # (9, 10)   (one pair isn't orderable on the predicted side)
```

The two series are aligned by position (row `i` is the same subject in both). Event flags
default to all-observed. The result fields:

- `concordance`: comparable-pairs concordance, 0.5 is chance (NaN if no pair is usable)
- `n_usable`, `n_pairs`, `frac_usable`: pair counts
- `resolution_times`: per usable pair, the time it became orderable. Off by default — pass
  `resolution_times=True` to fill it in (see [Performance](#performance))

For an inverse-probability-of-censoring weighted (IPCW) version, which upweights longer-time
pairs to undo the bias toward short survivors, call `symmetric_concordance_ipcw`. It fits a
built-in Kaplan-Meier censoring curve unless you pass your own via `censoring=` (a fitted
lifelines `KaplanMeierFitter`, any object with `.predict`, or a callable `G(t)`):

```python
import numpy as np
from lifelines import KaplanMeierFitter
from symmetric_concordance import symmetric_concordance_ipcw

kmf = KaplanMeierFitter().fit(gold_times, 1 - np.asarray(gold_observed))
r = symmetric_concordance_ipcw(
    gold_times, pred_times, gold_observed, pred_observed, censoring=kmf
)
r.concordance   # IPCW-reweighted
```

Ties in time aren't orderable (strict `<`), so tied predictions get no half-credit. IPCW
weights are `1 / G(t)**2`, with `G` floored at `weight_floor` (default 0.05).

## Performance

`symmetric_concordance_index` counts pairs without enumerating them: the metric reduces to two
2-D dominance counts, each a Fenwick-tree sweep in O(n log n) time and O(n) memory. So a cohort
that a pairwise implementation cannot hold in memory is a few tens of milliseconds here, and a
bootstrap over it is ordinary rather than an overnight job.

| n | enumerating pairs | counting them | peak memory |
|---|---|---|---|
| 4,644 | 113 ms | 4 ms | 721 MiB → 1 MiB |
| 20,000 | 3.9 s | 17 ms | 13 GiB → 2 MiB |
| 50,000 | ~28 GiB, not runnable | 43 ms | 5 MiB |

Two things still enumerate pairs, because they need a value *per pair* rather than a count:

- `resolution_times=True`, which returns each usable pair's binding time. It is off by default
  for exactly this reason; every other field is populated either way, and both paths return
  identical counts.
- `symmetric_concordance_ipcw`. A pair's weight depends on `max(gold_t, pred_t)` across *both*
  members when the margins disagree, so the weighted total does not reduce to a dominance
  count. Budget O(n²) for it, or subsample.

Re-measure with `uv run python bench/bench_concordance.py`.

## Caveats

- `gold_times` and `pred_times` are each used only for *within-series* ordering, so they
  need not share a scale, just orient both so bigger = later event. (IPCW is the exception:
  its binding time `max(gold, pred)` needs them on the same time axis.)
- Pass `pred_observed` when predictions are censored. It defaults to all-observed, which
  brings back the bias this metric exists to avoid.
- Series are aligned by position; align by id yourself first if needed.
