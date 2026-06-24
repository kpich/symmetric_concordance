# symmetric-concordance

A concordance index (C-index) between two right-censored survival series.

`lifelines.concordance_index` only lets the outcome be censored; the predicted score is
treated as a plain number. This computes a C-index when both the gold series and the
predicted series can be right-censored. A pair is only counted when its order is known on
both sides (the smaller time is an event), so if the predicted side has no events nothing is
orderable and you get NaN instead of a misleadingly perfect score.

## Install

Not on PyPI yet (coming soon). For now:

```bash
pip install -e .
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
- `resolution_times`: per usable pair, the time it became orderable

For the IPCW-reweighted version, call `symmetric_concordance_ipcw`. It fits a built-in
Kaplan-Meier censoring curve unless you pass your own via `censoring=` (a fitted lifelines
`KaplanMeierFitter`, any object with `.predict`, or a callable `G(t)`):

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

## Caveats

- `gold_times` and `pred_times` are each used only for *within-series* ordering, so they
  need not share a scale, just orient both so bigger = later event. (IPCW is the exception:
  its binding time `max(gold, pred)` needs them on the same time axis.)
- Pass `pred_observed` when predictions are censored. It defaults to all-observed, which
  brings back the bias this metric exists to avoid.
- Series are aligned by position; align by id yourself first if needed.

## Development

```bash
make dev     # uv sync
make test
make ruff
make mypy
make hooks   # install the pre-commit hook
```
