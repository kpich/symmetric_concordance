# symmetric-concordance

A concordance index (C-index) between two right-censored survival series.

`lifelines.concordance_index` only lets the outcome be censored; the predicted score is
treated as a plain number. This computes a C-index when both the gold series and the
predicted series can be right-censored. A pair is only counted when its order is known on
both sides (the smaller time is an event), so if the predicted side has no events nothing is
orderable and you get NaN instead of a misleadingly perfect score.

## Install

```bash
pip install -e .
pip install -e ".[ipcw]"   # adds lifelines, for KaplanMeierFitter interop
```

## Usage

```python
from symmetric_concordance import symmetric_concordance_index

gold_times    = [10, 20, 30, 40, 50]
gold_observed = [1, 1, 1, 1, 1]
pred_times    = [12, 18, 33, 38, 51]
pred_observed = [1, 1, 1, 1, 1]

r = symmetric_concordance_index(gold_times, pred_times, gold_observed, pred_observed)
r.concordance         # 1.0
r.concordance_ipcw    # 1.0
r.n_usable, r.n_pairs # (10, 10)
```

The two series are aligned by position (row `i` is the same subject in both). Event flags
default to all-observed. The result fields:

- `concordance`: comparable-pairs concordance, 0.5 is chance
- `concordance_ipcw`: IPCW-reweighted version (NaN when `ipcw=False`)
- `n_usable`, `n_pairs`, `frac_usable`: pair counts
- `resolution_times`: per usable pair, the time it became orderable

IPCW uses a built-in Kaplan-Meier estimate of the censoring curve by default. Pass a fitted
lifelines `KaplanMeierFitter` (or any object with `.predict`, or a callable `G(t)`) to use
your own:

```python
import numpy as np
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter().fit(gold_times, 1 - np.asarray(gold_observed))
r = symmetric_concordance_index(
    gold_times, pred_times, gold_observed, pred_observed, censoring=kmf
)
```

Ties in time aren't orderable (strict `<`), so tied predictions get no half-credit. IPCW
weights are `1 / G(t)**2`, with `G` floored at `weight_floor` (default 0.05).

## Development

```bash
make dev     # uv sync
make test
make ruff
make mypy
make hooks   # install the pre-commit hook
```
