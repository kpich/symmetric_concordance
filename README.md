# symmetric-concordance

Concordance index (C-index) between **two right-censored survival series**.

## Why

The textbook C-index (Harrell; Uno's IPCW variant) compares a *fully-observed
marker* against a *censored outcome*. lifelines' `concordance_index(event_times,
predicted_scores, event_observed)` is exactly that — it consults only the
**outcome's** censoring flag and treats the predicted marker as a plain number.

That breaks when the "marker" is itself right-censored — e.g. a predicted
event time that, for some subjects, is only a *censoring* time (the model never
made a firm call). Fed in as a raw score, a censored prediction at time `C`
becomes indistinguishable from an event at `C`; the prediction's own event flag
is silently discarded. If un-resolved predictions fall back to the gold date,
that leaks the answer and inflates the index:

```python
from lifelines.utils import concordance_index

times = [10, 20, 30, 40, 50]
# Predictions are *copies of the gold dates*, but all censored (no real call).
concordance_index(times, times, [1, 1, 1, 1, 1])   # -> 1.0  (spurious!)
```

This package fixes it by using **both** series' event indicators to decide
whether a pair is orderable. For a pair `(i, j)` in one right-censored margin,
the order is known only if the **smaller** time is an event (Harrell's
comparability rule). A pair is *usable* only if it is orderable in **both**
margins, and *concordant* if the two margins agree on direction. The "handle"
lives in the pair-counting logic — not in any Kaplan-Meier fit. (The optional
IPCW reweighting uses a censoring curve `G(t)`, but that only corrects the
short-time bias; it is not part of the core fix.)

```python
from symmetric_concordance import symmetric_concordance_index

# Same leak case: predictions are censored, so nothing is orderable -> NaN.
r = symmetric_concordance_index(
    times, times, gold_observed=[1, 1, 1, 1, 1], pred_observed=[0, 0, 0, 0, 0]
)
r.n_usable      # 0
r.concordance   # nan  (correctly: "no information")
```

## Does this exist elsewhere?

Not that we could find. lifelines' `concordance_index` and scikit-survival's
`concordance_index_censored` / `concordance_index_ipcw` all assume an
uncensored risk score with censoring only in the outcome. A symmetric variant
that tolerates censoring in **both** series is absent from both.

## Install

```bash
pip install -e .          # core (numpy only)
pip install -e ".[ipcw]"  # + lifelines, for KaplanMeierFitter interop
```

Requires Python ≥ 3.10. The core has a single dependency, `numpy`.

## Usage

### Basic

```python
from symmetric_concordance import symmetric_concordance_index

gold_times    = [10, 20, 30, 40, 50]
gold_observed = [1, 1, 1, 1, 1]
pred_times    = [12, 18, 33, 38, 51]   # same ordering as gold
pred_observed = [1, 1, 1, 1, 1]

r = symmetric_concordance_index(gold_times, pred_times, gold_observed, pred_observed)
print(r.concordance)              # 1.0
print(r.concordance_ipcw)         # 1.0
print(r.n_usable, r.n_pairs)      # 10 10
print(float(r))                   # 1.0  (float() returns the unweighted index)
```

The two series are aligned **position-wise**: row `i` of the gold arrays and row
`i` of the predicted arrays describe the same subject. If your data lives in two
DataFrames keyed by an id, merge them first (`gold.merge(pred, on="id")`) and
pass the aligned columns. Event flags default to all-observed, recovering the
ordinary fully-observed case.

The result is a `SymmetricConcordanceResult` dataclass:

| field | meaning |
| --- | --- |
| `concordance` | comparable-pairs concordance using both event flags (0.5 = chance) |
| `concordance_ipcw` | same, IPCW-reweighted (`nan` if `ipcw=False`) |
| `n_usable` | pairs orderable in both margins |
| `n_pairs` | total pairs, `n * (n - 1) / 2` |
| `frac_usable` | `n_usable / n_pairs` |
| `resolution_times` | per usable pair, binding time `max(gold_t*, pred_t*)` |

### lifelines interop

By default IPCW uses a built-in numpy product-limit estimator of the censoring
curve, fit from the gold series. You can instead hand in a lifelines
`KaplanMeierFitter` (or any object with a `.predict`, or a plain callable
`G(t)`) — the results are identical:

```python
import numpy as np
from lifelines import KaplanMeierFitter
from symmetric_concordance import symmetric_concordance_index

# Censoring KM: fit on the gold follow-up with the event flag *flipped*.
kmf = KaplanMeierFitter().fit(gold_times, 1 - np.asarray(gold_observed))

r = symmetric_concordance_index(
    gold_times, pred_times, gold_observed, pred_observed, censoring=kmf
)
print(r.concordance_ipcw)
```

To skip IPCW entirely, pass `ipcw=False` (then `concordance_ipcw` is `nan` and
no censoring curve is fit).

## Notes

- Comparisons use a strict `<` on times, so exact time ties are treated as not
  orderable and tied predictions receive **no half-credit** (unlike lifelines'
  `concordance_index`).
- IPCW weights are `1 / G(t)**2` with `G` clipped from below at `weight_floor`
  (default `0.05`) to keep late, sparsely-observed pairs from exploding.

## Development

The Makefile uses [uv](https://docs.astral.sh/uv/), which manages a project-local `.venv`
automatically — nothing is installed into your global/base Python. `uv run` creates and
syncs the env on demand (including the `dev` dependency group), so the targets work without
any manual activation; `make dev` (`uv sync`) is just an explicit way to pre-build the env.

```bash
make dev     # uv sync — create/sync .venv with the package + dev tools
make test    # run the test suite (uv run pytest)
make ruff    # ruff check + ruff format --check
make mypy    # strict type-check of src + tests
```
