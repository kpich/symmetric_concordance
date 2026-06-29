"""Concordance index (C-index) between two right-censored survival series.

Unlike lifelines' ``concordance_index`` -- which only tolerates censoring in the
outcome and treats the predicted marker as a fully-observed number -- this
package allows *both* series to be right-censored.
"""

from .concordance import (
    SymmetricConcordanceResult,
    symmetric_concordance_index,
    symmetric_concordance_ipcw,
)

__all__ = [
    "SymmetricConcordanceResult",
    "symmetric_concordance_index",
    "symmetric_concordance_ipcw",
]

__version__ = "0.2.0.dev0"
