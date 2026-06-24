"""Concordance index (C-index) between two right-censored survival series.

Unlike lifelines' ``concordance_index`` -- which only tolerates censoring in the
outcome and treats the predicted marker as a fully-observed number -- this
package allows *both* series to be right-censored.
"""

from .censoring import KaplanMeierCensoring
from .concordance import SymmetricConcordanceResult, symmetric_concordance_index

__all__ = [
    "KaplanMeierCensoring",
    "SymmetricConcordanceResult",
    "symmetric_concordance_index",
]

__version__ = "0.1.0"
