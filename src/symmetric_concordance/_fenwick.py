"""A minimal Fenwick tree (binary indexed tree) of counts.

Private helper behind the O(n log n) pair-counting path in
:mod:`symmetric_concordance.concordance`. Both counts the metric needs are 2-D
dominance counts, and a dominance count is a sweep in one coordinate plus
prefix-counting in the other -- which is exactly what this structure does in
``O(log n)`` per operation and ``O(n)`` memory.

Plain Python ints, no numpy: the arrays involved are tiny and the per-operation
cost is a handful of integer ops, so numpy's per-call overhead would dominate.
"""


class Fenwick:
    """Counts over indices ``0 .. size - 1``, with prefix queries.

    Parameters
    ----------
    size
        Number of distinct indices that may be added. ``0`` is allowed and
        yields a tree that can only be queried (always ``0``).
    """

    __slots__ = ("_size", "_tree")

    def __init__(self, size: int) -> None:
        self._size = size
        self._tree = [0] * (size + 1)  # 1-based internally

    def add(self, index: int) -> None:
        """Record one item at ``index``."""
        i = index + 1
        while i <= self._size:
            self._tree[i] += 1
            i += i & -i

    def pref(self, count: int) -> int:
        """Return how many recorded items have an index ``< count``.

        ``pref(0)`` is ``0`` and ``pref(size)`` is the total recorded, so a
        strict "below" query needs no off-by-one juggling at the call site.
        """
        i = min(count, self._size)
        total = 0
        while i > 0:
            total += self._tree[i]
            i -= i & -i
        return total
