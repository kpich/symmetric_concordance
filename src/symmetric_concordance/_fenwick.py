"""A Fenwick tree (binary indexed tree) of counts, ``O(log n)`` per operation."""


class Fenwick:
    """Counts over indices ``0 .. size - 1``, with prefix queries.

    Parameters
    ----------
    size
        Number of distinct indices that may be added.
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
        strict "below" query needs no off-by-one at the call site.
        """
        i = min(count, self._size)
        total = 0
        while i > 0:
            total += self._tree[i]
            i -= i & -i
        return total
