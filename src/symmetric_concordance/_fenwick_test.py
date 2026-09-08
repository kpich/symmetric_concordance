"""Unit tests for :mod:`symmetric_concordance._fenwick`."""

import random

from symmetric_concordance._fenwick import Fenwick


def test_prefix_counts_items_strictly_below() -> None:
    fen = Fenwick(5)
    for i in (0, 2, 2, 4):
        fen.add(i)
    assert fen.pref(0) == 0
    assert fen.pref(1) == 1
    assert fen.pref(3) == 3
    assert fen.pref(5) == 4


def test_empty_tree_is_queryable() -> None:
    assert Fenwick(0).pref(0) == 0
    assert Fenwick(0).pref(3) == 0
    assert Fenwick(4).pref(4) == 0


def test_single_index() -> None:
    fen = Fenwick(1)
    fen.add(0)
    assert fen.pref(0) == 0
    assert fen.pref(1) == 1


def test_matches_brute_force_on_random_sequences() -> None:
    rng = random.Random(0)
    for _ in range(200):
        size = rng.randint(1, 40)
        fen = Fenwick(size)
        added: list[int] = []
        for _ in range(rng.randint(0, 60)):
            if added and rng.random() < 0.5:
                k = rng.randint(0, size)
                assert fen.pref(k) == sum(1 for a in added if a < k)
            else:
                i = rng.randrange(size)
                fen.add(i)
                added.append(i)
        for k in range(size + 1):
            assert fen.pref(k) == sum(1 for a in added if a < k)
