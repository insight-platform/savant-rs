"""Tests for savant_rs.utils.AtomicCounter (also re-exported as
savant_rs.atomic_counter.AtomicCounter)."""

from __future__ import annotations

from savant_rs.utils import AtomicCounter


class TestAtomicCounter:
    def test_initial_value(self):
        c = AtomicCounter(0)
        assert c.get == 0

    def test_next_increments(self):
        c = AtomicCounter(0)
        assert c.next == 0
        assert c.next == 1
        assert c.next == 2

    def test_set(self):
        c = AtomicCounter(0)
        c.set(100)
        assert c.get == 100

    def test_set_and_next(self):
        c = AtomicCounter(10)
        assert c.next == 10
        c.set(50)
        assert c.next == 50
        assert c.next == 51

    def test_large_initial(self):
        c = AtomicCounter(1_000_000)
        assert c.get == 1_000_000
