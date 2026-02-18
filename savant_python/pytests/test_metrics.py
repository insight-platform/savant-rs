"""Tests for savant_rs.metrics – CounterFamily, GaugeFamily,
delete_metric_family, set_extra_labels."""

from __future__ import annotations

import pytest

from savant_rs.metrics import (
    CounterFamily,
    GaugeFamily,
    delete_metric_family,
    set_extra_labels,
)


# ── CounterFamily ────────────────────────────────────────────────────────


class TestCounterFamily:
    def test_create(self):
        cf = CounterFamily.get_or_create_counter_family(
            "test_counter_create",
            "A test counter",
            ["method", "status"],
            None,
        )
        assert cf is not None

    def test_set_and_get(self):
        cf = CounterFamily.get_or_create_counter_family(
            "test_counter_set_get",
            "desc",
            ["label"],
            None,
        )
        cf.set(10, ["a"])
        val = cf.get(["a"])
        assert val == 10

    def test_inc(self):
        cf = CounterFamily.get_or_create_counter_family(
            "test_counter_inc",
            "desc",
            ["label"],
            None,
        )
        cf.set(5, ["x"])
        cf.inc(3, ["x"])
        val = cf.get(["x"])
        assert val == 8

    def test_delete(self):
        cf = CounterFamily.get_or_create_counter_family(
            "test_counter_delete",
            "desc",
            ["label"],
            None,
        )
        cf.set(1, ["del"])
        deleted = cf.delete(["del"])
        assert deleted is not None
        assert cf.get(["del"]) is None

    def test_get_nonexistent(self):
        cf = CounterFamily.get_or_create_counter_family(
            "test_counter_nonexist",
            "desc",
            ["label"],
            None,
        )
        assert cf.get(["nope"]) is None

    def test_get_counter_family(self):
        CounterFamily.get_or_create_counter_family(
            "test_counter_retrieve",
            "desc",
            ["label"],
            None,
        )
        cf = CounterFamily.get_counter_family("test_counter_retrieve")
        assert cf is not None

    def test_get_counter_family_not_found(self):
        cf = CounterFamily.get_counter_family("nonexistent_counter_xyz")
        assert cf is None


# ── GaugeFamily ──────────────────────────────────────────────────────────


class TestGaugeFamily:
    def test_create(self):
        gf = GaugeFamily.get_or_create_gauge_family(
            "test_gauge_create",
            "A test gauge",
            ["region"],
            None,
        )
        assert gf is not None

    def test_set_and_get(self):
        gf = GaugeFamily.get_or_create_gauge_family(
            "test_gauge_set_get",
            "desc",
            ["label"],
            None,
        )
        gf.set(3.14, ["a"])
        val = gf.get(["a"])
        assert val == pytest.approx(3.14)

    def test_delete(self):
        gf = GaugeFamily.get_or_create_gauge_family(
            "test_gauge_delete",
            "desc",
            ["label"],
            None,
        )
        gf.set(1.0, ["del"])
        deleted = gf.delete(["del"])
        assert deleted is not None
        assert gf.get(["del"]) is None

    def test_get_gauge_family(self):
        GaugeFamily.get_or_create_gauge_family(
            "test_gauge_retrieve",
            "desc",
            ["label"],
            None,
        )
        gf = GaugeFamily.get_gauge_family("test_gauge_retrieve")
        assert gf is not None

    def test_get_gauge_family_not_found(self):
        gf = GaugeFamily.get_gauge_family("nonexistent_gauge_xyz")
        assert gf is None


# ── delete_metric_family ─────────────────────────────────────────────────


class TestDeleteMetricFamily:
    def test_delete(self):
        CounterFamily.get_or_create_counter_family(
            "test_delete_family",
            "desc",
            ["label"],
            None,
        )
        delete_metric_family("test_delete_family")
        # After deletion, get_counter_family should return None
        assert CounterFamily.get_counter_family("test_delete_family") is None


# ── set_extra_labels ─────────────────────────────────────────────────────


class TestSetExtraLabels:
    def test_set(self):
        # Should not raise
        set_extra_labels({"env": "test", "host": "localhost"})

    def test_empty(self):
        set_extra_labels({})
