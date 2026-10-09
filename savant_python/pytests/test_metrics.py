"""Tests for savant_rs.metrics – CounterFamily, GaugeFamily, HistogramFamily,
delete_metric_family, set_extra_labels."""

from __future__ import annotations

import sys

import pytest

from savant_rs.metrics import (
    CounterFamily,
    GaugeFamily,
    HistogramFamily,
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


# ── HistogramFamily ──────────────────────────────────────────────────────


class TestHistogramFamily:
    def test_create(self):
        hf = HistogramFamily.get_or_create_histogram_family(
            "test_histogram_create",
            "A test histogram",
            ["route"],
            [0.1, 1.0],
            "seconds",
        )
        assert hf is not None
        assert hf.buckets == [0.1, 1.0]

    def test_default_buckets(self):
        hf = HistogramFamily.get_or_create_histogram_family("test_histogram_default")
        assert hf.buckets == [
            0.005,
            0.01,
            0.025,
            0.05,
            0.1,
            0.25,
            0.5,
            1.0,
            2.5,
            5.0,
            10.0,
        ]

    def test_observe_and_get(self):
        hf = HistogramFamily.get_or_create_histogram_family(
            "test_histogram_observe",
            "desc",
            ["label"],
            [1.0, 5.0],
        )
        assert hf.get(["a"]) is None
        for v in (0.5, 1.0, 3.0, 100.0):
            hf.observe(v, ["a"])
        total, count, buckets = hf.get(["a"])
        assert total == pytest.approx(104.5)
        assert count == 4
        assert buckets == [(1.0, 2), (5.0, 3), (float("inf"), 4)]

    def test_observe_without_labels(self):
        hf = HistogramFamily.get_or_create_histogram_family(
            "test_histogram_no_labels", buckets=[1.0]
        )
        hf.observe(0.5)
        assert hf.get() == (0.5, 1, [(1.0, 1), (float("inf"), 1)])

    def test_delete(self):
        hf = HistogramFamily.get_or_create_histogram_family(
            "test_histogram_delete",
            "desc",
            ["label"],
            [1.0],
        )
        hf.observe(2.0, ["del"])
        deleted = hf.delete(["del"])
        assert deleted == (2.0, 1, [(1.0, 0), (float("inf"), 1)])
        assert hf.get(["del"]) is None

    def test_wrong_labels(self):
        hf = HistogramFamily.get_or_create_histogram_family(
            "test_histogram_wrong_labels", "desc", ["label"]
        )
        with pytest.raises(ValueError):
            hf.observe(1.0, [])
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError):
                hf.observe(bad, ["a"])
        assert hf.get(["a"]) is None

    @pytest.mark.parametrize(
        "label_names, buckets",
        [
            (["le"], None),
            ([], [2.0, 1.0]),
            ([], [1.0, 1.0]),
            ([], [1.0, float("inf")]),
            ([], [1.0, sys.float_info.max]),
        ],
    )
    def test_invalid_config(self, label_names, buckets):
        with pytest.raises(ValueError):
            HistogramFamily.get_or_create_histogram_family(
                "test_histogram_invalid", "desc", label_names, buckets
            )
        assert HistogramFamily.get_histogram_family("test_histogram_invalid") is None

    def test_get_histogram_family(self):
        HistogramFamily.get_or_create_histogram_family("test_histogram_retrieve")
        assert (
            HistogramFamily.get_histogram_family("test_histogram_retrieve") is not None
        )
        delete_metric_family("test_histogram_retrieve")
        assert HistogramFamily.get_histogram_family("test_histogram_retrieve") is None

    def test_get_histogram_family_not_found(self):
        assert HistogramFamily.get_histogram_family("nonexistent_histogram_xyz") is None


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
