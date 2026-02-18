"""Tests for savant_rs.utils.symbol_mapper – model/object registration,
lookup, and management."""

from __future__ import annotations

import pytest

from savant_rs.utils.symbol_mapper import (
    RegistrationPolicy,
    build_model_object_key,
    clear_symbol_maps,
    dump_registry,
    get_model_id,
    get_model_name,
    get_object_id,
    get_object_ids,
    get_object_label,
    get_object_labels,
    is_model_registered,
    is_object_registered,
    parse_compound_key,
    register_model_objects,
    validate_base_key,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure each test starts with a clean symbol map."""
    clear_symbol_maps()
    yield
    clear_symbol_maps()


# ── RegistrationPolicy ───────────────────────────────────────────────────


class TestRegistrationPolicy:
    def test_variants(self):
        assert RegistrationPolicy.Override is not None
        assert RegistrationPolicy.ErrorIfNonUnique is not None


# ── register & lookup ─────────────────────────────────────────────────────


class TestRegisterAndLookup:
    def test_register_model_objects(self):
        model_id = register_model_objects(
            "detector",
            {0: "person", 1: "car", 2: "bike"},
            RegistrationPolicy.ErrorIfNonUnique,
        )
        assert isinstance(model_id, int)

    def test_get_model_id(self):
        register_model_objects(
            "detector", {0: "person"}, RegistrationPolicy.Override
        )
        mid = get_model_id("detector")
        assert isinstance(mid, int)

    def test_get_model_name(self):
        register_model_objects(
            "detector", {0: "person"}, RegistrationPolicy.Override
        )
        mid = get_model_id("detector")
        name = get_model_name(mid)
        assert name == "detector"

    def test_get_model_name_not_found(self):
        name = get_model_name(999999)
        assert name is None

    def test_is_model_registered(self):
        assert not is_model_registered("nomodel")
        register_model_objects(
            "mymodel", {0: "obj"}, RegistrationPolicy.Override
        )
        assert is_model_registered("mymodel")

    def test_get_object_id(self):
        register_model_objects(
            "det", {0: "person", 1: "car"}, RegistrationPolicy.Override
        )
        model_id, obj_id = get_object_id("det", "car")
        assert isinstance(model_id, int)
        assert obj_id == 1

    def test_get_object_label(self):
        register_model_objects(
            "det", {0: "person", 1: "car"}, RegistrationPolicy.Override
        )
        mid = get_model_id("det")
        label = get_object_label(mid, 0)
        assert label == "person"

    def test_get_object_label_not_found(self):
        register_model_objects(
            "det", {0: "person"}, RegistrationPolicy.Override
        )
        mid = get_model_id("det")
        label = get_object_label(mid, 999)
        assert label is None

    def test_is_object_registered(self):
        register_model_objects(
            "det", {0: "person"}, RegistrationPolicy.Override
        )
        assert is_object_registered("det", "person")
        assert not is_object_registered("det", "unknown")

    def test_get_object_ids_batch(self):
        register_model_objects(
            "det", {0: "person", 1: "car"}, RegistrationPolicy.Override
        )
        results = get_object_ids("det", ["person", "car", "unknown"])
        assert len(results) == 3
        # person and car should have ids, unknown should be None
        assert results[0][1] is not None
        assert results[1][1] is not None
        assert results[2][1] is None

    def test_get_object_labels_batch(self):
        register_model_objects(
            "det", {0: "person", 1: "car"}, RegistrationPolicy.Override
        )
        mid = get_model_id("det")
        results = get_object_labels(mid, [0, 1, 999])
        assert len(results) == 3
        assert results[0][1] == "person"
        assert results[1][1] == "car"
        assert results[2][1] is None


# ── String utilities ─────────────────────────────────────────────────────


class TestStringUtilities:
    def test_build_model_object_key(self):
        key = build_model_object_key("model", "object")
        assert isinstance(key, str)
        assert "model" in key
        assert "object" in key

    def test_parse_compound_key(self):
        key = build_model_object_key("model", "object")
        model, obj = parse_compound_key(key)
        assert model == "model"
        assert obj == "object"

    def test_validate_base_key(self):
        result = validate_base_key("valid_key")
        assert isinstance(result, str)


# ── dump & clear ─────────────────────────────────────────────────────────


class TestDumpAndClear:
    def test_dump_empty(self):
        dump = dump_registry()
        assert isinstance(dump, str)

    def test_dump_with_data(self):
        register_model_objects(
            "det", {0: "person"}, RegistrationPolicy.Override
        )
        dump = dump_registry()
        assert "det" in dump

    def test_clear(self):
        register_model_objects(
            "det", {0: "person"}, RegistrationPolicy.Override
        )
        clear_symbol_maps()
        assert not is_model_registered("det")
