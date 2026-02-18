"""Tests for savant_rs.match_query – expressions, MatchQuery, QueryFunctions,
and resolver helpers."""

from __future__ import annotations

import json

import pytest

from savant_rs.match_query import (
    EtcdCredentials,
    FloatExpression,
    IntExpression,
    MatchQuery,
    QueryFunctions,
    StringExpression,
    TlsConfig,
    config_resolver_name,
    env_resolver_name,
    etcd_resolver_name,
    register_config_resolver,
    register_env_resolver,
    register_utility_resolver,
    unregister_resolver,
    update_config_resolver,
    utility_resolver_name,
)
from savant_rs.primitives import VideoFrame, VideoFrameContent
from savant_rs.primitives.geometry import RBBox
from savant_rs.utils import BBoxMetricType


# ── FloatExpression ───────────────────────────────────────────────────────


class TestFloatExpression:
    def test_eq(self):
        e = FloatExpression.eq(1.0)
        assert e is not None

    def test_ne(self):
        assert FloatExpression.ne(2.0) is not None

    def test_gt(self):
        assert FloatExpression.gt(0.5) is not None

    def test_ge(self):
        assert FloatExpression.ge(0.5) is not None

    def test_lt(self):
        assert FloatExpression.lt(10.0) is not None

    def test_le(self):
        assert FloatExpression.le(10.0) is not None

    def test_between(self):
        assert FloatExpression.between(0.0, 1.0) is not None

    def test_one_of(self):
        assert FloatExpression.one_of(1.0, 2.0, 3.0) is not None


# ── IntExpression ─────────────────────────────────────────────────────────


class TestIntExpression:
    def test_eq(self):
        assert IntExpression.eq(1) is not None

    def test_ne(self):
        assert IntExpression.ne(2) is not None

    def test_gt(self):
        assert IntExpression.gt(0) is not None

    def test_ge(self):
        assert IntExpression.ge(0) is not None

    def test_lt(self):
        assert IntExpression.lt(100) is not None

    def test_le(self):
        assert IntExpression.le(100) is not None

    def test_between(self):
        assert IntExpression.between(0, 10) is not None

    def test_one_of(self):
        assert IntExpression.one_of(1, 2, 3) is not None


# ── StringExpression ──────────────────────────────────────────────────────


class TestStringExpression:
    def test_eq(self):
        assert StringExpression.eq("hello") is not None

    def test_ne(self):
        assert StringExpression.ne("hello") is not None

    def test_contains(self):
        assert StringExpression.contains("sub") is not None

    def test_not_contains(self):
        assert StringExpression.not_contains("sub") is not None

    def test_starts_with(self):
        assert StringExpression.starts_with("pre") is not None

    def test_ends_with(self):
        assert StringExpression.ends_with("suf") is not None

    def test_one_of(self):
        assert StringExpression.one_of("a", "b", "c") is not None


# ── MatchQuery construction ──────────────────────────────────────────────


class TestMatchQueryConstruction:
    def test_idle(self):
        q = MatchQuery.idle()
        assert q is not None

    def test_namespace(self):
        q = MatchQuery.namespace(StringExpression.eq("ns"))
        assert q is not None

    def test_label(self):
        q = MatchQuery.label(StringExpression.eq("person"))
        assert q is not None

    def test_confidence(self):
        q = MatchQuery.confidence(FloatExpression.ge(0.5))
        assert q is not None

    def test_id(self):
        q = MatchQuery.id(IntExpression.eq(42))
        assert q is not None

    def test_track_id(self):
        q = MatchQuery.track_id(IntExpression.eq(1))
        assert q is not None

    def test_and_(self):
        q = MatchQuery.and_(
            MatchQuery.namespace(StringExpression.eq("ns")),
            MatchQuery.label(StringExpression.eq("lbl")),
        )
        assert q is not None

    def test_or_(self):
        q = MatchQuery.or_(
            MatchQuery.namespace(StringExpression.eq("a")),
            MatchQuery.namespace(StringExpression.eq("b")),
        )
        assert q is not None

    def test_not_(self):
        q = MatchQuery.not_(MatchQuery.idle())
        assert q is not None

    def test_stop_if_false(self):
        q = MatchQuery.stop_if_false(MatchQuery.idle())
        assert q is not None

    def test_stop_if_true(self):
        q = MatchQuery.stop_if_true(MatchQuery.idle())
        assert q is not None

    def test_box_queries(self):
        assert MatchQuery.box_x_center(FloatExpression.ge(0.0)) is not None
        assert MatchQuery.box_y_center(FloatExpression.ge(0.0)) is not None
        assert MatchQuery.box_width(FloatExpression.ge(0.0)) is not None
        assert MatchQuery.box_height(FloatExpression.ge(0.0)) is not None
        assert MatchQuery.box_area(FloatExpression.ge(0.0)) is not None
        assert MatchQuery.box_width_to_height_ratio(FloatExpression.ge(0.0)) is not None
        assert MatchQuery.box_angle(FloatExpression.ge(0.0)) is not None

    def test_track_box_queries(self):
        assert MatchQuery.track_box_x_center(FloatExpression.ge(0.0)) is not None
        assert MatchQuery.track_box_y_center(FloatExpression.ge(0.0)) is not None
        assert MatchQuery.track_box_width(FloatExpression.ge(0.0)) is not None
        assert MatchQuery.track_box_height(FloatExpression.ge(0.0)) is not None
        assert MatchQuery.track_box_area(FloatExpression.ge(0.0)) is not None
        assert (
            MatchQuery.track_box_width_to_height_ratio(FloatExpression.ge(0.0))
            is not None
        )
        assert MatchQuery.track_box_angle(FloatExpression.ge(0.0)) is not None

    def test_parent_queries(self):
        assert MatchQuery.parent_id(IntExpression.eq(0)) is not None
        assert MatchQuery.parent_namespace(StringExpression.eq("ns")) is not None
        assert MatchQuery.parent_label(StringExpression.eq("lbl")) is not None
        assert MatchQuery.parent_defined() is not None

    def test_defined_checks(self):
        assert MatchQuery.confidence_defined() is not None
        assert MatchQuery.track_id_defined() is not None
        assert MatchQuery.box_angle_defined() is not None
        assert MatchQuery.track_box_angle_defined() is not None
        assert MatchQuery.attributes_empty() is not None

    def test_attribute_defined(self):
        q = MatchQuery.attribute_defined("ns", "key")
        assert q is not None

    def test_frame_queries(self):
        assert MatchQuery.frame_source_id(StringExpression.eq("cam")) is not None
        assert MatchQuery.frame_is_key_frame() is not None
        assert MatchQuery.frame_width(IntExpression.eq(1920)) is not None
        assert MatchQuery.frame_height(IntExpression.eq(1080)) is not None
        assert MatchQuery.frame_no_video() is not None
        assert MatchQuery.frame_transcoding_is_copy() is not None
        assert MatchQuery.frame_attribute_exists("ns", "key") is not None
        assert MatchQuery.frame_attributes_empty() is not None

    def test_eval(self):
        q = MatchQuery.eval("true")
        assert q is not None

    def test_attributes_jmes_query(self):
        q = MatchQuery.attributes_jmes_query("length(@) > `0`")
        assert q is not None

    def test_frame_attributes_jmes_query(self):
        q = MatchQuery.frame_attributes_jmes_query("length(@) > `0`")
        assert q is not None

    def test_box_metric(self):
        bbox = RBBox(5.0, 5.0, 10.0, 10.0)
        q = MatchQuery.box_metric(bbox, BBoxMetricType.IoU, FloatExpression.ge(0.5))
        assert q is not None

    def test_track_box_metric(self):
        bbox = RBBox(5.0, 5.0, 10.0, 10.0)
        q = MatchQuery.track_box_metric(
            bbox, BBoxMetricType.IoU, FloatExpression.ge(0.5)
        )
        assert q is not None

    def test_with_children(self):
        q = MatchQuery.with_children(
            MatchQuery.label(StringExpression.eq("child")),
            IntExpression.ge(1),
        )
        assert q is not None


# ── MatchQuery serialization ─────────────────────────────────────────────


class TestMatchQuerySerialization:
    def test_json_roundtrip(self):
        q = MatchQuery.namespace(StringExpression.eq("test"))
        j = q.json
        parsed = json.loads(j)
        assert parsed is not None
        q2 = MatchQuery.from_json(j)
        assert q2 is not None

    def test_json_pretty(self):
        q = MatchQuery.idle()
        jp = q.json_pretty
        assert len(jp) > 0

    def test_yaml_roundtrip(self):
        q = MatchQuery.label(StringExpression.eq("person"))
        y = q.yaml
        assert len(y) > 0
        q2 = MatchQuery.from_yaml(y)
        assert q2 is not None


# ── MatchQuery actual filtering ──────────────────────────────────────────


class TestMatchQueryFiltering:
    @pytest.fixture()
    def frame(self):
        f = VideoFrame(
            source_id="cam",
            framerate="30/1",
            width=640,
            height=480,
            content=VideoFrameContent.none(),
        )
        f.create_object("det", "person", detection_box=RBBox(0, 0, 10, 10))
        f.create_object("det", "car", detection_box=RBBox(0, 0, 20, 20))
        f.create_object("cls", "person", detection_box=RBBox(0, 0, 30, 30))
        return f

    def test_access_objects_with_query(self, frame):
        q = MatchQuery.label(StringExpression.eq("person"))
        view = frame.access_objects(q)
        assert len(view) == 2

    def test_access_objects_namespace(self, frame):
        q = MatchQuery.namespace(StringExpression.eq("det"))
        view = frame.access_objects(q)
        assert len(view) == 2


# ── QueryFunctions ───────────────────────────────────────────────────────


class TestQueryFunctions:
    @pytest.fixture()
    def frame(self):
        f = VideoFrame(
            source_id="cam",
            framerate="30/1",
            width=640,
            height=480,
            content=VideoFrameContent.none(),
        )
        f.create_object("det", "person", detection_box=RBBox(0, 0, 10, 10))
        f.create_object("det", "car", detection_box=RBBox(0, 0, 20, 20))
        return f

    def test_filter(self, frame):
        all_objs = frame.get_all_objects()
        q = MatchQuery.label(StringExpression.eq("person"))
        filtered = QueryFunctions.filter(all_objs, q)
        assert len(filtered) == 1

    def test_partition(self, frame):
        all_objs = frame.get_all_objects()
        q = MatchQuery.label(StringExpression.eq("person"))
        matched, unmatched = QueryFunctions.partition(all_objs, q)
        assert len(matched) == 1
        assert len(unmatched) == 1


# ── Resolver helpers ─────────────────────────────────────────────────────


class TestResolverNames:
    def test_utility_resolver_name(self):
        name = utility_resolver_name()
        assert isinstance(name, str) and len(name) > 0

    def test_env_resolver_name(self):
        assert isinstance(env_resolver_name(), str)

    def test_etcd_resolver_name(self):
        assert isinstance(etcd_resolver_name(), str)

    def test_config_resolver_name(self):
        assert isinstance(config_resolver_name(), str)


class TestResolverRegistration:
    def test_register_utility_resolver(self):
        register_utility_resolver()

    def test_register_env_resolver(self):
        register_env_resolver()

    def test_register_config_resolver(self):
        register_config_resolver({"key": "value"})

    def test_update_config_resolver(self):
        register_config_resolver({"k": "v"})
        update_config_resolver({"k": "v2", "k2": "v3"})

    def test_unregister_resolver(self):
        name = config_resolver_name()
        register_config_resolver({"k": "v"})
        unregister_resolver(name)


# ── EtcdCredentials / TlsConfig ──────────────────────────────────────────


class TestEtcdCredentials:
    def test_create(self):
        cred = EtcdCredentials(username="user", password="pass")
        assert cred is not None


class TestTlsConfig:
    def test_create(self):
        tls = TlsConfig("ca.pem", "cert.pem", "key.pem")
        assert tls is not None
