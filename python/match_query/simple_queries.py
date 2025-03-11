from savant_rs.match_query import FloatExpression as FE
from savant_rs.match_query import IntExpression as IE
from savant_rs.match_query import MatchQuery as MQ
from savant_rs.match_query import StringExpression as SE
from savant_rs.primitives.geometry import RBBox
from savant_rs.utils import BBoxMetricType

q = MQ.and_(MQ.id(IE.eq(5)), MQ.label(SE.eq("hello")))
print(q.yaml, "\n", q.json)

q = MQ.or_(MQ.namespace(SE.eq("model1")), MQ.namespace(SE.eq("model2")))
print(q.yaml, "\n", q.json)

q = MQ.not_(MQ.id(IE.eq(5)))
print(q.yaml, "\n", q.json)

q = MQ.stop_if_false(MQ.frame_is_key_frame())
print(q.yaml, "\n", q.json)

q = MQ.stop_if_true(MQ.not_(MQ.frame_is_key_frame()))
print(q.yaml, "\n", q.json)

# More than one person among the children of the object
q = MQ.with_children(MQ.label(SE.eq("person")), IE.ge(1))
print(q.yaml, "\n", q.json)

q = MQ.eval("1 + 1 == 2")
print(q.yaml, "\n", q.json)

q = MQ.eval(
    """(etcd("pipeline_status", false) == true || env("PIPELINE_STATUS", false) == true) && frame.keyframe"""
)
print(q.yaml, "\n", q.json)

q = MQ.box_metric(RBBox(0.5, 0.5, 0.5, 0.5, 0.0), BBoxMetricType.IoU, FE.gt(0.5))
print(q.yaml, "\n", q.json)
