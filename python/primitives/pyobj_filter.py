from threading import Barrier, Thread
from timeit import default_timer as timer

from savant_rs.match_query import FloatExpression as FE
from savant_rs.match_query import MatchQuery as Q
from savant_rs.match_query import StringExpression as SE
from savant_rs.match_query import register_utility_resolver
from savant_rs.primitives import (IdCollisionResolutionPolicy, VideoFrameBatch,
                                  VideoObject)
from savant_rs.primitives.geometry import RBBox
from savant_rs.utils import enable_dl_detection, gen_frame

enable_dl_detection()

import random

register_utility_resolver()

T = 1
N = 10


def thread_python(barrier):
    global results

    def and_(*l):
        return lambda o: all(f(o) for f in l)

    def or_(*l):
        return lambda o: any(f(o) for f in l)

    def label_endswith(x):
        def label_endswith_helper(o):
            return o.label.endswith(x)

        return label_endswith_helper

    def angle():
        def angle_helper(o):
            return o.detection_box.angle is not None

        return angle_helper

    def expr():
        return and_(
            or_(
                lambda o: o.namespace in ["created_by_2", "created_by_4"],
                or_(label_endswith("2"), label_endswith("4"), label_endswith("6")),
            ),
            angle(),
            or_(lambda o: o.confidence > 0.5, lambda o: o.confidence < 0.3),
        )

    run_expr = expr()

    objects = [
        VideoObject(
            id=1,
            namespace="created_by_{}".format(i),
            label="person_{}".format(i),
            detection_box=RBBox(0.1, 0.2, 0.3, 0.4, 30.0),
            confidence=random.random(),
            attributes=[],
            track_id=None,
            track_box=None,
        )
        for i in range(N)
    ]

    barrier.wait()
    t = timer()
    for o in objects:
        run_expr(o)
    results.append((timer() - t) * 1000_000)


b = Barrier(T + 1)
threads = []
results = []

for i in range(T):
    t = Thread(target=thread_python, args=(b,))
    t.start()
    threads.append(t)

b.wait()
tim = timer()

for t in threads:
    t.join()

print("Python Query\t\t", (timer() - tim) * 1000_000)


def thread_full(barrier):
    global results
    f = gen_frame()
    f.delete_objects(Q.idle())
    for i in range(0, N):
        f.add_object(
            VideoObject(
                id=i,
                namespace="created_by_{}".format(i),
                label="person_{}".format(i),
                detection_box=RBBox(0.1, 0.2, 0.3, 0.4, 30.0),
                confidence=random.random(),
                attributes=[],
                track_id=None,
                track_box=None,
            ),
            IdCollisionResolutionPolicy.Error,
        )

    full_expr = Q.eval(
        """ 
    (contains(("created_by_4", "created_by_2"), namespace) || ends_with(label, "2") || ends_with(label, "4") || ends_with(label, "6")) &&
    !is_empty(bbox.angle) &&
    (confidence > 0.5 || confidence < 0.3)
    """
    )

    barrier.wait()
    t = timer()
    f.access_objects(full_expr)
    results.append((timer() - t) * 1000_000)


b = Barrier(T + 1)
threads = []
results = []

for i in range(T):
    t = Thread(target=thread_full, args=(b,))
    t.start()
    threads.append(t)

b.wait()
tim = timer()

for t in threads:
    t.join()

print("Full Query\t\t", (timer() - tim) * 1000_000)


def thread_decomposed(barrier):
    global results
    f = gen_frame()
    f.delete_objects(Q.idle())
    for i in range(0, N):
        f.add_object(
            VideoObject(
                id=i,
                namespace="created_by_{}".format(i),
                label="person_{}".format(i),
                detection_box=RBBox(0.1, 0.2, 0.3, 0.4, 30.0),
                confidence=random.random(),
                attributes=[],
                track_id=None,
                track_box=None,
            ),
            IdCollisionResolutionPolicy.Error,
        )

    decomposed_expr = Q.and_(
        Q.or_(
            Q.eval("""namespace == "created_by_4" """),
            Q.eval("""namespace == "created_by_2" """),
            Q.eval("""ends_with(label,"2")"""),
            Q.eval("""ends_with(label,"4")"""),
            Q.eval("""ends_with(label,"6")"""),
        ),
        Q.eval("""!is_empty(bbox.angle)"""),
        Q.eval("""confidence > 0.5 || confidence < 0.3"""),
    )

    barrier.wait()
    t = timer()
    f.access_objects(decomposed_expr)
    results.append((timer() - t) * 1000_000)


b = Barrier(T + 1)
threads = []
results = []

for i in range(T):
    t = Thread(target=thread_decomposed, args=(b,))
    t.start()
    threads.append(t)

b.wait()
tim = timer()

for t in threads:
    t.join()

print("Decomposed Query\t", (timer() - tim) * 1000_000)


def measure_batch_full():
    batch = VideoFrameBatch()
    for id in range(0, T):
        f = gen_frame()
        f.source_id = f"source_{id}"
        f.delete_objects(Q.idle())
        for i in range(0, N):
            f.add_object(
                VideoObject(
                    id=i,
                    namespace="created_by_{}".format(i),
                    label="person_{}".format(i),
                    detection_box=RBBox(0.1, 0.2, 0.3, 0.4, 30.0),
                    confidence=random.random(),
                    attributes=[],
                    track_id=None,
                    track_box=None,
                ),
                IdCollisionResolutionPolicy.Error,
            )
        batch.add(id, f)

    full_expr = Q.eval(
        """ 
        (contains(("created_by_4", "created_by_2"), namespace) || ends_with(label, "2") || ends_with(label, "4") || ends_with(label, "6")) &&
        !is_empty(bbox.angle) &&
        (confidence > 0.5 || confidence < 0.3)
        """
    )

    t = timer()
    res = batch.access_objects(full_expr)
    print("Batch Full Query\t", (timer() - t) * 1000_000)


measure_batch_full()


def measure_batch_full_dsl():
    batch = VideoFrameBatch()
    for id in range(0, T):
        f = gen_frame()
        f.source_id = f"source_{id}"
        f.delete_objects(Q.idle())
        for i in range(0, N):
            f.add_object(
                VideoObject(
                    id=i,
                    namespace="created_by_{}".format(i),
                    label="person_{}".format(i),
                    detection_box=RBBox(0.1, 0.2, 0.3, 0.4, 30.0),
                    confidence=random.random(),
                    attributes=[],
                    track_id=None,
                    track_box=None,
                ),
                IdCollisionResolutionPolicy.Error,
            )
        batch.add(id, f)

    optimized_expr = Q.and_(
        Q.or_(
            Q.namespace(SE.eq("created_by_4")),
            Q.namespace(SE.eq("created_by_2")),
            Q.namespace(SE.ends_with("2")),
            Q.namespace(SE.ends_with("4")),
            Q.namespace(SE.ends_with("6")),
        ),
        Q.box_angle_defined(),
        Q.or_(Q.confidence(FE.gt(0.5)), Q.confidence(FE.lt(0.3))),
    )

    t = timer()
    res = batch.access_objects(optimized_expr)
    print("Batch Full Optimized Query\t", (timer() - t) * 1000_000)


measure_batch_full_dsl()
