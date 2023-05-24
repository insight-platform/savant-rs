from pprint import pprint

from savant_rs.primitives import PolygonalArea, Point, Segment, IntersectionKind
from timeit import default_timer as timer

area = PolygonalArea([Point(-1, 1), Point(1, 1), Point(1, -1), Point(-1, -1)], ["up", None, "down", None])
assert area.is_self_intersecting() == False

bad_area = PolygonalArea([Point(-1, -1), Point(1, 1), Point(1, -1), Point(-1, 1)], ["up", None, "down", None])
assert bad_area.is_self_intersecting() == True

bad_area2 = PolygonalArea([Point(-1, -1), Point(1, 1), Point(1, 1), Point(-1, 1)], ["up", None, "down", None])
assert bad_area2.is_self_intersecting() == True

good_area2 = PolygonalArea([Point(-1, -1), Point(1, 1), Point(-1, 1)], ["up", None, "down"])
assert good_area2.is_self_intersecting() == False

crosses_031 = Segment(Point(-2, 1), Point(2, 1))
crosses_013 = Segment(Point(2, 1), Point(-2, 1))
crosses_31 = Segment(Point(-2, 0), Point(2, 0))
crosses_20 = Segment(Point(0, -2), Point(0, 2))
leaves_vertex = Segment(Point(0, 0), Point(2, 2))
crosses_vertices = Segment(Point(-2, -2), Point(2, 2))
crosses_whole_edge = Segment(Point(-2, 1), Point(2, 1))
enters_vertex = Segment(Point(2, 2), Point(0, 0))
outside = Segment(Point(-2, 2), Point(2, 2))
inside = Segment(Point(-0.5, -0.5), Point(0.5, 0.5))

l = [crosses_31, crosses_20, crosses_031, crosses_013, leaves_vertex, crosses_vertices, crosses_whole_edge, enters_vertex, outside, inside]

t = timer()

res = None
for _ in range(10_000):
    res = area.crossed_by_segments(l)

print("Spent", timer() - t)
pprint(list(zip(l, res)))

r = res[0]
assert (r.kind == IntersectionKind.Cross)
assert (r.edges == [(3, None), (1, None)])

r = res[1]
assert (r.kind == IntersectionKind.Cross)
assert (r.edges == [(2, "down"), (0, "up")])

r = res[2]
assert (r.kind == IntersectionKind.Cross)
assert (r.edges == [(0, "up"), (3, None), (1, None)])

r = res[3]
assert (r.kind == IntersectionKind.Cross)
assert (r.edges == [(0, "up"), (1, None), (3, None)])