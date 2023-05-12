from pprint import pprint

from savant_rs.primitives import PolygonalArea, Point, Segment
from timeit import default_timer as timer

area = PolygonalArea([Point(-1, 1), Point(1, 1), Point(1, -1), Point(-1, -1)], ["up", None, "down", None])

crosses_13 = Segment(Point(-2, 0), Point(2, 0))
crosses_02 = Segment(Point(0, -2), Point(0, 2))
leaves_vertex = Segment(Point(0, 0), Point(2, 2))
crosses_vertices = Segment(Point(-2, -2), Point(2, 2))
crosses_whole_edge = Segment(Point(-2, 1), Point(2, 1))
enters_vertex = Segment(Point(2, 2), Point(0, 0))
outside = Segment(Point(-2, 2), Point(2, 2))
inside = Segment(Point(-0.5, -0.5), Point(0.5, 0.5))

l = [crosses_13, crosses_02, leaves_vertex, crosses_vertices, crosses_whole_edge, enters_vertex, outside, inside]

t = timer()

res = None
for _ in range(10_000_000):
    res = area.crossed_by_segments(l)

print("Spent", timer() - t)
pprint(list(zip(l, res)))
