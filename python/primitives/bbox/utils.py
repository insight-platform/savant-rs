from savant_rs.primitives.geometry import BBox, RBBox, solely_owned_areas

red = RBBox.ltrb(0.0, 2.0, 2.0, 4.0)
green = RBBox.ltrb(1.0, 3.0, 5.0, 5.0)
yellow = RBBox.ltrb(1.0, 1.0, 3.0, 6.0)
purple = RBBox.ltrb(4.0, 0.0, 7.0, 2.0)

areas = solely_owned_areas([red, green, yellow, purple], parallel=True)

red = areas[0]
green = areas[1]
yellow = areas[2]
purple = areas[3]

assert red == 2.0
assert green == 4.0
assert yellow == 5.0
assert purple == 6.0
