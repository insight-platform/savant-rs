from savant_rs.utils import gen_frame

frame = gen_frame()

x = []
frame.set_pyobject("ns","x", x)

new_x = frame.get_pyobject("ns", "x")
assert new_x == x

x.append(1)
assert new_x == x

new_x2 = frame.get_pyobject("ns", "x")
assert new_x2 == x
assert new_x2 == new_x

attaches = frame.list_pyobjects()
assert attaches == [('ns', 'x')]

new_x = frame.delete_pyobject("ns", "x")
assert new_x == x

frame.clear_pyobjects()
