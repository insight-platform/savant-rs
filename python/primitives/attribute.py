from savant_rs.primitives import Attribute, AttributeValue

attr = Attribute(
    namespace="some",
    name="attr",
    hint="x",
    values=[
        AttributeValue.bytes(dims=[8, 3, 8, 8], blob=bytes(3 * 8 * 8), confidence=None),
        AttributeValue.bytes_from_list(dims=[4, 1], blob=[0, 1, 2, 3], confidence=None),
        AttributeValue.integer(1, confidence=0.5),
        AttributeValue.float(1.0, confidence=0.5),
    ],
)
print(attr.json)

vals = attr.values

view = attr.values_view
print(len(view))
print(view[2])

attr2 = Attribute.from_json(attr.json)
print(attr2.json)

x = dict(x=5)
temp_py_attr = Attribute(
    namespace="some",
    name="attr",
    hint="x",
    values=[AttributeValue.temporary_python_object(x)],
)

x["y"] = 6

o = temp_py_attr.values[0].as_temporary_python_object()
print(o)
