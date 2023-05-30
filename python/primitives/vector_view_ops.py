from savant_rs.utils import gen_frame, register_plugin_function, is_plugin_function_registered, UserFunctionKind
from savant_rs.video_object_query import Query as Q, IntExpression as IE

register_plugin_function("../../target/release/libsample_plugin.so", "map_modifier", UserFunctionKind.ObjectMapModifier,
                         "sample.map_modifier")

assert is_plugin_function_registered("sample.map_modifier")

register_plugin_function("../../target/release/libsample_plugin.so", "inplace_modifier",
                         UserFunctionKind.ObjectInplaceModifier,
                         "sample.inplace_modifier")

assert is_plugin_function_registered("sample.inplace_modifier")

f = gen_frame()

objects = f.access_objects(Q.idle()).filter(Q.id(IE.one_of(1, 2)))

new_objects = objects.map_udf("sample.map_modifier")
assert new_objects[0].label == "modified_test"
assert objects[0].label == "test"

objects.foreach_udf("sample.inplace_modifier")
assert objects[0].label == "modified_test"
