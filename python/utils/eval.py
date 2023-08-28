from savant_rs.utils import eval_expr
from savant_rs.video_object_query import register_env_resolver, register_utility_resolver

register_env_resolver()
register_utility_resolver()

print(eval_expr("1 + 1"))
print(eval_expr("""p = env("PATH", ""); (is_string(p), p)"""))

