from savant_rs.match_query import register_env_resolver, register_utility_resolver
from savant_rs.utils import eval_expr

register_env_resolver()
register_utility_resolver()

print(eval_expr("1 + 1"))

print(eval_expr("""p = env("PATH", ""); (is_string(p), p)"""))  # uncached
print(eval_expr("""p = env("PATH", ""); (is_string(p), p)"""))  # cached
