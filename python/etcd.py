from pathlib import Path

from savant_rs.match_query import (EtcdCredentials, TlsConfig,
                                   register_env_resolver,
                                   register_etcd_resolver)
from savant_rs.utils import eval_expr

register_env_resolver()

# if not set env var RUN_ETCD_TESTS=1, skip
if not eval_expr('env("RUN_ETCD_TESTS", 0)') == 0:
    print("Skipping etcd tests")
    exit(0)

# read ca from file to string
ca = Path("../../etcd_dynamic_state/assets/certs/ca.crt").read_text()
cert = Path("../../etcd_dynamic_state/assets/certs/client.crt").read_text()
key = Path("../../etcd_dynamic_state/assets/certs/client.key").read_text()

conf = TlsConfig(
    ca,
    cert,
    key,
)

creds = EtcdCredentials("root", "qwerty")

register_etcd_resolver(
    hosts=["https://127.0.0.1:2379"], credentials=creds, tls_config=conf, watch_path=""
)

print(eval_expr('etcd("foo/bar", "default")'))
