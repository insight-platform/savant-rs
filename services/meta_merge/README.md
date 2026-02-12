# Meta Merge

Meta Merge is a Python-extendable service that merges metadata from multiple ingress streams into a single egress stream. It receives video frames with the same UUID from multiple ingress sockets, merges their metadata using user-defined Python handlers, and forwards the resulting frames downstream. This service is useful for scenarios where parallel processing pipelines produce partial metadata that needs to be consolidated before further processing.

**License**: Apache 2
