"""Backward-compat shim. Import from loci.cache directly."""
from loci.cache import *  # noqa: F401, F403
from loci.cache import check_memory_cache, format_cache_response, CacheHit  # noqa: F401
