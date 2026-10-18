"""Backward-compat shim. Import from loci.litellm_client directly."""
from loci.litellm_client import *  # noqa: F401, F403
from loci.litellm_client import call_litellm_proxy, LiteLLMResponse  # noqa: F401
