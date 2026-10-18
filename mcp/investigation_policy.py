"""Backward-compat shim. Import from loci.investigation_policy directly."""
from loci.investigation_policy import *  # noqa: F401, F403
from loci.investigation_policy import (  # noqa: F401
    cap_perspectives_to_tier, evaluate_pre_answer_for_refusal,
    filter_findings_for_verification, push_refusal_step_to_trajectory,
    get_investigation_policy_config,
)
