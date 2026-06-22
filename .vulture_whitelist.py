# vulture whitelist — intentionally-defined names that are not called locally
# but are part of the public API, MCP tool surface, or hook interface.
#
# Keep entries alphabetical within each section.

# ── MCP tool functions (registered via @mcp.tool() decorator, called by the framework)
_ = causal_edges_list
_ = investigation_start
_ = investigation_store
_ = investigation_note
_ = investigation_load
_ = investigation_list
_ = investigation_share
_ = investigation_unshare
_ = investigation_search
_ = investigation_evidence_precheck
_ = investigation_entity_lookup
_ = investigation_related_cases
_ = investigation_finding_provenance
_ = investigation_pre_answer_check
_ = investigation_reflect
_ = investigation_reason
_ = investigation_export
_ = investigation_import
_ = audit_log
_ = code_memory_correlate
_ = memory_health
_ = memory_self_check
_ = memory_consolidate
_ = memory_demote
_ = memory_promote
_ = memory_retract
_ = memory_restore
_ = memory_confidence
_ = memory_hints
_ = memory_hints_resource
_ = memory_route
_ = memory_surface
_ = procedure_attempt
_ = procedure_search
_ = rag_context_search
_ = reflection_loop_seed
_ = reflection_loop_tick
_ = reflection_loop_status
_ = conflict_list
_ = conflict_resolve

# ── Public API exported by memcheck modules
_ = run_contradiction
_ = run_contradiction_llm
_ = verify_and_merge
_ = extract_json

# ── MLOps public entry points (called by loop.py or CLI)
_ = apply_decay
_ = adapt
_ = weibull_retention
_ = boundary_samples
_ = hard_negatives
_ = build_anchor
_ = measure_drift

# ── A2A server skill handlers (dispatched dynamically by method name)
_ = skill_memory_recall
_ = skill_memory_remember
_ = skill_memory_stats
_ = skill_session_search
_ = skill_memory_sleep
_ = skill_gpu_inference
_ = skill_memory_prime

# ── glymphatic sweep steps (called by main() via skip set)
_ = sweep_verdicts
_ = sweep_orphans
_ = sweep_edges
_ = sweep_duplicates
_ = check_content_shift

# ── event_log public interface
_ = append
_ = replay
_ = compact

# ── spreading activation public interface
_ = run_spreading_activation

# ── Signal handler parameters (received by OS convention, not read in body)
_ = signum   # signal number param in _sigterm(signum, frame)
_ = frame    # stack frame param in _sigterm(signum, frame)

# ── Server socket variables assigned for binding but not read after bind
_ = client_address  # assigned in TCPServer.__init__ or similar; used by framework
