# vulture whitelist — intentionally-defined names that are not called locally
# but are part of the public API, MCP tool surface, or hook interface.
#
# Every entry is a claim that vulture's report about that name is wrong, and every
# claim needs a reason someone can re-check. Put new entries under a section that
# says why the name is reachable — which framework calls it, which __all__ exports
# it — not just that it is.
#
# scripts/check_vulture_whitelist.py runs in CI and fails on an entry that no
# longer suppresses anything. An inert entry is either debris to delete or a name
# that has since become genuinely dead and is now hidden by its own exemption;
# both need a human, so neither is allowed to sit here silently.
#
# Keep entries alphabetical within each section.

# ── MCP tool functions (registered via @mcp.tool() decorator, called by the framework)
_ = causal_edges_list
_ = investigation_search
_ = investigation_evidence_precheck
_ = investigation_entity_lookup
_ = investigation_related_cases
_ = entity_list
_ = entity_timeline
_ = investigation_pre_answer_check
_ = investigation_reason
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
_ = contract_declare
_ = contract_query
_ = contract_check
_ = wiring_obligation_declare
_ = wiring_obligation_list
_ = wiring_obligation_resolve
_ = rag_context_search
# reflection_loop_seed / reflection_loop_tick / investigation_verify_all were
# here as tool surface with no local caller. scripts/loci_groom.py now calls
# all three (#194), so vulture sees real references and the entries would
# suppress nothing.
_ = reflection_loop_status
_ = conflict_list
_ = conflict_resolve

# ── Public API exported by memcheck modules

# ── MLOps public entry points (called by loop.py or CLI)

# ── A2A server skill handlers (dispatched dynamically by method name)

# ── glymphatic sweep steps (called by main() via skip set)

# ── event_log public interface

# ── spreading activation public interface

# ── Signal handler parameters (received by OS convention, not read in body)
_ = signum   # signal number param in _sigterm(signum, frame)
_ = frame    # stack frame param in _sigterm(signum, frame)

# ── Server socket variables assigned for binding but not read after bind
_ = client_address  # assigned in TCPServer.__init__ or similar; used by framework

# ── a2a_server HTTP route handlers (registered via @app.get / @app.post)
# The router holds the only reference; no source line names them as a callee.
_ = agent_card_rfc002
_ = agent_card_legacy_alias
_ = bootstrap
_ = a2a_endpoint
_ = get_task

# ── MCP surface added after this list was last revised
# health is @mcp.custom_route("/health"); the other three are @mcp.tool().
# A tool missing from this list is a finding, not a gap: `mcp.list_tools()`
# is the authority, and mcp/tests asserts its count.
_ = health
_ = finding_resolve
_ = loci_health
_ = retrieval_selftest

# ── socketserver hooks and tuning attributes, read by the stdlib base class
# MemcheckDaemon subclasses ThreadingUnixStreamServer; _Handler subclasses
# BaseRequestHandler. Both names are called through the base, never directly.
_ = handle
_ = handle_error
_ = daemon_threads
_ = request_queue_size

# ── memcheck.VerdictEngine methods reachable from outside this tree
# VerdictEngine is in memcheck/engine.py's __all__ and re-exported from
# memcheck/__init__.py, so "no in-repo caller" does not mean "no caller".
_ = in_memory
_ = recall_decision

# ── sqlite3 connection API
# `conn.row_factory = sqlite3.Row` is a write the stdlib reads back; there is
# no root model that makes this resolvable, so it is suppressed by name.
_ = row_factory

# ── FastMCP settings written by main()
# `mcp.settings.port` is read inside FastMCP.run() when it builds the uvicorn
# config; there is no in-repo read, so vulture calls the write dead.
_ = port
