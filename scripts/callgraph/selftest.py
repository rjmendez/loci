"""Standalone health check for this checkout: builds the graph over
fixtures/ (no git, no network) and once over the real corpus at HEAD (git
read, still no network/venv/LadybugDB dependency), then asserts the
SPECIFIC findings this tool exists to produce -- not merely "did not
raise". Runnable directly (`cg selftest`) and imported by
tests/test_selftest.py so pytest exercises the identical checks.

Deliberately does NOT re-run the three historical-revision bug regressions
(BUG B/C/D at their specific pre-fix commits) -- those already have
dedicated, thorough coverage in tests/test_cli_step10_12.py and
tests/test_pipeline_real_corpus.py, and re-doing three more full-corpus
git-blob builds here would blow the 5s budget for no new signal. This file
checks the same dispatch shapes those bugs came from, over the fast
fixtures/ build, plus the one real-corpus invariant that would catch a
regression in any of them structurally: the hard gate (no registered
function ever reported dead) and the registration-surface counts.

ACCEPTANCE (build_steps step 13): green from a clean checkout, no venv, no
network, no LadybugDB; the full run (fixture build + the one HEAD corpus
build + every assertion) finishes in well under 5s.

stdlib only.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .analyze.deadcode import registered_but_dead
from .analyze.flags import rank_flags
from .analyze.literalaudit import near_miss_pairs, orphans
from .analyze.nameaudit import dangling_globals
from .model import Confidence
from .pipeline import build_graph
from .tests.helpers import build_fixture_store


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class SelftestReport:
    checks: list = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def failed(self):
        return [c for c in self.checks if not c.ok]


def _run(checks: list, name: str, fn) -> None:
    """Runs one check function; a raised exception (including a failed
    `assert`) is recorded as a failing Check instead of aborting the rest
    of the run -- one broken dispatch shape should never hide the other
    thirteen results."""
    try:
        fn()
        checks.append(Check(name, True))
    except AssertionError as exc:
        checks.append(Check(name, False, f"assertion failed: {exc}"))
    except Exception as exc:  # tool error inside a check itself
        checks.append(Check(name, False, f"{exc.__class__.__name__}: {exc}"))


# -- fixture-based checks, one per dispatch shape in the design ------------


def _check_dec():
    store, _, _ = build_fixture_store(["decorator_registry.py"])
    reg_id = "reg:decorator_registry.py::mcp.tool"
    reg = store.get(reg_id)
    assert reg is not None and reg.attrs["mechanism"] == "decorator"
    members = {e.dst for e in store.out_edges(reg_id, "REGISTERS")}
    assert members == {"fn:decorator_registry.py::registered_tool"}
    entry = store.get("entry:mcp-tool:registered_tool")
    assert entry is not None
    enters = store.out_edges(entry.id, "ENTERS")
    assert len(enters) == 1 and enters[0].dst == "fn:decorator_registry.py::registered_tool"
    # the decorator no rule recognizes must NOT spawn a registry
    assert store.get("reg:decorator_registry.py::some_unrecognized_decorator") is None
    mystery_dbs = store.out_edges("fn:decorator_registry.py::mystery", "DECORATED_BY")
    assert mystery_dbs and mystery_dbs[0].attrs["classification"] == "unknown"


def _check_manloop():
    store, _, _ = build_fixture_store(["registry_manloop.py"])
    reg_id = "reg:registry_manloop.py::register"
    reg = store.get(reg_id)
    assert reg is not None and reg.attrs["mechanism"] == "manifest-tuple"
    members = {e.dst for e in store.out_edges(reg_id, "REGISTERS")}
    assert members == {"fn:registry_manloop.py::tool_a", "fn:registry_manloop.py::tool_b"}
    # the unrelated "call a method ON each tuple element" loop is NOT MAN-LOOP
    assert store.get("reg:registry_manloop.py::_reset_cache") is None


def _check_mandict():
    store, _, _ = build_fixture_store(["registry_mandict.py"])
    reg_id = "reg:registry_mandict.py::_SKILL_MAP"
    reg = store.get(reg_id)
    assert reg is not None and reg.attrs["mechanism"] == "manifest-dict"
    keys = {e.attrs["key"] for e in store.out_edges(reg_id, "REGISTERS")}
    assert keys == {"one", "two"}
    # a config dict of literal values must never be mistaken for a dispatch table
    assert store.get("reg:registry_mandict.py::_CONFIG_MAP") is None
    # the `.get(...)` dispatch callsite fans out to both registered skills
    sites = [
        n for n in store.nodes_of_kind("CALLSITE")
        if n.attrs.get("enclosing_fn") == "fn:registry_mandict.py::dispatch" and n.attrs.get("callee") == "handler"
    ]
    assert len(sites) == 1
    call_edge = store.out_edges(sites[0].id, "CALLS")[0]
    assert call_edge.dst == "?" and call_edge.attrs["reason"] == "dict-dispatch-fanout"
    disp_targets = {e.dst for e in store.out_edges(sites[0].id, "DISPATCHES")}
    assert disp_targets == {"fn:registry_mandict.py::skill_one", "fn:registry_mandict.py::skill_two"}


def _check_reg_fn_injects():
    # Load BOTH callers, mirroring register() being called twice in the
    # real corpus (tests do this) -- the re-entrant, widened-to-fanout shape.
    store, _, _ = build_fixture_store([
        "registry_injection.py", "registry_injection_caller.py", "registry_injection_caller2.py",
    ])
    nid = "name:registry_injection.py::_get_thing"
    injects = store.in_edges(nid, "INJECTS")
    assert len(injects) == 2
    values = {e.attrs["value"] for e in injects}
    assert values == {
        "fn:registry_injection_caller.py::real_thing",
        "fn:registry_injection_caller2.py::other_thing",
    }
    use_thing_id = "fn:registry_injection.py::use_thing"
    sites = [n for n in store.nodes_of_kind("CALLSITE") if n.attrs.get("enclosing_fn") == use_thing_id]
    assert len(sites) == 1
    edge = store.out_edges(sites[0].id, "CALLS")[0]
    assert edge.dst == "?" and edge.attrs["reason"] == "injected-global-multi-value"
    assert edge.attrs["alternatives"] == 2
    disp_targets = {e.dst for e in store.out_edges(sites[0].id, "DISPATCHES")}
    assert disp_targets == values
    assert all(e.confidence == Confidence.PROBABLE for e in store.out_edges(sites[0].id, "DISPATCHES"))
    # the deps["helper_a"] key-injection side is ALSO called from both
    # callers (caller2.py's dict passes a different bare name for it) --
    # two INJECTS edges too, same shape as the direct-param side above.
    helper_injects = store.in_edges("name:registry_injection.py::_helper_a", "INJECTS")
    assert len(helper_injects) == 2
    assert all(e.attrs["param"] == "deps" and e.attrs["key"] == "helper_a" for e in helper_injects)


def _check_rootcli():
    store, _, _ = build_fixture_store(["registry_rootcli.py"])
    entry = store.get("entry:cli:registry_rootcli.py")
    assert entry is not None and entry.attrs["trust"] == "declared-in-source"
    enters = store.out_edges(entry.id, "ENTERS")
    assert len(enters) == 1 and enters[0].dst == "fn:registry_rootcli.py::main"


def _check_dangling_global():
    # THIS IS BUG C, in miniature: a `global`-declared write with no
    # module-level binding, correctly NOT flagged for an ident that IS
    # bound at module level in the SAME module, with the real slot named
    # in a DIFFERENT module.
    store, _, _ = build_fixture_store(["dangling_global.py", "dangling_global_real_slot.py"])
    findings = {f.slot.id: f for f in dangling_globals(store)}
    assert "name:dangling_global.py::_symbol_index_cache" in findings
    assert "name:dangling_global.py::_symbol_index_count" in findings
    assert "name:dangling_global.py::_real_counter" not in findings
    real = findings["name:dangling_global.py::_symbol_index_cache"].real_slots
    assert len(real) == 1 and real[0].path == "dangling_global_real_slot.py" and real[0].line == 7


def _check_literal_orphans():
    # THIS IS BUG D, in miniature: a produced path with zero consumers and
    # a consumed path with zero producers, paired by --near-miss on their
    # shared stem, while a genuinely matched pair (notes.jsonl) is excluded
    # from the orphan lists entirely.
    store, _, _ = build_fixture_store(["literals_produce.py", "literals_consume.py", "literals_keys.py"])
    producer_only, consumer_only = orphans(store, flavour="path")
    p_texts = {g.match_text for g in producer_only}
    c_texts = {g.match_text for g in consumer_only}
    assert "graph.ladybug" in p_texts and "notes.jsonl" not in p_texts
    assert "graph.kuzu" in c_texts and "notes.jsonl" not in c_texts
    pairs = near_miss_pairs(producer_only, consumer_only)
    matches = [p for p in pairs if p.shared_stem == "graph"]
    assert len(matches) == 1
    assert {matches[0].producer.match_text, matches[0].consumer.match_text} == {"graph.ladybug", "graph.kuzu"}
    kp, kc = orphans(store, flavour="key")
    assert "cfg_write_only" in {g.match_text for g in kp}
    assert "cfg_lookup_only" in {g.match_text for g in kc}


def _check_flags():
    # THIS IS BUG B, in miniature: a constant-initialized local reassigned
    # only inside guarded branches ranks ABOVE one that is never reassigned
    # at all, by the guard-exits-that-skip-it / assignments-made ratio.
    store, _, _ = build_fixture_store(["flags_shapes.py"])
    rows = rank_flags(store)
    by_ident = {r.ident: r for r in rows}
    assert "ok" in by_ident and "step" in by_ident
    assert "parts" not in by_ident and "result" not in by_ident  # non-constant inits excluded
    assert by_ident["ok"].score > by_ident["step"].score
    assert rows[0].ident == "ok"
    assert by_ident["ok"].escape_form == "dict-value"


def _check_calls_ladder():
    store, _, _ = build_fixture_store(["calls_shapes.py", "calls_target.py"])

    def _edge_for(fn_qualname):
        fid = f"fn:calls_shapes.py::{fn_qualname}"
        sites = [n for n in store.nodes_of_kind("CALLSITE") if n.attrs.get("enclosing_fn") == fid]
        assert len(sites) == 1, fn_qualname
        return store.out_edges(sites[0].id, "CALLS")[0]

    e = _edge_for("caller_name_def_local")
    assert e.dst == "fn:calls_shapes.py::helper" and e.confidence == Confidence.PROVEN
    e = _edge_for("caller_module_attribute_corpus")
    assert e.dst == "fn:calls_target.py::target_fn" and e.confidence == Confidence.PROVEN
    e = _edge_for("caller_module_attribute_external")
    assert e.dst.startswith("ext:") and e.confidence == Confidence.PROVEN
    e = _edge_for("caller_local_import")
    assert e.dst.startswith("ext:") and e.confidence == Confidence.PROVEN
    e = _edge_for("caller_param_call")
    assert e.dst == "?"
    e = _edge_for("caller_unknown_attribute")
    assert e.dst == "?"


def _check_dispatch():
    store, _, _ = build_fixture_store(["dispatch_shapes.py", "dispatch_target.py"])

    def _edge_for(fn_qualname, callee=None):
        fid = f"fn:dispatch_shapes.py::{fn_qualname}"
        sites = [n for n in store.nodes_of_kind("CALLSITE") if n.attrs.get("enclosing_fn") == fid]
        if callee is not None:
            sites = [n for n in sites if n.attrs.get("callee") == callee]
        assert len(sites) == 1, (fn_qualname, callee)
        return store.out_edges(sites[0].id, "CALLS")[0]

    e = _edge_for("call_unique_method")
    assert e.dst == "fn:dispatch_shapes.py::OnlyOwner.unique_op" and e.confidence == Confidence.PROBABLE
    e = _edge_for("call_ambiguous_method")
    assert e.dst == "?" and e.attrs["reason"] == "ambiguous-method-name" and e.attrs["alternatives"] == 2
    fid = "fn:dispatch_shapes.py::call_getattr_literal"
    site = next(n for n in store.nodes_of_kind("CALLSITE") if n.attrs.get("enclosing_fn") == fid)
    e = store.out_edges(site.id, "CALLS")[0]
    assert e.dst == "fn:dispatch_target.py::target_fn" and e.confidence == Confidence.PROVEN
    fid = "fn:dispatch_shapes.py::call_getattr_variable"
    site = next(n for n in store.nodes_of_kind("CALLSITE") if n.attrs.get("enclosing_fn") == fid)
    e = store.out_edges(site.id, "CALLS")[0]
    assert e.dst == "?" and e.attrs["reason"] == "computed-getattr"


def _check_cross_module_names():
    store, _, _ = build_fixture_store(["names_cross_module.py", "names_cross_module_target.py"])
    nid = "name:names_cross_module_target.py::CONFIG_VALUE"
    reads = store.in_edges(nid, "READS_NAME")
    assert len(reads) == 1
    assert reads[0].src == "fn:names_cross_module.py::read_it"
    assert reads[0].attrs.get("cross_module") is True


def _check_nesting():
    store, _, _ = build_fixture_store(["nesting.py"])
    assert store.get("fn:nesting.py::outer.<locals>.inner") is not None
    lambdas = [n for n in store.nodes_of_kind("FUNCTION") if n.attrs.get("is_lambda") and n.path == "nesting.py"]
    assert len(lambdas) == 1
    # a lambda passed straight into a registering call is a real,
    # addressable FUNCTION node -- not silently dropped.
    assert lambdas[0].id.startswith("fn:nesting.py::")


def _check_reexport():
    store, _, _ = build_fixture_store(["reexport.py", "reexport_source.py"])
    target = "fn:reexport_source.py::symbol_impact"
    aliases_import = store.out_edges("name:reexport.py::symbol_impact", "ALIASES")
    assert len(aliases_import) == 1 and aliases_import[0].dst == target
    # `runner = symbol_impact` is a bare module-level alias to the IMPORTED
    # name slot, not straight to the function -- a one-more-hop ALIASES
    # chain (NAME -> NAME -> FUNCTION), exactly the "one-or-more ALIASES
    # hops" rung 2 of the CALLS resolution ladder is built to walk.
    aliases_bare = store.out_edges("name:reexport.py::runner", "ALIASES")
    assert len(aliases_bare) == 1 and aliases_bare[0].dst == "name:reexport.py::symbol_impact"
    chained = store.out_edges(aliases_bare[0].dst, "ALIASES")
    assert len(chained) == 1 and chained[0].dst == target
    definers = store.in_edges(target, "DEFINES")
    assert len(definers) == 1  # the re-exports don't create a second definition


def _check_lazy_import():
    store, _, _ = build_fixture_store(["lazy_import.py"])
    edges = list(store.edges_of_kind("IMPORTS"))
    lazy = [e for e in edges if e.attrs.get("scope") == "function-local" and e.src == "mod:lazy_import.py"]
    assert any(e.attrs.get("module") == "numpy" and e.attrs.get("enclosing_fn") == "resolve_vllm" for e in lazy)
    assert any(e.attrs.get("module") == "textwrap" and e.attrs.get("enclosing_fn") == "Widget.render" for e in lazy)
    module_level = [e for e in edges if e.attrs.get("scope") != "function-local" and e.attrs.get("module") == "json"]
    assert module_level


# -- one real-corpus check: the hard gate + registration-surface counts ---


def _check_real_corpus():
    result = build_graph(rev="HEAD")
    store = result.store
    assert result.meta.file_count == 116, result.meta.file_count
    assert result.meta.error_count == 0, result.meta.errors
    bad = registered_but_dead(store)
    assert bad == [], [n.id for n in bad]
    from collections import Counter
    by_rule = Counter(e.attrs["rule"] for e in store.edges_of_kind("REGISTERS"))
    assert by_rule["DEC-tool"] == 42, dict(by_rule)
    assert by_rule["DEC-route"] == 6, dict(by_rule)
    assert by_rule["DEC-mcp-route"] == 1, dict(by_rule)
    assert by_rule["MAN-LOOP"] == 31, dict(by_rule)
    assert by_rule["MAN-DICT"] == 13, dict(by_rule)
    unmatched = [e for e in store.edges_of_kind("DECORATED_BY") if e.attrs["classification"] == "unknown"]
    assert unmatched == [], [(e.src, e.attrs["raw"]) for e in unmatched]


_ALL_CHECKS = [
    ("DEC: @mcp.tool()-style decorator registers fn + entrypoint; unknown decorator classified unknown", _check_dec),
    ("MAN-LOOP: manifest tuple registers both members; unrelated bare-name loop is not MAN-LOOP", _check_manloop),
    ("MAN-DICT: _SKILL_MAP registers + dispatch fans out; literal-valued config dict is not a registry", _check_mandict),
    ("REG-FN/INJECTS: register()-param + deps[key] injection; re-entrant registration widens to DISPATCHES", _check_reg_fn_injects),
    ("ROOT-CLI: __main__ guard ENTERS the called function", _check_rootcli),
    ("BUG C shape: DANGLING-GLOBAL flags global-only slot and names the real slot in another module", _check_dangling_global),
    ("BUG D shape: path-literal orphans found + paired by --near-miss on shared stem", _check_literal_orphans),
    ("BUG B shape: guarded-reassignment flag ranks above a never-reassigned one", _check_flags),
    ("CALLS ladder rungs 1-3: proven in-corpus/external resolution; param-call and unknown-attr land on ?", _check_calls_ladder),
    ("Dispatch rungs 5-6 + getattr-literal: unique-method probable, ambiguous unresolved, getattr(lit) proven", _check_dispatch),
    ("Cross-module READS_NAME via an imported module's attribute", _check_cross_module_names),
    ("Qualname nesting: closure-nested def and a lambda passed to a registering call", _check_nesting),
    ("ALIASES: re-export and bare module-level alias both resolve to the one definition", _check_reexport),
    ("Lazy imports: function-local scope + enclosing_fn recorded, distinct from module-level", _check_lazy_import),
    ("Real corpus @ HEAD: hard gate (zero registered-but-dead) + registration-surface counts", _check_real_corpus),
]


def run_selftest() -> SelftestReport:
    t0 = time.time()
    checks: list = []
    for name, fn in _ALL_CHECKS:
        _run(checks, name, fn)
    return SelftestReport(checks=checks, elapsed_s=time.time() - t0)
