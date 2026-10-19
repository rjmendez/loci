"""scripts/hermes_config_check.py: read-only contract check of a Hermes config.yaml (Tripwire-Governor S8).

Every config here is a FIXTURE written to tmp_path (a compliant one and one violation per rule). The live
Hermes config is never read and no secret appears in this file; fixture keys are obvious placeholders.
The parser tests double as the guard that the stdlib YAML subset reads the constructs Hermes configs use.

Mutation log (each applied to scripts/hermes_config_check.py; the named tests went red; then reverted):
- INT_RULES max_iterations 40 -> 250: test_compliant_fixture_has_no_violations, test_cli_compliant_config_exits_zero_and_says_so
- INT_RULES max_concurrent_children 2 -> 3 / max_spawn_depth 1 -> 2: the same two tests
- the integer check drops the type test (``val != want`` only, so true == 1 passes): test_boolean_does_not_satisfy_an_integer_rule
- the boolean rule accepts any truthy value (``not val`` instead of ``val is not want``): test_string_true_does_not_satisfy_a_boolean_rule
- the missing-key branch removed (missing reads as compliant): test_missing_key_is_a_violation
- api.anthropic.com removed from FORBIDDEN_HOSTS: test_direct_endpoint_is_a_violation[anthropic]
- gateway.ai.cloudflare.com removed from FORBIDDEN_HOSTS: test_direct_endpoint_is_a_violation[cloudflare]
- the copilot substring test removed: test_direct_endpoint_is_a_violation[copilot]
- subdomain match (endswith) removed: test_subdomain_of_a_forbidden_host_is_a_violation
- the LiteLLM host exemption removed: test_litellm_host_override_is_honoured, test_cli_litellm_host_flag_is_passed_through
- seat_billed also exempts anthropic: test_seat_billed_does_not_exempt_anthropic
- seat_billed ignored for copilot: test_seat_billed_copilot_entry_is_allowed
- the provider-key check removed: test_direct_provider_key_is_a_violation
- the walk stops at lists (custom_providers entries unchecked): test_direct_endpoint_is_a_violation[cloudflare]
- the exit code of a violation changed from 1 to 0: test_cli_reports_every_violation_and_exits_one
- an unreadable config reports compliant: test_cli_unreadable_config_fails_closed
- duplicate keys silently keep the last value: test_duplicate_key_is_rejected, test_cli_unparsable_config_fails_closed
- the comment stripper keeps comments (URLs in comments become findings): test_compliant_fixture_has_no_violations
- the comment stripper cuts at '#' inside quotes: test_hash_inside_quotes_is_not_a_comment
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "hermes_config_check.py"


def _load():
    spec = importlib.util.spec_from_file_location("hermes_config_check_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


hc = _load()

BYPASS = "bypasses the LiteLLM gateway; route it through the LiteLLM base_url"
SEAT = " or mark the entry seat_billed: true"

# Same shape as a real Hermes config: top-level sections, commented-out examples (with forbidden URLs
# in the comments, which must NOT count), a list of provider dicts, nested guardrail maps.
COMPLIANT = """\
_config_version: 49
model:
  default: "auto-router"
  # Direct Anthropic API: base_url: "https://api.anthropic.com/v1"   (a comment, not a config value)
  provider: "custom"
  api_key: "sk-fixture-placeholder"
  base_url: "https://litellm.lab.floppydicks.net:4443/v1"
  context_length: 65536
terminal:
  backend: local
  timeout: 180
tool_loop_guardrails:
  warnings_enabled: true
  hard_stop_enabled: true
  non_interactive_hard_stop_enabled: true
  hard_stop_after:
    exact_failure: 5
delegation:
  max_iterations: 40 # Max tool-calling turns per child
  max_concurrent_children: 2
  max_spawn_depth: 1
  worktree_isolation: true
  # provider: "openrouter"
mcp_servers:
  loci:
    url: https://loci.lab.floppydicks.net:4443/mcp
custom_providers:
  - name: LiteLLM auto-router
    base_url: https://litellm.lab.floppydicks.net:4443/v1
    key_env: LITELLM_API_KEY
    model: clef-omni
"""


def swap(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"fixture drifted: {old!r} appears {text.count(old)} times"
    return text.replace(old, new)


def write(tmp_path: Path, text: str, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def violations(text: str, hosts=hc.DEFAULT_LITELLM_HOSTS):
    return hc.check_config(hc.parse_yaml_subset(text), hosts)


# ---------------------------------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------------------------------
def test_compliant_fixture_has_no_violations():
    assert violations(COMPLIANT) == []


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("max_iterations: 40", "max_iterations: 250", "delegation.max_iterations: expected 40, found 250"),
        ("max_iterations: 40", "max_iterations: 39", "delegation.max_iterations: expected 40, found 39"),
        ("max_iterations: 40", "max_iterations: 41", "delegation.max_iterations: expected 40, found 41"),
        ("max_concurrent_children: 2", "max_concurrent_children: 10", "delegation.max_concurrent_children: expected 2, found 10"),
        ("max_concurrent_children: 2", "max_concurrent_children: 1", "delegation.max_concurrent_children: expected 2, found 1"),
        ("max_spawn_depth: 1", "max_spawn_depth: 2", "delegation.max_spawn_depth: expected 1, found 2"),
        ("worktree_isolation: true", "worktree_isolation: false", "delegation.worktree_isolation: expected true, found false"),
        ("\n  hard_stop_enabled: true", "\n  hard_stop_enabled: false", "tool_loop_guardrails.hard_stop_enabled: expected true, found false"),
    ],
)
def test_each_rule_reports_its_exact_violation(old, new, message):
    assert violations(swap(COMPLIANT, old, new)) == [message]


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("  max_iterations: 40 # Max tool-calling turns per child\n", "delegation.max_iterations: missing, expected 40"),
        ("  max_concurrent_children: 2\n", "delegation.max_concurrent_children: missing, expected 2"),
        ("  max_spawn_depth: 1\n", "delegation.max_spawn_depth: missing, expected 1"),
        ("  worktree_isolation: true\n", "delegation.worktree_isolation: missing, expected true"),
        ("  hard_stop_enabled: true\n", "tool_loop_guardrails.hard_stop_enabled: missing, expected true"),
    ],
)
def test_missing_key_is_a_violation(line, message):
    assert violations(swap(COMPLIANT, line, "")) == [message]


def test_missing_section_reports_every_key_of_it():
    text = swap(COMPLIANT, "tool_loop_guardrails:\n  warnings_enabled: true\n  hard_stop_enabled: true\n  non_interactive_hard_stop_enabled: true\n  hard_stop_after:\n    exact_failure: 5\n", "")
    assert violations(text) == ["tool_loop_guardrails.hard_stop_enabled: missing, expected true"]


def test_boolean_does_not_satisfy_an_integer_rule():
    # True == 1 in Python, so max_spawn_depth: true must still fail the "== 1" rule.
    assert violations(swap(COMPLIANT, "max_spawn_depth: 1", "max_spawn_depth: true")) == [
        "delegation.max_spawn_depth: expected 1, found true"
    ]


def test_quoted_number_does_not_satisfy_an_integer_rule():
    assert violations(swap(COMPLIANT, "max_iterations: 40", 'max_iterations: "40"')) == [
        'delegation.max_iterations: expected 40, found "40"'
    ]


def test_string_true_does_not_satisfy_a_boolean_rule():
    assert violations(swap(COMPLIANT, "\n  hard_stop_enabled: true", '\n  hard_stop_enabled: "true"')) == [
        'tool_loop_guardrails.hard_stop_enabled: expected true, found "true"'
    ]


def test_null_value_counts_as_missing():
    assert violations(swap(COMPLIANT, "max_spawn_depth: 1", "max_spawn_depth:")) == [
        "delegation.max_spawn_depth: missing, expected 1"
    ]


def test_several_violations_are_all_reported_in_rule_order():
    text = swap(COMPLIANT, "max_iterations: 40", "max_iterations: 250")
    text = swap(text, "\n  hard_stop_enabled: true", "\n  hard_stop_enabled: false")
    assert violations(text) == [
        "delegation.max_iterations: expected 40, found 250",
        "tool_loop_guardrails.hard_stop_enabled: expected true, found false",
    ]


def test_non_mapping_config_is_a_violation():
    assert hc.check_config(["not", "a", "mapping"]) == ["<root>: config is not a mapping"]


# ---------------------------------------------------------------------------------------------------
# provider entries / gateway bypass
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("url", "host", "tail"),
    [
        ("https://gateway.ai.cloudflare.com/v1/acct/oxalis-hermes/openai", "gateway.ai.cloudflare.com", ""),
        ("https://api.anthropic.com/v1", "api.anthropic.com", ""),
        ("https://api.githubcopilot.com", "api.githubcopilot.com", SEAT),
    ],
    ids=["cloudflare", "anthropic", "copilot"],
)
def test_direct_endpoint_is_a_violation(url, host, tail):
    text = swap(
        COMPLIANT,
        "base_url: https://litellm.lab.floppydicks.net:4443/v1\n    key_env",
        f"base_url: {url}\n    key_env",
    )
    assert violations(text) == [f"custom_providers[0].base_url: forbidden endpoint host {host} {BYPASS}{tail}"]


def test_litellm_endpoint_is_not_a_violation():
    # positive twin of the three tests above: the same entry pointed at the LiteLLM base_url
    assert violations(COMPLIANT) == []


def test_direct_endpoint_in_the_model_section_is_a_violation():
    text = swap(COMPLIANT, 'base_url: "https://litellm.lab.floppydicks.net:4443/v1"', 'base_url: "https://api.anthropic.com/v1"')
    assert violations(text) == [f"model.base_url: forbidden endpoint host api.anthropic.com {BYPASS}"]


def test_subdomain_of_a_forbidden_host_is_a_violation():
    text = swap(COMPLIANT, "base_url: https://litellm.lab.floppydicks.net:4443/v1\n    key_env", "base_url: https://eu.api.anthropic.com/v1\n    key_env")
    assert violations(text) == [f"custom_providers[0].base_url: forbidden endpoint host eu.api.anthropic.com {BYPASS}"]


def test_lookalike_host_is_not_flagged_as_the_forbidden_host():
    # "notapi.anthropic.com" is not api.anthropic.com or a subdomain of it
    text = swap(COMPLIANT, "base_url: https://litellm.lab.floppydicks.net:4443/v1\n    key_env", "base_url: https://notapi.anthropic.com/v1\n    key_env")
    assert violations(text) == []


def test_credentials_in_a_url_are_not_echoed():
    text = swap(
        COMPLIANT,
        "base_url: https://litellm.lab.floppydicks.net:4443/v1\n    key_env",
        "base_url: https://user:hunter2secret@api.anthropic.com/v1\n    key_env",
    )
    out = violations(text)
    assert out == [f"custom_providers[0].base_url: forbidden endpoint host api.anthropic.com {BYPASS}"]
    assert "hunter2secret" not in out[0]


@pytest.mark.parametrize(
    ("provider", "tail"),
    [("anthropic", ""), ("copilot", SEAT), ("github-copilot", SEAT), ("Anthropic", "")],
)
def test_direct_provider_key_is_a_violation(provider, tail):
    text = swap(COMPLIANT, 'provider: "custom"', f'provider: "{provider}"')
    assert violations(text) == [
        f"model.provider: direct provider {provider!r} bypasses the LiteLLM gateway; route it through the LiteLLM base_url{tail}"
    ]


def test_custom_provider_value_is_not_a_violation():
    assert violations(swap(COMPLIANT, 'provider: "custom"', 'provider: "openrouter"')) == []


COPILOT_ENTRY = """\
  - name: Copilot seat
    base_url: https://api.githubcopilot.com
    provider: copilot
    seat_billed: %s
"""


def test_seat_billed_copilot_entry_is_allowed():
    assert violations(COMPLIANT + COPILOT_ENTRY % "true") == []


def test_unmarked_copilot_entry_is_reported_twice_once_per_key():
    assert violations(COMPLIANT + COPILOT_ENTRY % "false") == [
        f"custom_providers[1].base_url: forbidden endpoint host api.githubcopilot.com {BYPASS}{SEAT}",
        f"custom_providers[1].provider: direct provider 'copilot' {BYPASS}{SEAT}",
    ]


def test_seat_billed_string_is_not_the_boolean():
    assert violations(COMPLIANT + COPILOT_ENTRY % '"true"') == [
        f"custom_providers[1].base_url: forbidden endpoint host api.githubcopilot.com {BYPASS}{SEAT}",
        f"custom_providers[1].provider: direct provider 'copilot' {BYPASS}{SEAT}",
    ]


def test_seat_billed_does_not_exempt_anthropic():
    entry = (
        "  - name: Direct\n"
        "    base_url: https://api.anthropic.com/v1\n"
        "    provider: anthropic\n"
        "    seat_billed: true\n"
    )
    assert violations(COMPLIANT + entry) == [
        f"custom_providers[1].base_url: forbidden endpoint host api.anthropic.com {BYPASS}",
        f"custom_providers[1].provider: direct provider 'anthropic' {BYPASS}",
    ]


def test_seat_billed_on_one_entry_does_not_cover_a_sibling():
    sibling = "  - name: Other\n    base_url: https://api.githubcopilot.com\n"
    assert violations(COMPLIANT + COPILOT_ENTRY % "true" + sibling) == [
        f"custom_providers[2].base_url: forbidden endpoint host api.githubcopilot.com {BYPASS}{SEAT}"
    ]


def test_litellm_host_override_is_honoured():
    # A different LiteLLM host is treated as a violation-free endpoint only when named; the default is not.
    text = swap(COMPLIANT, "base_url: https://litellm.lab.floppydicks.net:4443/v1\n    key_env", "base_url: https://api.anthropic.com/v1\n    key_env")
    assert violations(text, ("api.anthropic.com",)) == []
    assert violations(text) == [f"custom_providers[0].base_url: forbidden endpoint host api.anthropic.com {BYPASS}"]


# ---------------------------------------------------------------------------------------------------
# the CLI
# ---------------------------------------------------------------------------------------------------
def test_cli_compliant_config_exits_zero_and_says_so(tmp_path, capsys):
    path = write(tmp_path, COMPLIANT)
    assert hc.main(["--config", str(path)]) == 0
    captured = capsys.readouterr()
    assert captured.out == f"Hermes config contract OK: {path}\n"
    assert captured.err == ""


def test_cli_reports_every_violation_and_exits_one(tmp_path, capsys):
    text = swap(COMPLIANT, "max_iterations: 40", "max_iterations: 250")
    text = swap(text, "worktree_isolation: true\n", "")
    text = swap(text, "base_url: https://litellm.lab.floppydicks.net:4443/v1\n    key_env", "base_url: https://gateway.ai.cloudflare.com/v1/a/b/openai\n    key_env")
    path = write(tmp_path, text)
    assert hc.main(["--config", str(path)]) == 1
    assert capsys.readouterr().out == (
        "VIOLATION delegation.max_iterations: expected 40, found 250\n"
        "VIOLATION delegation.worktree_isolation: missing, expected true\n"
        f"VIOLATION custom_providers[0].base_url: forbidden endpoint host gateway.ai.cloudflare.com {BYPASS}\n"
        f"3 violation(s) in {path}\n"
    )


def test_cli_litellm_host_flag_is_passed_through(tmp_path, capsys):
    text = swap(COMPLIANT, "base_url: https://litellm.lab.floppydicks.net:4443/v1\n    key_env", "base_url: https://api.anthropic.com/v1\n    key_env")
    path = write(tmp_path, text)
    assert hc.main(["--config", str(path), "--litellm-host", "api.anthropic.com"]) == 0
    assert capsys.readouterr().out == f"Hermes config contract OK: {path}\n"


def test_cli_unreadable_config_fails_closed(tmp_path, capsys):
    missing = tmp_path / "nope.yaml"
    assert hc.main(["--config", str(missing)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(f"ERROR: cannot read {missing}: FileNotFoundError")


def test_cli_unparsable_config_fails_closed(tmp_path, capsys):
    path = write(tmp_path, "delegation:\n  max_iterations: 40\n  max_iterations: 250\n")
    assert hc.main(["--config", str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"ERROR: cannot parse {path}: line 3: duplicate key 'max_iterations'\n"


def test_cli_never_writes(tmp_path, capsys):
    text = swap(COMPLIANT, "max_iterations: 40", "max_iterations: 250")
    path = write(tmp_path, text)
    before_bytes = path.read_bytes()
    before_mtime = path.stat().st_mtime_ns
    before_listing = sorted(p.name for p in tmp_path.iterdir())
    assert hc.main(["--config", str(path)]) == 1
    capsys.readouterr()
    assert path.read_bytes() == before_bytes
    assert path.stat().st_mtime_ns == before_mtime
    assert sorted(p.name for p in tmp_path.iterdir()) == before_listing == ["config.yaml"]


def test_cli_default_config_is_the_live_hermes_path_without_reading_it():
    # only the default's value is checked; main() is not called without --config
    assert hc.DEFAULT_CONFIG == "C:/Users/rjmendez-admin/AppData/Local/hermes/config.yaml"


# ---------------------------------------------------------------------------------------------------
# the YAML subset parser
# ---------------------------------------------------------------------------------------------------
def test_parser_reads_the_compliant_fixture_structure_exactly():
    cfg = hc.parse_yaml_subset(COMPLIANT)
    assert cfg["delegation"] == {
        "max_iterations": 40,
        "max_concurrent_children": 2,
        "max_spawn_depth": 1,
        "worktree_isolation": True,
    }
    assert cfg["tool_loop_guardrails"] == {
        "warnings_enabled": True,
        "hard_stop_enabled": True,
        "non_interactive_hard_stop_enabled": True,
        "hard_stop_after": {"exact_failure": 5},
    }
    assert cfg["custom_providers"] == [
        {
            "name": "LiteLLM auto-router",
            "base_url": "https://litellm.lab.floppydicks.net:4443/v1",
            "key_env": "LITELLM_API_KEY",
            "model": "clef-omni",
        }
    ]
    assert cfg["model"]["base_url"] == "https://litellm.lab.floppydicks.net:4443/v1"
    assert cfg["_config_version"] == 49


def test_parser_scalars():
    text = (
        "a: true\nb: False\nc: yes\nd: no\ne: ~\nf: null\ng: 12\nh: -3\ni: 1.5\nj: 007x\n"
        "k: 'it''s'\nl: \"tab\\there\"\nm:\nn: ''\n"
    )
    assert hc.parse_yaml_subset(text) == {
        "a": True, "b": False, "c": True, "d": False, "e": None, "f": None, "g": 12, "h": -3, "i": 1.5,
        "j": "007x", "k": "it's", "l": "tab\there", "m": None, "n": "",
    }


def test_hash_inside_quotes_is_not_a_comment():
    text = 'a: "x # y"  # real comment\nb: \'p # q\'\nc: url#frag\nd: plain # gone\n'
    assert hc.parse_yaml_subset(text) == {"a": "x # y", "b": "p # q", "c": "url#frag", "d": "plain"}


def test_parser_sequences_flow_and_nesting():
    text = (
        "empty_list: []\nempty_map: {}\nnums: [1, 2, three]\nmap: {a: 1, b: [x, y]}\n"
        "plain:\n- one\n- two\n"
        "indented:\n  - three\n  -\n    k: v\n"
        "dicts:\n  - name: a\n    n: 1\n  - name: b\n    sub:\n      z: 9\n"
        "after: 1\n"
    )
    assert hc.parse_yaml_subset(text) == {
        "empty_list": [],
        "empty_map": {},
        "nums": [1, 2, "three"],
        "map": {"a": 1, "b": ["x", "y"]},
        "plain": ["one", "two"],
        "indented": ["three", {"k": "v"}],
        "dicts": [{"name": "a", "n": 1}, {"name": "b", "sub": {"z": 9}}],
        "after": 1,
    }


def test_parser_key_with_comment_and_following_sibling_is_null():
    assert hc.parse_yaml_subset("max_concurrent_sessions: # note\ngroup: true\n") == {
        "max_concurrent_sessions": None,
        "group": True,
    }


def test_parser_block_scalars():
    text = "lit: |\n  line one\n  line two\nfold: >\n  a\n  b\nnext: 1\n"
    assert hc.parse_yaml_subset(text) == {"lit": "line one\nline two\n", "fold": "a b\n", "next": 1}


def test_parser_ignores_leading_document_marker_and_bom():
    assert hc.parse_yaml_subset("\ufeff---\na: 1\n") == {"a": 1}


def test_parser_handles_mixed_line_endings():
    # the live Hermes config mixes CRLF and LF lines
    assert hc.parse_yaml_subset("a:\r\n  b: 1\r\nc:\n  d: [x]\n  e: 2\r\n") == {"a": {"b": 1}, "c": {"d": ["x"], "e": 2}}


def test_parser_quoted_keys_and_urls_with_colons():
    assert hc.parse_yaml_subset('"a b": 1\nurl: http://h:8080/p\n') == {"a b": 1, "url": "http://h:8080/p"}


@pytest.mark.parametrize(
    ("text", "pattern"),
    [
        ("a: 1\na: 2\n", r"line 2: duplicate key 'a'"),
        ("a: &x 1\n", r"line 1: anchors, aliases and tags are not supported"),
        ("a: *x\n", r"line 1: anchors, aliases and tags are not supported"),
        ("a: !tag 1\n", r"line 1: anchors, aliases and tags are not supported"),
        ("a:\n\tb: 1\n", r"line 2: tab indentation is not supported"),
        ("a: one\n  two\n", r"line 2: multi-line plain scalars are not supported"),
        ("a: 1\n---\nb: 2\n", r"line 2: multi-document YAML is not supported"),
        ("a: [1,\n  2]\n", r"line 1: multi-line flow sequence is not supported"),
        ("a: {b: 1\n", r"line 1: multi-line flow mapping is not supported"),
        ("a: 'unterminated\n", r"line 1: unterminated quoted string"),
        ("- - x\n", r"line 1: nested block sequences on one line are not supported"),
        ("a: 1\n b: 2\n", r"line 2: multi-line plain scalars are not supported"),
        ("a:\n  b: 1\n c: 2\n", r"line 3: unexpected indentation"),
        ("just a line\n", r"line 1: expected 'key: value'"),
    ],
)
def test_parser_rejects_what_it_does_not_support(text, pattern):
    with pytest.raises(hc.ConfigParseError, match=pattern):
        hc.parse_yaml_subset(text)


def test_parser_empty_document_is_none_and_checks_report_not_a_mapping():
    assert hc.parse_yaml_subset("# only a comment\n\n") is None
    assert hc.check_config(None) == ["<root>: config is not a mapping"]
