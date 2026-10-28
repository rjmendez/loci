#!/usr/bin/env python3
"""Check a Hermes ``config.yaml`` against the Tripwire-Governor S8 hardening contract. READ-ONLY.

The script opens the config for reading only, prints every violation with its key path and exits
non-zero if there is any. It never writes, creates or edits a file.

    python scripts/hermes_config_check.py
    python scripts/hermes_config_check.py --config /path/to/config.yaml --litellm-host litellm.lab.floppydicks.net

Contract (design section 4 and decision 6):

* ``delegation.max_iterations == 40``, ``delegation.max_concurrent_children == 2``,
  ``delegation.max_spawn_depth == 1``, ``delegation.worktree_isolation is true``;
* ``tool_loop_guardrails.hard_stop_enabled is true``;
* no provider entry reaches ``gateway.ai.cloudflare.com``, ``api.anthropic.com`` or a Copilot endpoint
  except through the LiteLLM ``base_url``. Every URL found anywhere in the config is checked, and so is
  every ``provider:`` key naming ``anthropic`` or a Copilot provider. A dict marked ``seat_billed: true``
  may keep a Copilot entry (Copilot cannot be proxied; it is seat-billed, outside the dollar caps).

Exit codes: 0 compliant, 1 violations found, 2 the config could not be read or parsed (fail closed: an
unreadable config is never reported as compliant).

Stdlib only. PyYAML is not a dependency of this repo, so the file is read with a small YAML-subset
parser (block mappings and sequences, scalars, single-line flow collections, comments, quoted strings,
block scalars). Anything outside that subset (anchors, aliases, tags, multi-document files, duplicate
keys, multi-line flow collections, multi-line plain scalars) raises ``ConfigParseError``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

DEFAULT_CONFIG = "C:/Users/rjmendez-admin/AppData/Local/hermes/config.yaml"
DEFAULT_LITELLM_HOSTS = ("litellm.lab.floppydicks.net", "127.0.0.1", "localhost")

# key path -> required exact value (booleans are compared by type, so ``true`` never satisfies ``1``)
INT_RULES = (
    ("delegation", "max_iterations", 40),
    ("delegation", "max_concurrent_children", 2),
    ("delegation", "max_spawn_depth", 1),
)
BOOL_RULES = (
    ("delegation", "worktree_isolation", True),
    ("tool_loop_guardrails", "hard_stop_enabled", True),
)
FORBIDDEN_HOSTS = ("gateway.ai.cloudflare.com", "api.anthropic.com")
FORBIDDEN_PROVIDERS = ("anthropic", "copilot", "github-copilot", "copilot-acp")
_URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)


class ConfigParseError(ValueError):
    """The config uses YAML outside the supported subset, or is malformed."""


# --------------------------------------------------------------------------------------------------
# YAML subset parser
# --------------------------------------------------------------------------------------------------
_INT_RE = re.compile(r"^[-+]?(0|[1-9][0-9]*)$")
_FLOAT_RE = re.compile(r"^[-+]?([0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)([eE][-+]?[0-9]+)?$")
_TRUE = {"true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON"}
_FALSE = {"false", "False", "FALSE", "no", "No", "NO", "off", "Off", "OFF"}
_NULL = {"", "~", "null", "Null", "NULL"}


def _strip_comment(line: str) -> str:
    """Remove a trailing ``# comment`` that sits outside quotes (``#`` must follow whitespace)."""
    quote = ""
    prev = " "
    for i, ch in enumerate(line):
        if quote:
            if quote == '"' and ch == "\\":
                continue
            if ch == quote:
                quote = ""
        elif ch in "\"'" and prev in " \t:-[{,":
            quote = ch
        elif ch == "#" and prev in " \t":
            return line[:i].rstrip()
        prev = ch
    return line.rstrip()


def _unquote(token: str, where: str) -> str:
    q = token[0]
    if len(token) < 2 or token[-1] != q:
        raise ConfigParseError(f"{where}: unterminated quoted string")
    body = token[1:-1]
    if q == "'":
        return body.replace("''", "'")
    out: List[str] = []
    i = 0
    escapes = {"n": "\n", "t": "\t", '"': '"', "\\": "\\", "/": "/", "r": "\r", "0": "\0"}
    while i < len(body):
        c = body[i]
        if c == "\\":
            i += 1
            if i >= len(body) or body[i] not in escapes:
                raise ConfigParseError(f"{where}: unsupported escape in double-quoted string")
            out.append(escapes[body[i]])
        else:
            out.append(c)
        i += 1
    return "".join(out)


def _scalar(token: str, where: str) -> Any:
    token = token.strip()
    if token and token[0] in "\"'":
        return _unquote(token, where)
    if token[:1] in ("&", "*", "!"):
        raise ConfigParseError(f"{where}: anchors, aliases and tags are not supported")
    if token in _NULL:
        return None
    if token in _TRUE:
        return True
    if token in _FALSE:
        return False
    if _INT_RE.match(token):
        return int(token)
    if _FLOAT_RE.match(token):
        return float(token)
    return token


def _split_flow(body: str, where: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    quote = ""
    cur: List[str] = []
    for ch in body:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    if quote or depth != 0:
        raise ConfigParseError(f"{where}: unbalanced flow collection")
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def _find_key_colon(text: str) -> int:
    """Index of the ``:`` that ends a mapping key, or -1 when ``text`` is not a ``key: value`` line."""
    if text[:1] in "\"'":
        q = text[0]
        i = 1
        while i < len(text):
            if text[i] == q and not (q == "'" and text[i + 1 : i + 2] == "'"):
                break
            i += 2 if (q == "'" and text[i + 1 : i + 2] == "'") or (q == '"' and text[i] == "\\") else 1
        j = i + 1
        return j if text[j : j + 1] == ":" and (len(text) == j + 1 or text[j + 1] in " \t") else -1
    if text[:1] in "[{":
        return -1
    for i, ch in enumerate(text):
        if ch == ":" and (i + 1 == len(text) or text[i + 1] in " \t"):
            return i
    return -1


def _flow(token: str, where: str) -> Any:
    token = token.strip()
    if token.startswith("["):
        if not token.endswith("]"):
            raise ConfigParseError(f"{where}: multi-line flow sequence is not supported")
        return [_flow(p, where) for p in _split_flow(token[1:-1], where)]
    if token.startswith("{"):
        if not token.endswith("}"):
            raise ConfigParseError(f"{where}: multi-line flow mapping is not supported")
        result: dict = {}
        for part in _split_flow(token[1:-1], where):
            colon = _find_key_colon(part)
            if colon < 0:
                raise ConfigParseError(f"{where}: flow mapping entry without a key")
            key = _scalar(part[:colon], where)
            if key in result:
                raise ConfigParseError(f"{where}: duplicate key {key!r}")
            result[key] = _flow(part[colon + 1 :], where)
        return result
    return _scalar(token, where)


class _Line:
    __slots__ = ("indent", "text", "no")

    def __init__(self, indent: int, text: str, no: int) -> None:
        self.indent = indent
        self.text = text
        self.no = no


def _tokenize(text: str) -> Tuple[List[_Line], List[str]]:
    raw = text.lstrip("\ufeff").splitlines()
    lines: List[_Line] = []
    seen_doc = False
    for no, line in enumerate(raw, 1):
        if line.startswith("---") and line.strip() == "---":
            if seen_doc or lines:
                raise ConfigParseError(f"line {no}: multi-document YAML is not supported")
            seen_doc = True
            continue
        if line.startswith("...") and line.strip() == "...":
            raise ConfigParseError(f"line {no}: multi-document YAML is not supported")
        stripped = _strip_comment(line)
        if not stripped.strip():
            continue
        lead = stripped[: len(stripped) - len(stripped.lstrip())]
        if "\t" in lead:
            raise ConfigParseError(f"line {no}: tab indentation is not supported")
        lines.append(_Line(len(lead), stripped.strip(), no))
    return lines, raw


class _Parser:
    def __init__(self, lines: List[_Line], raw: List[str]) -> None:
        self.lines = lines
        self.raw = raw
        self.pos = 0

    def _block_scalar(self, header: str, parent_indent: int, line_no: int) -> str:
        """Collect the raw lines of a ``|`` or ``>`` block scalar (indicator in ``header``)."""
        folded = header.startswith(">")
        body: List[str] = []
        i = line_no  # raw index of the line after the header (line_no is 1-based)
        block_indent: Optional[int] = None
        while i < len(self.raw):
            ln = self.raw[i]
            if not ln.strip():
                body.append("")
                i += 1
                continue
            ind = len(ln) - len(ln.lstrip())
            if ind <= parent_indent:
                break
            if block_indent is None:
                block_indent = ind
            body.append(ln[block_indent:] if ind >= block_indent else ln.strip())
            i += 1
        while self.pos < len(self.lines) and self.lines[self.pos].no <= i:
            self.pos += 1
        text = (" " if folded else "\n").join(body).rstrip("\n ")
        return text + "\n" if text else ""

    def parse(self) -> Any:
        if not self.lines:
            return None
        first = self.lines[0]
        value = self._block(first.indent)
        if self.pos < len(self.lines):
            bad = self.lines[self.pos]
            raise ConfigParseError(f"line {bad.no}: unexpected content or indentation")
        return value

    def _block(self, indent: int) -> Any:
        line = self.lines[self.pos]
        if line.text == "-" or line.text.startswith("- "):
            return self._sequence(indent)
        return self._mapping(indent)

    def _value(self, rest: str, indent: int, line: _Line, allow_same_indent_seq: bool) -> Any:
        """Value of ``key: rest`` / ``- rest``; the owning line has already been consumed."""
        where = f"line {line.no}"
        if rest[:1] in ("|", ">"):
            return self._block_scalar(rest, indent, line.no)
        if rest:
            if rest[:1] in "[{":
                return _flow(rest, where)
            val = _scalar(rest, where)
            if self.pos < len(self.lines) and self.lines[self.pos].indent > indent:
                raise ConfigParseError(
                    f"line {self.lines[self.pos].no}: multi-line plain scalars are not supported"
                )
            return val
        nxt = self.lines[self.pos] if self.pos < len(self.lines) else None
        if nxt is not None and nxt.indent > indent:
            return self._block(nxt.indent)
        if (
            allow_same_indent_seq
            and nxt is not None
            and nxt.indent == indent
            and (nxt.text == "-" or nxt.text.startswith("- "))
        ):
            return self._sequence(indent)
        return None

    def _mapping(self, indent: int) -> dict:
        result: dict = {}
        while self.pos < len(self.lines):
            line = self.lines[self.pos]
            if line.indent < indent:
                break
            if line.indent > indent:
                raise ConfigParseError(f"line {line.no}: unexpected indentation")
            if line.text == "-" or line.text.startswith("- "):
                break
            colon = _find_key_colon(line.text)
            if colon < 0:
                raise ConfigParseError(f"line {line.no}: expected 'key: value'")
            key = _scalar(line.text[:colon], f"line {line.no}")
            if isinstance(key, (dict, list)) or key is None:
                raise ConfigParseError(f"line {line.no}: unsupported mapping key")
            if key in result:
                raise ConfigParseError(f"line {line.no}: duplicate key {key!r}")
            self.pos += 1
            result[key] = self._value(line.text[colon + 1 :].strip(), indent, line, True)
        return result

    def _sequence(self, indent: int) -> list:
        result: list = []
        while self.pos < len(self.lines):
            line = self.lines[self.pos]
            if line.indent != indent or not (line.text == "-" or line.text.startswith("- ")):
                if line.indent > indent:
                    raise ConfigParseError(f"line {line.no}: unexpected indentation")
                break
            rest = line.text[1:].strip()
            if rest == "-" or rest.startswith("- "):
                raise ConfigParseError(f"line {line.no}: nested block sequences on one line are not supported")
            if rest and _find_key_colon(rest) >= 0 and rest[:1] not in "[{|>":
                # "- key: value": the mapping continues on following lines at the key's column.
                offset = line.text.index(rest)
                self.lines[self.pos] = _Line(indent + offset, rest, line.no)
                result.append(self._mapping(indent + offset))
            else:
                self.pos += 1
                result.append(self._value(rest, indent, line, False))
        return result


def parse_yaml_subset(text: str) -> Any:
    """Parse ``text`` (the supported YAML subset) into dicts, lists and scalars."""
    lines, raw = _tokenize(text)
    return _Parser(lines, raw).parse()


# --------------------------------------------------------------------------------------------------
# Contract checks
# --------------------------------------------------------------------------------------------------
def _fmt(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return type(value).__name__
    text = json.dumps(value)
    return text if len(text) <= 40 else text[:37] + '..."'


def _lookup(cfg: Any, parts: Iterable[str]) -> Tuple[bool, Any]:
    cur = cfg
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


def _is_seat_billed(node: dict) -> bool:
    return node.get("seat_billed") is True


def _forbidden_host(host: str, litellm_hosts: Tuple[str, ...]) -> Optional[str]:
    host = host.lower().rstrip(".")
    if host in litellm_hosts:
        return None
    for banned in FORBIDDEN_HOSTS:
        if host == banned or host.endswith("." + banned):
            return banned
    if "copilot" in host:
        return "copilot"
    return None


def _walk_providers(node: Any, path: str, seat_billed: bool, litellm_hosts: Tuple[str, ...], out: List[str]) -> None:
    if isinstance(node, dict):
        seat = seat_billed or _is_seat_billed(node)
        for key, val in node.items():
            sub = f"{path}.{key}" if path else str(key)
            if key == "provider" and isinstance(val, str) and val.strip().lower() in FORBIDDEN_PROVIDERS:
                name = val.strip().lower()
                if not (name != "anthropic" and seat):
                    out.append(
                        f"{sub}: direct provider {val.strip()!r} bypasses the LiteLLM gateway; "
                        "route it through the LiteLLM base_url"
                        + ("" if name == "anthropic" else " or mark the entry seat_billed: true")
                    )
            _walk_providers(val, sub, seat, litellm_hosts, out)
    elif isinstance(node, list):
        for i, val in enumerate(node):
            _walk_providers(val, f"{path}[{i}]", seat_billed, litellm_hosts, out)
    elif isinstance(node, str):
        for url in _URL_RE.findall(node):
            host = urlparse(url).hostname or ""
            banned = _forbidden_host(host, litellm_hosts)
            if banned is None:
                continue
            if banned == "copilot" and seat_billed:
                continue
            out.append(
                f"{path}: forbidden endpoint host {host} bypasses the LiteLLM gateway; "
                "route it through the LiteLLM base_url"
                + ("" if banned != "copilot" else " or mark the entry seat_billed: true")
            )


def check_config(cfg: Any, litellm_hosts: Iterable[str] = DEFAULT_LITELLM_HOSTS) -> List[str]:
    """Return one message per contract violation, each starting with the offending key path."""
    hosts = tuple(h.lower() for h in litellm_hosts)
    out: List[str] = []
    if not isinstance(cfg, dict):
        return ["<root>: config is not a mapping"]
    for section, key, want in INT_RULES:
        found, val = _lookup(cfg, (section, key))
        path = f"{section}.{key}"
        if not found or val is None:
            out.append(f"{path}: missing, expected {want}")
        elif type(val) is not int or val != want:
            out.append(f"{path}: expected {want}, found {_fmt(val)}")
    for section, key, want in BOOL_RULES:
        found, val = _lookup(cfg, (section, key))
        path = f"{section}.{key}"
        if not found or val is None:
            out.append(f"{path}: missing, expected {str(want).lower()}")
        elif val is not want:
            out.append(f"{path}: expected {str(want).lower()}, found {_fmt(val)}")
    _walk_providers(cfg, "", False, hosts, out)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Hermes config.yaml to check (read only)")
    parser.add_argument(
        "--litellm-host",
        action="append",
        default=None,
        help="host of the LiteLLM base_url (repeatable); default: %s" % ", ".join(DEFAULT_LITELLM_HOSTS),
    )
    args = parser.parse_args(argv)
    try:
        text = Path(args.config).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        print(f"ERROR: cannot read {args.config}: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 2
    try:
        cfg = parse_yaml_subset(text)
    except ConfigParseError as exc:
        print(f"ERROR: cannot parse {args.config}: {exc}", file=sys.stderr)
        return 2
    violations = check_config(cfg, args.litellm_host or DEFAULT_LITELLM_HOSTS)
    for v in violations:
        print(f"VIOLATION {v}")
    if violations:
        print(f"{len(violations)} violation(s) in {args.config}")
        return 1
    print(f"Hermes config contract OK: {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
