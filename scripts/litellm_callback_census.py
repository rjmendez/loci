#!/usr/bin/env python3
"""Census of the LiteLLM callbacks named in config.yaml: does each one resolve, and from where. Read-only.

Slice S4 of the Tripwire-Governor design. For every entry of ``litellm_settings.callbacks`` it asks
a given interpreter, with a given set of extra ``sys.path`` entries, whether the callback's module can
be found, and reports the file it would load, that file's sha256 and which ``sys.path`` entry it came
from. Nothing is imported: parent packages are walked one level at a time with the interpreter's own
``sys.meta_path`` finders (what ``importlib.util.find_spec`` does for one name), so no ``__init__.py``
(litellm's is heavy) and no callback module is executed, no model is loaded, and the network is never
touched. The child runs with ``PYTHONDONTWRITEBYTECODE=1`` so a census does not write ``__pycache__``
into the trees it inspects.

    python scripts/litellm_callback_census.py --config ~/AppData/Local/litellm/config.yaml \\
        --python F:/services/litellm/venv/Scripts/python.exe --path F:/services/litellm/control-worktree
    python scripts/litellm_callback_census.py --config cfg.yaml --path RELEASE_TREE --compare-path DEV_TREE --json

Semantics, matching LiteLLM's ``get_instance_fn``: an entry ``a.b.c`` is module ``a.b`` plus attribute ``c``.
An entry with no dot is a LiteLLM built-in callback name and is not checkable here (resolved is null, not a
failure). A callback is ``resolved`` when its module is found and the attribute is not provably missing: the
module source is parsed (never run) for a top-level binding of the attribute; a module with ``__getattr__`` or
unparseable source gives ``attr_defined: null`` and is not held against it.

The ``PYTHONPATH`` of the child is exactly the ``--path`` entries (the caller's own is not inherited), the same
way the proxy is launched. ``--compare-path`` runs a second census with that path set and compares the files
by sha256 (``match``, ``mismatch``, ``only_left``, ``only_right``, ``neither``, ``unhashable``).

Only dotted module paths are ever written to the output: the config is not echoed, and a callbacks entry that
is not a dotted identifier is shown as ``<invalid entry #N>`` (it could be a pasted secret). Exits non-zero,
with "unresolved" in the message, when any primary-census callback is unresolved, and with a message when no
callbacks were found.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

MARKER = "CENSUS_JSON:"
_DOTTED = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")

# Runs inside the target interpreter (``python -c``), so it is self-contained and avoids new syntax.
PROBE_SOURCE = r'''
import ast, json, sys, types

MARKER = "CENSUS_JSON:"


def find(name, path):
    for finder in sys.meta_path:
        fs = getattr(finder, "find_spec", None)
        if fs is None:
            continue
        spec = fs(name, path)
        if spec is not None:
            return spec
    return None


def where(spec):
    return spec.origin if getattr(spec, "has_location", False) and spec.origin else None


def bound_names(body, out):
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        out.add(n.id)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            if isinstance(node.target, ast.Name):
                out.add(node.target.id)
        elif isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                out.add(a.asname or a.name)
        elif isinstance(node, ast.If):
            bound_names(node.body, out)
            bound_names(node.orelse, out)
        elif isinstance(node, ast.Try):
            bound_names(node.body, out)
            for h in node.handlers:
                bound_names(h.body, out)
            bound_names(node.orelse, out)
            bound_names(node.finalbody, out)
        elif isinstance(node, (ast.With, ast.For, ast.While)):
            bound_names(node.body, out)


def attr_defined(origin, attr):
    if not origin or not origin.endswith(".py"):
        return None
    try:
        with open(origin, "rb") as fh:
            tree = ast.parse(fh.read(), origin)
    except (OSError, SyntaxError, ValueError):
        return None
    names = set()
    bound_names(tree.body, names)
    if "__getattr__" in names:
        return None
    return attr in names


def resolve(module, attr):
    parts = module.split(".")
    spec = None
    prefix = None
    for i in range(len(parts)):
        name = ".".join(parts[: i + 1])
        if i == 0:
            path = None
        else:
            locations = spec.submodule_search_locations
            if locations is None:
                return {"found": False, "origin": None, "kind": None, "resolved_prefix": prefix, "attr_defined": None,
                        "reason": "%r is not a package (it resolves to %s)" % (prefix, where(spec) or "a non-file module")}
            path = list(locations)
        found = find(name, path)
        if found is None:
            if i == 0:
                reason = "top-level module %r not found on sys.path" % name
            else:
                reason = "module %r not found in package %r (resolved from %s)" % (
                    name, prefix, where(spec) or "namespace package")
            return {"found": False, "origin": None, "kind": None, "resolved_prefix": prefix, "attr_defined": None,
                    "reason": reason}
        spec = found
        prefix = name
        if spec.submodule_search_locations is not None and name not in sys.modules:
            # an empty stand-in (no code runs) so a namespace child can compute its parent path
            stub = types.ModuleType(name)
            stub.__path__ = list(spec.submodule_search_locations)
            sys.modules[name] = stub
    origin = where(spec)
    if origin is None:
        kind = "namespace" if spec.submodule_search_locations is not None else str(spec.origin)
    else:
        kind = "package" if spec.submodule_search_locations is not None else "module"
    return {"found": True, "origin": origin, "kind": kind, "resolved_prefix": prefix, "reason": None,
            "attr_defined": attr_defined(origin, attr)}


def main():
    request = json.loads(sys.stdin.read())
    results = [resolve(m, a) for m, a in request["pairs"]]
    interp = {"executable": sys.executable, "version": sys.version.split()[0], "prefix": sys.prefix,
              "sys_path": [str(p) for p in sys.path]}
    sys.stdout.write("\n" + MARKER + json.dumps({"interpreter": interp, "results": results}) + "\n")


main()
'''

Resolver = Callable[[list], dict]


def _strip_comment(line: str) -> str:
    quote = ""
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i]
    return line


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def parse_callbacks(text: str) -> list[str]:
    """Raw entries of ``litellm_settings.callbacks``; ``[]`` when the key is absent. Reads nothing else."""
    lines = [_strip_comment(ln).rstrip() for ln in text.splitlines()]
    start = next((i for i, ln in enumerate(lines) if re.fullmatch(r"litellm_settings\s*:\s*", ln)), None)
    if start is None:
        return []
    block = []
    for ln in lines[start + 1:]:
        if ln.strip() and _indent(ln) == 0 and not ln.lstrip().startswith("-"):
            break
        block.append(ln)
    body = [ln for ln in block if ln.strip()]
    if not body:
        return []
    key_indent = _indent(body[0])
    for n, ln in enumerate(body):
        m = re.fullmatch(r"\s*callbacks\s*:\s*(.*)", ln)
        if _indent(ln) != key_indent or not m:
            continue
        rest = m.group(1).strip()
        if rest.startswith("["):
            inner = rest.strip("[]")
            return [_unquote(p) for p in inner.split(",") if p.strip()]
        if rest:
            return [_unquote(rest)]
        out = []
        for nxt in body[n + 1:]:
            stripped = nxt.lstrip()
            if stripped.startswith("-") and _indent(nxt) >= key_indent:
                out.append(_unquote(stripped[1:].strip()))
            else:
                break
        return out
    return []


def sha256_of(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolved_from(origin: Optional[str], sys_path: list) -> Optional[str]:
    """The sys.path entry that contains ``origin`` (longest match), or None."""
    if not origin:
        return None
    norm = os.path.normcase(os.path.abspath(origin))
    best = None
    for entry in sys_path:
        if not entry:
            continue
        e = os.path.normcase(os.path.abspath(entry))
        if norm.startswith(e.rstrip("\\/") + os.sep) and (best is None or len(e) > len(best[0])):
            best = (e, entry)
    return best[1] if best else None


def subprocess_resolver(python: str, paths: list, timeout: float = 60.0) -> Resolver:
    """A resolver that asks ``python`` (PYTHONPATH = ``paths`` only) to look each module up without importing it."""
    def resolve(pairs: list) -> dict:
        exe = shutil.which(python) or (python if os.path.isfile(python) else None)
        if exe is None:
            raise SystemExit(f"litellm_callback_census: interpreter not found: {python}")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        if paths:
            env["PYTHONPATH"] = os.pathsep.join(str(p) for p in paths)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        with tempfile.TemporaryDirectory(prefix="census_cwd_") as cwd:
            try:
                proc = subprocess.run([exe, "-c", PROBE_SOURCE], input=json.dumps({"pairs": pairs}), capture_output=True,
                                      text=True, encoding="utf-8", env=env, cwd=cwd, timeout=timeout)
            except subprocess.TimeoutExpired:
                raise SystemExit(f"litellm_callback_census: interpreter {python} did not answer within {timeout:g} s")
        line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith(MARKER)), None)
        if proc.returncode != 0 or line is None:
            raise SystemExit(f"litellm_callback_census: interpreter {python} failed (exit {proc.returncode}): "
                             f"{proc.stderr.strip()[-400:]}")
        return json.loads(line[len(MARKER):])
    return resolve


def census(entries: list[str], resolver: Resolver) -> dict[str, Any]:
    """One row per config entry, in config order."""
    pairs, slots = [], []
    for i, entry in enumerate(entries):
        if not _DOTTED.match(entry):
            slots.append(("invalid", i))
        elif "." not in entry:
            slots.append(("builtin", i))
        else:
            module, attr = entry.rsplit(".", 1)
            slots.append(("path", len(pairs)))
            pairs.append([module, attr])
    answer = resolver(pairs) if pairs else {"interpreter": {}, "results": []}
    results = answer["results"]
    if len(results) != len(pairs):
        raise SystemExit(f"litellm_callback_census: resolver returned {len(results)} results for {len(pairs)} modules")
    sys_path = answer.get("interpreter", {}).get("sys_path", [])
    rows = []
    for i, (entry, (kind, ref)) in enumerate(zip(entries, slots)):
        row: dict[str, Any] = {"index": i, "entry": entry, "module": None, "attr": None, "resolved": None, "reason": None,
                               "origin": None, "kind": None, "sha256": None, "resolved_from": None,
                               "resolved_prefix": None, "attr_defined": None}
        if kind == "invalid":
            row.update(entry=f"<invalid entry #{i + 1}>", resolved=False, reason="not a dotted python path")
        elif kind == "builtin":
            row.update(reason="no dot: a LiteLLM built-in callback name, not checkable as a module path")
        else:
            r = results[ref]
            module, attr = entry.rsplit(".", 1)
            origin = r.get("origin")
            ok = bool(r["found"]) and r.get("attr_defined") is not False
            reason = r.get("reason")
            if r["found"] and r.get("attr_defined") is False:
                reason = f"module {module!r} resolves to {origin} but defines no top-level name {attr!r}"
            row.update(module=module, attr=attr, resolved=ok, reason=reason, origin=origin, kind=r.get("kind"),
                       sha256=sha256_of(origin) if r["found"] else None,
                       resolved_from=_resolved_from(origin, sys_path), resolved_prefix=r.get("resolved_prefix"),
                       attr_defined=r.get("attr_defined"))
        rows.append(row)
    return {"interpreter": answer.get("interpreter", {}), "rows": rows,
            "unresolved": [r["entry"] for r in rows if r["resolved"] is False]}


def compare(left: dict, right: dict) -> list[dict]:
    """Per-callback comparison of two censuses of the same config by file sha256."""
    out = []
    for a, b in zip(left["rows"], right["rows"]):
        ra, rb = a["resolved"], b["resolved"]
        if ra is None and rb is None:
            continue
        if ra and rb:
            status = ("unhashable" if not (a["sha256"] and b["sha256"])
                      else "match" if a["sha256"] == b["sha256"] else "mismatch")
        elif ra:
            status = "only_left"
        elif rb:
            status = "only_right"
        else:
            status = "neither"
        out.append({"entry": a["entry"], "status": status, "left_origin": a["origin"], "right_origin": b["origin"],
                    "left_sha256": a["sha256"], "right_sha256": b["sha256"]})
    return out


def render_text(report: dict[str, Any]) -> str:
    it = report["interpreter"]
    out = [f"interpreter: {it.get('executable')} (python {it.get('version')})",
           f"PYTHONPATH: {os.pathsep.join(report['pythonpath']) or '-'}", ""]
    for r in report["rows"]:
        state = {True: "RESOLVED", False: "UNRESOLVED", None: "n/a"}[r["resolved"]]
        out.append(f"{r['index'] + 1}. {r['entry']}  {state}")
        if r["resolved"] is not False and r["origin"]:
            out.append(f"     file: {r['origin']}")
            out.append(f"     sha256: {r['sha256']}   from sys.path entry: {r['resolved_from']}")
        if r["reason"]:
            out.append(f"     reason: {r['reason']}")
        if r["resolved"] is False and r["resolved_prefix"]:
            out.append(f"     deepest package found: {r['resolved_prefix']}")
    if "comparison" in report:
        out += ["", f"comparison (left = --path, right = --compare-path {os.pathsep.join(report['compare_pythonpath'])}):"]
        for c in report["comparison"]:
            out.append(f"  {c['status']:>10}  {c['entry']}")
            if c["status"] in ("match", "mismatch"):
                out.append(f"              left  {c['left_sha256']}\n              right {c['right_sha256']}")
    out.append("")
    out.append(f"{len(report['rows'])} callbacks, {len(report['unresolved'])} unresolved")
    return "\n".join(out)


def build_report(config_text: str, config_name: str, resolver: Resolver, paths: list,
                 compare_resolver: Optional[Resolver] = None, compare_paths: Optional[list] = None) -> dict[str, Any]:
    entries = parse_callbacks(config_text)
    if not entries:
        raise SystemExit(f"litellm_callback_census: no callbacks found under litellm_settings in {config_name}")
    primary = census(entries, resolver)
    report: dict[str, Any] = {"config": config_name, "pythonpath": [str(p) for p in paths], **primary}
    if compare_resolver is not None:
        other = census(entries, compare_resolver)
        report["compare_pythonpath"] = [str(p) for p in (compare_paths or [])]
        report["comparison"] = compare(primary, other)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, required=True, help="LiteLLM config.yaml (only litellm_settings.callbacks is read)")
    ap.add_argument("--python", default=sys.executable, help="interpreter to ask (default: this one); use the proxy's own")
    ap.add_argument("--path", action="append", default=[], help="extra sys.path entry (PYTHONPATH), repeatable")
    ap.add_argument("--compare-path", action="append", default=[],
                    help="second path set, same interpreter; compares each callback's file by sha256, repeatable")
    ap.add_argument("--timeout", type=float, default=60.0, help="seconds to wait for the interpreter")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    args = ap.parse_args(argv)
    try:
        text = args.config.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise SystemExit(f"litellm_callback_census: cannot read config {args.config}: {exc.strerror}")
    cmp_resolver = subprocess_resolver(args.python, args.compare_path, args.timeout) if args.compare_path else None
    report = build_report(text, str(args.config), subprocess_resolver(args.python, args.path, args.timeout), args.path,
                          cmp_resolver, args.compare_path)
    print(json.dumps(report, indent=1) if args.json else render_text(report))
    if report["unresolved"]:
        raise SystemExit(f"litellm_callback_census: {len(report['unresolved'])} unresolved: {', '.join(report['unresolved'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
