#!/usr/bin/env python3
"""Configure and verify local Loci backend settings.

Writes only to a user config path (default: ~/.loci/backends.toml), never to repo
tracked config files, and never prints secret values.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib import error, request

try:
    import tomllib
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Python 3.11+ with tomllib is required") from exc


_DEFAULT_CONFIG = "~/.loci/backends.toml"
_DEFAULT_OR_KEY_FILE = "~/.openrouter"
_DEFAULT_AB_KEY_FILE = "~/.abliteration"
_ROLE_SET = {"triage", "coding", "reasoning", "synthesis", "redteam"}


def _expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _toml_quote(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return _toml_quote(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        parts: list[str] = []
        for k, v in value.items():
            parts.append(f"{k}={_toml_value(v)}")
        return "{ " + ", ".join(parts) + " }"
    raise TypeError(f"Unsupported TOML value type: {type(value).__name__}")


def _dump_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []
    for section in sorted(data):
        section_data = data.get(section)
        if not isinstance(section_data, dict):
            continue
        lines.append(f"[{section}]")
        nested_sections: list[tuple[str, dict[str, Any]]] = []
        for key in sorted(section_data):
            value = section_data[key]
            if isinstance(value, dict) and all(isinstance(k, str) for k in value):
                if any(isinstance(v, dict) for v in value.values()):
                    nested_sections.append((key, value))
                else:
                    lines.append(f"{key}={_toml_value(value)}")
            else:
                lines.append(f"{key}={_toml_value(value)}")
        lines.append("")
        for sub_name, sub_values in sorted(nested_sections, key=lambda x: x[0]):
            lines.append(f"[{section}.{sub_name}]")
            for k in sorted(sub_values):
                lines.append(f"{k}={_toml_value(sub_values[k])}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _safe_write_toml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(_dump_toml(data), encoding="utf-8")
    tmp_path.replace(path)


def _bool_flag(value: str) -> bool:
    token = value.strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected one of on/off,true/false, got: {value}")


def _parse_role_assignments(values: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in values or []:
        if "=" not in raw:
            raise argparse.ArgumentTypeError(f"Expected role=value, got: {raw}")
        role, value = raw.split("=", 1)
        role_key = role.strip().lower().replace("_", "-")
        if role_key not in _ROLE_SET:
            raise argparse.ArgumentTypeError(f"Unknown role '{role}'. Valid: {sorted(_ROLE_SET)}")
        mapped = value.strip()
        if not mapped:
            raise argparse.ArgumentTypeError(f"Empty value for role '{role_key}'")
        out[role_key] = mapped
    return out


def _read_key_file(path: Path) -> str:
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if raw.startswith("export "):
            raw = raw[len("export "):].strip()
        if "=" in raw:
            _, raw = raw.split("=", 1)
            raw = raw.strip()
        raw = raw.strip().strip('"').strip("'")
        if raw:
            return raw
    return ""


def _ensure_user_path(config_path: Path, repo_root: Path) -> list[str]:
    failures: list[str] = []
    try:
        config_path.relative_to(repo_root)
    except ValueError:
        return failures
    failures.append(
        f"Refusing --apply to repo path '{config_path}'. Use a user config path (default: {_DEFAULT_CONFIG})."
    )
    return failures


def _provider_auth_check(name: str, base_url: str, api_key: str) -> tuple[bool, str]:
    if not base_url:
        return False, f"{name}: missing url"
    if not api_key:
        return False, f"{name}: missing key"
    url = base_url.rstrip("/") + "/models"
    req = request.Request(url, method="GET")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    try:
        with request.urlopen(req, timeout=12) as resp:  # nosec: B310
            ok = 200 <= int(resp.status) < 300
            return ok, f"{name}: auth check status {resp.status}"
    except error.HTTPError as exc:
        return False, f"{name}: auth check http {exc.code}"
    except Exception as exc:
        return False, f"{name}: auth check failed ({type(exc).__name__})"


def _tmux_session_exists(name: str) -> bool:
    try:
        proc = subprocess.run(
            ["tmux", "has-session", "-t", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _verify_config(cfg: dict[str, Any], args: argparse.Namespace) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    notes: list[str] = []

    cloud = cfg.get("cloud") if isinstance(cfg.get("cloud"), dict) else {}
    openrouter = cfg.get("openrouter") if isinstance(cfg.get("openrouter"), dict) else {}
    abliteration = cfg.get("abliteration") if isinstance(cfg.get("abliteration"), dict) else {}
    tmux = cfg.get("tmux_offload") if isinstance(cfg.get("tmux_offload"), dict) else {}

    cloud_enabled = bool(cloud.get("enabled", False))
    if cloud_enabled:
        or_key = str(openrouter.get("key", "")).strip()
        ab_key = str(abliteration.get("key", "")).strip()
        if not or_key and not ab_key:
            failures.append("cloud.enabled=true but neither [openrouter].key nor [abliteration].key is configured")
        if or_key and not str(openrouter.get("url", "")).strip():
            failures.append("openrouter key set but [openrouter].url missing")
        if ab_key and not str(abliteration.get("url", "")).strip():
            failures.append("abliteration key set but [abliteration].url missing")
    else:
        notes.append("cloud tier disabled")

    if "tmux_offload" in cfg:
        mapping = tmux.get("role_sessions", {})
        if mapping and not isinstance(mapping, dict):
            failures.append("[tmux_offload].role_sessions must be a table/object")

    tmux_enabled = bool(tmux.get("enabled", False))
    require_session = bool(tmux.get("require_mapped_session", False))
    should_check_tmux = tmux_enabled and (args.check_tmux_sessions or require_session)
    if should_check_tmux:
        sessions = tmux.get("role_sessions", {})
        if not isinstance(sessions, dict) or not sessions:
            failures.append("tmux offload session checks enabled but no [tmux_offload].role_sessions mappings found")
        else:
            missing: list[str] = []
            for role, session in sessions.items():
                if not _tmux_session_exists(str(session)):
                    missing.append(f"{role}={session}")
            if missing:
                failures.append("missing tmux sessions: " + ", ".join(missing))
            else:
                notes.append("tmux session checks passed")

    if args.check_provider_auth:
        checks: list[tuple[str, str, str]] = [
            ("openrouter", str(openrouter.get("url", "")).strip(), str(openrouter.get("key", "")).strip()),
            ("abliteration", str(abliteration.get("url", "")).strip(), str(abliteration.get("key", "")).strip()),
        ]
        for name, url, key in checks:
            if not key:
                notes.append(f"{name}: auth check skipped (no key)")
                continue
            ok, msg = _provider_auth_check(name, url, key)
            if ok:
                notes.append(msg)
            else:
                failures.append(msg)

    return failures, notes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply and verify ~/.loci/backends.toml cloud/offload configuration."
    )
    parser.add_argument("--config", default=_DEFAULT_CONFIG, help="Target backend config file path.")
    parser.add_argument("--apply", action="store_true", help="Apply requested config changes to --config.")
    parser.add_argument("--check-provider-auth", action="store_true",
                        help="Verify configured cloud keys by probing each provider /models endpoint.")
    parser.add_argument("--check-tmux-sessions", action="store_true",
                        help="Verify mapped tmux sessions exist when tmux offload is enabled.")

    parser.add_argument("--cloud-tier", type=_bool_flag,
                        help="Set [cloud].enabled (on/off).")

    parser.add_argument("--openrouter-key-file", default=_DEFAULT_OR_KEY_FILE,
                        help="Path to file containing OpenRouter API key.")
    parser.add_argument("--openrouter-url", help="Set [openrouter].url.")
    parser.add_argument("--openrouter-model", help="Set [openrouter].model.")
    parser.add_argument("--openrouter-role-model", action="append",
                        help="Set [openrouter.<role>].model via role=model; repeatable.")

    parser.add_argument("--abliteration-key-file", default=_DEFAULT_AB_KEY_FILE,
                        help="Path to file containing Abliteration API key.")
    parser.add_argument("--abliteration-url", help="Set [abliteration].url.")
    parser.add_argument("--abliteration-model", help="Set [abliteration].model.")
    parser.add_argument("--abliteration-role-model", action="append",
                        help="Set [abliteration.<role>].model via role=model; repeatable.")

    parser.add_argument("--tmux-offload", type=_bool_flag,
                        help="Set [tmux_offload].enabled (on/off).")
    parser.add_argument("--tmux-role-session", action="append",
                        help="Set [tmux_offload].role_sessions via role=session; repeatable.")
    parser.add_argument("--tmux-expensive-roles",
                        help="Comma list for [tmux_offload].expensive_roles.")
    parser.add_argument("--tmux-require-mapped-session", type=_bool_flag,
                        help="Set [tmux_offload].require_mapped_session (on/off).")
    parser.add_argument("--strict-key-files", action="store_true",
                        help="Fail apply if a key file path is configured but unreadable/empty.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    config_path = _expand(args.config)

    try:
        cfg = _read_toml(config_path)
    except Exception as exc:
        print(f"ERROR: failed to parse config '{config_path}': {type(exc).__name__}", file=sys.stderr)
        return 2

    apply_failures: list[str] = []
    if args.apply:
        apply_failures.extend(_ensure_user_path(config_path, repo_root))
        if not apply_failures:
            cloud = dict(cfg.get("cloud") or {})
            openrouter = dict(cfg.get("openrouter") or {})
            abliteration = dict(cfg.get("abliteration") or {})
            tmux = dict(cfg.get("tmux_offload") or {})

            if args.cloud_tier is not None:
                cloud["enabled"] = bool(args.cloud_tier)

            or_file = _expand(args.openrouter_key_file)
            ab_file = _expand(args.abliteration_key_file)
            or_key = _read_key_file(or_file)
            ab_key = _read_key_file(ab_file)
            if args.strict_key_files:
                if args.openrouter_key_file and not or_key:
                    apply_failures.append(f"OpenRouter key file empty/unreadable: {or_file}")
                if args.abliteration_key_file and not ab_key:
                    apply_failures.append(f"Abliteration key file empty/unreadable: {ab_file}")

            if args.openrouter_url is not None:
                openrouter["url"] = args.openrouter_url
            if args.openrouter_model is not None:
                openrouter["model"] = args.openrouter_model
            role_models_or = _parse_role_assignments(args.openrouter_role_model)
            for role, model in role_models_or.items():
                role_block = dict(openrouter.get(role) or {})
                role_block["model"] = model
                openrouter[role] = role_block
            if or_key:
                openrouter["key"] = or_key

            if args.abliteration_url is not None:
                abliteration["url"] = args.abliteration_url
            if args.abliteration_model is not None:
                abliteration["model"] = args.abliteration_model
            role_models_ab = _parse_role_assignments(args.abliteration_role_model)
            for role, model in role_models_ab.items():
                role_block = dict(abliteration.get(role) or {})
                role_block["model"] = model
                abliteration[role] = role_block
            if ab_key:
                abliteration["key"] = ab_key

            if args.tmux_offload is not None:
                tmux["enabled"] = bool(args.tmux_offload)
            if args.tmux_require_mapped_session is not None:
                tmux["require_mapped_session"] = bool(args.tmux_require_mapped_session)
            if args.tmux_expensive_roles is not None:
                roles = [r.strip().lower().replace("_", "-")
                         for r in args.tmux_expensive_roles.split(",") if r.strip()]
                tmux["expensive_roles"] = roles
            session_map = _parse_role_assignments(args.tmux_role_session)
            if session_map:
                merged = dict(tmux.get("role_sessions") or {})
                merged.update(session_map)
                tmux["role_sessions"] = merged

            cfg["cloud"] = cloud
            cfg["openrouter"] = openrouter
            cfg["abliteration"] = abliteration
            cfg["tmux_offload"] = tmux

            if not apply_failures:
                _safe_write_toml(config_path, cfg)

    if apply_failures:
        for failure in apply_failures:
            print(f"FAIL: {failure}")
        return 2

    failures, notes = _verify_config(cfg, args)
    print(f"Config: {config_path}")
    print(f"Apply: {'yes' if args.apply else 'no'}")
    print("Keys: "
          f"openrouter={'present' if str((cfg.get('openrouter') or {}).get('key', '')).strip() else 'missing'}, "
          f"abliteration={'present' if str((cfg.get('abliteration') or {}).get('key', '')).strip() else 'missing'}")
    if notes:
        print("Notes:")
        for note in notes:
            print(f"  - {note}")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("Verification: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
