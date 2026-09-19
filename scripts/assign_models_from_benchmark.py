#!/usr/bin/env python3
"""Derive local role-model assignments from benchmark output + installed tags.

This script does not modify runtime defaults in code. It emits a backends.toml
snippet you can copy into ~/.loci/backends.toml after running
scripts/bench_model_catalog_quality.py.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ROLE_TO_FIELD = {
    "synthesis": "gen_model",
    "escalation": "verify_model",
    "cheap_fanout": "compress_model",
    "guardian": "guardian_model",
}


def parse_ollama_list(output: str) -> set[str]:
    tags: set[str] = set()
    for line in output.splitlines():
        s = line.strip()
        if not s or s.startswith("NAME "):
            continue
        tag = s.split()[0]
        if tag:
            tags.add(tag)
    return tags


def installed_tags() -> set[str]:
    p = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False, timeout=5)
    if p.returncode != 0:
        return set()
    return parse_ollama_list(p.stdout or "")


def benchmark_winners(payload: dict[str, Any]) -> dict[str, list[str]]:
    winners: dict[str, list[str]] = {}
    for row in payload.get("summary", []):
        role = row.get("role")
        if not isinstance(role, str):
            continue
        row_winners = row.get("quality_winners")
        if isinstance(row_winners, list):
            winners.setdefault(role, [])
            for model in row_winners:
                if isinstance(model, str) and model not in winners[role]:
                    winners[role].append(model)
    return winners


def choose_first_available(candidates: list[str], installed: set[str]) -> tuple[str, bool]:
    for model in candidates:
        if model in installed:
            return model, True
    if candidates:
        return candidates[0], False
    return "", False


def choose_redteam(winners: dict[str, list[str]], installed: set[str]) -> tuple[str, bool]:
    preferred_roles = ("synthesis", "escalation", "cheap_fanout")
    ordered: list[str] = []
    for role in preferred_roles:
        for model in winners.get(role, []):
            if model not in ordered:
                ordered.append(model)
    for model in ordered:
        lower = model.lower()
        if ("heretic" in lower or "abliterated" in lower) and model in installed:
            return model, True
    for model in sorted(installed):
        lower = model.lower()
        if "heretic" in lower or "abliterated" in lower:
            return model, True
    default = "hf.co/slevinw/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF:Q4_K_M"
    return default, default in installed


def render_toml(assignments: dict[str, str]) -> str:
    lines = ["[ollama]"]
    for key in ("gen_model", "verify_model", "compress_model", "guardian_model", "redteam_model"):
        value = assignments.get(key, "")
        if value:
            lines.append(f'{key} = "{value}"')
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark-json", required=True, help="Path to bench_model_catalog_quality JSON output")
    p.add_argument("--ollama-list-file", help="Optional path containing `ollama list` output (for offline use)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(Path(args.benchmark_json).read_text(encoding="utf-8"))
    winners = benchmark_winners(payload)
    installed = (
        parse_ollama_list(Path(args.ollama_list_file).read_text(encoding="utf-8"))
        if args.ollama_list_file
        else installed_tags()
    )

    assignments: dict[str, str] = {}
    warnings: list[str] = []
    for role, field in ROLE_TO_FIELD.items():
        model, present = choose_first_available(winners.get(role, []), installed)
        if model:
            assignments[field] = model
        if model and not present:
            warnings.append(f"{field}: selected benchmark winner '{model}' but tag is not installed locally")
        if not model:
            warnings.append(f"{field}: no winner found for role '{role}' in benchmark summary")

    redteam_model, redteam_present = choose_redteam(winners, installed)
    assignments["redteam_model"] = redteam_model
    if not redteam_present:
        warnings.append(f"redteam_model: '{redteam_model}' is not installed locally")

    print(render_toml(assignments).rstrip())
    if warnings:
        print("\n# Warnings")
        for warning in warnings:
            print(f"# - {warning}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
