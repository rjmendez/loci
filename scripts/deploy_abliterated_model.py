#!/usr/bin/env python3
"""Opt-in deployer for heavier local Ollama generation models.

Downloads one researched GGUF quant, rewrites a checked-in Modelfile to point at it,
registers the model with `ollama create`, then does a tiny `/api/generate` smoke test
and reports Ollama residency / best-effort GPU state.

This is additive only. It does NOT change Loci's defaults; switching generation to the
new tag is a separate explicit step via LOCI_OLLAMA_GEN_MODEL or ~/.loci/backends.toml.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib import error, request


_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_MODELS_DIR = Path.home() / ".loci" / "models" / "ollama"
_RESERVE_BYTES = 2 * 1024 ** 3
_TIMEOUT = 30.0


@dataclass(frozen=True)
class ModelSpec:
    choice: str
    repo: str
    template: str
    filename_prefix: str
    tag_prefix: str
    allowed_quants: tuple[str, ...]
    default_quant: str
    prompt: str
    expected_bytes: dict[str, int]

    def filename(self, quant: str) -> str:
        return f"{self.filename_prefix}-{quant.upper()}.gguf"

    def tag(self, quant: str) -> str:
        return f"{self.tag_prefix}:{quant.lower().replace('_', '')}"


@dataclass(frozen=True)
class Plan:
    spec: ModelSpec
    quant: str
    tag: str
    template_path: Path
    gguf_path: Path
    local_modelfile: Path
    expected_bytes: int


_SPECS: dict[str, ModelSpec] = {
    "14b-coder": ModelSpec(
        choice="14b-coder",
        repo="bartowski/Qwen2.5-Coder-14B-Instruct-abliterated-GGUF",
        template="Modelfile.qwen2.5-coder-14b-abliterated",
        filename_prefix="Qwen2.5-Coder-14B-Instruct-abliterated",
        tag_prefix="loci-qwen25-coder-14b-abliterated",
        allowed_quants=("q4_k_m", "q5_k_m"),
        default_quant="q4_k_m",
        prompt="Reply with exactly: ready",
        expected_bytes={
            "q4_k_m": 8988111200,
            "q5_k_m": 10508874080,
        },
    ),
    "24b-mistral": ModelSpec(
        choice="24b-mistral",
        repo="bartowski/huihui-ai_Mistral-Small-24B-Instruct-2501-abliterated-GGUF",
        template="Modelfile.mistral-small-24b-abliterated",
        filename_prefix="huihui-ai_Mistral-Small-24B-Instruct-2501-abliterated",
        tag_prefix="loci-mistral-small-24b-abliterated",
        allowed_quants=("q5_k_m", "q6_k"),
        default_quant="q5_k_m",
        prompt="Reply with exactly: ready",
        expected_bytes={
            "q5_k_m": 16763984896,
            "q6_k": 19345939456,
        },
    ),
}


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Download and register one opt-in abliterated Ollama model for Loci.")
    ap.add_argument("model", choices=sorted(_SPECS),
                    help="14b-coder (12-16GB VRAM pick) or 24b-mistral (24GB+ VRAM pick)")
    ap.add_argument("--quant", choices=("q4_k_m", "q5_k_m", "q6_k"),
                    help="override the default researched quant for the chosen model")
    ap.add_argument("--models-dir", type=Path, default=_MODELS_DIR,
                    help=f"where GGUFs and rewritten Modelfiles live (default: {_MODELS_DIR})")
    ap.add_argument("--tag",
                    help="override the Ollama tag to create (default: researched loci-* tag)")
    ap.add_argument("--ollama-url",
                    default=(os.environ.get("LOCI_OLLAMA_GEN_URL")
                             or os.environ.get("OLLAMA_BASE_URL")
                             or os.environ.get("OLLAMA_URL")
                             or "http://localhost:11434"),
                    help="Ollama HTTP base URL for health/smoke checks (default: env -> localhost)")
    ap.add_argument("--keep-alive", default="30m",
                    help='keep_alive sent to the smoke-test generate call (default: "30m")')
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan only; do no network/process/file side effects")
    return ap


def build_plan(model: str, *, quant: str | None, models_dir: Path, tag: str | None) -> Plan:
    spec = _SPECS[model]
    chosen = (quant or spec.default_quant).lower()
    if chosen not in spec.allowed_quants:
        raise ValueError(f"{model} supports only: {', '.join(spec.allowed_quants)}")
    expected = spec.expected_bytes[chosen]
    root = models_dir.expanduser().resolve() / spec.choice / chosen
    return Plan(
        spec=spec,
        quant=chosen,
        tag=tag or spec.tag(chosen),
        template_path=_REPO / "ollama" / spec.template,
        gguf_path=root / spec.filename(chosen),
        local_modelfile=root / f"{spec.template}.local",
        expected_bytes=expected,
    )


def _render_modelfile(template_text: str, gguf_path: Path) -> str:
    lines = template_text.splitlines()
    if not lines or not lines[0].startswith("FROM "):
        raise ValueError("Modelfile template must start with FROM")
    lines[0] = f"FROM {gguf_path}"
    return "\n".join(lines) + "\n"


def _human_bytes(n: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(n)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024
    return f"{n}B"


def _require_tool(names: tuple[str, ...], label: str) -> str:
    for name in names:
        path = shutil.which(name)
        if path:
            return name
    raise RuntimeError(f"{label} not found. Install it first and re-run.")


def _json_request(method: str, url: str, payload: dict | None = None, timeout: float = _TIMEOUT) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url, data=data, headers=headers, method=method.upper())
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body or "{}")


def _ensure_ollama_ready(base_url: str) -> None:
    try:
        _json_request("GET", f"{base_url.rstrip('/')}/api/tags", timeout=10.0)
    except Exception as exc:
        raise RuntimeError(f"Ollama is not reachable at {base_url}: {exc}") from exc


def _existing_file_state(path: Path, expected_bytes: int) -> str:
    if not path.exists():
        return "missing"
    actual = path.stat().st_size
    if actual == expected_bytes:
        return "ready"
    if 0 < actual < expected_bytes:
        return "partial"
    return "mismatch"


def _disk_usage_path(path: Path) -> Path:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe


def _ensure_space(plan: Plan) -> None:
    state = _existing_file_state(plan.gguf_path, plan.expected_bytes)
    if state == "ready":
        return
    if state == "partial":
        actual = plan.gguf_path.stat().st_size
        raise RuntimeError(
            f"partial GGUF present at {plan.gguf_path} "
            f"({_human_bytes(actual)} of {_human_bytes(plan.expected_bytes)}). "
            "Refusing to continue; remove it or complete the download manually.")
    if state == "mismatch":
        actual = plan.gguf_path.stat().st_size
        raise RuntimeError(
            f"existing GGUF size mismatch at {plan.gguf_path} "
            f"({_human_bytes(actual)} on disk, expected {_human_bytes(plan.expected_bytes)}).")

    free = shutil.disk_usage(_disk_usage_path(plan.gguf_path.parent)).free
    need = plan.expected_bytes + max(_RESERVE_BYTES, plan.expected_bytes // 10)
    if free < need:
        raise RuntimeError(
            f"insufficient disk at {plan.gguf_path.parent}: need at least "
            f"{_human_bytes(need)} free before download, have {_human_bytes(free)}.")


def _download(hf_cli: str, plan: Plan) -> None:
    plan.gguf_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [hf_cli, "download", plan.spec.repo, "--include", plan.gguf_path.name,
           "--local-dir", str(plan.gguf_path.parent)]
    print("[deploy] downloading:", " ".join(cmd))
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"download failed (exit {result.returncode})")
    state = _existing_file_state(plan.gguf_path, plan.expected_bytes)
    if state != "ready":
        raise RuntimeError(f"download finished but {plan.gguf_path.name} is {state}, not complete")


def _write_modelfile(plan: Plan) -> None:
    template = plan.template_path.read_text()
    rendered = _render_modelfile(template, plan.gguf_path)
    plan.local_modelfile.write_text(rendered)


def _ollama_env(base_url: str) -> dict[str, str]:
    env = dict(os.environ)
    env["OLLAMA_HOST"] = base_url
    return env


def _create_model(plan: Plan, *, base_url: str) -> None:
    cmd = ["ollama", "create", plan.tag, "-f", str(plan.local_modelfile)]
    print("[deploy] creating:", " ".join(cmd))
    result = subprocess.run(cmd, text=True, env=_ollama_env(base_url))
    if result.returncode != 0:
        raise RuntimeError(f"ollama create failed (exit {result.returncode})")


def _smoke_test(plan: Plan, *, base_url: str, keep_alive: str) -> dict:
    body = {
        "model": plan.tag,
        "prompt": plan.spec.prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {"num_predict": 16, "temperature": 0.0},
    }
    out = _json_request("POST", f"{base_url.rstrip('/')}/api/generate", body, timeout=120.0)
    text = (out.get("response") or "").strip()
    if not text:
        raise RuntimeError("smoke test returned an empty response")
    return out


def _ollama_ps(base_url: str) -> list[dict]:
    try:
        return (_json_request("GET", f"{base_url.rstrip('/')}/api/ps", timeout=15.0)
                .get("models") or [])
    except Exception:
        return []


def _gpu_state() -> dict:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"available": False, "note": "nvidia-smi not found", "gpus": []}
    try:
        args = [
            exe,
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        p = subprocess.run(args, capture_output=True, text=True, timeout=10)
        if p.returncode != 0:
            return {"available": False, "note": "nvidia-smi returned non-zero", "gpus": []}
        gpus = []
        for line in p.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "index": parts[0], "name": parts[1], "mem_used_mib": parts[2],
                    "mem_total_mib": parts[3], "util_pct": parts[4],
                })
        return {"available": bool(gpus), "gpus": gpus}
    except Exception as exc:
        return {"available": False, "note": f"{type(exc).__name__}: {exc}", "gpus": []}


def _print_summary(plan: Plan, *, smoke: dict | None, ps: list[dict], gpu: dict) -> None:
    print(f"[deploy] tag: {plan.tag}")
    print(f"[deploy] GGUF: {plan.gguf_path} ({_human_bytes(plan.expected_bytes)})")
    if smoke is not None:
        text = (smoke.get("response") or "").strip().replace("\n", " ")
        print(f"[deploy] smoke ok: {text[:120]}")
    if ps:
        names = []
        for model in ps:
            label = model.get("name") or model.get("model") or "<unknown>"
            if model.get("size_vram"):
                label += f" [{_human_bytes(int(model['size_vram']))} VRAM]"
            names.append(label)
        print("[deploy] /api/ps resident:", ", ".join(names))
    else:
        print("[deploy] /api/ps resident: unavailable")
    if gpu.get("available"):
        for g in gpu["gpus"]:
            print(f"[deploy] GPU{g['index']} {g['name']}: "
                  f"{g['mem_used_mib']}/{g['mem_total_mib']} MiB, util {g['util_pct']}%")
    elif gpu.get("note"):
        print(f"[deploy] GPU state: {gpu['note']}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = build_plan(args.model, quant=args.quant, models_dir=args.models_dir, tag=args.tag)
        if args.dry_run:
            _print_summary(plan, smoke=None, ps=[], gpu={"available": False})
            print(f"[deploy] dry-run only. To switch Loci later: export LOCI_OLLAMA_GEN_MODEL={plan.tag}")
            return 0

        _require_tool(("ollama",), "ollama")
        hf_cli = _require_tool(("huggingface-cli", "hf"), "huggingface-cli / hf")
        _ensure_ollama_ready(args.ollama_url)
        _ensure_space(plan)
        if _existing_file_state(plan.gguf_path, plan.expected_bytes) != "ready":
            _download(hf_cli, plan)
        else:
            print(f"[deploy] reusing existing GGUF: {plan.gguf_path}")
        _write_modelfile(plan)
        _create_model(plan, base_url=args.ollama_url)
        smoke = _smoke_test(plan, base_url=args.ollama_url, keep_alive=args.keep_alive)
        ps = _ollama_ps(args.ollama_url)
        gpu = _gpu_state()
        _print_summary(plan, smoke=smoke, ps=ps, gpu=gpu)
        print(f"[deploy] switch Loci with: export LOCI_OLLAMA_GEN_MODEL={plan.tag}")
        return 0
    except (RuntimeError, ValueError, error.URLError) as exc:
        print(f"[deploy] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
