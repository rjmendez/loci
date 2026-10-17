import collections
import datetime
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mcp"))
sys.path.insert(0, str(ROOT))

import backends
import llm_local
import verify
from deep_think_loci.grounding import ground_gate
from scripts import local_deep_think as ldt


def main() -> None:
    run_started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    store: list[dict] = []
    calls: list[dict] = []

    def start_fn(**kwargs):
        return json.dumps({"status": "created", "manifest": {"id": kwargs.get("investigation_id")}})

    def store_fn(**kwargs):
        fid = str(uuid.uuid4())
        rec = {
            "id": fid,
            "text": kwargs.get("text", ""),
            "source": kwargs.get("source", ""),
            "confidence": kwargs.get("confidence", "medium"),
            "finding_type": kwargs.get("finding_type", "inferred"),
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        store.append(rec)
        return json.dumps({"stored": True, "finding_id": fid, "type": rec["finding_type"]})

    def load_fn(**_kwargs):
        return json.dumps({"findings": list(store)})

    def search_fn(*, query, collection_name, limit):
        _ = (query, collection_name, limit)
        return []

    def generate_probe(prompt, model="", fmt=None, max_tokens=256, temperature=0.2, **_kwargs):
        res = llm_local.generate(
            prompt=prompt,
            model=model,
            fmt=fmt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        calls.append(
            {
                "requested_model": model,
                "fmt": fmt,
                "max_tokens": max_tokens,
                "ok": bool(res.get("ok")) if isinstance(res, dict) else False,
                "served_model": (res or {}).get("model") if isinstance(res, dict) else None,
                "tier": (res or {}).get("tier", "ollama") if isinstance(res, dict) else "unknown",
                "why": (res or {}).get("why") if isinstance(res, dict) else "non-dict",
                "text_len": len((res or {}).get("text", "")) if isinstance(res, dict) else 0,
            }
        )
        return res

    args = ldt.parse_args(
        [
            "Assess fleet queue saturation governance controls and likely root causes from prior memory.",
            "--investigation-id",
            "validate-hybrid-deepthink-mode-20260919",
            "--title",
            "Hybrid deep-think routing validation 2026-09-19",
            "--red-team",
            "--strict-grounding",
            "--ideas-per-model",
            "2",
            "--retrieval-limit",
            "8",
            "--evidence-chars",
            "4200",
            "--no-learn-procedures",
        ]
    )
    config = ldt._resolve_models(args)
    result = ldt.run_chain(
        config,
        deps={
            "generate": generate_probe,
            "search_collection": search_fn,
            "gate": ground_gate.gate,
            "start": start_fn,
            "load": load_fn,
            "store": store_fn,
            "verify": verify.verify_finding,
        },
    )

    by_tier = collections.Counter(c.get("tier", "unknown") for c in calls)
    by_ok = collections.Counter("ok" if c.get("ok") else "not_ok" for c in calls)
    summary = {
        "run_started_utc": run_started,
        "run_finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "routing_inputs": {
            "LOCI_CONFIG": os.environ.get("LOCI_CONFIG", str(Path.home() / ".loci" / "backends.toml")),
            "OLLAMA_BASE_URL": os.environ.get("OLLAMA_BASE_URL", ""),
            "LOCI_OLLAMA_GEN_URL": os.environ.get("LOCI_OLLAMA_GEN_URL", ""),
            "OLLAMA_GEN_URL": os.environ.get("OLLAMA_GEN_URL", ""),
            "LOCI_VLLM_FALLBACK": os.environ.get("LOCI_VLLM_FALLBACK", ""),
            "QDRANT_URL": os.environ.get("QDRANT_URL", ""),
            "resolved_ollama_url": backends.ollama_url(),
            "resolved_ollama_gen_url": backends.ollama_gen_url(),
            "resolved_vllm_url": backends.vllm_url(),
            "resolved_vllm_model": backends.vllm_model(),
            "resolved_qdrant_url": backends.qdrant()[0],
        },
        "deep_think_config": {
            "collections": config.collections,
            "ideate_models": config.ideate_models,
            "verify_model": config.verify_model,
            "synthesize_model": config.synthesize_model,
            "self_reflect_model": config.self_reflect_model,
            "redteam_model": config.redteam_model,
            "strict_grounding": config.strict_grounding,
            "red_team": config.red_team,
            "retrieval_limit": config.retrieval_limit,
            "evidence_chars": config.evidence_chars,
            "ideas_per_model": config.ideas_per_model,
        },
        "generation_observed": {
            "calls": len(calls),
            "by_tier": dict(by_tier),
            "by_ok": dict(by_ok),
            "calls_detail": calls,
        },
        "result": result,
    }

    out_path = ROOT / "artifacts" / "deepthink" / "validate-hybrid-deepthink-mode-run1.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
