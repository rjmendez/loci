#!/usr/bin/env python3
"""Live supervisor demo with fabricated source data and a real Ollama supervisor."""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.swarm_supervisor import make_loci_evidence_fn, plan_source_routing, supervise_and_correct  # noqa: E402

OLLAMA_URL = "http://100.73.200.19:11434"
DEFAULT_MODEL = os.environ.get("LOCI_SWARM_SUPERVISOR_MODEL", "llama3.1-agent:latest")


def ollama_gen(*, prompt: str, model: str = DEFAULT_MODEL, fmt: str = "json", max_tokens: int = 1400,
               temperature: float = 0.1, **_: Any) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json" if fmt == "json" else fmt,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"ok": False, "text": "", "error": str(exc)}
    return {"ok": True, "text": data.get("response", ""), "model": data.get("model", model)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Demonstrate closed-loop swarm supervision.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    gen_transcript: list[dict[str, Any]] = []

    def gen_fn(**kwargs: Any) -> dict[str, Any]:
        result = ollama_gen(model=args.model, **kwargs)
        gen_transcript.append({
            "prompt_excerpt": str(kwargs.get("prompt") or "")[:800],
            "response": result,
        })
        return result

    task = (
        "Identify all spider species observed within 5 miles of PLACEHOLDER_LOCATION: "
        "Example NC. Use only stubbed source data below; do not infer the user's real GPS location."
    )
    sources = [
        {"name": "DuckDuckGo", "capability": "general web search; recent/local mentions; no structured GPS filtering or photo-verification"},
        {"name": "Wikipedia", "capability": "general encyclopedic taxonomy/background; no local observation records or GPS filtering"},
        {"name": "iNaturalist", "capability": "GPS-filterable, date-filterable, photo-verified species observations near a location"},
    ]
    fabricated_source_data = {
        "DuckDuckGo": ["Local nature blog mentions orb-weaver spiders in Example County, but gives no GPS/photo observation record."],
        "Wikipedia": ["Argiope aurantia and Phidippus audax are common North American spiders; page is not evidence of local observation."],
        "iNaturalist": [
            "Observation INAT-EX-101: Argiope aurantia, photo=yes, research_grade=yes, 2.1 miles from PLACEHOLDER_LOCATION: Example NC.",
            "Observation INAT-EX-102: Phidippus audax, photo=yes, research_grade=yes, 3.4 miles from PLACEHOLDER_LOCATION: Example NC.",
        ],
    }
    findings = [{
        "source": "Wikipedia",
        "sub_need": "GPS/photo-verified local spider observations",
        "claim": "Argiope aurantia is confirmed within 5 miles because Wikipedia says it is common in North America.",
        "evidence": "Wikipedia background says it is common in North America; no local GPS/photo observation is cited.",
    }]

    def loci_verify_fn(*, claim: str, context: str, **_: Any) -> dict[str, Any]:
        _ = claim
        source = "iNaturalist" if "INAT-EX-" in context else "Wikipedia"
        supported = source == "iNaturalist" and "INAT-EX-101" in context
        rationale = (
            "stubbed iNaturalist evidence contains GPS/photo observation IDs"
            if supported
            else "stubbed evidence has no GPS/photo observation IDs; Wikipedia is background only"
        )
        return {"supported": supported, "rationale": rationale}

    evidence_fn = make_loci_evidence_fn(loci_verify_fn)

    def worker_fn(**kwargs: Any) -> dict[str, Any]:
        return {
            "source": "iNaturalist",
            "sub_need": kwargs["finding"]["sub_need"],
            "claim": "Argiope aurantia and Phidippus audax have photo-verified observations within 5 miles of PLACEHOLDER_LOCATION: Example NC.",
            "evidence": "INAT-EX-101 and INAT-EX-102 are research-grade, photo=yes, 2.1 and 3.4 miles from PLACEHOLDER_LOCATION: Example NC.",
            "correction_context": {"supervisor_guidance": kwargs["guidance"], "routing_node": kwargs["routing_node"]},
        }

    plan = plan_source_routing(task, sources, gen_fn)
    correction = supervise_and_correct(task, findings, plan, gen_fn, worker_fn, max_rounds=2, evidence_fn=evidence_fn)
    transcript = {
        "demo": "closed-loop swarm_supervisor spider evidence correction",
        "ollama_url": OLLAMA_URL,
        "model": args.model,
        "location_notice": "PLACEHOLDER_LOCATION: Example NC; fabricated source data; no user GPS used.",
        "task": task,
        "available_sources": sources,
        "fabricated_source_data": fabricated_source_data,
        "routing_plan": plan,
        "initial_worker_findings": findings,
        "closed_loop_result": correction,
        "live_supervisor_calls": gen_transcript,
    }
    print(json.dumps(transcript, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
