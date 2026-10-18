"""Lightweight LiteLLM proxy client for Loci MCP."""
import json, os, time, urllib.request, urllib.error
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

LITELLM_PROXY_URL = os.getenv("LOCI_LITELLM_PROXY_URL", "http://localhost:4000")
LITELLM_API_KEY = os.getenv("LOCI_LITELLM_API_KEY", "")
LITELLM_TIMEOUT_S = float(os.getenv("LOCI_LITELLM_TIMEOUT_S", "30"))

@dataclass
class LiteLLMResponse:
    model: str; content: str; prompt_tokens: int; completion_tokens: int; total_tokens: int; latency_ms: float; raw: Dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> Dict[str, Any]:
        return {"model": self.model, "content": self.content, "usage": {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens, "total_tokens": self.total_tokens}, "latency_ms": round(self.latency_ms, 1)}

def call_litellm_proxy(model: str, messages: List[Dict[str, str]], max_tokens: int = 1024, temperature: float = 0.0) -> LiteLLMResponse:
    url = f"{LITELLM_PROXY_URL.rstrip('/')}/v1/chat/completions"
    payload = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}).encode("utf-8")
    headers = {"Content-Type": "application/json", "Content-Length": str(len(payload))}
    if LITELLM_API_KEY:
        headers["Authorization"] = f"Bearer {LITELLM_API_KEY}"
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=LITELLM_TIMEOUT_S) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise ValueError(f"LiteLLM proxy error {e.code}: {body}") from e
    latency_ms = (time.monotonic() - t0) * 1000
    choice = raw["choices"][0]["message"]["content"]
    usage = raw.get("usage", {})
    return LiteLLMResponse(model=raw.get("model", model), content=choice, prompt_tokens=usage.get("prompt_tokens", 0), completion_tokens=usage.get("completion_tokens", 0), total_tokens=usage.get("total_tokens", 0), latency_ms=latency_ms, raw=raw)
