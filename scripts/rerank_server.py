"""Out-of-process cross-encoder reranker speaking the llama.cpp ``/rerank`` protocol.

Point Loci at it with ``RERANK_HTTP_URL=http://127.0.0.1:<port>/rerank`` (see ``mcp/reranker.py``,
``_HttpCrossEncoder``). It exists so the reranker runs on a GPU from a venv of its own: the Loci
interpreter ships a CPU-only torch, and its 20 s rerank budget cut most searches short.

    F:\\venvs\\rerank\\Scripts\\python.exe scripts/rerank_server.py --device cuda:1 --port 8765

``--device`` takes torch's card numbering, which need not match ``nvidia-smi``'s.

Request:  POST /rerank  {"query": str, "documents": [str, ...]}
Response: {"model": str, "results": [{"index": int, "relevance_score": float}, ...]}  (one per document)
GET /health answers {"status": "ok", "model": str, "device": str}.

Binds loopback only. The service loads the model before it listens: a cold load takes ~40 s, longer
than Loci's 10 s client timeout.
"""
import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, List, Optional, Sequence

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
MAX_BODY_BYTES = 8 * 1024 * 1024

Predict = Callable[[Sequence[Sequence[str]]], Sequence[float]]


def load_predict(model_name: str, device: str) -> Predict:
    """The sentence-transformers CrossEncoder's ``predict`` on *device*, serialised behind a lock."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, device=device)
    lock = threading.Lock()  # one CUDA stream per process: serialise scoring

    def predict(pairs):
        with lock:
            return [float(s) for s in model.predict([list(p) for p in pairs])]

    return predict


class _Bad(Exception):
    pass


def parse_request(raw: bytes):
    """``(query, documents)`` from a request body; raises ``_Bad`` naming the first thing wrong."""
    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _Bad(f"body is not JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise _Bad("body must be a JSON object")
    query, docs = body.get("query"), body.get("documents")
    if not isinstance(query, str):
        raise _Bad("query must be a string")
    if not isinstance(docs, list) or not all(isinstance(d, str) for d in docs):
        raise _Bad("documents must be a list of strings")
    return query, docs


def make_handler(get_predict: Callable[[], Predict], model_name: str, device: str):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "model": model_name, "device": device})
            else:
                self._send(404, {"error": f"no route {self.path}"})

        def do_POST(self):
            if self.path != "/rerank":
                self._send(404, {"error": f"no route {self.path}"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                self._send(413, {"error": f"body over {MAX_BODY_BYTES} bytes"})
                return
            try:
                query, docs = parse_request(self.rfile.read(length))
            except _Bad as exc:
                self._send(400, {"error": str(exc)})
                return
            if not docs:
                self._send(200, {"model": model_name, "results": []})
                return
            try:
                scores = get_predict()([[query, d] for d in docs])
            except Exception as exc:  # a model/CUDA failure must be an error the client sees, not a hang
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._send(200, {"model": model_name, "results": [
                {"index": i, "relevance_score": float(s)} for i, s in enumerate(scores)]})

    return Handler


def make_server(predict: Optional[Predict], host: str, port: int, model_name: str = DEFAULT_MODEL,
                device: str = "cpu", preload: bool = False) -> ThreadingHTTPServer:
    """A server on (*host*, *port*); *predict* None loads the real model on the first request, or now with
    *preload* (a cold load takes ~40 s, longer than Loci's 10 s client timeout, so the service preloads)."""
    state: List[Predict] = []
    guard = threading.Lock()
    if predict is None and preload:
        state.append(load_predict(model_name, device))

    def get_predict() -> Predict:
        if predict is not None:
            return predict
        with guard:
            if not state:
                state.append(load_predict(model_name, device))
        return state[0]

    return ThreadingHTTPServer((host, port), make_handler(get_predict, model_name, device))


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=os.environ.get("RERANK_MODEL", DEFAULT_MODEL))
    ap.add_argument("--device", default=os.environ.get("RERANK_DEVICE", "cuda:0"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    srv = make_server(None, args.host, args.port, args.model, args.device, preload=True)
    print(f"rerank_server: {args.model} on {args.device} at http://{args.host}:{args.port}/rerank", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
