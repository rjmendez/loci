"""Transport auth gate (_BearerAuthMiddleware): the one module that authenticates every HTTP tool call and
binds per-agent identity into the ASGI scope. It was only exercised indirectly before.

Mutation runs to prove the negatives bite (each reverted after):
  - `_match`: `hmac.compare_digest(presented, self._token)` -> `presented == self._token or True` ->
    test_wrong_token_is_401 and test_no_auth_header_is_401 FAILED (everything authenticated). Restored.
  - `__call__`: removed the `/health` exempt check -> test_health_is_exempt FAILED (401 on /health). Restored.
  - `__call__`: dropped `scope[caller_identity.SCOPE_KEY] = agent` -> test_agent_token_binds_identity FAILED.

Run: pytest mcp/tests/test_bearer_auth_middleware.py -v
"""
import asyncio
import sys
import unittest
from pathlib import Path

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import caller_identity  # noqa: E402
import server  # noqa: E402


def _run(mw, path="/mcp", auth=None):
    """Drive the ASGI middleware once; return (status, captured_scope_or_None).
    captured_scope is the scope the downstream app saw (None if the request was rejected)."""
    headers = []
    if auth is not None:
        headers.append((b"authorization", auth.encode("latin-1")))
    scope = {"type": "http", "path": path, "headers": headers}
    seen = {}

    async def downstream(s, receive, send):
        seen["scope"] = s
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    captured = {"status": None}

    async def send(ev):
        if ev["type"] == "http.response.start":
            captured["status"] = ev["status"]

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    mw._app = downstream
    asyncio.run(mw.__call__(scope, receive, send))
    return captured["status"], seen.get("scope")


class BearerAuthMiddlewareTest(unittest.TestCase):
    def _mw(self):
        return server._BearerAuthMiddleware(
            app=None, token="shared-secret-123",
            agent_tokens={"agent-tok-abc": "hermes"},
        )

    def test_shared_token_authenticates_but_binds_no_agent(self) -> None:
        status, scope = _run(self._mw(), auth="Bearer shared-secret-123")
        self.assertEqual(status, 200)
        self.assertNotIn(caller_identity.SCOPE_KEY, scope)   # shared token cannot tell callers apart

    def test_agent_token_binds_identity(self) -> None:
        status, scope = _run(self._mw(), auth="Bearer agent-tok-abc")
        self.assertEqual(status, 200)
        self.assertEqual(scope.get(caller_identity.SCOPE_KEY), "hermes")

    def test_wrong_token_is_401(self) -> None:
        status, scope = _run(self._mw(), auth="Bearer not-the-secret")
        self.assertEqual(status, 401)
        self.assertIsNone(scope)

    def test_no_auth_header_is_401(self) -> None:
        status, scope = _run(self._mw(), auth=None)
        self.assertEqual(status, 401)
        self.assertIsNone(scope)

    def test_bearer_prefix_is_case_insensitive(self) -> None:
        status, _ = _run(self._mw(), auth="bearer shared-secret-123")
        self.assertEqual(status, 200)

    def test_health_is_exempt(self) -> None:
        # /health passes with no token at all
        status, scope = _run(self._mw(), path="/health", auth=None)
        self.assertEqual(status, 200)
        self.assertIsNotNone(scope)


if __name__ == "__main__":
    unittest.main()
