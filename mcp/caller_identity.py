"""Who is calling: identity bound by the transport, not claimed by the caller.

Investigation ACLs used to trust ``requesting_agent_id``, a tool argument any
caller can set to any value. That makes the ACL advisory: a caller that names a
member is let in. This module supplies the identity the *transport* vouches for,
and the ACL check in ``inv_store._acl_access_denied`` treats a caller-supplied
id as a narrowing filter on top of it, never as a way in.

Where the identity comes from, per transport:

* **MCP streamable-http / SSE with per-agent tokens.** ``LOCI_MCP_AGENT_TOKENS``
  maps agent ids to bearer tokens. ``_BearerAuthMiddleware`` (server.py) matches
  the presented token and writes the agent id into the ASGI scope under
  :data:`SCOPE_KEY`. The MCP SDK hands that Starlette request to the tool call
  (``request_ctx``), and :func:`bound_agent_id` reads it back. A caller cannot
  set a scope key; only a token it holds can.
* **A2A.** The A2A server binds a bootstrap session token to the agent id it was
  issued for. It calls :func:`bound` around work done for that sender.
* **Tests and in-process callers** use :func:`bound` directly.

Where no identity can be bound, and what happens instead:

* **stdio MCP.** There is no authentication channel at all; the server process
  was spawned by its one client. The caller *is* the process identity,
  ``HERMES_AGENT_ID``, and that is what the ACL checks.
* **streamable-http / SSE with only the shared ``LOCI_MCP_TOKEN``, or on
  loopback with no token.** Every caller holds the same secret (or none), so the
  transport cannot tell them apart. The process identity is used, exactly as for
  stdio. Agents that need to be told apart must each get their own token.

In every case ``requesting_agent_id`` can only narrow: access requires both the
bound (or process) identity *and* the named agent to be allowed.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import os
from typing import Iterator, Optional


#: ASGI scope key the HTTP auth middleware sets for a per-agent token.
SCOPE_KEY = "loci.bound_agent_id"

_BOUND: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "loci_bound_agent_id", default=None
)


@contextlib.contextmanager
def bound(agent_id: Optional[str]) -> Iterator[None]:
    """Bind ``agent_id`` as the authenticated caller for the enclosed work."""
    token = _BOUND.set(str(agent_id) if agent_id else None)
    try:
        yield
    finally:
        _BOUND.reset(token)


def _from_mcp_request() -> Optional[str]:
    try:
        from mcp.server.lowlevel.server import request_ctx
    except Exception:  # SDK layout changed or not importable: no binding
        return None
    try:
        ctx = request_ctx.get()
    except LookupError:
        return None
    request = getattr(ctx, "request", None)
    scope = getattr(request, "scope", None)
    if isinstance(scope, dict):
        value = scope.get(SCOPE_KEY)
        if value:
            return str(value)
    return None


def bound_agent_id() -> Optional[str]:
    """The transport-authenticated agent id for this call, or None."""
    value = _BOUND.get()
    if value:
        return value
    return _from_mcp_request()


def process_agent_id() -> str:
    """This server process's own agent id (``HERMES_AGENT_ID``)."""
    return os.environ.get("HERMES_AGENT_ID", "")


def load_agent_tokens(environ: Optional[dict] = None) -> dict[str, str]:
    """Return ``{token: agent_id}`` from ``LOCI_MCP_AGENT_TOKENS[_FILE]``.

    ``LOCI_MCP_AGENT_TOKENS`` is either a JSON object ``{"agent": "token"}`` or
    ``agent=token`` pairs separated by commas. ``LOCI_MCP_AGENT_TOKENS_FILE``
    names a JSON file in the object form (preferred: keeps secrets out of the
    unit file). Malformed input raises ``ValueError`` so the server refuses to
    start rather than silently serving with no per-agent binding.
    """
    env = os.environ if environ is None else environ
    mapping: dict[str, str] = {}
    raw_file = str(env.get("LOCI_MCP_AGENT_TOKENS_FILE", "") or "").strip()
    raw = str(env.get("LOCI_MCP_AGENT_TOKENS", "") or "").strip()
    sources = []
    if raw_file:
        with open(os.path.expanduser(raw_file)) as fh:
            sources.append(fh.read())
    if raw:
        sources.append(raw)
    for text in sources:
        text = text.strip()
        if not text:
            continue
        if text.startswith("{"):
            obj = json.loads(text)
            if not isinstance(obj, dict):
                raise ValueError("LOCI_MCP_AGENT_TOKENS must be a JSON object of agent -> token")
            pairs = obj.items()
        else:
            pairs = []
            for part in text.split(","):
                part = part.strip()
                if not part:
                    continue
                if "=" not in part:
                    raise ValueError("LOCI_MCP_AGENT_TOKENS has an entry with no '=token' (entry not echoed: it may be a secret)")
                agent, tok = part.split("=", 1)
                pairs.append((agent, tok))
        for agent, tok in pairs:
            agent, tok = str(agent).strip(), str(tok).strip()
            if not agent or not tok:
                raise ValueError("LOCI_MCP_AGENT_TOKENS has an empty agent id or token")
            if tok in mapping and mapping[tok] != agent:
                raise ValueError("LOCI_MCP_AGENT_TOKENS maps one token to two agents")
            mapping[tok] = agent
    return mapping
