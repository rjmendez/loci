"""Minimal Loci MCP client for scheduled jobs: call(tool, **args) -> parsed JSON/text.
Endpoint from LOCI_MCP_URL (default http://127.0.0.1:8000/mcp); optional LOCI_MCP_TOKEN bearer."""
import asyncio, json, os, time

TIMEOUT_S = float(os.environ.get("LOCI_MCP_TIMEOUT_S", "300"))        # per-call ceiling
# Pauses between attempts when Loci refuses connections (it is restarting). Only connection-level failures are
# retried: nothing reached the server, so even a write such as investigation_store cannot be applied twice.
RETRY_DELAYS = tuple(float(x) for x in os.environ.get("LOCI_MCP_RETRY_DELAYS", "5,15,30").split(",") if x.strip())


class LociUnreachable(RuntimeError):
    """Connection refused/failed before any request was served (Loci down or restarting). Safe to retry."""


class LociTimeout(TimeoutError):
    """The call outlived its ceiling. NOT retried: the server may still be running it."""


class LociError(RuntimeError):
    """Any other failure, with the real cause named (an MCP TaskGroup ExceptionGroup says nothing by itself)."""

URL = os.environ.get("LOCI_MCP_URL", "http://127.0.0.1:8000/mcp")


def _transport(tok):
    """Context manager yielding (read, write). mcp >= 2 renamed the helper and moved headers onto an httpx client."""
    hdr = {"Authorization": f"Bearer {tok}"} if tok else None
    try:
        from mcp.client.streamable_http import streamable_http_client
    except ImportError:
        from mcp.client.streamable_http import streamablehttp_client      # mcp 1.x
        return streamablehttp_client(URL, headers=hdr)
    from mcp.shared._httpx_utils import create_mcp_http_client
    return streamable_http_client(URL, http_client=create_mcp_http_client(headers=hdr))


def _require_auth(tok):
    """Fail with a readable message instead of an MCP ExceptionGroup when the server wants a bearer token."""
    if tok:
        return
    import urllib.request, urllib.error
    req = urllib.request.Request(URL, data=b"{}", headers={"content-type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=10)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise RuntimeError("Loci MCP at %s answered 401: set LOCI_MCP_TOKEN" % URL) from None
    except OSError as e:
        raise LociUnreachable("Loci MCP at %s unreachable: %s" % (URL, e)) from None


async def _call(tool, args):
    from mcp import ClientSession
    _require_auth(os.environ.get("LOCI_MCP_TOKEN"))
    async with _transport(os.environ.get("LOCI_MCP_TOKEN")) as streams:
        async with ClientSession(streams[0], streams[1]) as s:
            await s.initialize()
            res = await s.call_tool(tool, args)
            txt = "".join(c.text for c in res.content if getattr(c, "text", None))
            try:
                out = json.loads(txt)
                return json.loads(out["result"]) if isinstance(out, dict) and isinstance(out.get("result"), str) else out
            except Exception:
                return txt


def _leaves(exc):
    if isinstance(exc, BaseExceptionGroup):
        return [x for e in exc.exceptions for x in _leaves(e)]
    return [exc]


def _is_connect_failure(leaf):
    # refused / could not connect = the request never left. A reset mid-call is NOT here: it may have been applied.
    return isinstance(leaf, (ConnectionRefusedError, LociUnreachable)) or type(leaf).__name__ in ("ConnectError", "ConnectTimeout")


def call_with(tool, args, timeout=None, delays=None, sleep=time.sleep):
    """One tool call with a per-call ceiling and bounded retries while Loci refuses connections."""
    timeout = TIMEOUT_S if timeout is None else timeout
    pauses = list(RETRY_DELAYS if delays is None else delays)
    while True:
        try:
            return asyncio.run(asyncio.wait_for(_call(tool, args), timeout))
        except LociUnreachable:
            if not pauses:
                raise
        except BaseExceptionGroup as eg:
            leaves = _leaves(eg)
            if not any(_is_connect_failure(x) for x in leaves):
                raise LociError("%s: %s: %s" % (tool, type(leaves[0]).__name__, leaves[0])) from eg
            if not pauses:
                raise LociUnreachable("Loci MCP at %s unreachable: %s: %s" % (URL, type(leaves[0]).__name__, leaves[0])) from eg
        except TimeoutError:
            raise LociTimeout("%s exceeded %g s" % (tool, timeout)) from None
        sleep(pauses.pop(0))


def call(tool, **args):
    return call_with(tool, args)


if __name__ == "__main__":
    import sys
    print(json.dumps(call(sys.argv[1], **json.loads(sys.argv[2] if len(sys.argv) > 2 else "{}")), indent=1)[:1500])
