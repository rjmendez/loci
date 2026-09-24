"""Session-wide isolation for Loci's pytest suites.

Why this exists: mcp/server.py and its backends resolve every store from the
environment -- HOME, LOCI_MEMORY_DIR, ~/.loci/backends.toml, the repo .env files
(mcp/.env with override=True) -- and most of that happens at *import* time. A
test run on the machine in daily use therefore read and wrote the live
investigation store, the live Qdrant collection and ~/.hermes, while the same
run on a clean CI runner found nothing and passed. The memcheck audit-log
incident recorded in mcp/tests/conftest.py is the worked example.

``install()`` is called from a conftest at import time -- before pytest imports
any test module, and so before ``server`` is imported -- and does three things:

1. Points every store at a fresh temp root: HOME, HERMES_HOME, LOCI_CONFIG,
   LOCI_MEMORY_DIR, MNEMOSYNE_DATA_DIR; backend URLs at an unreachable port;
   API keys removed; any inherited variable whose value lies under the real
   ~/.loci or ~/.hermes removed.
2. Stops ``dotenv.load_dotenv`` from reading this checkout's own .env files,
   which would otherwise put the live QDRANT_URL/QDRANT_API_KEY back.
   Explicit paths (a test's tmp_path) still load.
3. Installs an audit hook that refuses -- and records -- any open, sqlite
   connect, listdir, mkdir, rename, remove or rmtree under the real ~/.loci or
   ~/.hermes. The refusal raises ``LiveStoreAccessError`` (a PermissionError, so
   fail-open code degrades rather than crashes), and because fail-open code may
   swallow it, the conftest also fails the test from the record.

Opt-out, for an intentional live smoke test only: ``LOCI_TESTS_LIVE=1``.
Nothing else disables it.
"""
from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

OPT_OUT_VAR = "LOCI_TESTS_LIVE"
UNREACHABLE = "http://127.0.0.1:1"

REPO_ROOT = Path(__file__).resolve().parent.parent

# Endpoints whose unset default is a localhost probe: pin them to a port nothing
# listens on, so "unset" cannot quietly resolve to the developer's live service.
# Only the base variable of each chain is pinned; the higher-precedence overrides
# (OLLAMA_GEN_URL, LOCI_OLLAMA_GEN_URL, OLLAMA_URL) are removed below, so a test
# that sets OLLAMA_BASE_URL to its own stub server is still the one that wins.
_UNREACHABLE_URLS = (
    "QDRANT_URL",
    "OLLAMA_BASE_URL",
    "LOCI_LOCAL_OLLAMA",
    "VLLM_BASE_URL",
    "LOCI_LOCAL_VLLM",
)

# Endpoints whose unset value means "feature off", and credentials. Removed
# rather than pinned, so the code under test sees the same thing as on CI.
_REMOVED = (
    "OLLAMA_URL",
    "OLLAMA_GEN_URL",
    "LOCI_OLLAMA_GEN_URL",
    "QDRANT_API_KEY",
    "QDRANT_KEY",
    "EMBED_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ABLITERATION_API_KEY",
    "ABLITERATION_BASE_URL",
    "GITHUB_COPILOT_OAUTH_TOKEN",
    "LOCI_MCP_TOKEN",
    "HERMES_MCP_TOKEN",
    "HERMES_MEMORY_DIR",
    "LOCI_MEMORY_MD_DIR",
    "LOCI_VLLM_FALLBACK",
    "LOCI_EVENT_LOG",
    "RERANK_HTTP_URL",
    "MNEMOSYNE_LLM_BASE_URL",
    "MNEMOSYNE_EMBEDDING_API_URL",
    "EMBED_WORKER_URL",
    "HERMES_A2A_URL",
    "MRPINK_A2A_URL",
)

_SEARCH_PATH_VARS = {"PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "MANPATH", "VIRTUAL_ENV"}

# Audit events whose arguments carry filesystem paths (positions to check).
_PATH_EVENTS = {
    "open": (0,),
    "sqlite3.connect": (0,),
    "os.listdir": (0,),
    "os.scandir": (0,),
    "os.mkdir": (0,),
    "os.remove": (0,),
    "os.rmdir": (0,),
    "os.rename": (0, 1),
    "os.symlink": (0, 1),
    "os.link": (0, 1),
    "os.truncate": (0,),
    "os.chmod": (0,),
    "os.utime": (0,),
    "shutil.rmtree": (0,),
    "shutil.copyfile": (0, 1),
    "shutil.move": (0, 1),
    "glob.glob": (0,),
}


class LiveStoreAccessError(PermissionError):
    """A test touched the real ~/.loci or ~/.hermes."""


_state: dict = {"installed": False, "root": None, "forbidden": (), "violations": []}


def _real_homes() -> set[str]:
    homes = set()
    for var in ("HOME", "USERPROFILE"):
        value = os.environ.get(var)
        if value:
            homes.add(value)
    try:
        import pwd

        homes.add(pwd.getpwuid(os.getuid()).pw_dir)
    except Exception:
        pass
    try:
        homes.add(str(Path.home()))
    except Exception:
        pass
    return {h for h in homes if h and os.path.abspath(h) != os.path.abspath(os.sep)}


def _forbidden_roots() -> tuple[str, ...]:
    roots = set()
    for home in _real_homes():
        for name in (".loci", ".hermes"):
            candidate = os.path.join(home, name)
            roots.add(os.path.abspath(candidate))
            roots.add(os.path.realpath(candidate))
    return tuple(sorted(roots))


def is_forbidden(path) -> bool:
    """True when ``path`` lies under the real ~/.loci or ~/.hermes."""
    if isinstance(path, int) or path is None:
        return False
    try:
        text = os.fsdecode(os.fspath(path))
    except TypeError:
        return False
    if not text or text == ":memory:" or text.startswith("file:"):
        return False
    absolute = os.path.abspath(os.path.expanduser(text)) if text.startswith("~") else os.path.abspath(text)
    for root in _state["forbidden"]:
        if absolute == root or absolute.startswith(root + os.sep):
            return True
    return False


def _audit(event: str, args: tuple) -> None:
    positions = _PATH_EVENTS.get(event)
    if positions is None or not _state["installed"]:
        return
    for pos in positions:
        if pos < len(args) and is_forbidden(args[pos]):
            message = f"{event}({args[pos]!r}) reaches a live Loci store"
            _state["violations"].append(message)
            raise LiveStoreAccessError(message)


def _guard_dotenv() -> None:
    """Keep this checkout's .env files out of the test process."""
    try:
        import dotenv
        import dotenv.main
    except Exception:
        return
    original = dotenv.main.load_dotenv
    if getattr(original, "_loci_hermetic", False):
        return
    blocked = {
        os.path.realpath(REPO_ROOT / ".env"),
        os.path.realpath(REPO_ROOT / "mcp" / ".env"),
    }

    def load_dotenv(dotenv_path=None, *args, **kwargs):
        # No explicit path means find_dotenv(), which walks up from the caller
        # and would land on the repo .env: refuse that as well.
        if dotenv_path is None and kwargs.get("stream") is None and not args:
            return False
        if dotenv_path is not None and os.path.realpath(os.fspath(dotenv_path)) in blocked:
            return False
        return original(dotenv_path, *args, **kwargs)

    load_dotenv._loci_hermetic = True
    dotenv.load_dotenv = load_dotenv
    dotenv.main.load_dotenv = load_dotenv


def install() -> dict:
    """Isolate this process. Idempotent; a no-op under LOCI_TESTS_LIVE=1."""
    if _state["installed"] or os.environ.get(OPT_OUT_VAR) == "1":
        return _state

    too_late = [m for m in ("server", "backends", "qdrant_ops", "inv_store") if m in sys.modules]
    if too_late:
        raise RuntimeError(
            f"loci_hermetic.install() ran after {too_late} were imported; those modules "
            "read the environment at import time, so isolation would not apply to them."
        )

    forbidden = _forbidden_roots()
    root = Path(tempfile.mkdtemp(prefix="loci-tests-"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)

    # An inherited variable that names a live store (LOCI_EVENT_LOG=~/.hermes/...,
    # an exported LOCI_MEMORY_DIR) goes. Search-path variables are left alone:
    # dropping PATH to scrub one entry would break far more than it protects.
    _state["forbidden"] = forbidden
    for key, value in list(os.environ.items()):
        if key in _SEARCH_PATH_VARS or not value:
            continue
        if is_forbidden(value):
            del os.environ[key]
    _state["forbidden"] = ()
    for key in _REMOVED:
        os.environ.pop(key, None)

    home = root / "home"
    home.mkdir()
    os.environ.update({
        "HOME": str(home),
        "USERPROFILE": str(home),
        "HERMES_HOME": str(home / ".hermes"),
        "LOCI_CONFIG": str(root / "no-such-backends.toml"),
        "LOCI_MEMORY_DIR": str(root / "memory-sessions"),
        "MNEMOSYNE_DATA_DIR": str(root / "mnemosyne"),
    })
    for key in _UNREACHABLE_URLS:
        os.environ[key] = UNREACHABLE

    _guard_dotenv()
    _state.update(root=root, forbidden=forbidden, installed=True)
    sys.addaudithook(_audit)
    return _state


def root() -> "Path | None":
    return _state["root"]


def forbidden_roots() -> tuple[str, ...]:
    return _state["forbidden"]


def installed() -> bool:
    return _state["installed"]


def violation_count() -> int:
    return len(_state["violations"])


def take_violations(since: int = 0) -> list[str]:
    """Return and clear the recorded live-store accesses from index ``since`` on."""
    out = _state["violations"][since:]
    del _state["violations"][since:]
    return out


def report_header() -> str:
    if not _state["installed"]:
        return f"loci-hermetic: DISABLED ({OPT_OUT_VAR}=1) -- tests may touch live stores"
    return f"loci-hermetic: isolated under {_state['root']}"


def pytest_hooks() -> dict:
    """Conftest members that turn a recorded access into a failure.

    Use as ``globals().update(loci_hermetic.pytest_hooks())`` in a conftest.
    The audit hook already refused the access, but most store code is fail-open
    and would turn that refusal into a quiet degraded result -- a pass -- so the
    test that caused it is failed from the record instead.
    """
    import pytest

    @pytest.fixture(autouse=True)
    def _fail_on_live_store_access():
        # Also puts os.environ back as the test found it. Tests that set or pop
        # variables directly (not via monkeypatch) leaked into every later test:
        # one popped LOCI_MEMORY_DIR, another left QDRANT_API_KEY behind, and the
        # isolation above was silently undone for the rest of the session.
        start = violation_count()
        environ = dict(os.environ)
        yield
        if dict(os.environ) != environ:
            os.environ.clear()
            os.environ.update(environ)
        violations = take_violations(start)
        if violations:
            pytest.fail("test reached a live Loci store:\n  " + "\n  ".join(violations),
                        pytrace=False)

    def pytest_report_header(config):  # noqa: ARG001
        return report_header()

    def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
        # Accesses outside any test: collection, module import, session fixtures.
        violations = take_violations()
        if violations:
            sys.stderr.write("\nloci-hermetic: live-store access outside a test:\n  "
                             + "\n  ".join(violations) + "\n")
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    return {
        "_fail_on_live_store_access": _fail_on_live_store_access,
        "pytest_report_header": pytest_report_header,
        "pytest_sessionfinish": pytest_sessionfinish,
    }
