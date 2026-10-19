"""The LadybugDB store must not read "available" when no database can be opened.

The 2026-10-18 honesty investigation found, live: ``loci_health`` said ``ladybug: available`` / ``status: ok`` while
``code_graph_ingest`` parsed 14 files and stored nothing (``ingested: {}``) and every query returned zero rows. Three
things lied at once: ``available()`` only meant "the module imported", ``readable_probe()`` answered True whenever the
database file did not exist (which it never did, because no open ever succeeded), and the open error was logged at
DEBUG. On Windows the cause is that LadybugDB's native module links ``libssl-3-x64.dll`` / ``libcrypto-3-x64.dll`` while
CPython ships ``libssl-3.dll`` / ``libcrypto-3.dll``.

The engine is replaced by a recording fake whose ``Database`` raises the real error text, so the failure is reproduced
on any machine, and every failing case has a working-engine twin.

Mutation runs (each applied to the named file, confirmed red, reverted):
- ladybug_store.py ``available``: back to ``return bool(self.ok)`` -> test_a_store_whose_engine_cannot_open_is_not_available_and_says_why red
- ladybug_store.py ``readable_probe``: missing file back to ``return True`` -> the same test red
- ladybug_store.py ``engine_probe``: swallow the error (ok stays True) -> the same test red
- ladybug_store.py ENGINE_FAIL_TTL_S 20 -> 21 -> test_a_failing_engine_is_rechecked_after_exactly_its_ttl red
- ladybug_store.py ENGINE_OK_TTL_S 300 -> 301 -> test_a_working_engine_is_rechecked_after_exactly_its_ttl red
- ladybug_store.py engine_probe: probe the real db_path instead of a temp dir -> test_the_probe_never_touches_the_real_database_path red
- ladybug_store.py _windows_ssl_aliases: skip the copy -> test_the_openssl_aliases_are_copied_under_the_x64_names red
- ladybug_store.py _windows_ssl_aliases: drop the platform guard -> test_nothing_is_done_off_windows red
- graph_tools.py: remove the empty-ingest guard -> test_ingest_that_stored_nothing_is_an_error red
"""
import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph import ladybug_store as K  # noqa: E402

DLL_ERROR = "DLL load failed while importing _lbug: The specified module could not be found."


def make_engine(fail=None):
    """A recording stand-in for the ``ladybug`` module. ``fail`` is the RuntimeError text Database raises, or None."""
    calls = []
    state = {"fail": fail}

    class Database:
        def __init__(self, path, read_only=False):
            calls.append(("open", str(path)))
            if state["fail"]:
                raise RuntimeError(state["fail"])
            self.path = path

        def close(self):
            calls.append(("close_db", str(self.path)))

    class Connection:
        def __init__(self, db):
            self.db = db

        def execute(self, query):
            calls.append(("exec", query))

        def close(self):
            calls.append(("close_conn", None))

    return types.SimpleNamespace(Database=Database, Connection=Connection), calls, state


@pytest.fixture
def engine(monkeypatch):
    fake, calls, state = make_engine(DLL_ERROR)
    monkeypatch.setattr(K, "ladybug", fake)
    monkeypatch.setattr(K, "_HAS_LADYBUG", True)
    return calls, state


def opens(calls):
    return [c for c in calls if c[0] == "open"]


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_a_store_whose_engine_cannot_open_is_not_available_and_says_why(tmp_path, engine):
    store = K.LadybugStore(str(tmp_path / "graph.ladybug"))
    assert not os.path.exists(store.db_path)                      # the state the old readable_probe called fine
    assert store.available() is False
    assert store.readable_probe() is False
    assert store.last_open_error == f"RuntimeError({DLL_ERROR!r})"


def test_a_working_engine_is_available_and_reports_no_error(tmp_path, monkeypatch):
    fake, calls, _ = make_engine(None)
    monkeypatch.setattr(K, "ladybug", fake)
    monkeypatch.setattr(K, "_HAS_LADYBUG", True)
    store = K.LadybugStore(str(tmp_path / "graph.ladybug"))
    assert store.available() is True
    assert store.readable_probe() is True                         # fresh store, no file yet: fine because the engine works
    assert store.last_open_error == ""
    assert [c for c in calls if c[0] == "exec"] == [("exec", "RETURN 1")]


def test_a_store_without_the_module_is_unavailable_with_a_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(K, "ladybug", None)
    monkeypatch.setattr(K, "_HAS_LADYBUG", False)
    store = K.LadybugStore(str(tmp_path / "graph.ladybug"))
    assert store.engine_probe() == (False, "ladybug not importable")
    assert store.available() is False


def test_a_failing_engine_is_rechecked_after_exactly_its_ttl(tmp_path, engine):
    calls, state = engine
    store = K.LadybugStore(str(tmp_path / "graph.ladybug"))
    clock = FakeClock()
    store._clock = clock
    assert store.available() is False and len(opens(calls)) == 1
    clock.t += 19.9
    assert store.available() is False and len(opens(calls)) == 1     # cached: no second open yet
    clock.t += 0.1                                                    # 20.0 s after the probe
    state["fail"] = None                                              # the DLL problem was fixed meanwhile
    assert store.available() is True and len(opens(calls)) == 2
    assert store.last_open_error == ""


def test_a_working_engine_is_rechecked_after_exactly_its_ttl(tmp_path, monkeypatch):
    fake, calls, state = make_engine(None)
    monkeypatch.setattr(K, "ladybug", fake)
    monkeypatch.setattr(K, "_HAS_LADYBUG", True)
    store = K.LadybugStore(str(tmp_path / "graph.ladybug"))
    clock = FakeClock()
    store._clock = clock
    assert store.available() is True and len(opens(calls)) == 1
    clock.t += 299.9
    assert store.available() is True and len(opens(calls)) == 1
    clock.t += 0.1                                                    # 300.0 s after the probe
    state["fail"] = DLL_ERROR                                         # the engine broke since
    assert store.available() is False and len(opens(calls)) == 2


def test_the_probe_never_touches_the_real_database_path(tmp_path, monkeypatch):
    fake, calls, _ = make_engine(None)
    monkeypatch.setattr(K, "ladybug", fake)
    monkeypatch.setattr(K, "_HAS_LADYBUG", True)
    real = tmp_path / "graph.ladybug"
    store = K.LadybugStore(str(real))
    store.engine_probe()
    (path,) = [c[1] for c in opens(calls)]
    assert path != str(real) and not path.startswith(str(tmp_path))
    assert path.endswith("probe.ladybug")
    assert not os.path.exists(os.path.dirname(path))                  # the throwaway directory is removed
    assert not os.path.exists(real)


def test_a_session_that_cannot_open_records_its_reason(tmp_path, engine):
    store = K.LadybugStore(str(tmp_path / "graph.ladybug"))
    with store._session(write=True) as conn:
        assert conn is None
    assert store.last_open_error == f"RuntimeError({DLL_ERROR!r})"


# ------------------------------------------------------------------ Windows OpenSSL aliases

def _openssl_tree(tmp_path, with_crypto=True):
    base = tmp_path / "py"
    dlls = base / "DLLs"
    dlls.mkdir(parents=True)
    (dlls / "libssl-3.dll").write_bytes(b"ssl-bytes")
    if with_crypto:
        (dlls / "libcrypto-3.dll").write_bytes(b"crypto-bytes!")
    return base, dlls


def test_the_openssl_aliases_are_copied_under_the_x64_names(tmp_path):
    base, dlls = _openssl_tree(tmp_path)
    target = tmp_path / "loci" / "dll"
    added = []
    result = K._windows_ssl_aliases("win32", base, target, add_dll_directory=added.append)
    assert result == ([str(target), str(dlls)], "")
    assert (target / "libssl-3-x64.dll").read_bytes() == b"ssl-bytes"
    assert (target / "libcrypto-3-x64.dll").read_bytes() == b"crypto-bytes!"
    assert added == [str(target), str(dlls)]


def test_an_existing_alias_of_the_same_size_is_not_copied_again(tmp_path):
    base, _ = _openssl_tree(tmp_path)
    target = tmp_path / "dll"
    copies = []
    real_copy = K.shutil.copy2
    K._windows_ssl_aliases("win32", base, target, copy=lambda s, d: (copies.append(d.name), real_copy(s, d)),
                           add_dll_directory=lambda d: None)
    assert copies == ["libssl-3-x64.dll", "libcrypto-3-x64.dll"]
    copies.clear()
    K._windows_ssl_aliases("win32", base, target, copy=lambda s, d: copies.append(d.name),
                           add_dll_directory=lambda d: None)
    assert copies == []                                               # idempotent: nothing to refresh
    (target / "libssl-3-x64.dll").write_bytes(b"stale")               # a different size is refreshed
    K._windows_ssl_aliases("win32", base, target, copy=lambda s, d: copies.append(d.name),
                           add_dll_directory=lambda d: None)
    assert copies == ["libssl-3-x64.dll"]


def test_nothing_is_done_off_windows(tmp_path):
    base, _ = _openssl_tree(tmp_path)
    target = tmp_path / "dll"
    added = []
    assert K._windows_ssl_aliases("linux", base, target, add_dll_directory=added.append) == ([], "")
    assert added == [] and not target.exists()


def test_a_missing_openssl_dll_is_reported_not_raised(tmp_path):
    base, dlls = _openssl_tree(tmp_path, with_crypto=False)
    added = []
    dirs, err = K._windows_ssl_aliases("win32", base, tmp_path / "dll", add_dll_directory=added.append)
    assert dirs == [] and added == []
    assert err == f"OpenSSL DLL not found: {dlls / 'libcrypto-3.dll'}"


# ------------------------------------------------------------------ the tool must not report an empty success

class _Store:
    def __init__(self, counts, why=""):
        self.counts, self.last_open_error = counts, why

    def ingest_code(self, parsed):
        return self.counts

    def delete_code_under(self, path):
        return {}


@pytest.fixture
def ingest(monkeypatch, tmp_path):
    import graph_tools
    src = tmp_path / "pkg"
    src.mkdir()
    (src / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    def run(store):
        monkeypatch.setattr(graph_tools, "_get_ladybug", lambda: store)
        return json.loads(graph_tools.code_graph_ingest(str(src)))
    return run


def test_ingest_that_stored_nothing_is_an_error(ingest):
    out = ingest(_Store({}, f"RuntimeError({DLL_ERROR!r})"))
    assert out["error"] == f"code_graph_ingest stored nothing from 1 parsed file(s): RuntimeError({DLL_ERROR!r})"
    assert out["parsed_files"] == 1 and out["ingested"] == {}


def test_ingest_that_stored_nothing_without_a_known_reason_still_says_so(ingest):
    out = ingest(_Store({}))
    assert out["error"] == "code_graph_ingest stored nothing from 1 parsed file(s): the store accepted no rows"


def test_ingest_that_stored_rows_is_a_success(ingest):
    out = ingest(_Store({"files": 1, "symbols": 1, "defines": 1, "calls": 0, "imports": 0}))
    assert "error" not in out
    assert out["ingested"] == {"files": 1, "symbols": 1, "defines": 1, "calls": 0, "imports": 0}
    assert out["parsed_files"] == 1
