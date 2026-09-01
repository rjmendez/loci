"""A script's embedding config must follow the Qdrant collection it touches.

Four standalone cron scripts each read an env var that no other participant in
their own collection reads, so one setting had two names across two files:

  * swr_replay reads and writes mcp's `loci_memory` but embedded with
    MNEMOSYNE_EMBEDDING_MODEL, while qdrant_ops.py — which creates that
    collection and searches it — uses EMBED_MODEL. Root .env.example ships only
    the former, so an operator who followed it wrote consolidated abstractions
    into loci_memory in a space nothing searches. Dims permitting, the upsert
    succeeds and the memory is simply never retrievable again.
  * agentHER_relabeler is the mirror image: it writes the `mnemosyne`
    collection but embedded with EMBED_MODEL, while that collection's only
    reader (a2a_server) uses MNEMOSYNE_EMBEDDING_MODEL and drops every hit
    below a hard 0.59 score floor.
  * state_db_qdrant_sync created loci_sessions from MNEMOSYNE_EMBEDDING_DIM but
    validated returned vectors against a 768 literal, so a deployment on
    1536-dim embeddings dropped every chunk as "malformed" and exited 0.
  * ua-ingest read QDRANT_KEY, a name that appears in no .env.example, no doc
    and no compose file.

All of these look healthy from outside: they print progress and exit 0.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import pathlib
import sys
import types
from unittest import mock

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _fresh(mod: str):
    """Re-import a script so its module-level env reads happen under patch."""
    sys.modules.pop(mod, None)
    return importlib.import_module(mod)


def _load_dashed(modname: str, filename: str):
    """ua-ingest.py is not an importable name. `requests` is stubbed because the
    module pip-installs it on ImportError, and the CI test job has no network."""
    sys.modules.setdefault("requests", types.ModuleType("requests"))
    spec = importlib.util.spec_from_file_location(modname, SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def isolated_env(tmp_path):
    """A clean environment with HOME pointed somewhere empty, so the scripts'
    own ~/.hermes/.env and ~/.claude/settings.json loaders find nothing."""
    home = tmp_path / "home"
    home.mkdir()

    def apply(**overrides):
        env = {"HOME": str(home),
               "LOCI_ENV_FILE": str(tmp_path / "no-such.env"),
               "QDRANT_URL": "http://qdrant.invalid:6333"}
        env.update(overrides)
        return mock.patch.dict(os.environ, env, clear=True)

    return apply


# ── swr_replay: loci_memory is mcp's collection, so mcp's knobs win ───────────

def test_swr_replay_embeds_loci_memory_with_the_model_mcp_searches_it_with(isolated_env):
    with isolated_env(EMBED_MODEL="mcp-space",
                      MNEMOSYNE_EMBEDDING_MODEL="a2a-space"):
        assert _fresh("swr_replay").EMBED_MODEL == "mcp-space"


def test_swr_replay_still_honours_the_mnemosyne_name_alone(isolated_env):
    """Existing deployments set only this one; they must not silently move."""
    with isolated_env(MNEMOSYNE_EMBEDDING_MODEL="a2a-space"):
        assert _fresh("swr_replay").EMBED_MODEL == "a2a-space"


def test_swr_replay_follows_the_collection_prefix_mcp_owns(isolated_env):
    with isolated_env(QDRANT_COLLECTION_PREFIX="loci_memory_v2"):
        assert _fresh("swr_replay").COLLECTION == "loci_memory_v2"


def test_an_explicit_swr_collection_still_overrides_the_prefix(isolated_env):
    with isolated_env(SWR_COLLECTION="pinned",
                      QDRANT_COLLECTION_PREFIX="loci_memory_v2"):
        assert _fresh("swr_replay").COLLECTION == "pinned"


# ── agentHER: the mirror case, mnemosyne is the a2a server's collection ──────

def test_agenther_embeds_mnemosyne_with_the_model_its_reader_uses(isolated_env):
    with isolated_env(EMBED_MODEL="mcp-space",
                      MNEMOSYNE_EMBEDDING_MODEL="a2a-space"):
        mod = _fresh("agentHER_relabeler")
        assert mod.COLLECTION == "mnemosyne"
        assert mod.EMBED_MODEL == "a2a-space"


def test_agenther_still_honours_embed_model_alone(isolated_env):
    with isolated_env(EMBED_MODEL="mcp-space"):
        assert _fresh("agentHER_relabeler").EMBED_MODEL == "mcp-space"


# ── state_db_qdrant_sync: one source for the vector width ───────────────────

def test_the_validator_width_follows_the_same_env_the_collection_is_created_from(isolated_env):
    with isolated_env(MNEMOSYNE_EMBEDDING_DIM="1536"):
        assert _fresh("state_db_qdrant_sync").VECTOR_DIM == 1536


def test_a_correctly_sized_vector_is_kept_rather_than_dropped(isolated_env):
    """The defect in behavioural form: on a 1536-dim deployment every chunk the
    embed worker returned was counted as malformed and discarded."""
    with isolated_env(MNEMOSYNE_EMBEDDING_DIM="1536", EMBED_WORKER_URL="http://w.invalid"):
        sd = _fresh("state_db_qdrant_sync")
        vector = [0.1] * 1536
        worker = mock.Mock(stdout='{"status": "ok", "vectors": {"c1": "u1"}}')
        points = {"result": [{"id": "u1", "vector": {"dense": vector}}]}
        with mock.patch.object(sd.subprocess, "run", return_value=worker), \
             mock.patch.object(sd, "curl_json", return_value=points):
            id_to_vec, dropped = sd.embed_batch([{"id": "c1", "text": "hi"}], key="")
    assert dropped == 0
    assert id_to_vec == {"c1": vector}


def test_a_genuinely_wrong_width_is_still_dropped(isolated_env):
    """The guard must keep guarding — it is what stops a mixed-space upsert."""
    with isolated_env(MNEMOSYNE_EMBEDDING_DIM="1536", EMBED_WORKER_URL="http://w.invalid"):
        sd = _fresh("state_db_qdrant_sync")
        worker = mock.Mock(stdout='{"status": "ok", "vectors": {"c1": "u1"}}')
        points = {"result": [{"id": "u1", "vector": {"dense": [0.1] * 768}}]}
        with mock.patch.object(sd.subprocess, "run", return_value=worker), \
             mock.patch.object(sd, "curl_json", return_value=points):
            id_to_vec, dropped = sd.embed_batch([{"id": "c1", "text": "hi"}], key="")
    assert (id_to_vec, dropped) == ({}, 1)


# ── ua-ingest: the documented key name ──────────────────────────────────────

def test_ua_ingest_reads_the_key_name_both_env_examples_actually_ship(isolated_env):
    with isolated_env(QDRANT_API_KEY="documented"):
        assert _load_dashed("ua_ingest_impl", "ua-ingest.py").QDRANT_KEY == "documented"


def test_ua_ingest_keeps_the_undocumented_name_as_a_fallback(isolated_env):
    with isolated_env(QDRANT_KEY="legacy"):
        assert _load_dashed("ua_ingest_impl", "ua-ingest.py").QDRANT_KEY == "legacy"
