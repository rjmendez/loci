"""swr_replay's consolidated-point search sent Qdrant the upsert form of a named
vector, {"dense": vec}, which /points/search rejects with HTTP 400. The call
swallowed the error and returned [], which is indistinguishable from a query
that legitimately matched nothing, so the interleave step silently never ran."""
from __future__ import annotations

import importlib.util
import io
import json
import os
import pathlib
from unittest import mock

import pytest

MODULE = pathlib.Path(__file__).resolve().parent.parent / "swr_replay.py"


def load(env=None):
    e = {"QDRANT_URL": "http://qdrant.invalid:6333", "QDRANT_API_KEY": "k",
         **(env or {})}
    with mock.patch.dict(os.environ, e, clear=True):
        spec = importlib.util.spec_from_file_location("_swr_uut", MODULE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def swr():
    return load()


def _capture(mod, bodies, result=None):
    def fake(req, timeout=None):
        bodies.append(json.loads(req.data.decode()))
        payload = json.dumps({"result": result if result is not None else []})
        r = io.BytesIO(payload.encode())
        r.__enter__ = lambda s=r: s
        r.__exit__ = lambda s, *a: False
        return r
    return mock.patch.object(mod.urllib.request, "urlopen", fake)


def test_search_uses_named_vector_search_shape(swr):
    bodies = []
    with _capture(swr, bodies):
        swr._qdrant_search_consolidated("hermes_memory", [1.0, 2.0], 3)
    assert bodies, "no request issued"
    assert bodies[0]["vector"] == {"name": "dense", "vector": [1.0, 2.0]}, (
        "sent the upsert form; Qdrant answers 400 and the result is swallowed"
    )


def test_search_keeps_its_consolidated_filter(swr):
    bodies = []
    with _capture(swr, bodies):
        swr._qdrant_search_consolidated("hermes_memory", [1.0], 3)
    assert bodies[0]["filter"]["must"][0]["match"]["value"] == "consolidated"


def test_upsert_still_uses_the_bare_named_vector_form(swr):
    # Upsert genuinely takes {"dense": vec}; fixing search must not touch it.
    bodies = []
    with _capture(swr, bodies):
        swr._qdrant_upsert("hermes_memory", "p1", [1.0], {"a": 1})
    assert bodies[0]["points"][0]["vector"] == {"dense": [1.0]}
