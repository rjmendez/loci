"""embed() in amem_consolidation.py must bound the Ollama call and degrade, not hang.

Every other Ollama/Qdrant HTTP call site in this repo passes a timeout to
urlopen(); this one didn't, so a connected-but-silent Ollama server blocked
the whole consolidation run forever. Fixed to pass timeout= and to catch the
failure so one bad row degrades (returns None, same as vecmath.cosine's
"unanswerable" convention) instead of crashing/hanging the run.
"""

import importlib.util
import pathlib
import urllib.request
from unittest import mock

_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "amem_consolidation_impl", _SCRIPTS / "amem_consolidation.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


amem = _load()


def test_embed_passes_a_timeout_to_urlopen():
    with mock.patch.object(urllib.request, "urlopen") as m:
        m.return_value.__enter__.return_value.read.return_value = (
            b'{"data": [{"embedding": [0.1, 0.2]}]}'
        )
        amem.embed("hello")
        assert m.call_args is not None
        _, kwargs = m.call_args
        assert kwargs.get("timeout") is not None, (
            "embed() must pass a timeout to urlopen — a connected-but-silent "
            "server would otherwise block forever"
        )


def test_embed_degrades_to_none_on_timeout_instead_of_raising():
    with mock.patch.object(
        urllib.request, "urlopen", side_effect=TimeoutError("timed out")
    ):
        assert amem.embed("hello") is None
