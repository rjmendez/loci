import os
from pathlib import Path

import server


def test_server_memory_dir_is_not_live_user_store():
    actual = Path(server.MEMORY_DIR).expanduser().resolve()
    loci_env = Path(os.environ["LOCI_MEMORY_DIR"]).expanduser().resolve()
    hermes_env = Path(os.environ["HERMES_MEMORY_DIR"]).expanduser().resolve()
    live_loci = (Path.home() / ".loci" / "memory-sessions").resolve()
    live_hermes = (Path.home() / ".hermes" / "memory-sessions").resolve()

    assert actual == loci_env == hermes_env
    assert actual not in {live_loci, live_hermes}, (
        "server.MEMORY_DIR resolved to the live user memory store. "
        "Tests must never read/write production investigations or lock graph.ladybug."
    )
