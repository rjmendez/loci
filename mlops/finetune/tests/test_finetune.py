"""Characterization tests for mlops/finetune/ — collect.py, format_sft.py, train_lora.py.

These pin the CURRENT behaviour of the fine-tuning data pipeline, bugs included.
They are a safety net for a later refactor, not a specification of what the pipeline
*should* do. Assertions that lock in behaviour which is arguably wrong are flagged
with a ``BUG:`` comment.

No external services are touched:
  * Ollama is never contacted — ``subprocess.run`` is replaced by an in-process fake.
  * No GPU / Unsloth / transformers import ever happens — the unsloth backend only
    *emits* a script, and we assert on the emitted text (and that it compiles).
  * SQLite is used against throwaway files under ``tmp_path`` only.
  * ``train_lora._HERE`` is redirected at ``tmp_path`` so nothing is written into the
    repository working tree.
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import mlops.finetune.collect as C  # noqa: E402
import mlops.finetune.format_sft as F  # noqa: E402
import mlops.finetune.train_lora as T  # noqa: E402
import scripts.score_trace_collector as STC  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════════

def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _long(ch: str, n: int = 25) -> str:
    """A string comfortably above MIN_CONTENT_LEN (20)."""
    return ch * n


def _write_jsonl(path: Path, records) -> Path:
    """`records` may be dicts (json-encoded) or raw strings (written verbatim)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [r if isinstance(r, str) else json.dumps(r) for r in records]
    path.write_text("\n".join(lines) + "\n")
    return path


def _read_jsonl(path: Path) -> list:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _make_wm_db(path: Path, rows) -> Path:
    """rows: iterable of (content, session_id, source)."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE working_memory (content TEXT, session_id TEXT, source TEXT)"
    )
    conn.executemany("INSERT INTO working_memory VALUES (?, ?, ?)", list(rows))
    conn.commit()
    conn.close()
    return path


class FakeCompleted:
    def __init__(self, returncode=0):
        self.returncode = returncode


class FakeRun:
    """Records subprocess.run invocations and replays scripted return codes."""

    def __init__(self, codes=None):
        self.calls = []
        self.codes = list(codes or [])

    def __call__(self, cmd, **kwargs):
        self.calls.append((list(cmd), kwargs))
        code = self.codes.pop(0) if self.codes else 0
        return FakeCompleted(code)


# ══════════════════════════════════════════════════════════════════════════════════
# collect.py — helpers
# ══════════════════════════════════════════════════════════════════════════════════

def test_collect_sha256_is_plain_hexdigest_of_utf8():
    assert C._sha256("") == _sha("")
    assert C._sha256("abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    # non-ASCII goes through str.encode() default (utf-8)
    assert C._sha256("é") == _sha("é")
    assert len(C._sha256("x")) == 64


def test_collect_now_iso_is_utc_with_z_suffix_and_no_offset():
    got = C._now_iso()
    assert got.endswith("Z")
    assert "+00:00" not in got
    # e.g. 2026-08-07T12:34:56.789012Z  (microseconds present in practice)
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$", got), got


# ══════════════════════════════════════════════════════════════════════════════════
# collect.load_agentHER_from_db
# ══════════════════════════════════════════════════════════════════════════════════

def test_load_agentHER_missing_db_returns_empty_list_silently(tmp_path, capsys):
    assert C.load_agentHER_from_db(str(tmp_path / "nope.db")) == []
    assert capsys.readouterr().err == ""


def test_load_agentHER_filters_on_source_and_returns_fixed_four_key_shape(tmp_path):
    db = _make_wm_db(
        tmp_path / "m.db",
        [
            ("hello agentHER", "s1", "agentHER"),
            ("some other row", "s2", "manual"),
            ("second her", "s3", "agentHER"),
        ],
    )
    got = C.load_agentHER_from_db(str(db))
    assert got == [
        {"type": "agentHER", "content": "hello agentHER",
         "source": "mnemosyne", "session_id": "s1"},
        {"type": "agentHER", "content": "second her",
         "source": "mnemosyne", "session_id": "s3"},
    ]
    # NOTE: no "id" and no "collected_at" here — those are added by
    # build_unified_records, not by the reader.
    assert set(got[0]) == {"type", "content", "source", "session_id"}


def test_load_agentHER_coerces_sql_nulls_to_empty_strings(tmp_path):
    db = _make_wm_db(tmp_path / "m.db", [(None, None, "agentHER")])
    assert C.load_agentHER_from_db(str(db)) == [
        {"type": "agentHER", "content": "", "source": "mnemosyne", "session_id": ""}
    ]


def test_load_agentHER_missing_table_fails_open_with_stderr_note(tmp_path, capsys):
    db = tmp_path / "empty.db"
    sqlite3.connect(str(db)).close()  # valid sqlite file, no working_memory table
    assert C.load_agentHER_from_db(str(db)) == []
    err = capsys.readouterr().err
    assert "[collect] sqlite error reading agentHER rows" in err
    assert "working_memory" in err


def test_load_agentHER_non_sqlite_file_fails_open(tmp_path, capsys):
    bogus = tmp_path / "not.db"
    bogus.write_bytes(b"this is definitely not a sqlite database")
    assert C.load_agentHER_from_db(str(bogus)) == []
    assert "[collect] sqlite error" in capsys.readouterr().err


def test_load_agentHER_lets_non_sqlite_errors_propagate(tmp_path, monkeypatch):
    """Only sqlite3.Error is caught; anything else escapes."""
    db = _make_wm_db(tmp_path / "m.db", [("c", "s", "agentHER")])

    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(C.sqlite3, "connect", boom)
    with pytest.raises(RuntimeError, match="kaboom"):
        C.load_agentHER_from_db(str(db))


# ══════════════════════════════════════════════════════════════════════════════════
# collect.build_unified_records
# ══════════════════════════════════════════════════════════════════════════════════

def test_build_unified_records_order_is_neg_pos_corr_her_and_nothing_is_dropped():
    recs = C.build_unified_records(
        negatives=[{"content": "n1"}, {"content": "n2"}],
        positives=[{"content": "p1"}],
        corrections=[{"failed_content": "f", "corrected_content": "c"}],
        agentHER=[{"content": "h1"}],
        collected_at="T0",
    )
    assert [r["type"] for r in recs] == [
        "negative", "negative", "positive", "correction", "agentHER"
    ]
    assert len(recs) == 5


def test_build_unified_records_field_shape_and_sources():
    recs = C.build_unified_records(
        [{"content": "n1", "session_id": "sn"}],
        [{"content": "p1", "session_id": "sp"}],
        [{"failed_content": "f", "corrected_content": "c", "session_id": "sc"}],
        [{"content": "h1", "session_id": "sh"}],
        "2026-01-01T00:00:00Z",
    )
    for r in recs:
        assert set(r) == {"id", "type", "content", "source", "session_id", "collected_at"}
        assert r["collected_at"] == "2026-01-01T00:00:00Z"
        assert r["id"] == _sha(r["content"])
    assert [r["source"] for r in recs] == [
        "guard_log", "guard_log", "guard_log", "mnemosyne"
    ]
    assert [r["session_id"] for r in recs] == ["sn", "sp", "sc", "sh"]


def test_build_unified_records_correction_content_is_json_failed_then_corrected():
    recs = C.build_unified_records([], [], [
        {"failed_content": "F", "corrected_content": "C"},
    ], [], "T")
    # Exact serialization matters: it is the dedup key AND format_sft parses it back.
    assert recs[0]["content"] == '{"failed": "F", "corrected": "C"}'
    assert json.loads(recs[0]["content"]) == {"failed": "F", "corrected": "C"}


def test_build_unified_records_defaults_missing_fields_to_empty_string():
    recs = C.build_unified_records([{}], [{}], [{}], [{}], "T")
    assert [r["content"] for r in recs] == [
        "", "", '{"failed": "", "corrected": ""}', ""
    ]
    assert all(r["session_id"] == "" for r in recs)
    # empty content still gets a (constant) id — no record is filtered out
    assert recs[0]["id"] == _sha("")


def test_build_unified_records_all_empty_inputs_gives_empty_list():
    assert C.build_unified_records([], [], [], [], "T") == []


# ══════════════════════════════════════════════════════════════════════════════════
# collect.deduplicate
# ══════════════════════════════════════════════════════════════════════════════════

def test_deduplicate_keeps_first_occurrence_and_reports_removed_count():
    recs = [
        {"id": "a", "n": 1},
        {"id": "b", "n": 2},
        {"id": "a", "n": 3},
        {"id": "a", "n": 4},
    ]
    unique, n_removed = C.deduplicate(recs)
    assert [r["n"] for r in unique] == [1, 2]
    assert n_removed == 2


def test_deduplicate_empty_input():
    assert C.deduplicate([]) == ([], 0)


def test_deduplicate_collapses_across_types_because_id_is_content_only():
    """BUG: a negative and a positive with identical text share an id, so the
    positive is silently discarded and the pair is mislabelled as a negative."""
    recs = C.build_unified_records(
        [{"content": "ls -la"}], [{"content": "ls -la"}], [], [], "T"
    )
    unique, n_removed = C.deduplicate(recs)
    assert n_removed == 1
    assert len(unique) == 1
    assert unique[0]["type"] == "negative"  # the positive lost


def test_deduplicate_raises_keyerror_when_a_record_has_no_id():
    with pytest.raises(KeyError):
        C.deduplicate([{"type": "negative"}])


# ══════════════════════════════════════════════════════════════════════════════════
# collect.parse_args / main
# ══════════════════════════════════════════════════════════════════════════════════

def test_collect_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["collect.py"])
    args = C.parse_args()
    assert args.out == "mlops/finetune/data/"
    assert args.db == C._DEFAULT_DB
    assert args.hook_state == C._DEFAULT_STATE_DIR
    assert args.ollama  # defaults from OLLAMA_URL or localhost:11434


def test_collect_parse_args_hook_state_flag_maps_to_hook_state_attr(monkeypatch):
    monkeypatch.setattr(
        sys, "argv",
        ["collect.py", "--out", "/o", "--db", "/d.db", "--hook-state", "/hs",
         "--ollama", "http://x:1"],
    )
    args = C.parse_args()
    assert (args.out, args.db, args.hook_state, args.ollama) == (
        "/o", "/d.db", "/hs", "http://x:1"
    )


def _guard_logs(state_dir: Path, failures, successes):
    state_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(state_dir / "guard_bash_failures.log", failures)
    _write_jsonl(state_dir / "guard_bash_successes.log", successes)


def test_collect_main_end_to_end_writes_raw_traces_jsonl(tmp_path, monkeypatch, capsys):
    state = tmp_path / "hook-state"
    _guard_logs(
        state,
        failures=[
            {"canonical_command": "bad cmd", "count": 2, "session_id": "s1"},
            {"canonical_command": "ignored, count too low", "count": 1,
             "session_id": "s1"},
        ],
        successes=[{"command": "good cmd", "session_id": "s1"}],
    )
    db = _make_wm_db(tmp_path / "m.db", [("her row", "s9", "agentHER")])
    out = tmp_path / "data"

    # main() mutates score_trace_collector module globals; monkeypatch restores them.
    monkeypatch.setattr(STC, "STATE_DIR", STC.STATE_DIR)
    monkeypatch.setattr(STC, "MNEMOSYNE_DB", STC.MNEMOSYNE_DB)
    monkeypatch.setattr(STC, "OLLAMA_URL", STC.OLLAMA_URL, raising=False)
    monkeypatch.setattr(
        sys, "argv",
        ["collect.py", "--out", str(out), "--db", str(db),
         "--hook-state", str(state), "--ollama", "http://ollama.invalid:11434"],
    )

    C.main()

    # the --hook-state / --db / --ollama flags are applied by patching the
    # collector's module globals
    assert STC.STATE_DIR == str(state)
    assert STC.MNEMOSYNE_DB == str(db)
    assert STC.OLLAMA_URL == "http://ollama.invalid:11434"

    written = _read_jsonl(out / "raw_traces.jsonl")
    assert [r["type"] for r in written] == [
        "negative", "positive", "correction", "agentHER"
    ]
    assert written[0]["content"] == "bad cmd"
    assert written[1]["content"] == "good cmd"
    assert json.loads(written[2]["content"]) == {
        "failed": "bad cmd", "corrected": "good cmd"
    }
    assert written[3]["content"] == "her row"
    # one shared timestamp for the whole batch
    assert len({r["collected_at"] for r in written}) == 1

    outp = capsys.readouterr().out
    assert "negatives=1" in outp and "positives=1" in outp
    assert "corrections=1" in outp and "agentHER=1" in outp
    assert "deduped=0" in outp
    assert "wrote 4 records" in outp


def test_collect_main_creates_out_dir_and_survives_empty_sources(
    tmp_path, monkeypatch, capsys
):
    out = tmp_path / "deep" / "nested" / "data"
    monkeypatch.setattr(STC, "STATE_DIR", STC.STATE_DIR)
    monkeypatch.setattr(STC, "MNEMOSYNE_DB", STC.MNEMOSYNE_DB)
    monkeypatch.setattr(STC, "OLLAMA_URL", STC.OLLAMA_URL, raising=False)
    monkeypatch.setattr(
        sys, "argv",
        ["collect.py", "--out", str(out),
         "--db", str(tmp_path / "missing.db"),
         "--hook-state", str(tmp_path / "missing-state")],
    )

    C.main()

    raw = out / "raw_traces.jsonl"
    assert raw.exists()
    assert raw.read_text() == ""  # empty file, not absent
    assert "wrote 0 records" in capsys.readouterr().out


def test_collect_main_dedup_counter_reflects_content_collisions(
    tmp_path, monkeypatch, capsys
):
    state = tmp_path / "hs"
    # identical text on both sides of the guard logs
    _guard_logs(
        state,
        failures=[{"canonical_command": "same", "count": 2, "session_id": "s1"}],
        successes=[{"command": "same", "session_id": "s1"}],
    )
    monkeypatch.setattr(STC, "STATE_DIR", STC.STATE_DIR)
    monkeypatch.setattr(STC, "MNEMOSYNE_DB", STC.MNEMOSYNE_DB)
    monkeypatch.setattr(STC, "OLLAMA_URL", STC.OLLAMA_URL, raising=False)
    monkeypatch.setattr(
        sys, "argv",
        ["collect.py", "--out", str(tmp_path / "o"),
         "--db", str(tmp_path / "none.db"), "--hook-state", str(state)],
    )
    C.main()
    outp = capsys.readouterr().out
    assert "deduped=1" in outp
    assert "positives=0" in outp  # BUG: swallowed by the negative with the same text
    assert "wrote 2 records" in outp


def test_collect_reexports_unused_agenthr_loader():
    """load_agenthr_positives is imported but never called — dead import, kept
    here so a refactor that removes it is a deliberate choice."""
    assert callable(C.load_agenthr_positives)
    assert C.load_agenthr_positives is STC.load_agenthr_positives


# ══════════════════════════════════════════════════════════════════════════════════
# format_sft.py — constants + pairs_from_corrections
# ══════════════════════════════════════════════════════════════════════════════════

def test_format_sft_min_content_len_constant():
    assert F.MIN_CONTENT_LEN == 20


def test_pairs_from_corrections_shape_and_source():
    rec = {
        "type": "correction",
        "content": json.dumps({"failed": _long("f"), "corrected": _long("c")}),
        "session_id": "s7",
    }
    pairs = F.pairs_from_corrections([rec])
    assert pairs == [{
        "messages": [
            {"role": "user", "content": _long("f")},
            {"role": "assistant", "content": _long("c")},
        ],
        "source": "score_correction",
        "session_id": "s7",
    }]


def test_pairs_from_corrections_ignores_other_types():
    other = {"type": "agentHER", "content": json.dumps(
        {"failed": _long("f"), "corrected": _long("c")})}
    assert F.pairs_from_corrections([other]) == []
    assert F.pairs_from_corrections([{"content": "x"}]) == []  # no "type" key


@pytest.mark.parametrize("n,kept", [(19, False), (20, True), (21, True)])
def test_pairs_from_corrections_length_boundary_is_inclusive_at_20(n, kept):
    rec = {"type": "correction",
           "content": json.dumps({"failed": "f" * n, "corrected": "c" * 40})}
    assert bool(F.pairs_from_corrections([rec])) is kept
    rec2 = {"type": "correction",
            "content": json.dumps({"failed": "f" * 40, "corrected": "c" * n})}
    assert bool(F.pairs_from_corrections([rec2])) is kept


def test_pairs_from_corrections_skips_bad_json_and_missing_content():
    recs = [
        {"type": "correction", "content": "{not json"},
        {"type": "correction"},  # KeyError path
        {"type": "correction",
         "content": json.dumps({"failed": _long("f"), "corrected": _long("c")})},
    ]
    assert len(F.pairs_from_corrections(recs)) == 1


def test_pairs_from_corrections_defaults_session_id_to_empty_string():
    rec = {"type": "correction",
           "content": json.dumps({"failed": _long("f"), "corrected": _long("c")})}
    assert F.pairs_from_corrections([rec])[0]["session_id"] == ""


def test_pairs_from_corrections_missing_side_keys_are_filtered_by_length():
    rec = {"type": "correction", "content": json.dumps({"failed": _long("f")})}
    assert F.pairs_from_corrections([rec]) == []  # corrected defaults to "" (len 0)


@pytest.mark.parametrize("payload", ["[1, 2]", '"a string"', "123", "null"])
def test_pairs_from_corrections_crashes_on_non_object_json(payload):
    """BUG: only JSONDecodeError/KeyError are caught. Valid JSON that is not an
    object reaches ``sides.get`` and raises AttributeError, killing the run."""
    with pytest.raises(AttributeError):
        F.pairs_from_corrections([{"type": "correction", "content": payload}])


def test_pairs_from_corrections_crashes_on_null_content():
    """BUG: content=None raises TypeError out of json.loads (not caught)."""
    with pytest.raises(TypeError):
        F.pairs_from_corrections([{"type": "correction", "content": None}])


# ══════════════════════════════════════════════════════════════════════════════════
# format_sft.pairs_from_agentHER
# ══════════════════════════════════════════════════════════════════════════════════

def test_pairs_from_agentHER_uses_content_as_both_prompt_and_answer():
    rec = {"type": "agentHER", "content": _long("m"), "session_id": "s2"}
    assert F.pairs_from_agentHER([rec]) == [{
        "messages": [
            {"role": "user", "content": "Recall relevant memory for: " + _long("m")},
            {"role": "assistant", "content": _long("m")},
        ],
        "source": "agentHER",
        "session_id": "s2",
    }]


@pytest.mark.parametrize("n,kept", [(19, False), (20, True)])
def test_pairs_from_agentHER_length_boundary(n, kept):
    rec = {"type": "agentHER", "content": "m" * n}
    assert bool(F.pairs_from_agentHER([rec])) is kept


def test_pairs_from_agentHER_ignores_other_types_and_missing_content():
    assert F.pairs_from_agentHER([{"type": "correction", "content": _long("x")}]) == []
    assert F.pairs_from_agentHER([{"type": "agentHER"}]) == []


# ══════════════════════════════════════════════════════════════════════════════════
# format_sft dedup
# ══════════════════════════════════════════════════════════════════════════════════

def test_deduplicate_pairs_keys_on_messages_only_ignoring_source_and_session():
    msgs = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    pairs = [
        {"messages": msgs, "source": "score_correction", "session_id": "s1"},
        {"messages": list(msgs), "source": "agentHER", "session_id": "s2"},
    ]
    unique, removed = F.deduplicate_pairs(pairs)
    assert removed == 1
    assert unique[0]["source"] == "score_correction"  # first wins


def test_deduplicate_pairs_key_is_order_insensitive_within_a_message_dict():
    """sort_keys=True — dict key order inside a message does not create a new key."""
    a = {"messages": [{"role": "u", "content": "c"}], "source": "x"}
    b = {"messages": [{"content": "c", "role": "u"}], "source": "y"}
    assert F.deduplicate_pairs([a, b])[1] == 1


def test_deduplicate_pairs_preserves_order_and_empty_input():
    pairs = [{"messages": [{"c": i}]} for i in range(3)]
    unique, removed = F.deduplicate_pairs(pairs)
    assert unique == pairs and removed == 0
    assert F.deduplicate_pairs([]) == ([], 0)


def test_deduplicate_dpo_keys_on_chosen_only():
    chosen = [{"role": "user", "content": "u"},
              {"role": "assistant", "content": "a"}]
    pairs = [
        {"chosen": chosen, "rejected": [{"role": "assistant", "content": "z1"}]},
        {"chosen": list(chosen), "rejected": [{"role": "assistant", "content": "z2"}]},
    ]
    unique, removed = F.deduplicate_dpo(pairs)
    assert removed == 1
    assert unique[0]["rejected"][0]["content"] == "z1"


# ══════════════════════════════════════════════════════════════════════════════════
# format_sft.pairs_from_corrections_dpo
# ══════════════════════════════════════════════════════════════════════════════════

def test_pairs_from_corrections_dpo_shape():
    rec = {"type": "correction", "session_id": "s3",
           "content": json.dumps({"failed": _long("f"), "corrected": _long("c")})}
    assert F.pairs_from_corrections_dpo([rec]) == [{
        "prompt": "You are a helpful assistant.",
        "chosen": [
            {"role": "user", "content": _long("f")},
            {"role": "assistant", "content": _long("c")},
        ],
        "rejected": [
            {"role": "user", "content": _long("f")},
            {"role": "assistant", "content": _long("f")},
        ],
        "source": "score_correction_dpo",
        "session_id": "s3",
    }]


def test_pairs_from_corrections_dpo_drops_identity_pairs():
    same = _long("s")
    rec = {"type": "correction",
           "content": json.dumps({"failed": same, "corrected": same})}
    assert F.pairs_from_corrections_dpo([rec]) == []
    # ...but the SFT builder happily keeps the identity pair
    assert len(F.pairs_from_corrections([rec])) == 1


@pytest.mark.parametrize("n,kept", [(19, False), (20, True)])
def test_pairs_from_corrections_dpo_length_boundary(n, kept):
    rec = {"type": "correction",
           "content": json.dumps({"failed": "f" * n, "corrected": "c" * 40})}
    assert bool(F.pairs_from_corrections_dpo([rec])) is kept


def test_pairs_from_corrections_dpo_skips_bad_json_and_other_types():
    recs = [
        {"type": "correction", "content": "nope{"},
        {"type": "correction"},
        {"type": "agentHER", "content": _long("x")},
    ]
    assert F.pairs_from_corrections_dpo(recs) == []


# ══════════════════════════════════════════════════════════════════════════════════
# format_sft.parse_args / main
# ══════════════════════════════════════════════════════════════════════════════════

def test_format_sft_parse_args_defaults_and_required(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["format_sft.py", "--traces", "t", "--out", "o"])
    args = F.parse_args()
    assert (args.traces, args.out, args.min_pairs, args.mode) == ("t", "o", 50, "sft")

    monkeypatch.setattr(sys, "argv", ["format_sft.py"])
    with pytest.raises(SystemExit):
        F.parse_args()

    monkeypatch.setattr(
        sys, "argv",
        ["format_sft.py", "--traces", "t", "--out", "o", "--mode", "bogus"])
    with pytest.raises(SystemExit):
        F.parse_args()


def _traces_fixture(tmp_path: Path) -> Path:
    return _write_jsonl(tmp_path / "raw_traces.jsonl", [
        {"type": "correction", "session_id": "s1",
         "content": json.dumps({"failed": _long("f"), "corrected": _long("c")})},
        {"type": "agentHER", "session_id": "s2", "content": _long("m")},
        {"type": "negative", "content": _long("n")},   # never becomes a pair
        "",                                            # blank line skipped
        "{{{ not json",                                # bad line skipped
    ])


def _run_format(monkeypatch, traces, out, *extra):
    monkeypatch.setattr(
        sys, "argv",
        ["format_sft.py", "--traces", str(traces), "--out", str(out), *extra])
    F.main()


def test_format_sft_main_missing_traces_file_exits_1(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        _run_format(monkeypatch, tmp_path / "nope.jsonl", tmp_path / "o.jsonl")
    assert exc.value.code == 1
    assert "[format_sft] traces file not found" in capsys.readouterr().err


def test_format_sft_main_sft_mode_writes_only_sft_file(tmp_path, monkeypatch, capsys):
    traces = _traces_fixture(tmp_path)
    out = tmp_path / "sub" / "sft_pairs.jsonl"
    _run_format(monkeypatch, traces, out, "--min-pairs", "1")

    pairs = _read_jsonl(out)
    assert [p["source"] for p in pairs] == ["score_correction", "agentHER"]
    assert not (tmp_path / "sub" / "sft_pairs_dpo.jsonl").exists()

    cap = capsys.readouterr()
    assert "[format_sft] SFT: correction=1 agentHER=1 deduped=0 total=2" in cap.out
    assert "DPO" not in cap.out
    assert cap.err == ""


def test_format_sft_main_dpo_mode_writes_dpo_path_and_not_out(
    tmp_path, monkeypatch, capsys
):
    traces = _traces_fixture(tmp_path)
    out = tmp_path / "sft_pairs.jsonl"
    _run_format(monkeypatch, traces, out, "--mode", "dpo", "--min-pairs", "1")

    dpo = tmp_path / "sft_pairs_dpo.jsonl"
    assert dpo.exists()
    assert not out.exists()  # --out is never written in dpo-only mode
    assert [p["source"] for p in _read_jsonl(dpo)] == ["score_correction_dpo"]
    assert "[format_sft] DPO: pairs=1 deduped=0" in capsys.readouterr().out


def test_format_sft_main_both_mode_writes_two_files(tmp_path, monkeypatch, capsys):
    traces = _traces_fixture(tmp_path)
    out = tmp_path / "sft_pairs.jsonl"
    _run_format(monkeypatch, traces, out, "--mode", "both", "--min-pairs", "1")

    assert len(_read_jsonl(out)) == 2
    assert len(_read_jsonl(tmp_path / "sft_pairs_dpo.jsonl")) == 1
    cap = capsys.readouterr()
    assert "SFT:" in cap.out and "DPO:" in cap.out
    assert cap.err == ""  # total_written == 3 >= min_pairs 1


def test_format_sft_main_zero_dpo_pairs_warns_on_stderr(tmp_path, monkeypatch, capsys):
    traces = _write_jsonl(tmp_path / "t.jsonl",
                          [{"type": "agentHER", "content": _long("m")}])
    _run_format(monkeypatch, traces, tmp_path / "o.jsonl",
                "--mode", "dpo", "--min-pairs", "0")
    err = capsys.readouterr().err
    assert "WARNING: 0 DPO pairs" in err


def test_format_sft_main_min_pairs_warning_counts_total_written(
    tmp_path, monkeypatch, capsys
):
    traces = _traces_fixture(tmp_path)
    _run_format(monkeypatch, traces, tmp_path / "o.jsonl",
                "--mode", "both", "--min-pairs", "50")
    err = capsys.readouterr().err
    assert "WARNING: only 3 pairs (min=50)" in err  # 2 SFT + 1 DPO


def test_format_sft_main_min_pairs_zero_never_warns(tmp_path, monkeypatch, capsys):
    traces = _write_jsonl(tmp_path / "t.jsonl", [])
    _run_format(monkeypatch, traces, tmp_path / "o.jsonl", "--min-pairs", "0")
    assert "WARNING" not in capsys.readouterr().err


def test_format_sft_main_dpo_path_replaces_every_jsonl_occurrence(
    tmp_path, monkeypatch
):
    """BUG: str.replace with no count — a name containing '.jsonl' more than once
    gets mangled."""
    traces = _traces_fixture(tmp_path)
    out = tmp_path / "a.jsonl.jsonl"
    _run_format(monkeypatch, traces, out, "--mode", "dpo", "--min-pairs", "0")
    assert (tmp_path / "a_dpo.jsonl_dpo.jsonl").exists()


def test_format_sft_main_dpo_path_fallback_when_out_has_no_jsonl_suffix(
    tmp_path, monkeypatch
):
    traces = _traces_fixture(tmp_path)
    out = tmp_path / "pairs.txt"
    _run_format(monkeypatch, traces, out, "--mode", "dpo", "--min-pairs", "0")
    assert (tmp_path / "pairs.txt.dpo.jsonl").exists()


def test_format_sft_main_output_is_one_json_object_per_line(tmp_path, monkeypatch):
    traces = _traces_fixture(tmp_path)
    out = tmp_path / "o.jsonl"
    _run_format(monkeypatch, traces, out, "--min-pairs", "0")
    text = out.read_text()
    assert text.endswith("\n")
    assert all(json.loads(ln) for ln in text.splitlines())


# ══════════════════════════════════════════════════════════════════════════════════
# train_lora.load_pairs
# ══════════════════════════════════════════════════════════════════════════════════

def test_load_pairs_missing_file_exits_1(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        T.load_pairs(str(tmp_path / "nope.jsonl"), 10)
    assert exc.value.code == 1
    assert "[train] SFT file not found" in capsys.readouterr().err


def test_load_pairs_skips_blank_and_malformed_lines(tmp_path):
    p = _write_jsonl(tmp_path / "s.jsonl", [
        {"a": 1}, "", "   ", "not json", {"a": 2},
    ])
    assert T.load_pairs(str(p), 100) == [{"a": 1}, {"a": 2}]


def test_load_pairs_truncates_to_max_examples_keeping_file_order(tmp_path):
    p = _write_jsonl(tmp_path / "s.jsonl", [{"i": i} for i in range(10)])
    assert T.load_pairs(str(p), 3) == [{"i": 0}, {"i": 1}, {"i": 2}]


def test_load_pairs_max_examples_zero_yields_nothing(tmp_path):
    p = _write_jsonl(tmp_path / "s.jsonl", [{"i": 0}])
    assert T.load_pairs(str(p), 0) == []


def test_load_pairs_negative_max_examples_drops_from_the_end(tmp_path):
    """BUG-ish: no validation — a negative cap becomes a python slice."""
    p = _write_jsonl(tmp_path / "s.jsonl", [{"i": i} for i in range(5)])
    assert T.load_pairs(str(p), -2) == [{"i": 0}, {"i": 1}, {"i": 2}]


def test_load_pairs_empty_file_returns_empty(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text("")
    assert T.load_pairs(str(p), 10) == []


# ══════════════════════════════════════════════════════════════════════════════════
# train_lora modelfile building
# ══════════════════════════════════════════════════════════════════════════════════

def test_escape_modelfile_value_only_swaps_triple_double_quotes():
    assert T._escape_modelfile_value('a """b""" c') == "a '''b''' c"
    # everything else is untouched — including backslashes and newlines
    assert T._escape_modelfile_value('a\\nb\n"c"') == 'a\\nb\n"c"'
    assert T._escape_modelfile_value("") == ""


def test_build_modelfile_header_when_no_pairs():
    assert T.build_modelfile("llama3.2:latest", []) == (
        "FROM llama3.2:latest\n"
        "\n"
        "SYSTEM You are Loci, an autonomous coding agent with strong memory. "
        "You learn from past failures and apply corrected reasoning.\n"
    )


def test_build_modelfile_emits_message_lines_with_blank_separator():
    pairs = [{"messages": [
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
    ]}]
    lines = T.build_modelfile("m", pairs).split("\n")
    assert lines[0] == "FROM m"
    assert lines[-3:] == ["MESSAGE user U1", "MESSAGE assistant A1", ""]


def test_build_modelfile_skips_pairs_with_fewer_than_two_messages():
    pairs = [
        {"messages": [{"role": "user", "content": "only"}]},
        {"messages": []},
        {},  # no "messages" key at all
        {"messages": [{"role": "user", "content": "U"},
                      {"role": "assistant", "content": "A"}]},
    ]
    out = T.build_modelfile("m", pairs)
    assert out.count("MESSAGE user ") == 1
    assert "only" not in out


def test_build_modelfile_uses_position_not_role_and_ignores_extra_messages():
    """BUG: messages[0] is always labelled 'user' and messages[1] 'assistant',
    whatever the actual roles say; messages[2:] are silently dropped."""
    pairs = [{"messages": [
        {"role": "assistant", "content": "FIRST"},
        {"role": "user", "content": "SECOND"},
        {"role": "user", "content": "THIRD"},
    ]}]
    out = T.build_modelfile("m", pairs)
    assert "MESSAGE user FIRST" in out
    assert "MESSAGE assistant SECOND" in out
    assert "THIRD" not in out


def test_build_modelfile_missing_content_keys_become_empty_strings():
    pairs = [{"messages": [{"role": "user"}, {"role": "assistant"}]}]
    out = T.build_modelfile("m", pairs)
    assert "MESSAGE user \nMESSAGE assistant \n" in out


def test_build_modelfile_applies_triple_quote_escaping_to_both_sides():
    pairs = [{"messages": [
        {"role": "user", "content": 'x """ y'},
        {"role": "assistant", "content": '"""'},
    ]}]
    out = T.build_modelfile("m", pairs)
    assert "MESSAGE user x ''' y" in out
    assert "MESSAGE assistant '''" in out
    assert '"""' not in out


def test_build_modelfile_multiline_content_breaks_the_directive():
    """BUG: newlines inside a message are emitted verbatim, so the continuation
    lines are no longer valid MESSAGE directives."""
    pairs = [{"messages": [
        {"role": "user", "content": "line1\nline2"},
        {"role": "assistant", "content": "ok"},
    ]}]
    out = T.build_modelfile("m", pairs)
    assert "MESSAGE user line1\nline2\n" in out
    stray = [ln for ln in out.split("\n") if ln == "line2"]
    assert stray == ["line2"]


# ══════════════════════════════════════════════════════════════════════════════════
# train_lora.run_ollama_modelfile_backend
# ══════════════════════════════════════════════════════════════════════════════════

def _args(**kw):
    base = dict(sft="", base_model="llama3.2:latest",
                backend="ollama-modelfile", max_examples=200)
    base.update(kw)
    return argparse.Namespace(**base)


def test_ollama_backend_exits_1_when_no_pairs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(T, "_HERE", str(tmp_path))
    empty = tmp_path / "s.jsonl"
    empty.write_text("")
    with pytest.raises(SystemExit) as exc:
        T.run_ollama_modelfile_backend(_args(sft=str(empty)))
    assert exc.value.code == 1
    assert "no SFT pairs loaded" in capsys.readouterr().err


def test_ollama_backend_writes_modelfile_then_creates_and_probes(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(T, "_HERE", str(tmp_path))
    sft = _write_jsonl(tmp_path / "s.jsonl", [
        {"messages": [{"role": "user", "content": "U"},
                      {"role": "assistant", "content": "A"}]},
    ])
    runner = FakeRun([0, 0])
    monkeypatch.setattr(T.subprocess, "run", runner)

    T.run_ollama_modelfile_backend(_args(sft=str(sft)))

    mf = tmp_path / "Modelfile"
    assert mf.exists()
    assert mf.read_text().startswith("FROM llama3.2:latest\n")
    assert "MESSAGE user U" in mf.read_text()

    assert runner.calls[0][0] == [
        "ollama", "create", "loci-tuned", "-f", str(mf)]
    assert runner.calls[0][1]["capture_output"] is False
    assert runner.calls[0][1]["timeout"] > 0, "the bake must be bounded"
    assert runner.calls[1][0] == [
        "ollama", "run", "loci-tuned",
        "What is the most important memory about authentication?"]
    assert len(runner.calls) == 2

    out = capsys.readouterr().out
    assert "[train] wrote Modelfile with 1 examples" in out
    assert "[train] model created. Running probe..." in out


def test_ollama_backend_create_failure_exits_with_that_code_and_skips_probe(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(T, "_HERE", str(tmp_path))
    sft = _write_jsonl(tmp_path / "s.jsonl", [
        {"messages": [{"role": "user", "content": "U"},
                      {"role": "assistant", "content": "A"}]}])
    runner = FakeRun([7])
    monkeypatch.setattr(T.subprocess, "run", runner)

    with pytest.raises(SystemExit) as exc:
        T.run_ollama_modelfile_backend(_args(sft=str(sft)))
    assert exc.value.code == 7
    assert len(runner.calls) == 1  # probe never runs
    assert "ollama create failed (exit 7)" in capsys.readouterr().err


def test_ollama_backend_probe_failure_is_non_fatal(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(T, "_HERE", str(tmp_path))
    sft = _write_jsonl(tmp_path / "s.jsonl", [
        {"messages": [{"role": "user", "content": "U"},
                      {"role": "assistant", "content": "A"}]}])
    monkeypatch.setattr(T.subprocess, "run", FakeRun([0, 3]))

    T.run_ollama_modelfile_backend(_args(sft=str(sft)))  # no SystemExit
    assert "probe run failed (exit 3)" in capsys.readouterr().err


def test_ollama_backend_honours_max_examples(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(T, "_HERE", str(tmp_path))
    sft = _write_jsonl(tmp_path / "s.jsonl", [
        {"messages": [{"role": "user", "content": f"U{i}"},
                      {"role": "assistant", "content": f"A{i}"}]}
        for i in range(5)])
    monkeypatch.setattr(T.subprocess, "run", FakeRun())

    T.run_ollama_modelfile_backend(_args(sft=str(sft), max_examples=2))

    mf = (tmp_path / "Modelfile").read_text()
    assert "U0" in mf and "U1" in mf and "U2" not in mf
    assert "wrote Modelfile with 2 examples" in capsys.readouterr().out


# ══════════════════════════════════════════════════════════════════════════════════
# train_lora.run_unsloth_backend
# ══════════════════════════════════════════════════════════════════════════════════

def test_unsloth_backend_emits_three_files_without_touching_a_gpu(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(T, "_HERE", str(tmp_path))
    sft = tmp_path / "data" / "sft_pairs.jsonl"
    sft.parent.mkdir()
    sft.write_text("")

    T.run_unsloth_backend(_args(sft=str(sft), backend="unsloth"))

    script = tmp_path / "run_unsloth.py"
    export = tmp_path / "export_gguf.sh"
    modelfile = tmp_path / "Modelfile"
    assert script.exists() and export.exists() and modelfile.exists()
    # the sft file itself is NOT read by this backend
    out = capsys.readouterr().out
    assert str(script) in out and str(export) in out and str(modelfile) in out
    assert "Next steps (GPU machine required)" in out
    assert "pip install unsloth trl transformers datasets bitsandbytes" in out


def test_unsloth_emitted_script_is_valid_python_and_interpolated(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(T, "_HERE", str(tmp_path))
    sft = tmp_path / "s.jsonl"
    sft.write_text("")
    T.run_unsloth_backend(_args(sft=str(sft)))

    text = (tmp_path / "run_unsloth.py").read_text()
    compile(text, "run_unsloth.py", "exec")  # catches .format() brace damage
    assert f'SFT_PAIRS   = "{os.path.abspath(str(sft))}"' in text
    assert f'OUTPUT_DIR  = "{os.path.join(str(tmp_path), "loci-lora")}"' in text
    assert 'LORA_R      = 16' in text
    assert 'LORA_ALPHA  = 16' in text
    assert "MAX_SEQ_LEN = 2048" in text
    # the doubled braces in the template survive as single braces
    assert '{"text": text}' in text
    assert 'f"<|user|>\\n{content}\\n"' in text
    assert "{{" not in text


def test_unsloth_backend_ignores_base_model_flag(tmp_path, monkeypatch):
    """BUG: --base-model is honoured by the ollama backend but hard-coded to
    unsloth/llama-3.2-3b-instruct here."""
    monkeypatch.setattr(T, "_HERE", str(tmp_path))
    sft = tmp_path / "s.jsonl"
    sft.write_text("")
    T.run_unsloth_backend(_args(sft=str(sft), base_model="qwen2.5:7b"))
    text = (tmp_path / "run_unsloth.py").read_text()
    assert 'BASE_MODEL  = "unsloth/llama-3.2-3b-instruct"' in text
    assert "qwen2.5" not in text


def test_unsloth_backend_modelfile_points_at_gguf_and_has_no_messages(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(T, "_HERE", str(tmp_path))
    sft = tmp_path / "s.jsonl"
    sft.write_text("")
    T.run_unsloth_backend(_args(sft=str(sft)))
    gguf = os.path.join(str(tmp_path), "loci-lora", "loci-tuned.q4_k_m.gguf")
    text = (tmp_path / "Modelfile").read_text()
    assert text.startswith(f"FROM {gguf}\n")
    assert "SYSTEM You are Loci" in text
    assert "MESSAGE" not in text


def test_unsloth_backend_overwrites_an_existing_modelfile(tmp_path, monkeypatch):
    """Running --backend unsloth after --backend ollama-modelfile clobbers the
    baked few-shot Modelfile."""
    monkeypatch.setattr(T, "_HERE", str(tmp_path))
    (tmp_path / "Modelfile").write_text("FROM llama3.2:latest\nMESSAGE user hi\n")
    sft = tmp_path / "s.jsonl"
    sft.write_text("")
    T.run_unsloth_backend(_args(sft=str(sft)))
    assert "MESSAGE" not in (tmp_path / "Modelfile").read_text()


def test_unsloth_export_script_is_executable_and_references_the_paths(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(T, "_HERE", str(tmp_path))
    sft = tmp_path / "s.jsonl"
    sft.write_text("")
    T.run_unsloth_backend(_args(sft=str(sft)))

    export = tmp_path / "export_gguf.sh"
    mode = stat.S_IMODE(export.stat().st_mode)
    assert mode == 0o755
    text = export.read_text()
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text
    assert f'LORA_DIR="{os.path.join(str(tmp_path), "loci-lora")}"' in text
    assert f'MODELFILE="{os.path.join(str(tmp_path), "Modelfile")}"' in text
    assert "ollama create loci-tuned -f" in text
    # the loci-lora directory is never created by this function
    assert not (tmp_path / "loci-lora").exists()


# ══════════════════════════════════════════════════════════════════════════════════
# train_lora.parse_args / main dispatch
# ══════════════════════════════════════════════════════════════════════════════════

def test_train_lora_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_lora.py", "--sft", "s.jsonl"])
    args = T.parse_args()
    assert args.sft == "s.jsonl"
    assert args.base_model == "llama3.2:latest"
    assert args.backend == "ollama-modelfile"
    assert args.max_examples == 200


def test_train_lora_parse_args_rejects_unknown_backend(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["train_lora.py", "--sft", "s", "--backend", "peft"])
    with pytest.raises(SystemExit):
        T.parse_args()


def test_train_lora_parse_args_requires_sft(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_lora.py"])
    with pytest.raises(SystemExit):
        T.parse_args()


@pytest.mark.parametrize("backend,called,other", [
    ("ollama-modelfile", "run_ollama_modelfile_backend", "run_unsloth_backend"),
    ("unsloth", "run_unsloth_backend", "run_ollama_modelfile_backend"),
])
def test_train_lora_main_dispatches_on_backend(monkeypatch, backend, called, other):
    seen = {}
    monkeypatch.setattr(T, called, lambda a: seen.setdefault("args", a))
    monkeypatch.setattr(T, other, lambda a: pytest.fail("wrong backend dispatched"))
    monkeypatch.setattr(
        sys, "argv", ["train_lora.py", "--sft", "s.jsonl", "--backend", backend])
    T.main()
    assert seen["args"].backend == backend


# ══════════════════════════════════════════════════════════════════════════════════
# cross-stage integration: collect → format_sft → train_lora
# ══════════════════════════════════════════════════════════════════════════════════

def test_correction_content_round_trips_from_collect_into_sft_pairs(tmp_path):
    """The three stages agree on the correction envelope: collect json-encodes
    {failed, corrected} into `content`, format_sft decodes it, train_lora bakes it."""
    failed, corrected = _long("f"), _long("c")
    recs = C.build_unified_records(
        [], [], [{"failed_content": failed, "corrected_content": corrected,
                  "session_id": "s1"}], [], "T")
    pairs = F.pairs_from_corrections(recs)
    assert pairs[0]["messages"][0]["content"] == failed
    assert pairs[0]["messages"][1]["content"] == corrected

    modelfile = T.build_modelfile("base:tag", pairs)
    assert f"MESSAGE user {failed}" in modelfile
    assert f"MESSAGE assistant {corrected}" in modelfile


def test_short_guard_commands_are_dropped_by_the_formatter(tmp_path):
    """Real guard-log commands are usually shorter than MIN_CONTENT_LEN, so the
    pipeline silently produces zero pairs from them."""
    recs = C.build_unified_records(
        [], [], [{"failed_content": "ls -la", "corrected_content": "ls -l"}], [], "T")
    assert F.pairs_from_corrections(recs) == []
    assert F.pairs_from_corrections_dpo(recs) == []
