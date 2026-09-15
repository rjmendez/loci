import json
import string

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

import inv_store
import llm_tools
import model_json
import server
import text_ops


_VALID_ID_CHARS = string.ascii_letters + string.digits + "_-"
_PRINTABLE_NO_SURROGATES = st.characters(blacklist_categories=("Cs",))
_ADVERSARIAL_TEXT = st.one_of(
    st.sampled_from([
        "",
        "\n",
        "\r\n",
        "\x00",
        "../escape",
        "..\\escape",
        "/absolute/path",
        "C:\\absolute\\path",
        "\n{\"forged\": true}\n",
        "prefix\n{\"forged\": true}\n{\"forged\": false}",
        "line1\rline2",
        "\u2028{\"fake\": 1}",
        "\u2029{\"fake\": 2}",
        "ＡＢＣ-１２３",
        "𝖆𝖉𝖒𝖎𝖓",
    ]),
    st.text(_PRINTABLE_NO_SURROGATES, min_size=0, max_size=512),
)
_MALFORMED_UNICODE_TEXT = st.text(
    st.characters(min_codepoint=0, max_codepoint=0x10FFFF),
    min_size=0,
    max_size=4096,
)
_NESTING_DEPTH = st.integers(min_value=0, max_value=200)
_SCALAR_OR_TEXT = st.one_of(
    _ADVERSARIAL_TEXT,
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2 ** 31), max_value=(2 ** 31) - 1),
)
_LABEL_INPUT = st.one_of(
    st.lists(_SCALAR_OR_TEXT, min_size=0, max_size=5),
    _SCALAR_OR_TEXT,
    st.tuples(_SCALAR_OR_TEXT, _SCALAR_OR_TEXT),
    st.dictionaries(_ADVERSARIAL_TEXT, _SCALAR_OR_TEXT, min_size=0, max_size=3),
)
_MAX_CHARS_INPUT = st.one_of(
    st.integers(min_value=-(2 ** 31), max_value=(2 ** 31) - 1),
    _ADVERSARIAL_TEXT,
    st.none(),
    st.booleans(),
)
_INVESTIGATION_NOTE_FIELD = st.sampled_from([
    "context",
    "hypothesis",
    "next_step",
    "open_question_add",
    "open_question_remove",
    "checked_source",
    "closed_summary",
])


def _jsonl_roundtrip_entry(field: str, text: str) -> dict:
    base = {
        "finding_id": "finding-1",
        "ts": "2026-09-15T00:00:00+00:00",
        "record_type": "resolution",
    }
    base[field] = text
    if field == "note":
        base["resolution"] = "fixed"
    else:
        base["active"] = True
    return base


def _assert_inside(root, candidate):
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    assert resolved_candidate.is_dir()
    assert resolved_candidate.is_relative_to(resolved_root)


def _canonical_labels(labels) -> list[str]:
    if labels is None:
        seq = []
    elif isinstance(labels, list):
        seq = labels
    elif isinstance(labels, (tuple, set)):
        seq = list(labels)
    else:
        seq = [labels]
    return [str(item) for item in seq]


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    investigation_id=st.one_of(
        st.sampled_from([
            "",
            ".",
            "..",
            "../escape",
            "..\\escape",
            "/etc/passwd",
            "C:\\windows",
            "\x00",
            "ＡＢＣ-１２３",
            "𝖆𝖉𝖒𝖎𝖓",
        ]),
        st.text(st.characters(min_codepoint=0, max_codepoint=0x10FFFF), min_size=0, max_size=300),
        st.text(st.sampled_from(tuple(_VALID_ID_CHARS)), min_size=256, max_size=400),
    )
)
def test_inv_dir_fuzz_never_escapes_or_crashes(tmp_path, monkeypatch, investigation_id):
    root = tmp_path / f"inv-root-{len(list(tmp_path.iterdir()))}"
    root.mkdir()
    monkeypatch.setattr(inv_store, "_get_memory_dir", lambda: root)
    try:
        candidate = inv_store._inv_dir(investigation_id)
    except ValueError:
        return
    _assert_inside(root, candidate)


@pytest.mark.parametrize(
    ("path_factory", "field"),
    [
        (lambda inv_id: inv_store._finding_updates_path(inv_id), "note"),
        (lambda inv_id: inv_store._inv_dir(inv_id) / "retractions.jsonl", "reason"),
    ],
)
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(text=_ADVERSARIAL_TEXT)
def test_jsonl_append_fuzz_blocks_record_injection(tmp_path, monkeypatch, path_factory, field, text):
    root = tmp_path / f"jsonl-root-{len(list(tmp_path.iterdir()))}"
    root.mkdir()
    monkeypatch.setattr(inv_store, "_get_memory_dir", lambda: root)
    inv_id = "case-jsonl"
    path = path_factory(inv_id)
    entry = _jsonl_roundtrip_entry(field, text)

    inv_store._append_jsonl(path, entry)

    raw = path.read_text()
    assert raw.endswith("\n")
    assert len(raw.splitlines()) == 1
    assert inv_store._read_jsonl(path) == [entry]


@settings(max_examples=200, deadline=None)
@given(text=_MALFORMED_UNICODE_TEXT, depth=_NESTING_DEPTH)
def test_extract_json_object_fuzz_never_raises(text, depth):
    nested = ("{\"a\":" * depth) + "0" + ("}" * depth)
    wrapped = f"{text} ```json\n{nested}\n``` {text}"
    result = model_json.extract_json_object(wrapped)
    assert result is None or isinstance(result, dict)


@settings(max_examples=200, deadline=None)
@given(
    text=_MALFORMED_UNICODE_TEXT,
    labels=st.lists(_MALFORMED_UNICODE_TEXT, min_size=0, max_size=5),
    max_chars=st.integers(min_value=-64, max_value=2048),
)
def test_text_ops_fuzz_never_raises(text, labels, max_chars):
    label_result = text_ops.classify(
        text,
        labels,
        gen_fn=lambda prompt, *, fmt=None, max_tokens=256: {
            "text": labels[0] if labels else "",
            "ok": bool(labels),
        },
    )
    assert set(label_result) == {"label", "degraded"}

    truncated = text[: max(0, min(len(text), max_chars if max_chars > 0 else 0))]
    compress_result = text_ops.compress(
        text,
        max_chars=max_chars,
        gen_fn=lambda prompt, *, fmt=None, max_tokens=256: {"text": truncated, "ok": True},
    )
    assert set(compress_result) == {"text", "degraded"}
    assert isinstance(compress_result["text"], str)
    if max_chars > 0:
        assert len(compress_result["text"]) <= max_chars


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    text=_MALFORMED_UNICODE_TEXT,
    labels=_LABEL_INPUT,
    reply=_MALFORMED_UNICODE_TEXT,
)
def test_classify_text_tool_fuzz_never_raises(monkeypatch, text, labels, reply):
    monkeypatch.setattr(
        text_ops,
        "_resolve_gen_fn",
        lambda gen_fn: (lambda prompt, *, fmt=None, max_tokens=256: {"text": reply, "ok": True}),
    )
    result = llm_tools.classify_text(text, labels)
    parsed = json.loads(result)
    assert set(parsed) == {"label", "degraded"}
    assert parsed["label"] is None or parsed["label"] in _canonical_labels(labels)


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    text=_MALFORMED_UNICODE_TEXT,
    max_chars=_MAX_CHARS_INPUT,
)
def test_compress_text_tool_fuzz_never_raises(monkeypatch, text, max_chars):
    monkeypatch.setattr(
        text_ops,
        "_resolve_gen_fn",
        lambda gen_fn: (lambda prompt, *, fmt=None, max_tokens=256: {"text": text, "ok": True}),
    )
    parsed = json.loads(llm_tools.compress_text(text, max_chars=max_chars))
    assert set(parsed) == {"text", "degraded"}
    assert isinstance(parsed["text"], str)
    try:
        budget = int(max_chars)
    except Exception:
        budget = 600
    if budget > 0:
        assert len(parsed["text"]) <= budget
    else:
        assert parsed["text"] == ""


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(field=_INVESTIGATION_NOTE_FIELD, value=_SCALAR_OR_TEXT)
def test_investigation_note_tool_fuzz_never_raises(tmp_path, monkeypatch, field, value):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    server.investigation_start("case-note-fuzz", "fuzz note", "ctx")

    note_value = value
    if field == "checked_source" and isinstance(value, str) and ":" not in value:
        note_value = f"tool: {value}"

    parsed = json.loads(
        server.investigation_note("case-note-fuzz", field, note_value)
    )
    assert isinstance(parsed, dict)
    if "error" in parsed:
        assert isinstance(parsed["error"], str)
        return

    manifest = parsed["manifest"]
    assert parsed["updated"] == field
    assert all(isinstance(item, str) and item.strip() for item in manifest["open_questions"])
    assert manifest["closed_summary"] is None or (
        isinstance(manifest["closed_summary"], str) and manifest["closed_summary"].strip()
    )
    for tool_name, rows in manifest["checked_sources"].items():
        assert isinstance(tool_name, str) and tool_name.strip()
        for row in rows:
            assert isinstance(row["summary"], str) and row["summary"].strip()


def test_inv_dir_rejects_overlong_ids_with_value_error(tmp_path, monkeypatch):
    monkeypatch.setattr(inv_store, "_get_memory_dir", lambda: tmp_path)
    with pytest.raises(ValueError):
        inv_store._inv_dir("a" * 256)


def test_classify_text_scalar_labels_degrade_instead_of_raising(monkeypatch):
    monkeypatch.setattr(
        text_ops,
        "_resolve_gen_fn",
        lambda gen_fn: (lambda prompt, *, fmt=None, max_tokens=256: {"text": "anything", "ok": True}),
    )
    parsed = json.loads(llm_tools.classify_text("hello", 5))
    assert parsed == {"label": None, "degraded": True}


def test_investigation_note_rejects_none_checked_source_instead_of_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    server.investigation_start("case-note-regression", "fuzz note", "ctx")
    parsed = json.loads(server.investigation_note("case-note-regression", "checked_source", None))
    assert "error" in parsed
