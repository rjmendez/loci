import string

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

import inv_store
import model_json
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


def test_inv_dir_rejects_overlong_ids_with_value_error(tmp_path, monkeypatch):
    monkeypatch.setattr(inv_store, "_get_memory_dir", lambda: tmp_path)
    with pytest.raises(ValueError):
        inv_store._inv_dir("a" * 256)
