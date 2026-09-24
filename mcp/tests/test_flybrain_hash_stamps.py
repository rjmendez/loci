import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_banc_adapter as banc  # noqa: E402
import flybrain_hash_stamps as stamps  # noqa: E402
from flybrain_banc_adapter import BancAdapterError, BancAdapterErrorCode  # noqa: E402
from test_flybrain_banc_adapter import build_snapshot  # noqa: E402


class _CountingHasher:
    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, path):
        self.calls.append(Path(path).name)
        return stamps.sha256_file(path)


@pytest.fixture
def hasher(monkeypatch):
    counting = _CountingHasher()
    monkeypatch.setattr(banc, "sha256_file", counting)
    return counting


def _stamp_files(snap: Path) -> list[Path]:
    root = snap / stamps.HASH_STAMP_RELATIVE_DIR
    return sorted(root.glob("*.json")) if root.is_dir() else []


def _tamper_keep_size_and_mtime(path: Path) -> None:
    st = os.stat(path)
    data = bytearray(path.read_bytes())
    data[-3] = ord("9") if data[-3] != ord("9") else ord("8")
    path.write_bytes(bytes(data))
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))


def test_verify_file_sha256_unit(tmp_path):
    target = tmp_path / "f.bin"
    target.write_bytes(b"payload")
    sha = stamps.sha256_file(target)
    cache = stamps.HashStampCache(tmp_path / "stamps")
    first = stamps.verify_file_sha256(target, relative_path="f.bin", expected_sha256=sha, cache=cache)
    assert (first.method, first.matched, first.stamp_written) == (stamps.METHOD_HASHED, True, True)
    second = stamps.verify_file_sha256(target, relative_path="f.bin", expected_sha256=sha, cache=cache)
    assert (second.method, second.matched) == (stamps.METHOD_STAMP, True)
    forced = stamps.verify_file_sha256(target, relative_path="f.bin", expected_sha256=sha, cache=cache, force=True)
    assert (forced.method, forced.stamp_written) == (stamps.METHOD_HASHED, False)  # write-once
    bad = stamps.verify_file_sha256(target, relative_path="f.bin", expected_sha256="0" * 64, cache=cache)
    assert bad.matched is False and bad.stamp_written is False
    payload = json.loads(next((tmp_path / "stamps").glob("*.json")).read_text())
    assert payload["schema_version"] == stamps.HASH_STAMP_SCHEMA_VERSION
    assert payload["sha256"] == sha and payload["size_bytes"] == 7


def test_corrupt_stamp_is_ignored_and_replaced(tmp_path):
    target = tmp_path / "f.bin"
    target.write_bytes(b"payload")
    sha = stamps.sha256_file(target)
    cache = stamps.HashStampCache(tmp_path / "stamps")
    stamp = cache.stamp_path("f.bin", sha, stamps.FileStat.of(target))
    stamp.parent.mkdir(parents=True)
    stamp.write_text("{not json")
    check = stamps.verify_file_sha256(target, relative_path="f.bin", expected_sha256=sha, cache=cache)
    assert check.method == stamps.METHOD_HASHED and check.stamp_written is True
    assert cache.lookup("f.bin", sha, stamps.FileStat.of(target))


def test_second_open_uses_stamps_not_hashes(tmp_path, hasher):
    snap, _ = build_snapshot(tmp_path)
    hasher.calls.clear()  # ignore the hashes build_snapshot computed
    first = banc.open_banc_snapshot(tmp_path)
    assert len(hasher.calls) == 3
    assert set(first.hash_verification.values()) == {stamps.METHOD_HASHED}
    assert len(_stamp_files(snap)) == 3
    hasher.calls.clear()
    second = banc.open_banc_snapshot(tmp_path)
    assert hasher.calls == []
    assert set(second.hash_verification.values()) == {stamps.METHOD_STAMP}
    assert second.product_sha256 == first.product_sha256


def test_only_required_products_are_hashed_by_default(tmp_path, hasher):
    build_snapshot(tmp_path)
    hasher.calls.clear()  # ignore the hashes build_snapshot computed
    opened = banc.open_banc_snapshot(tmp_path, required_roles=(banc.ROLE_META,))
    assert hasher.calls == [Path(banc.BANC_PRODUCT_PATHS[banc.ROLE_META]).name]
    assert opened.hash_verification[banc.BANC_PRODUCT_PATHS[banc.ROLE_EDGELIST_V3]] == stamps.METHOD_SIZE_ONLY


def test_mtime_change_invalidates_stamp(tmp_path, hasher):
    snap, _ = build_snapshot(tmp_path)
    hasher.calls.clear()  # ignore the hashes build_snapshot computed
    banc.open_banc_snapshot(tmp_path)
    meta = snap / banc.BANC_PRODUCT_PATHS[banc.ROLE_META]
    st = os.stat(meta)
    os.utime(meta, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    hasher.calls.clear()
    opened = banc.open_banc_snapshot(tmp_path)
    assert hasher.calls == [meta.name]
    assert opened.hash_verification[banc.BANC_PRODUCT_PATHS[banc.ROLE_META]] == stamps.METHOD_HASHED


def test_tamper_with_new_mtime_fails_cached_open(tmp_path):
    snap, _ = build_snapshot(tmp_path)
    banc.open_banc_snapshot(tmp_path)
    nt = snap / banc.BANC_PRODUCT_PATHS[banc.ROLE_NT_PREDICTION]
    st = os.stat(nt)
    _tamper_keep_size_and_mtime(nt)
    os.utime(nt, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    with pytest.raises(BancAdapterError) as exc:
        banc.open_banc_snapshot(tmp_path)
    assert exc.value.code is BancAdapterErrorCode.INTEGRITY_MISMATCH


def test_explicit_verify_catches_same_size_same_mtime_tamper(tmp_path, hasher):
    """Documented limit of the stamp cache; the explicit verify closes it."""
    snap, _ = build_snapshot(tmp_path)
    hasher.calls.clear()  # ignore the hashes build_snapshot computed
    banc.open_banc_snapshot(tmp_path)
    _tamper_keep_size_and_mtime(snap / banc.BANC_PRODUCT_PATHS[banc.ROLE_NT_PREDICTION])
    banc.open_banc_snapshot(tmp_path)  # stamped: not detected
    with pytest.raises(BancAdapterError) as exc:
        banc.open_banc_snapshot(tmp_path, verify_hashes=True)
    assert exc.value.code is BancAdapterErrorCode.INTEGRITY_MISMATCH


def test_explicit_verify_hashes_every_listed_file_and_never_rewrites_stamps(tmp_path, hasher):
    snap, _ = build_snapshot(tmp_path)
    hasher.calls.clear()  # ignore the hashes build_snapshot computed
    banc.open_banc_snapshot(tmp_path)
    before = {path.name: path.stat().st_mtime_ns for path in _stamp_files(snap)}
    hasher.calls.clear()
    opened = banc.open_banc_snapshot(tmp_path, required_roles=(banc.ROLE_META,), verify_hashes=True)
    assert len(hasher.calls) == 3
    assert set(opened.hash_verification.values()) == {stamps.METHOD_HASHED}
    assert {path.name: path.stat().st_mtime_ns for path in _stamp_files(snap)} == before


def test_size_only_mode_hashes_nothing(tmp_path, hasher):
    build_snapshot(tmp_path)
    hasher.calls.clear()  # ignore the hashes build_snapshot computed
    opened = banc.open_banc_snapshot(tmp_path, verify_hashes=False)
    assert hasher.calls == []
    assert set(opened.hash_verification.values()) == {stamps.METHOD_SIZE_ONLY}


def test_sidecar_is_checked_before_any_hashing(tmp_path, hasher):
    snap, _ = build_snapshot(tmp_path)
    hasher.calls.clear()  # ignore the hashes build_snapshot computed
    (snap / "manifest" / "manifest.sha256").write_text("0" * 64 + "\n")
    with pytest.raises(BancAdapterError) as exc:
        banc.open_banc_snapshot(tmp_path)
    assert exc.value.code is BancAdapterErrorCode.INTEGRITY_MISMATCH
    assert hasher.calls == []


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores directory permissions")
def test_unwritable_stamp_dir_warns_but_verifies(tmp_path, hasher):
    snap, _ = build_snapshot(tmp_path)
    hasher.calls.clear()  # ignore the hashes build_snapshot computed
    stamp_dir = snap / stamps.HASH_STAMP_RELATIVE_DIR
    stamp_dir.mkdir(parents=True)
    stamp_dir.chmod(0o500)
    try:
        opened = banc.open_banc_snapshot(tmp_path)
    finally:
        stamp_dir.chmod(0o700)
    assert set(opened.hash_verification.values()) == {stamps.METHOD_HASHED}
    assert any("hash stamp not written" in warning for warning in opened.warnings)
