"""Investigation logs are UTF-8 on every platform.

The JSONL reader decodes UTF-8, but the writer opened files in the locale encoding.
On Windows that is cp1252, so an em dash was written as byte 0x97 and every later
investigation_search_fast over that file raised; a character outside cp1252 made
investigation_store fail outright. The writers now pin encoding="utf-8".

The tests record the encoding each open/fdopen/read_text is called with, so they
fail on a UTF-8-locale CI runner too, where the default would hide the bug.
"""

import builtins
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import inv_store  # noqa: E402

_TEXT = "café — → ✓ \U0001d518"  # cp1252 lacks the arrow, check mark and U+1D518


def _utf8_lines(path):
    """Lines of *path* decoded strictly as UTF-8 (text mode writes CRLF on Windows)."""
    return path.read_bytes().decode("utf-8", errors="strict").splitlines()


class InvStoreUtf8Test(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.opened = []

    def tearDown(self):
        self._tmp.cleanup()

    def _recording_open(self, *args, **kwargs):
        self.opened.append(kwargs.get("encoding"))
        return builtins.open(*args, **kwargs)

    def test_append_writes_utf8_and_reads_back(self):
        path = self.dir / "findings.jsonl"
        entry = {"id": "f1", "text": _TEXT}
        with mock.patch.object(inv_store, "open", self._recording_open, create=True):
            inv_store._append_jsonl(path, entry)
        self.assertEqual(self.opened, ["utf-8"])
        self.assertEqual(_utf8_lines(path), [inv_store._json_dumps(entry)])
        self.assertEqual(inv_store._read_jsonl(path), [entry])

    def test_binary_mode_lock_passes_no_encoding(self):
        path = self.dir / "blob.bin"
        with mock.patch.object(inv_store, "open", self._recording_open, create=True):
            with inv_store._locked_file(path, "ab") as fh:
                fh.write(b"\x97\x00")
        self.assertEqual(self.opened, [None])
        self.assertEqual(path.read_bytes(), b"\x97\x00")

    def test_rewrite_reads_and_writes_utf8(self):
        path = self.dir / "procedures.jsonl"
        kept = inv_store._json_dumps({"id": "b", "text": _TEXT})
        path.write_bytes((inv_store._json_dumps({"id": "a", "text": _TEXT}) + "\n" + kept + "\n").encode("utf-8"))
        fdopen_encodings, read_encodings = [], []
        real_fdopen, real_read_text = os.fdopen, Path.read_text

        def recording_fdopen(fd, *args, **kwargs):
            fdopen_encodings.append(kwargs.get("encoding"))
            return real_fdopen(fd, *args, **kwargs)

        def recording_read_text(self_path, *args, **kwargs):
            read_encodings.append(kwargs.get("encoding"))
            return real_read_text(self_path, *args, **kwargs)

        with mock.patch.object(inv_store.os, "fdopen", recording_fdopen), \
                mock.patch.object(Path, "read_text", recording_read_text):
            replaced = inv_store._rewrite_jsonl_preserving(
                path, lambda rec: {**rec, "status": "done"} if rec.get("id") == "a" else None)
        self.assertEqual(replaced, 1)
        self.assertEqual(read_encodings, ["utf-8"])
        self.assertEqual(fdopen_encodings, ["utf-8"])
        expected = inv_store._json_dumps({"id": "a", "text": _TEXT, "status": "done"}) + "\n" + kept + "\n"
        self.assertEqual(_utf8_lines(path), expected.splitlines())


if __name__ == "__main__":
    unittest.main()
