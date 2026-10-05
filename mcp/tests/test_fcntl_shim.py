"""The Windows fcntl shim must release locks, quickly, from any file position.

Regression: LOCK_UN used to fall through to a blocking lock (~10 s) and release nothing, so every
locked store operation on Windows cost ~10 s per release.
"""
import errno
import os
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="the shim is Windows-only")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fcntl as F  # noqa: E402  (resolves to mcp/fcntl.py on Windows)


@pytest.fixture
def lockfile(tmp_path):
    fds = []

    def open_fd():
        fd = os.open(str(tmp_path / "x.lock"), os.O_CREAT | os.O_RDWR)
        fds.append(fd)
        return fd

    yield open_fd
    for fd in fds:
        os.close(fd)


def test_unlock_is_fast_and_really_releases(lockfile):
    a, b = lockfile(), lockfile()
    F.flock(a, F.LOCK_EX | F.LOCK_NB)
    with pytest.raises(OSError) as held:
        F.flock(b, F.LOCK_EX | F.LOCK_NB)
    assert held.value.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK)  # refused because it is held
    t = time.monotonic()
    F.flock(a, F.LOCK_UN)
    assert time.monotonic() - t < 1.0
    F.flock(b, F.LOCK_EX | F.LOCK_NB)  # now free
    F.flock(b, F.LOCK_UN)


def test_unlock_works_after_the_position_moved(lockfile):
    a, b = lockfile(), lockfile()
    F.flock(a, F.LOCK_EX | F.LOCK_NB)
    os.write(a, b"some appended bytes")  # moves the file position, as a locked append does
    assert os.lseek(a, 0, os.SEEK_CUR) == 19
    F.flock(a, F.LOCK_UN)
    assert os.lseek(a, 0, os.SEEK_CUR) == 19  # position put back
    F.flock(b, F.LOCK_EX | F.LOCK_NB)
    F.flock(b, F.LOCK_UN)


def test_unlock_of_an_unheld_lock_is_harmless(lockfile):
    a = lockfile()
    F.flock(a, F.LOCK_UN)
    F.flock(a, F.LOCK_EX | F.LOCK_NB)
    F.flock(a, F.LOCK_UN)
    F.flock(a, F.LOCK_UN)
    F.flock(a, F.LOCK_EX | F.LOCK_NB)  # still lockable after the extra unlocks
    F.flock(a, F.LOCK_UN)
    assert os.fstat(a).st_size >= 0
