"""Regression coverage for private SQLite storage artifacts."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

import hermes_lcm.db_bootstrap as db_bootstrap_module
import hermes_lcm.maintenance as maintenance_module
import hermes_lcm.sqlite_util as sqlite_util_module
from hermes_lcm.maintenance import backup_database, rotate_backup_database
from hermes_lcm.store import MessageStore, build_message_fts_spec


_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires POSIX mode semantics")


@contextmanager
def _process_umask(mask: int):
    previous = os.umask(mask)
    try:
        yield
    finally:
        os.umask(previous)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _sqlite_artifacts(path: Path) -> list[Path]:
    return [path, *(path.with_name(path.name + suffix) for suffix in _SQLITE_SIDECAR_SUFFIXES)]


def _assert_private_sqlite_artifacts(path: Path) -> None:
    artifacts = [artifact for artifact in _sqlite_artifacts(path) if artifact.exists()]
    assert artifacts
    assert {artifact.name: _mode(artifact) for artifact in artifacts} == {
        artifact.name: 0o600 for artifact in artifacts
    }


def _seed_searchable_store(db_path: Path) -> None:
    store = MessageStore(db_path)
    try:
        store.append(
            "sidecar-race",
            {"role": "user", "content": "durable sidecar lifecycle token"},
        )
        store.commit()
    finally:
        store.close()


def _assert_searchable_store_integrity(store: MessageStore) -> None:
    conn = store.connection
    assert conn is not None
    assert conn.execute("SELECT content FROM messages").fetchone()[0] == (
        "durable sidecar lifecycle token"
    )
    assert store.search("lifecycle", limit=10)[0]["content"] == (
        "durable sidecar lifecycle token"
    )
    assert db_bootstrap_module.check_external_content_fts_integrity(
        conn, build_message_fts_spec()
    )["status"] == "pass"
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_message_store_creates_private_database_and_sidecars_under_umask_022(tmp_path):
    db_path = tmp_path / "database" / "lcm.db"

    with _process_umask(0o022):
        store = MessageStore(db_path)
        try:
            store.append("session", {"role": "user", "content": "private"})
            store.commit()

            assert _mode(db_path.parent) == 0o700
            assert db_path.with_name(db_path.name + "-wal").exists()
            assert db_path.with_name(db_path.name + "-shm").exists()
            _assert_private_sqlite_artifacts(db_path)
        finally:
            store.close()


def test_message_store_refuses_created_directory_swap_before_chmod(tmp_path, monkeypatch):
    shared_parent = tmp_path / "shared"
    shared_parent.mkdir(mode=0o777)
    shared_parent.chmod(0o777)
    db_dir = shared_parent / "database"
    db_path = db_dir / "lcm.db"
    displaced_dir = shared_parent / "database-displaced"
    unrelated_target = tmp_path / "unrelated-target"
    unrelated_target.mkdir(mode=0o755)
    unrelated_target.chmod(0o755)
    real_open = os.open
    real_path_chmod = Path.chmod
    swapped = False

    def swap_created_directory():
        nonlocal swapped
        if swapped or not db_dir.is_dir():
            return
        swapped = True
        db_dir.rename(displaced_dir)
        db_dir.symlink_to(unrelated_target, target_is_directory=True)

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        if dir_fd is not None and path == db_dir.name:
            swap_created_directory()
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    def swapping_path_chmod(path, mode, *, follow_symlinks=True):
        if path == db_dir:
            swap_created_directory()
        return real_path_chmod(path, mode, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(os, "open", swapping_open)
    monkeypatch.setattr(Path, "chmod", swapping_path_chmod)

    with pytest.raises(OSError):
        MessageStore(db_path)

    assert swapped is True
    assert _mode(unrelated_target) == 0o755


def test_message_store_tightens_compatible_existing_database_artifacts(tmp_path):
    db_dir = tmp_path / "existing"
    db_dir.mkdir(mode=0o755)
    db_dir.chmod(0o755)
    db_path = db_dir / "lcm.db"
    existing = sqlite3.connect(db_path)
    try:
        existing.execute("PRAGMA journal_mode=WAL")
        existing.execute("CREATE TABLE legacy (value TEXT)")
        existing.execute("INSERT INTO legacy VALUES ('retained')")
        existing.commit()

        wal_path = db_path.with_name(db_path.name + "-wal")
        shm_path = db_path.with_name(db_path.name + "-shm")
        assert wal_path.exists()
        assert shm_path.exists()
        for artifact in (db_path, wal_path, shm_path):
            artifact.chmod(0o644)

        with _process_umask(0o022):
            store = MessageStore(db_path)
            try:
                assert store.connection.execute("SELECT value FROM legacy").fetchone()[0] == "retained"
                assert _mode(db_dir) == 0o755
                _assert_private_sqlite_artifacts(db_path)
            finally:
                store.close()
    finally:
        existing.close()


@pytest.mark.parametrize("suffix", _SQLITE_SIDECAR_SUFFIXES)
def test_message_store_refuses_symlinked_sidecar_before_chmod(tmp_path, suffix):
    db_path = tmp_path / "lcm.db"
    target = tmp_path / "unrelated.txt"
    target.write_text("shared", encoding="utf-8")
    target.chmod(0o644)
    db_path.with_name(db_path.name + suffix).symlink_to(target)

    with pytest.raises(OSError, match="SQLite artifact"):
        MessageStore(db_path)

    assert _mode(target) == 0o644


@pytest.mark.parametrize("suffix", _SQLITE_SIDECAR_SUFFIXES)
def test_message_store_refuses_hardlinked_sidecar_before_chmod(tmp_path, suffix):
    db_path = tmp_path / "lcm.db"
    target = tmp_path / "unrelated.txt"
    target.write_text("shared", encoding="utf-8")
    target.chmod(0o644)
    os.link(target, db_path.with_name(db_path.name + suffix))

    with pytest.raises(OSError, match="SQLite artifact"):
        MessageStore(db_path)

    assert _mode(target) == 0o644


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_message_store_refuses_sidecar_link_swap_before_chmod(
    tmp_path,
    monkeypatch,
    link_kind,
):
    db_path = tmp_path / "lcm.db"
    sidecar = db_path.with_name(db_path.name + "-wal")
    sidecar.write_text("replace me", encoding="utf-8")
    target = tmp_path / "unrelated.txt"
    target.write_text("shared", encoding="utf-8")
    target.chmod(0o644)
    real_open = os.open
    swapped = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if not swapped and dir_fd is not None and path == sidecar.name:
            swapped = True
            sidecar.unlink()
            if link_kind == "symlink":
                sidecar.symlink_to(target)
            else:
                os.link(target, sidecar)
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(sqlite_util_module.os, "open", swapping_open)

    with pytest.raises(OSError):
        MessageStore(db_path)

    assert swapped is True
    assert _mode(target) == 0o644


def test_message_store_refuses_sidecar_replacement_after_open_before_fstat(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "lcm.db"
    sidecar = db_path.with_name(db_path.name + "-journal")
    replacement = tmp_path / "replacement-journal"
    sidecar.write_bytes(b"replace me")
    replacement.write_bytes(b"unrelated")
    sidecar.chmod(0o644)
    replacement.chmod(0o644)
    real_open = os.open
    swapped = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if dir_fd is None:
            return real_open(path, flags, mode)
        fd = real_open(path, flags, mode, dir_fd=dir_fd)
        if not swapped and dir_fd is not None and path == sidecar.name:
            swapped = True
            sidecar.unlink()
            replacement.rename(sidecar)
        return fd

    monkeypatch.setattr(sqlite_util_module.os, "open", swapping_open)

    with pytest.raises(OSError, match="directory entry changed while opening"):
        MessageStore(db_path)

    assert swapped is True
    assert _mode(sidecar) == 0o644


def test_message_store_tolerates_sidecar_disappearing_between_stat_and_open(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "lcm.db"
    _seed_searchable_store(db_path)
    journal = db_path.with_name(db_path.name + "-journal")
    journal.write_bytes(b"transient rollback journal")
    real_open = os.open
    disappeared = False

    def disappearing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal disappeared
        if not disappeared and dir_fd is not None and path == journal.name:
            disappeared = True
            journal.unlink()
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(sqlite_util_module.os, "open", disappearing_open)

    store = MessageStore(db_path)
    try:
        assert disappeared is True
        _assert_searchable_store_integrity(store)
    finally:
        store.close()


def test_message_store_tolerates_sidecar_unlinked_between_open_and_fstat(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "lcm.db"
    _seed_searchable_store(db_path)
    journal = db_path.with_name(db_path.name + "-journal")
    journal.write_bytes(b"transient rollback journal")
    real_open = os.open
    unlinked = False

    def unlinking_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal unlinked
        if dir_fd is None:
            return real_open(path, flags, mode)
        fd = real_open(path, flags, mode, dir_fd=dir_fd)
        if not unlinked and path == journal.name:
            unlinked = True
            journal.unlink()
        return fd

    monkeypatch.setattr(sqlite_util_module.os, "open", unlinking_open)

    store = MessageStore(db_path)
    try:
        assert unlinked is True
        _assert_searchable_store_integrity(store)
    finally:
        store.close()


def test_message_store_preserves_sqlite_memory_sentinel_and_cwd_permissions(tmp_path, monkeypatch):
    tmp_path.chmod(0o755)
    monkeypatch.chdir(tmp_path)

    store = MessageStore(":memory:")
    try:
        store.append("session", {"role": "user", "content": "memory-only"})
        store.commit()
        assert store.connection.execute("SELECT content FROM messages").fetchone()[0] == "memory-only"
    finally:
        store.close()

    assert _mode(tmp_path) == 0o755
    assert not (tmp_path / ":memory:").exists()


def test_maintenance_creates_private_backups_and_tightens_existing_slot(tmp_path):
    db_path = tmp_path / "database" / "lcm.db"
    store = MessageStore(db_path)
    store.append("session", {"role": "user", "content": "backup"})
    store.commit()

    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(mode=0o755)
    backup_dir.chmod(0o755)
    rotate_path = backup_dir / "rotate-latest.sqlite3"
    rotate_path.write_bytes(b"stale")
    rotate_sidecars = [
        rotate_path.with_name(rotate_path.name + suffix)
        for suffix in _SQLITE_SIDECAR_SUFFIXES
    ]
    for artifact in [rotate_path, *rotate_sidecars]:
        if not artifact.exists():
            artifact.write_bytes(b"stale")
        artifact.chmod(0o644)

    engine = SimpleNamespace(
        _store=store,
        _dag=SimpleNamespace(_conn=store.connection),
        _lifecycle=None,
        backup_dir=lambda: backup_dir,
        rotate_backup_path=lambda: rotate_path,
    )

    try:
        with _process_umask(0o022):
            timestamped = backup_database(engine)
            rotated = rotate_backup_database(engine)

        assert timestamped["ok"] is True
        assert rotated["ok"] is True
        assert _mode(backup_dir) == 0o700
        _assert_private_sqlite_artifacts(timestamped["backup_path"])
        _assert_private_sqlite_artifacts(rotate_path)
        assert not rotate_path.with_name(rotate_path.name + ".tmp").exists()

        for backup_path in (timestamped["backup_path"], rotate_path):
            with sqlite3.connect(backup_path) as restored:
                assert restored.execute("PRAGMA quick_check").fetchone()[0] == "ok"
                assert restored.execute("SELECT content FROM messages").fetchone()[0] == "backup"
    finally:
        store.close()


@pytest.mark.parametrize("operation", ["timestamped", "rotate"])
def test_maintenance_backups_work_without_fchmod(tmp_path, monkeypatch, operation):
    db_path = tmp_path / "database" / "lcm.db"
    store = MessageStore(db_path)
    store.append("session", {"role": "user", "content": "backup"})
    store.commit()
    backup_dir = tmp_path / "backups"
    rotate_path = backup_dir / "rotate-latest.sqlite3"
    engine = SimpleNamespace(
        _store=store,
        _dag=SimpleNamespace(_conn=store.connection),
        _lifecycle=None,
        backup_dir=lambda: backup_dir,
        rotate_backup_path=lambda: rotate_path,
    )

    if hasattr(maintenance_module, "_FCHMOD"):
        monkeypatch.setattr(maintenance_module, "_FCHMOD", None)
    else:  # RED control for the rejected head, which read os.fchmod directly.
        class _OSWithoutFchmod:
            fchmod = None

            def __getattr__(self, name):
                return getattr(os, name)

        monkeypatch.setattr(maintenance_module, "os", _OSWithoutFchmod())
    try:
        result = (
            backup_database(engine)
            if operation == "timestamped"
            else rotate_backup_database(engine)
        )

        assert result["ok"] is True
        assert _mode(backup_dir) == 0o700
        _assert_private_sqlite_artifacts(result["backup_path"])
    finally:
        store.close()


@pytest.mark.parametrize("operation", ["timestamped", "rotate"])
def test_maintenance_rejects_symlinked_backup_directory_before_chmod(
    tmp_path,
    operation,
):
    db_path = tmp_path / "database" / "lcm.db"
    store = MessageStore(db_path)
    store.append("session", {"role": "user", "content": "backup"})
    store.commit()

    backup_parent = tmp_path / "backups"
    backup_parent.mkdir()
    backup_dir = backup_parent / "lcm"
    unrelated_target = tmp_path / "unrelated-target"
    unrelated_target.mkdir(mode=0o755)
    unrelated_target.chmod(0o755)
    backup_dir.symlink_to(unrelated_target, target_is_directory=True)
    rotate_path = backup_dir / "rotate-latest.sqlite3"
    engine = SimpleNamespace(
        _store=store,
        _dag=SimpleNamespace(_conn=store.connection),
        _lifecycle=None,
        backup_dir=lambda: backup_dir,
        rotate_backup_path=lambda: rotate_path,
    )

    try:
        result = (
            backup_database(engine)
            if operation == "timestamped"
            else rotate_backup_database(engine)
        )

        assert result["ok"] is False
        assert _mode(unrelated_target) == 0o755
        assert list(unrelated_target.iterdir()) == []
    finally:
        store.close()


def test_rotate_backup_failure_preserves_existing_atomic_slot(tmp_path, monkeypatch):
    db_path = tmp_path / "database" / "lcm.db"
    store = MessageStore(db_path)
    store.append("session", {"role": "user", "content": "backup"})
    store.commit()

    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(mode=0o755)
    rotate_path = backup_dir / "rotate-latest.sqlite3"
    previous_backup = b"known-good-backup"
    rotate_path.write_bytes(previous_backup)
    rotate_path.chmod(0o644)
    engine = SimpleNamespace(
        _store=store,
        _dag=SimpleNamespace(_conn=store.connection),
        _lifecycle=None,
        rotate_backup_path=lambda: rotate_path,
    )

    def fail_backup(_destination):
        raise sqlite3.OperationalError("synthetic backup failure")

    monkeypatch.setattr(store, "backup", fail_backup)
    try:
        with _process_umask(0o022):
            result = rotate_backup_database(engine)

        assert result["ok"] is False
        assert result["error"] == "synthetic backup failure"
        assert rotate_path.read_bytes() == previous_backup
        assert _mode(backup_dir) == 0o700
        assert _mode(rotate_path) == 0o600
        assert not rotate_path.with_name(rotate_path.name + ".tmp").exists()
    finally:
        store.close()


def test_timestamped_backup_flush_failure_leaves_no_empty_artifact(tmp_path, monkeypatch):
    db_path = tmp_path / "database" / "lcm.db"
    store = MessageStore(db_path)
    backup_dir = tmp_path / "backups"
    engine = SimpleNamespace(
        _store=store,
        _dag=SimpleNamespace(_conn=store.connection),
        _lifecycle=None,
        backup_dir=lambda: backup_dir,
    )

    def fail_flush(_engine):
        raise sqlite3.OperationalError("synthetic flush failure")

    monkeypatch.setattr(maintenance_module, "flush_engine_connections", fail_flush)
    try:
        result = backup_database(engine)

        assert result["ok"] is False
        assert result["error"] == "synthetic flush failure"
        assert list(backup_dir.glob("*.sqlite3")) == []
    finally:
        store.close()


# ---------------------------------------------------------------------------
# POSIX lock-loss regression: _restrict_existing_sqlite_artifacts must not drop
# the calling process's own SQLite locks on the -shm / -wal sidecars.
#
# Root cause: the helper opened each artifact with os.open(O_RDWR) and closed
# the fd. POSIX fcntl locks are per-process, so closing ANY fd to a file drops
# ALL of that process's locks on it. SQLite holds a READ lock on the WAL-index
# (-shm) file for the lifetime of a connection; the helper's open/close cycle
# silently released it, so a later connection close could unlink the sidecars
# out from under a live gateway connection.
# ---------------------------------------------------------------------------


def _own_posix_locks_on(path: Path) -> list[tuple[str, str]]:
    """Return (fd, lockline) pairs for this process's fds pointing at ``path``."""
    target = str(path)
    held: list[tuple[str, str]] = []
    for fd_name in os.listdir("/proc/self/fd"):
        try:
            link = os.readlink(f"/proc/self/fd/{fd_name}")
        except OSError:
            continue
        if link != target:
            continue
        try:
            info = open(f"/proc/self/fdinfo/{fd_name}").read()
        except OSError:
            continue
        for line in info.splitlines():
            if line.startswith("lock:"):
                held.append((fd_name, line))
    return held


def _assert_own_shm_read_lock_preserved(db_path: Path) -> None:
    """Assert the process still holds its POSIX READ lock on the -shm file."""
    shm = db_path.with_name(db_path.name + "-shm")
    locks = _own_posix_locks_on(shm)
    assert locks, f"expected this process to hold a POSIX lock on {shm}"
    assert any("READ" in line for _, line in locks), (
        f"expected a READ lock on {shm}, got {locks}"
    )


def test_restrict_existing_artifacts_preserves_own_shm_read_lock(tmp_path):
    """The helper must not drop the caller's own WAL-index READ lock."""
    db_path = tmp_path / "lcm.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE t (v TEXT)")
        conn.execute("INSERT INTO t VALUES ('x')")
        conn.commit()
        shm = db_path.with_name(db_path.name + "-shm")
        assert shm.exists()
        _assert_own_shm_read_lock_preserved(db_path)

        sqlite_util_module._restrict_existing_sqlite_artifacts(db_path)

        _assert_own_shm_read_lock_preserved(db_path)
        # The connection must still be usable for a write + readback.
        conn.execute("INSERT INTO t VALUES ('y')")
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 2
    finally:
        conn.close()


def test_restrict_existing_artifacts_preserves_own_shm_read_lock_after_sidecar_restrict(tmp_path):
    """Lock preservation holds even when the helper re-opens every sidecar."""
    db_path = tmp_path / "lcm.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE t (v TEXT)")
        conn.execute("INSERT INTO t VALUES ('x')")
        conn.commit()
        for suffix in _SQLITE_SIDECAR_SUFFIXES:
            sidecar = db_path.with_name(db_path.name + suffix)
            if sidecar.exists():
                sidecar.chmod(0o644)
        _assert_own_shm_read_lock_preserved(db_path)

        sqlite_util_module._restrict_existing_sqlite_artifacts(db_path)

        _assert_own_shm_read_lock_preserved(db_path)
        conn.execute("INSERT INTO t VALUES ('y')")
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 2
    finally:
        conn.close()


def test_restrict_existing_artifacts_preserves_locks_and_sidecars_in_subprocess(tmp_path):
    """Real subprocess regression: locks stay, sidecars remain, two-way writes work.

    A gateway process holds the database open in WAL mode. A separate reader
    process opens the same database, runs the restrict helper, writes, reads
    back, and closes. The gateway's sidecars must survive the reader's close
    (the reader must not have dropped its own locks and then unlinked the
    sidecars out from under the gateway), and both processes must be able to
    write and read back.
    """
    db_path = tmp_path / "lcm.db"
    plugin_dir = Path(__file__).resolve().parent.parent

    gateway_code = f"""
import sqlite3, time
conn = sqlite3.connect({str(db_path)!r}, timeout=5)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("CREATE TABLE IF NOT EXISTS t (v TEXT)")
conn.execute("INSERT INTO t VALUES ('gateway')")
conn.commit()
print("GATEWAY_READY", flush=True)
time.sleep(30)
"""

    reader_code = f"""
import importlib.util, os, sqlite3, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location(
    "sqlite_util", {str(plugin_dir / "sqlite_util.py")!r})
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
db = {str(db_path)!r}
conn = sqlite3.connect(db, timeout=5)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("CREATE TABLE IF NOT EXISTS t (v TEXT)")
conn.execute("INSERT INTO t VALUES ('reader')")
conn.commit()
m._restrict_existing_sqlite_artifacts(Path(db))
# Two-way write + readback after the helper ran.
conn.execute("INSERT INTO t VALUES ('reader-after')")
conn.commit()
rows = [r[0] for r in conn.execute("SELECT v FROM t ORDER BY v").fetchall()]
print("READER_ROWS=" + "|".join(rows), flush=True)
conn.close()
"""

    gateway = subprocess.Popen(
        [sys.executable, "-c", gateway_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        ready = gateway.stdout.readline().strip()
        assert ready == "GATEWAY_READY", f"gateway failed to start: {ready}"

        reader = subprocess.run(
            [sys.executable, "-c", reader_code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert reader.returncode == 0, f"reader failed: {reader.stderr}"

        # Sidecars must still exist while the gateway holds the database open.
        wal = db_path.with_name(db_path.name + "-wal")
        shm = db_path.with_name(db_path.name + "-shm")
        assert wal.exists(), "gateway WAL sidecar was unlinked by the reader"
        assert shm.exists(), "gateway SHM sidecar was unlinked by the reader"

        # The reader's two-way write + readback must have succeeded.
        assert "READER_ROWS=gateway|reader|reader-after" in reader.stdout, (
            f"reader write/readback failed: {reader.stdout}"
        )

        # The gateway must still be able to write and read back.
        gateway_conn = sqlite3.connect(db_path, timeout=5)
        try:
            gateway_conn.execute("INSERT INTO t VALUES ('gateway-after')")
            gateway_conn.commit()
            rows = [
                r[0]
                for r in gateway_conn.execute("SELECT v FROM t ORDER BY v").fetchall()
            ]
            assert rows == ["gateway", "gateway-after", "reader", "reader-after"], rows
        finally:
            gateway_conn.close()
    finally:
        gateway.kill()
        gateway.wait()
