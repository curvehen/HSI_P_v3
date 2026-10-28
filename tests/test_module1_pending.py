import json

import pytest

from hsi_ml.storage import PendingUploads, upload_or_queue


class FakeDrive:
    def __init__(self, exc=None):
        self.exc = exc
        self.calls = []

    def upload(self, local, remote_path, overwrite=False):
        self.calls.append((remote_path, overwrite))
        if self.exc:
            raise self.exc
        return "id"


@pytest.fixture
def pend(tmp_path):
    return PendingUploads(tmp_path / ".pending" / "pending.json")


@pytest.fixture
def f(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("x")
    return p


def test_add_and_entries(pend, f):
    pend.add(f, "x/a.txt")
    assert len(pend) == 1
    e = pend.entries()[0]
    assert e["remote"] == "x/a.txt" and e["attempts"] == 0 and e["overwrite"] is False


def test_add_missing_file_rejected(pend, tmp_path):
    with pytest.raises(FileNotFoundError):
        pend.add(tmp_path / "nope.txt", "x/nope.txt")
    assert len(pend) == 0


def test_same_remote_replaces_and_keeps_attempts(pend, f):
    pend.add(f, "x/a.txt")
    pend.flush(FakeDrive(exc=OSError("net")))          # attempts -> 1
    pend.add(f, "x/a.txt", overwrite=True)
    es = pend.entries()
    assert len(es) == 1 and es[0]["attempts"] == 1 and es[0]["overwrite"] is True


def test_flush_without_drive_keeps_all(pend, f):
    pend.add(f, "x/a.txt")
    assert pend.flush(None)["remaining"] == 1
    assert len(pend) == 1


def test_flush_uploads_and_clears(pend, f):
    pend.add(f, "x/a.txt")
    d = FakeDrive()
    s = pend.flush(d)
    assert s["uploaded"] == 1 and s["remaining"] == 0
    assert d.calls == [("x/a.txt", False)]
    assert len(pend) == 0


def test_flush_existing_target_treated_as_done(pend, f):
    pend.add(f, "x/a.txt")
    s = pend.flush(FakeDrive(exc=FileExistsError("x")))
    assert s["already_exists"] == 1 and s["remaining"] == 0


def test_flush_failure_keeps_and_counts(pend, f):
    pend.add(f, "x/a.txt")
    s = pend.flush(FakeDrive(exc=OSError("net")))
    assert s["failed"] == 1 and s["remaining"] == 1
    assert pend.entries()[0]["attempts"] == 1


def test_flush_missing_local_dropped(pend, f):
    pend.add(f, "x/a.txt")
    f.unlink()
    s = pend.flush(FakeDrive())
    assert s["dropped_missing"] == 1 and s["remaining"] == 0


def test_corrupt_json_moved_aside(pend):
    pend.path.parent.mkdir(parents=True)
    pend.path.write_text("{not json")
    assert pend.entries() == []
    assert pend.path.with_suffix(".corrupt").exists()


def test_malformed_entries_filtered(pend):
    pend.path.parent.mkdir(parents=True)
    pend.path.write_text(json.dumps([{"local": "a"}, "junk",
                                     {"local": "a", "remote": "b"}]))
    assert len(pend.entries()) == 1


def test_upload_or_queue_no_drive(pend, f):
    assert upload_or_queue(None, f, "x/a.txt", pend) == "queued"
    assert len(pend) == 1


def test_upload_or_queue_success(pend, f):
    assert upload_or_queue(FakeDrive(), f, "x/a.txt", pend) == "uploaded"
    assert len(pend) == 0


def test_upload_or_queue_failure_queues(pend, f):
    assert upload_or_queue(FakeDrive(exc=OSError("net")), f, "x/a.txt", pend) == "queued"
    assert len(pend) == 1


def test_upload_or_queue_exists_propagates_not_queued(pend, f):
    with pytest.raises(FileExistsError):
        upload_or_queue(FakeDrive(exc=FileExistsError("x")), f, "x/a.txt", pend)
    assert len(pend) == 0


def test_upload_or_queue_missing_local(pend, tmp_path):
    with pytest.raises(FileNotFoundError):
        upload_or_queue(FakeDrive(), tmp_path / "nope", "x/n", pend)
