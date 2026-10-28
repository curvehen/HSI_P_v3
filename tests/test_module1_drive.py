import httplib2
import pytest
from google.auth.exceptions import RefreshError
from googleapiclient.errors import HttpError

from fake_drive import FakeService, make_downloader
from hsi_ml.logging_setup import current_secrets
from hsi_ml.storage import drive as drive_mod
from hsi_ml.storage.drive import (
    MAX_TRIES, SCOPES, DriveAuthError, DriveStore, _is_retryable,
)


def http_error(status, content=b""):
    return HttpError(httplib2.Response({"status": str(status)}), content)


@pytest.fixture
def svc():
    return FakeService()


@pytest.fixture
def store(svc, monkeypatch, no_sleep):
    monkeypatch.setattr(drive_mod, "MediaIoBaseDownload", make_downloader(svc))
    return DriveStore(root_name="root_t", service=svc)


@pytest.fixture
def local(tmp_path):
    p = tmp_path / "up.txt"
    p.write_bytes(b"hello")
    return p


# ------------------------------------------------------------ 同名重複
def test_duplicate_files_newest_updated_and_warned(store, svc, local, caplog):
    d = store.ensure_path("d")
    old = svc.add("up.txt", d, content=b"old")
    new = svc.add("up.txt", d, content=b"new")
    local.write_bytes(b"changed")
    with caplog.at_level("WARNING", logger="hsi_ml.drive"):
        fid = store.upload(local, "d/up.txt", overwrite=True)
    assert fid == new["id"]
    assert new["content"] == b"changed" and old["content"] == b"old"
    assert "duplicate" in caplog.text


def test_duplicate_files_block_append_only(store, svc, local):
    d = store.ensure_path("d")
    svc.add("up.txt", d)
    svc.add("up.txt", d)
    with pytest.raises(FileExistsError):
        store.upload(local, "d/up.txt")


def test_duplicate_folders_oldest_used(store, svc, caplog):
    first = svc.add("dup", store.root_id, folder=True)
    svc.add("dup", store.root_id, folder=True)
    with caplog.at_level("WARNING", logger="hsi_ml.drive"):
        assert store.ensure_path("dup") == first["id"]
    assert "duplicate" in caplog.text


# ------------------------------------------------------------ 列表
def test_list_names_paginates(store, svc):
    d = store.ensure_path("d")
    svc.max_page = 2
    for n in "edcba":
        svc.add(f"{n}.txt", d)
    assert store.list_names("d") == ["a.txt", "b.txt", "c.txt", "d.txt", "e.txt"]


def test_list_names_dedupes_and_warns(store, svc, caplog):
    d = store.ensure_path("d")
    svc.add("a.txt", d)
    svc.add("a.txt", d)
    with caplog.at_level("WARNING", logger="hsi_ml.drive"):
        assert store.list_names("d") == ["a.txt"]
    assert "duplicate" in caplog.text


def test_list_names_excludes_folders(store, svc):
    d = store.ensure_path("d")
    svc.add("sub", d, folder=True)
    svc.add("f.txt", d)
    assert store.list_names("d") == ["f.txt"]


# ------------------------------------------------------------ 重試
def test_retry_then_success(store, svc, local):
    svc.fail["create"] = [http_error(503), http_error(429)]
    assert store.upload(local, "d/up.txt")
    assert store.exists("d/up.txt")


def test_non_retryable_not_retried(store, svc, local):
    before = svc.calls["create"]
    svc.fail["create"] = [http_error(404)]
    with pytest.raises(HttpError):
        store.upload(local, "d/up.txt")
    assert svc.calls["create"] == before + 1


def test_retry_exhausted_raises(store, svc, local):
    before = svc.calls["create"]
    svc.fail["create"] = [http_error(503) for _ in range(MAX_TRIES)]
    with pytest.raises(HttpError):
        store.upload(local, "d/up.txt")
    assert svc.calls["create"] == before + MAX_TRIES


def test_retry_uses_backoff_sleep(store, svc, local, monkeypatch):
    sleeps = []
    monkeypatch.setattr(drive_mod.time, "sleep", lambda s: sleeps.append(s))
    svc.fail["create"] = [http_error(500), http_error(500)]
    store.upload(local, "d/up.txt")
    assert len(sleeps) == 2 and sleeps[1] >= sleeps[0] - 0.5


@pytest.mark.parametrize("exc,expected", [
    (http_error(429), True),
    (http_error(500), True),
    (http_error(503), True),
    (http_error(403, b'{"reason":"rateLimitExceeded"}'), True),
    (http_error(403, b'{"reason":"forbidden"}'), False),
    (http_error(404), False),
    (OSError("net"), True),
    (httplib2.HttpLib2Error("x"), True),
    (FileNotFoundError("x"), False),
    (PermissionError("x"), False),
    (ValueError("x"), False),
])
def test_is_retryable(exc, expected):
    assert _is_retryable(exc) is expected


# ------------------------------------------------------------ 下載
def test_download_roundtrip(store, local, tmp_path):
    store.upload(local, "d/up.txt")
    dest = tmp_path / "out" / "got.txt"
    assert store.download("d/up.txt", dest) is True
    assert dest.read_bytes() == b"hello"
    assert not (tmp_path / "out" / "got.txt.part").exists()


def test_download_missing_returns_false(store, tmp_path):
    dest = tmp_path / "got.txt"
    assert store.download("d/none.txt", dest) is False
    assert not dest.exists()


def test_download_failure_leaves_no_partial_and_keeps_old(store, svc, local, tmp_path):
    store.upload(local, "d/up.txt")
    dest = tmp_path / "got.txt"
    dest.write_bytes(b"old")
    svc.fail["download"] = [http_error(404)]
    with pytest.raises(HttpError):
        store.download("d/up.txt", dest)
    assert dest.read_bytes() == b"old"
    assert not (tmp_path / "got.txt.part").exists()


def test_download_retries_and_overwrites_partial(store, svc, local, tmp_path):
    store.upload(local, "d/up.txt")
    dest = tmp_path / "got.txt"
    svc.fail["download"] = [http_error(503)]
    assert store.download("d/up.txt", dest) is True
    assert dest.read_bytes() == b"hello"


# ------------------------------------------------------------ 認證
def _env(monkeypatch):
    monkeypatch.setenv("GDRIVE_CLIENT_ID", "cid-123456")
    monkeypatch.setenv("GDRIVE_CLIENT_SECRET", "csecret-123456")
    monkeypatch.setenv("GDRIVE_REFRESH_TOKEN", "rtoken-123456")


def _patch_auth(monkeypatch, svc, refresh_exc=None, build_exc=None):
    captured = {}

    class FakeCreds:
        def __init__(self, **kw):
            captured.update(kw)
            self.token = None

        def refresh(self, request):
            if refresh_exc:
                raise refresh_exc
            self.token = "access-token-abcdef"

    def fake_build(*a, **k):
        if build_exc:
            raise build_exc
        return svc

    monkeypatch.setattr(drive_mod, "Credentials", FakeCreds)
    monkeypatch.setattr(drive_mod, "build", fake_build)
    return captured


def test_missing_env_raises_auth_error_with_names_only():
    with pytest.raises(DriveAuthError) as ei:
        DriveStore()
    msg = str(ei.value)
    assert "GDRIVE_REFRESH_TOKEN" in msg and "GDRIVE_CLIENT_ID" in msg


def test_refresh_error_becomes_auth_error_without_leak(monkeypatch, svc):
    _env(monkeypatch)
    _patch_auth(monkeypatch, svc, refresh_exc=RefreshError("bad rtoken-123456"))
    with pytest.raises(DriveAuthError) as ei:
        DriveStore()
    assert "rtoken-123456" not in str(ei.value)
    assert ei.value.__cause__ is None


def test_success_uses_drive_file_scope_and_registers_token(monkeypatch, svc):
    _env(monkeypatch)
    cap = _patch_auth(monkeypatch, svc)
    s = DriveStore(root_name="root_t")
    assert s.root_id
    assert cap["scopes"] == SCOPES
    assert cap["refresh_token"] == "rtoken-123456"
    assert "access-token-abcdef" in current_secrets()


def test_try_connect_none_when_env_missing():
    assert DriveStore.try_connect() is None


def test_try_connect_none_on_refresh_error(monkeypatch, svc):
    _env(monkeypatch)
    _patch_auth(monkeypatch, svc, refresh_exc=RefreshError("x"))
    assert DriveStore.try_connect() is None


def test_try_connect_none_on_unexpected_error(monkeypatch, svc):
    _env(monkeypatch)
    _patch_auth(monkeypatch, svc, build_exc=RuntimeError("boom"))
    assert DriveStore.try_connect() is None


def test_try_connect_ok(monkeypatch, svc):
    _env(monkeypatch)
    _patch_auth(monkeypatch, svc)
    assert isinstance(DriveStore.try_connect("root_t"), DriveStore)
