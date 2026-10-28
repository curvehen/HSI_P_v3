import httplib2
import pytest
from google.auth.exceptions import RefreshError
from googleapiclient.errors import HttpError

from fake_drive import FakeService, make_downloader
from hsi_ml.logging_setup import current_secrets
from hsi_ml.storage import drive as drive_mod
from hsi_ml.storage.drive import (
    MAX_TRIES, SCOPES, DriveAuthError, DriveStore, _is_retryable, _q,
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


def _env(monkeypatch):
    monkeypatch.setenv("GDRIVE_CLIENT_ID", "cid-123456")
    monkeypatch.setenv("GDRIVE_CLIENT_SECRET", "csecret-123456")
    monkeypatch.setenv("GDRIVE_REFRESH_TOKEN", "rtoken-123456")


# ------------------------------------------------------------ 基本
def test_scope_is_drive_file_only():
    assert SCOPES == ["https://www.googleapis.com/auth/drive.file"]


def test_q_escapes_quote_and_backslash():
    assert _q("a'b\\c") == "a\\'b\\\\c"


def test_root_and_nested_folders_created_once(store, svc):
    assert len(svc.find("root_t")) == 1
    a = store.ensure_path("x/y/z")
    n_creates = svc.calls["create"]
    assert store.ensure_path("x/y/z") == a
    assert svc.calls["create"] == n_creates          # 第二次走快取


@pytest.mark.parametrize("bad", ["../a.txt", "a/./b.txt", "a/b/"])
def test_bad_paths_rejected(store, local, bad):
    with pytest.raises(ValueError):
        store.upload(local, bad)


def test_upload_exists_list(store, local):
    assert not store.exists("d/up.txt")
    fid = store.upload(local, "d/up.txt")
    assert fid and store.exists("d/up.txt")
    assert store.list_names("d") == ["up.txt"]


def test_upload_missing_local(store, tmp_path):
    with pytest.raises(FileNotFoundError):
        store.upload(tmp_path / "nope.txt", "d/nope.txt")


def test_name_with_quote(store, tmp_path):
    p = tmp_path / "it's.txt"
    p.write_bytes(b"x")
    store.upload(p, "d/it's.txt")
    assert store.exists("d/it's.txt")


# ------------------------------------------------------------ append-only
def test_append_only_refuses_overwrite(store, svc, local):
    store.upload(local, "d/up.txt")
    local.write_bytes(b"changed")
    with pytest.raises(FileExistsError):
        store.upload(local, "d/up.txt")
    assert svc.find("up.txt")[0]["content"] == b"hello"


def test_overwrite_true_updates_same_file(store, svc, local):
    fid = store.upload(local, "d/up.txt")
    local.write_bytes(b"changed")
    assert store.upload(local, "d/up.txt", overwrite=True) == fid
    assert svc.calls["update"] == 1
    assert len(svc.find("up.txt")) == 1
    assert svc.find("up.txt")[0]["content"] == b"changed"


# ------------------------------------------------------------ 同名重複
def test_duplicate_files_newest_used_and_warned(store, svc, local, caplog):
    d = store.ensure_path("d")
    old = svc.add("up.txt", d, content=b"old")
    new = svc.add("up.txt", d, content=b"new")
    local.write_byt
