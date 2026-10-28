"""Google Drive 存取(個人 Gmail):OAuth refresh token + drive.file scope。

- 憑證只讀環境變數 GDRIVE_CLIENT_ID / GDRIVE_CLIENT_SECRET / GDRIVE_REFRESH_TOKEN。
- drive.file scope 只看得到本程式建立的檔案,看不到你手動放進 Drive 的檔案。
- 預測與結果類檔案 append-only:upload(overwrite=False) 遇到同名檔案拋 FileExistsError。
- Drive 允許同名檔案並存:查詢時偵測重複並記錄警告。
  資料夾取最早建立者;檔案取最近修改者。
- OAuth consent screen 若停留在 Testing 狀態,refresh token 約 7 日失效(未核實),
  請發佈為 In production。
已知限制:「查詢後再建立」不是原子操作,並發寫入可能產生同名重複檔。
請靠 workflow concurrency 與運行鎖檔避免(模組 10)。
"""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import httplib2
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from .. import config
from ..logging_setup import register_secret

log = logging.getLogger("hsi_ml.drive")

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
TOKEN_URI = "https://oauth2.googleapis.com/token"
FOLDER_MIME = "application/vnd.google-apps.folder"
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_TRIES = 4
BASE_DELAY = 2.0
MAX_DELAY = 30.0
RESUMABLE_MIN_BYTES = 5 * 1024 * 1024
_AUTH_ENV = ("GDRIVE_CLIENT_ID", "GDRIVE_CLIENT_SECRET", "GDRIVE_REFRESH_TOKEN")

T = TypeVar("T")


class DriveAuthError(RuntimeError):
    """refresh token 失效、被撤銷或缺失,需要重新授權。訊息不含任何憑證值。"""


def _is_retryable(exc: BaseException) -> bool:
    """429、5xx、限流型 403、網絡錯誤可重試;本地檔案錯誤不重試。"""
    if isinstance(exc, (FileNotFoundError, PermissionError,
                        FileExistsError, IsADirectoryError)):
        return False
    if isinstance(exc, HttpError):
        try:
            status = int(exc.resp.status)
        except (AttributeError, TypeError, ValueError):
            return False
        if status in RETRY_STATUS:
            return True
        if status == 403:  # 限流型 403 的 reason 字串未核實
            content = exc.content or b""
            return b"rateLimitExceeded" in content or b"userRateLimitExceeded" in content
        return False
    return isinstance(exc, (OSError, httplib2.HttpLib2Error))


def _retry(fn: Callable[[], T], tries: int = MAX_TRIES, base: float = BASE_DELAY,
           what: str = "drive call") -> T:
    """指數退避加抖動。最後一次失敗或不可重試的錯誤直接拋出。

    日誌只記錄錯誤類型,不記錄訊息。測試可 monkeypatch time.sleep。
    """
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if attempt == tries - 1 or not _is_retryable(e):
                raise
            delay = min(base ** attempt, MAX_DELAY) + random.uniform(0, 0.5)
            log.warning("%s: retry %d/%d after %s", what, attempt + 1, tries - 1,
                        type(e).__name__)
            time.sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover


def _q(s: str) -> str:
    """轉義 Drive 查詢字串中的反斜線與單引號。"""
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _parts(path: str) -> list[str]:
    parts = [p for p in path.strip().strip("/").split("/") if p]
    if any(p in (".", "..") for p in parts):
        raise ValueError("path must not contain '.' or '..'")
    return parts


def _split(remote_path: str) -> tuple[str, str]:
    """'a/b/c.parquet' -> ('a/b', 'c.parquet')。無檔名拋 ValueError。"""
    parts = _parts(remote_path)
    if not parts:
        raise ValueError("remote path has no file name")
    return "/".join(parts[:-1]), parts[-1]


def _build_service() -> Any:
    missing = config.missing_secrets(_AUTH_ENV)
    if missing:
        raise DriveAuthError(f"missing env var(s): {', '.join(missing)}")
    import os
    creds = Credentials(
        token=None,
        refresh_token=os.environ["GDRIVE_REFRESH_TOKEN"].strip(),
        client_id=os.environ["GDRIVE_CLIENT_ID"].strip(),
        client_secret=os.environ["GDRIVE_CLIENT_SECRET"].strip(),
        token_uri=TOKEN_URI, scopes=SCOPES,
    )
    try:
        creds.refresh(Request())
    except RefreshError:
        # from None:不帶出原始例外,避免訊息洩漏
        raise DriveAuthError("refresh token rejected or revoked; re-authorize") from None
    register_secret(creds.token)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


class DriveStore:
    """Drive 檔案存取。所有遠端路徑相對於 root_name 資料夾。"""

    def __init__(self, root_name: str = config.DRIVE_ROOT_NAME,
                 service: Optional[Any] = None) -> None:
        """service 可注入(測試用 mock);省略則用環境變數憑證建立。

        Raises: DriveAuthError(憑證缺失或 token 失效);其他網絡錯誤原樣拋出。
        """
        self.svc = service if service is not None else _build_service()
        self._folder_cache: dict[tuple[str, str], str] = {}
        self.root_id = self._ensure_folder(root_name, "root")

    @classmethod
    def try_connect(cls, root_name: str = config.DRIVE_ROOT_NAME) -> Optional["DriveStore"]:
        """連線失敗返回 None,讓管線降級(本地暫存、稍後補傳、發警報)。"""
        try:
            return cls(root_name)
        except DriveAuthError as e:
            log.error("drive auth failed: %s", e)
        except Exception as e:  # noqa: BLE001
            log.error("drive connect failed: %s", type(e).__name__)
        return None

    # ------------------------------------------------------------ 查詢
    def _find(self, name: str, parent: str, folder: bool) -> Optional[dict[str, Any]]:
        op = "=" if folder else "!="
        q = (f"name='{_q(name)}' and '{parent}' in parents and trashed=false "
             f"and mimeType{op}'{FOLDER_MIME}'")
        order = "createdTime" if folder else "modifiedTime desc"
        res = _retry(lambda: self.svc.files().list(
            q=q, fields="files(id,name,size,createdTime,modifiedTime)",
            pageSize=10, orderBy=order, spaces="drive").execute(), what="find")
        files = res.get("files", [])
        if len(files) > 1:
            log.warning("duplicate %s on drive: n=%d (using %s)", 
                        "folders" if folder else "files", len(files),
                        "oldest" if folder else "newest")
        return files[0] if files else None

    def _ensure_folder(self, name: str, parent: str) -> str:
        key = (parent, name)
        if key in self._folder_cache:
            return self._folder_cache[key]
        f = self._find(name, parent, folder=True)
        if f is None:
            meta = {"name": name, "mimeType": FOLDER_MIME, "parents": [parent]}
            f = _retry(lambda: self.svc.files().create(
                body=meta, fields="id").execute(), what="mkdir")
        self._folder_cache[key] = f["id"]
        return f["id"]

    def ensure_path(self, remote_dir: str) -> str:
        """建立(若不存在)多層資料夾,回傳最末層 folder id。空字串回傳 root。"""
        pid = self.root_id
        for part in _parts(remote_dir):
            pid = self._ensure_folder(part, pid)
        return pid

    def exists(self, remote_path: str) -> bool:
        """檔案是否存在。副作用:可能建立資料夾。"""
        d, name = _split(remote_path)
        return self._find(name, self.ensure_path(d), folder=False) is not None

    def list_names(self, remote_dir: str) -> list[str]:
        """列出資料夾內的檔案名稱(不含子資料夾),排序並去重。

        副作用:可能建立資料夾。有同名重複檔時記錄警告。
        """
        pid = self.ensure_path(remote_dir)
        q = f"'{pid}' in parents and trashed=false and mimeType!='{FOLDER_MIME}'"
        names: list[str] = []
        token: Optional[str] = None
        while True:
            res = _retry(lambda: self.svc.files().list(
                q=q, fields="nextPageToken,files(name)", pageSize=200,
                pageToken=token, spaces="drive").execute(), what="list")
            names += [f["name"] for f in res.get("files", [])]
            token = res.get("nextPageToken")
            if not token:
                break
        unique = sorted(set(names))
        if len(unique) != len(names):
            log.warning("duplicate file names in %s: n=%d",
                        remote_dir, len(names) - len(unique))
        return unique

    def upload(self, local: str | Path, remote_path: str, overwrite: bool = False) -> str:
        """上傳檔案,回傳 Drive file id。

        overwrite=False(預設,append-only):遠端已有同名檔案就拋 FileExistsError。
        overwrite=True:更新現有檔案(有重複時更新最近修改者);不存在則建立。
        Raises: FileNotFoundError(本地檔案不存在)、FileExistsError、
                ValueError(路徑不合法)、HttpError(重試用盡)。
        副作用:可能建立資料夾;日誌只記錄遠端路徑。
        已知限制:逾時後重試 create,若首次其實已成功,可能產生同名重複檔
        (之後查詢時會警告並取最新者)。
        """
        lp = Path(local)
        if not lp.is_file():
            raise FileNotFoundError(f"local file not found: {lp.name}")
        d, name = _split(remote_path)
        parent = self.ensure_path(d)
        existing = self._find(name, parent, folder=False)
        if existing and not overwrite:
            raise FileExistsError(f"append-only: {remote_path} already exists")
        resumable = lp.stat().st_size >= RESUMABLE_MIN_BYTES

        def _do() -> dict[str, Any]:
            media = MediaFileUpload(str(lp), resumable=resumable)
            if existing:
                return self.svc.files().update(
                    fileId=existing["id"], media_body=media, fields="id").execute()
            return self.svc.files().create(
                body={"name": name, "parents": [parent]},
                media_body=media, fields="id").execute()

        fid = _retry(_do, what="upload")["id"]
        log.info("uploaded %s", remote_path)
        return fid

    def download(self, remote_path: str, local: str | Path) -> bool:
        """下載到本地。檔案不存在回傳 False;成功回傳 True。

        先寫 .part 臨時檔,完成後 os.replace,所以失敗不會留下半個檔案。
        Raises: ValueError、HttpError(重試用盡)、OSError。
        """
        d, name = _split(remote_path)
        f = self._find(name, self.ensure_path(d), folder=False)
        if f is None:
            return False
        dest = Path(local)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")

        def _do() -> None:
            req = self.svc.files().get_media(fileId=f["id"])
            with open(tmp, "wb") as fh:
                dl = MediaIoBaseDownload(fh, req)
                done = False
                while not done:
                    _, done = dl.next_chunk()

        try:
            _retry(_do, what="download")
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
        log.info("downloaded %s", remote_path)
        return True
