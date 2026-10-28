"""Drive 失敗時的待補傳清單。

清單存在本地 JSON(預設 data/.pending/pending.json)。下次運行先補傳。
重要限制:GitHub runner 是全新環境,job 結束後本地檔案會消失。
所以在 Actions 上,這份清單和被暫存的檔案「不會」自動保留到下一次運行。
解決方式留到模組 10(例如 Actions cache,但 public repo 的 cache 有可見性風險,
需要先評估),在此之前,Drive 失敗時一定要靠警報 email 人手處理。
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

from .. import config

log = logging.getLogger("hsi_ml.pending")


class _Uploader(Protocol):
    def upload(self, local: str | Path, remote_path: str, overwrite: bool = False) -> str: ...


class PendingUploads:
    """待補傳清單。以 remote 路徑為鍵:同一 remote 再次登記會取代舊紀錄。非執行緒安全。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else config.data_dir() / ".pending" / "pending.json"

    # ---------------------------------------------------------- 讀寫
    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError("not a list")
        except (OSError, ValueError):  # JSONDecodeError 是 ValueError 的子類
            bad = self.path.with_suffix(".corrupt")
            try:
                os.replace(self.path, bad)
            except OSError:
                pass
            log.error("pending list unreadable; moved aside as %s", bad.name)
            return []
        return [e for e in data if isinstance(e, dict)
                and isinstance(e.get("local"), str) and isinstance(e.get("remote"), str)]

    def _save(self, entries: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)  # 原子寫入

    # ---------------------------------------------------------- 公開 API
    def entries(self) -> list[dict[str, Any]]:
        """目前的待補傳紀錄(複本)。"""
        return self._load()

    def __len__(self) -> int:
        return len(self._load())

    def add(self, local: str | Path, remote: str, overwrite: bool = False) -> None:
        """登記一個待補傳檔案。local 存為絕對路徑。同 remote 已存在則取代並保留 attempts。

        Raises: FileNotFoundError(本地檔案不存在,無法補傳就不登記)。
        """
        lp = Path(local).resolve()
        if not lp.is_file():
            raise FileNotFoundError(f"cannot queue missing file: {lp.name}")
        entries = self._load()
        attempts = 0
        kept: list[dict[str, Any]] = []
        for e in entries:
            if e["remote"] == remote:
                attempts = int(e.get("attempts", 0))
            else:
                kept.append(e)
        kept.append({
            "local": str(lp), "remote": remote, "overwrite": bool(overwrite),
            "queued_utc": datetime.now(timezone.utc).isoformat(),
            "attempts": attempts,
        })
        self._save(kept)
        log.warning("queued for later upload: %s (pending=%d)", remote, len(kept))

    def flush(self, drive: Optional[_Uploader]) -> dict[str, int]:
        """嘗試補傳全部紀錄,回傳計數。

        - drive 為 None:不做任何事,全部保留。
        - 成功:移出清單。
        - FileExistsError(append-only 遠端已有同名檔):視為已完成並移出,記錄警告。
          注意:不會比較內容。
        - 本地檔案已消失:移出並計入 dropped_missing(數據可能已遺失,需人手檢查)。
        - 其他錯誤:保留,attempts 加 1。
        日誌只記錄遠端路徑與錯誤類型。
        """
        stats = {"uploaded": 0, "already_exists": 0, "dropped_missing": 0,
                 "failed": 0, "remaining": 0}
        entries = self._load()
        if drive is None or not entries:
            stats["remaining"] = len(entries)
            return stats
        remaining: list[dict[str, Any]] = []
        for e in entries:
            lp = Path(e["local"])
            if not lp.is_file():
                stats["dropped_missing"] += 1
                log.error("pending file missing, dropped: %s", e["remote"])
                continue
            try:
                drive.upload(lp, e["remote"], overwrite=bool(e.get("overwrite", False)))
                stats["uploaded"] += 1
            except FileExistsError:
                stats["already_exists"] += 1
                log.warning("pending upload target already exists: %s", e["remote"])
            except Exception as ex:  # noqa: BLE001
                e["attempts"] = int(e.get("attempts", 0)) + 1
                remaining.append(e)
                stats["failed"] += 1
                log.error("pending upload failed: %s (%s)", e["remote"], type(ex).__name__)
        self._save(remaining)
        stats["remaining"] = len(remaining)
        return stats


def upload_or_queue(drive: Optional[_Uploader], local: str | Path, remote: str,
                    pending: PendingUploads, overwrite: bool = False) -> str:
    """嘗試上傳;Drive 不可用或上傳失敗就登記待補傳。回傳 'uploaded' 或 'queued'。

    Raises:
      FileNotFoundError:本地檔案不存在。
      FileExistsError:append-only 遠端已有同名檔案。這是呼叫端邏輯錯誤,
        不會被吞掉也不會排隊。
    """
    if not Path(local).is_file():
        raise FileNotFoundError(f"local file not found: {Path(local).name}")
    if drive is None:
        pending.add(local, remote, overwrite)
        return "queued"
    try:
        drive.upload(local, remote, overwrite=overwrite)
        return "uploaded"
    except FileExistsError:
        raise
    except Exception as e:  # noqa: BLE001
        log.error("upload failed (%s); queueing: %s", type(e).__name__, remote)
        pending.add(local, remote, overwrite)
        return "queued"
