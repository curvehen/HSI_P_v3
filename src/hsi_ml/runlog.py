"""運行日誌:每步開始時間、耗時、狀態、筆數、錯誤類型與截斷後的訊息。

禁止記錄 DataFrame 內容或預測值:
- meta 只接受簡單型別,數據結構一律轉成 shape 描述。
- 錯誤訊息遮罩 secret 並截斷(MAX_ERR_LEN)。
輸出欄位對應全局規範的 run_log 表(另加 meta)。
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import pandas as pd

from .logging_setup import mask_text

log = logging.getLogger("hsi_ml.runlog")

MAX_ERR_LEN = 300
MAX_META_LEN = 300
STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_FAILED_SOFT = "failed_soft"

COLUMNS = ["run_id", "universe_version", "step", "status", "start_utc",
           "duration_s", "n_rows", "error_type", "error_msg", "meta"]


def _safe_meta_value(v: Any) -> Any:
    if isinstance(v, (pd.DataFrame, pd.Series, np.ndarray)):
        return f"<{type(v).__name__} shape={tuple(v.shape)}>"
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return f"<{type(v).__name__}>"


class RunLog:
    """一次運行的步驟紀錄。非執行緒安全(單一管線順序使用)。"""

    def __init__(self, run_id: Optional[str] = None, universe_version: str = "") -> None:
        self.run_id: str = run_id or uuid.uuid4().hex[:12]
        self.universe_version = universe_version
        self.rows: list[dict[str, Any]] = []

    @contextmanager
    def step(self, name: str, soft: bool = False, **meta: Any) -> Iterator[dict[str, Any]]:
        """記錄一個步驟。

        用法:
            with rl.step("fetch_yf") as ctx:
                ...
                ctx["rows"] = n
        soft=True:Exception 只記錄為 failed_soft 並繼續(用於 LLM、新聞等可降級步驟)。
        soft=False:記錄為 failed 並重新拋出。
        KeyboardInterrupt 等 BaseException 一律記錄為 failed 並重新拋出。
        副作用:追加一列到 self.rows;寫一行日誌(只含步驟名、狀態、耗時、筆數)。
        """
        ctx: dict[str, Any] = {"rows": None}
        t0 = time.perf_counter()
        start = datetime.now(timezone.utc)
        status = STATUS_OK
        err_type: Optional[str] = None
        err_msg: Optional[str] = None
        try:
            yield ctx
        except Exception as e:  # noqa: BLE001
            status = STATUS_FAILED_SOFT if soft else STATUS_FAILED
            err_type = type(e).__name__
            err_msg = mask_text(str(e))[:MAX_ERR_LEN]
            if not soft:
                raise
        except BaseException as e:
            status = STATUS_FAILED
            err_type = type(e).__name__
            err_msg = mask_text(str(e))[:MAX_ERR_LEN]
            raise
        finally:
            dur = round(time.perf_counter() - t0, 3)
            rows = ctx.get("rows")
            n_rows = int(rows) if isinstance(rows, (int, np.integer)) else None
            meta_str = None
            if meta:
                safe = {k: _safe_meta_value(v) for k, v in meta.items()}
                meta_str = mask_text(str(safe))[:MAX_META_LEN]
            self.rows.append({
                "run_id": self.run_id, "universe_version": self.universe_version,
                "step": name, "status": status, "start_utc": start,
                "duration_s": dur, "n_rows": n_rows,
                "error_type": err_type, "error_msg": err_msg, "meta": meta_str,
            })
            log.info("step=%s status=%s dur=%.2fs rows=%s", name, status, dur, n_rows)

    def has_failure(self) -> bool:
        """是否有 status == 'failed' 的步驟(failed_soft 不算)。"""
        return any(r["status"] == STATUS_FAILED for r in self.rows)

    def has_soft_failure(self) -> bool:
        """是否有 failed_soft 的步驟。"""
        return any(r["status"] == STATUS_FAILED_SOFT for r in self.rows)

    def to_frame(self) -> pd.DataFrame:
        """轉成 DataFrame(固定欄位與 dtype,空紀錄也有 schema)。"""
        df = pd.DataFrame(self.rows, columns=COLUMNS)
        df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True)
        df["duration_s"] = df["duration_s"].astype("float64")
        df["n_rows"] = df["n_rows"].astype("Int64")
        for c in ("run_id", "universe_version", "step", "status",
                  "error_type", "error_msg", "meta"):
            df[c] = df[c].astype("object")
        return df

    def to_parquet(self, path: str | Path) -> Path:
        """寫成 Parquet(自動建立父目錄)。回傳路徑。Raises: OSError。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_parquet(p, index=False)
        return p

    def summary(self) -> dict[str, Any]:
        """只回傳狀態與筆數(不含錯誤訊息、meta),可安全寫入日誌或 email。"""
        return {
            "run_id": self.run_id,
            "n_steps": len(self.rows),
            "n_ok": sum(r["status"] == STATUS_OK for r in self.rows),
            "n_failed": sum(r["status"] == STATUS_FAILED for r in self.rows),
            "n_failed_soft": sum(r["status"] == STATUS_FAILED_SOFT for r in self.rows),
            "steps": [{"step": r["step"], "status": r["status"], "n_rows": r["n_rows"]}
                      for r in self.rows],
        }
