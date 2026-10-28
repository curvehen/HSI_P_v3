"""日誌設定與 secret 遮罩。

Actions 日誌在 public repo 是公開的,所以:
- 名稱含 KEY / SECRET / TOKEN / PASSWORD 的環境變數值(長度 >= 6)一律遮罩。
- 遮罩發生在「格式化之後」,涵蓋 msg、args、異常堆疊。
- DataFrame / Series / ndarray 作為 args 時只記錄 shape,不記錄內容。
- 第三方庫(root logger)的日誌也經過同一個遮罩 handler。
限制:f-string 直接把 DataFrame 轉成文字,本模組無法偵測,呼叫端不可這樣寫。
"""
from __future__ import annotations

import logging
import os
import re
import sys
from typing import IO, Any, Optional

import numpy as np
import pandas as pd

_SENSITIVE_NAME = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD)", re.I)
MIN_SECRET_LEN = 6
MASK = "***"
_HANDLER_FLAG = "_hsi_ml_handler"
_extra_secrets: set[str] = set()

_NOISY_LOGGERS = ("urllib3", "googleapiclient", "google.auth", "httpx",
                  "httpcore", "yfinance", "peewee")


def register_secret(value: Optional[str]) -> None:
    """登記額外要遮罩的值(例如執行期取得的 access token)。

    副作用:修改模組級集合。長度不足 MIN_SECRET_LEN 的值會被忽略。
    """
    if value and len(value.strip()) >= MIN_SECRET_LEN:
        _extra_secrets.add(value.strip())


def current_secrets() -> list[str]:
    """回傳目前要遮罩的值,長的在前(避免短 secret 是長 secret 的子字串時遮罩不全)。

    每次呼叫都重新掃描環境變數,所以 setup 之後才設定的變數也會被遮罩。
    若值含空白,另外加入「去空白版本」(例如 Google 分組顯示的 App Password)。
    """
    vals: set[str] = set(_extra_secrets)
    for name, raw in os.environ.items():
        if not raw or not _SENSITIVE_NAME.search(name):
            continue
        v = raw.strip()
        if len(v) >= MIN_SECRET_LEN:
            vals.add(v)
        compact = v.replace(" ", "")
        if len(compact) >= MIN_SECRET_LEN:
            vals.add(compact)
    return sorted(vals, key=len, reverse=True)


def mask_text(text: str) -> str:
    """把文字中所有 secret 值替換成 ***。純函數(只讀環境變數)。"""
    for s in current_secrets():
        if s in text:
            text = text.replace(s, MASK)
    return text


def _scrub_arg(a: Any) -> Any:
    if isinstance(a, (pd.DataFrame, pd.Series, np.ndarray)):
        return f"<{type(a).__name__} shape={tuple(a.shape)}>"
    return a


class SecretMaskFilter(logging.Filter):
    """遮罩 msg、args、異常堆疊與 stack_info;數據結構只記錄 shape。

    必須掛在 handler 上(不是 logger 上),logger 層的 filter 不會處理子 logger 的記錄。
    """

    def mask(self, text: str) -> str:
        return mask_text(text)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            args = record.args
            if isinstance(args, tuple):
                record.args = tuple(_scrub_arg(a) for a in args)
            elif isinstance(args, dict):
                record.args = {k: _scrub_arg(v) for k, v in args.items()}
            record.msg = mask_text(record.getMessage())  # 先格式化再遮罩
            record.args = ()
            if record.exc_info:
                # 預先寫入 exc_text,Formatter 會直接使用,不會再生成未遮罩版本
                record.exc_text = mask_text(
                    logging.Formatter().formatException(record.exc_info))
            elif record.exc_text:
                record.exc_text = mask_text(record.exc_text)
            if record.stack_info:
                record.stack_info = mask_text(record.stack_info)
        except Exception:  # noqa: BLE001
            # 寧可丟失訊息,也不可洩漏
            record.msg = "<log message suppressed: filter error>"
            record.args = ()
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
        return True


def setup_logging(level: int = logging.INFO, stream: Optional[IO[str]] = None) -> logging.Logger:
    """設定 root logger 的遮罩 handler(含第三方庫的日誌),回傳 'hsi_ml' logger。

    - 冪等:重複呼叫只更新等級,不會重複加 handler。
    - 現有 root handler 也會被加上遮罩 filter。
    - 副作用:修改 root logger、降低部分第三方 logger 的等級至 WARNING。
    """
    root = logging.getLogger()
    root.setLevel(level)
    for h in root.handlers:
        if not any(isinstance(f, SecretMaskFilter) for f in h.filters):
            h.addFilter(SecretMaskFilter())
    if not any(getattr(h, _HANDLER_FLAG, False) for h in root.handlers):
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s | %(message)s"))
        handler.addFilter(SecretMaskFilter())
        setattr(handler, _HANDLER_FLAG, True)
        root.addHandler(handler)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    return logging.getLogger("hsi_ml")

