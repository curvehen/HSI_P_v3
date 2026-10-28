"""單根 bar 的有效性判斷。只標記,不修改價格。

重要:yfinance 的指數(例如 ^HSI)volume 常為 0(未核實)。
若 market_daily 內指數的 volume 為 0 或缺失,必須把該 symbol 放進 volume_exempt_symbols,
否則整個指數會被標為 zero_volume。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config

BAR_REQUIRED_COLUMNS = ("symbol", "open", "high", "low", "close", "volume",
                        "source_type", "quality_flag")


@dataclass(frozen=True)
class ValidityConfig:
    """有效性規則。數值是我的預設假設,未經驗證,請按實際數據調整。

    allowed_source_types:只有 real / proxy 可用於標籤(全局規範 A4)。
    ok_quality_flags:視為「沒問題」的 quality_flag 值;其他非空值都視為異常。
        模組 2 實際使用的旗標名稱未定,所以用「白名單」而不是猜測黑名單。
    logic_tol:OHLC 邏輯檢查的相對容差(例如 1e-6)。預設 0 即嚴格。
    max_calendar_gap_days:與上一根 bar 相隔的日曆日上限(停牌後復牌判斷)。
        10 是暫定值,需用港股長假期(例如農曆新年)的實際間隔核對。
    """
    allowed_source_types: tuple[str, ...] = config.TRAINING_SOURCE_TYPES
    ok_quality_flags: tuple[str, ...] = ("", "ok")
    require_positive_volume: bool = True
    volume_exempt_symbols: tuple[str, ...] = ()
    logic_tol: float = 0.0
    max_calendar_gap_days: int = 10


def bar_invalid_reason(df: pd.DataFrame, vcfg: ValidityConfig = ValidityConfig()) -> pd.Series:
    """逐列回傳「第一個」失敗原因;有效則為 None。索引與 df 相同。

    原因(按優先序):missing_ohlc、nonpositive_price、ohlc_logic、zero_volume、
    bad_source_type、bad_quality_flag。
    Raises: ValueError(缺欄位)。副作用:無。
    """
    missing = [c for c in BAR_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    o, h, l, c = (pd.to_numeric(df[x], errors="coerce")
                  for x in ("open", "high", "low", "close"))
    vol = pd.to_numeric(df["volume"], errors="coerce")
    reason = pd.Series([None] * len(df), index=df.index, dtype="object")

    def mark(cond: pd.Series, text: str) -> None:
        m = cond.fillna(False).astype(bool) & reason.isna()
        reason.loc[m] = text

    mark(o.isna() | h.isna() | l.isna() | c.isna(), "missing_ohlc")
    mark((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0), "nonpositive_price")
    tol = vcfg.logic_tol * c.abs()
    mark((h < np.maximum(o, c) - tol) | (l > np.minimum(o, c) + tol) | (h < l),
         "ohlc_logic")
    if vcfg.require_positive_volume:
        exempt = df["symbol"].isin(vcfg.volume_exempt_symbols)
        mark(~(vol > 0) & ~exempt, "zero_volume")
    mark(~df["source_type"].isin(vcfg.allowed_source_types), "bad_source_type")
    qf = df["quality_flag"]
    flagged = qf.notna() & ~qf.astype(str).str.strip().isin(vcfg.ok_quality_flags)
    mark(flagged, "bad_quality_flag")
    return reason
