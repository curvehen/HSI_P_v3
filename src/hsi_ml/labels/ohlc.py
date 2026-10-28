"""由 market_daily 生成標籤 gap / up / dn / ret(相對 c0 的 log ratio)。

定義(c0 = 該標的「自己」上一根 bar 的收市價,不用 HSI 日曆代替):
    gap = ln(open  / c0)
    up  = ln(high  / c0)
    dn  = ln(low   / c0)
    ret = ln(close / c0)

價格口徑:
- c0_mode="raw"(預設):open/high/low/close/c0 全用原始價格欄位。
  除息日的 gap 會包含股息跳空(這是真實可交易的價差)。
- c0_mode="div_adjusted":把上一日收市價換算到「當日價格水平」:
      c0 = adj_close[t-1] * close[t] / adj_close[t]
  之後與當日未調整的 OHLC 比較,等於扣除除息缺口。
  假設 adj_close 只反映股息與拆股調整(yfinance 的行為,未核實)。
- 兩種模式都不會改寫原始價格,也不混用調整後與未調整的 OHLC。

標籤只在 T 日收市後才可用:label_available_at_utc = T 日 bar 的 available_at_utc。
不處理:缺少整根 bar(中間漏日)與停牌的區別。漏日只會被「日曆間隔上限」間接攔截。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .validity import ValidityConfig, bar_invalid_reason

LABEL_COLUMNS = ["symbol", "target_date", "c0", "gap", "up", "dn", "ret",
                 "label_valid", "invalid_reason", "label_available_at_utc"]
MARKET_REQUIRED = ("symbol", "market_date", "open", "high", "low", "close",
                   "adj_close", "volume", "source_type", "quality_flag",
                   "available_at_utc")


@dataclass(frozen=True)
class LabelConfig:
    c0_mode: Literal["raw", "div_adjusted"] = "raw"
    validity: ValidityConfig = field(default_factory=ValidityConfig)


def _to_naive_date(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s)
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    return d.dt.normalize()

def build_labels(market: pd.DataFrame, cfg: LabelConfig = LabelConfig()) -> pd.DataFrame:
    """生成 labels 表(全局規範 A10,另加 label_available_at_utc)。

    輸入:market_daily 長表(欄位見 MARKET_REQUIRED)。
    輸出:每個輸入 bar 一列,按 (symbol, target_date) 排序。
        label_valid=False 的列,c0 與四個標籤一律為 NaN,原因在 invalid_reason。
    invalid_reason 優先序:
        本根 bar 的原因(見 validity.bar_invalid_reason)
        > no_prev_bar(該標的第一根 bar)
        > prev_bar_invalid(上一根 bar 本身無效,c0 不可靠)
        > calendar_gap(與上一根相隔日曆日超過上限,例如停牌後復牌)
        > missing_adj_close(div_adjusted 模式下缺少或非正的 adj_close)
    Raises: ValueError(缺欄位,或 (symbol, market_date) 重複)。副作用:無(不改輸入)。
    """
    missing = [c for c in MARKET_REQUIRED if c not in market.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    df = market.copy()
    df["_date"] = _to_naive_date(df["market_date"])
    if df.duplicated(["symbol", "_date"]).any():
        raise ValueError("duplicate (symbol, market_date) rows")
    for c in ("open", "high", "low", "close", "adj_close"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df = df.sort_values(["symbol", "_date"]).reset_index(drop=True)

    reason = bar_invalid_reason(df, cfg.validity).astype("object")
    bar_ok = reason.isna()

    grp = df.groupby("symbol", sort=False)
    prev_close = grp["close"].shift(1)
    prev_adj = grp["adj_close"].shift(1)
    prev_date = grp["_date"].shift(1)
    prev_ok = bar_ok.groupby(df["symbol"], sort=False).shift(1)
    has_prev = prev_date.notna()
    prev_ok = prev_ok.where(has_prev, False).fillna(False).astype(bool)

    def mark(cond: pd.Series, text: str) -> None:
        m = cond.fillna(False).astype(bool) & reason.isna()
        reason.loc[m] = text

    mark(~has_prev, "no_prev_bar")
    mark(has_prev & ~prev_ok, "prev_bar_invalid")
    gap_days = (df["_date"] - prev_date).dt.days
    mark(gap_days > cfg.validity.max_calendar_gap_days, "calendar_gap")

    if cfg.c0_mode == "raw":
        c0 = prev_close.copy()
    elif cfg.c0_mode == "div_adjusted":
        adj_bad = ~((df["adj_close"] > 0) & (prev_adj > 0))
        mark(adj_bad, "missing_adj_close")
        c0 = prev_adj * df["close"] / df["adj_close"]
    else:  # pragma: no cover
        raise ValueError(f"unknown c0_mode: {cfg.c0_mode}")
    mark(~(c0 > 0), "bad_c0")

    valid = reason.isna()
    out = pd.DataFrame({"symbol": df["symbol"], "target_date": df["_date"]})
    out["c0"] = c0.where(valid)
    for name, col in (("gap", "open"), ("up", "high"), ("dn", "low"), ("ret", "close")):
        vals = pd.Series(np.nan, index=df.index, dtype="float64")
        vals.loc[valid] = np.log(df.loc[valid, col] / c0.loc[valid])
        out[name] = vals
    out["label_valid"] = valid.to_numpy()
    out["invalid_reason"] = reason
    out["label_available_at_utc"] = pd.to_datetime(df["available_at_utc"], utc=True)
    return out[LABEL_COLUMNS]


def valid_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """只回傳 label_valid=True 的列(訓練與評估用)。"""
    return labels.loc[labels["label_valid"]].reset_index(drop=True)


def label_summary(labels: pd.DataFrame) -> dict[str, int]:
    """只回傳計數,可安全寫入日誌:total、valid、各 invalid_reason 的數量。"""
    out = {"total": int(len(labels)), "valid": int(labels["label_valid"].sum())}
    for k, v in labels.loc[~labels["label_valid"], "invalid_reason"].value_counts().items():
        out[f"invalid:{k}"] = int(v)
    return out

