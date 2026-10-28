"""ATR baseline:由 ATR 推導 gap / ret / up / dn 的 quantile。

推導(atr_rel = ATR / close,取目標日 T 之前最後一根有效 bar;c0 = 該 bar 的 close):
    z_q = Φ^-1(q)                      標準常態分位數
    h_q = Φ^-1((1 + q) / 2)            半常態分位數,用於 high 與 low 的距離
    gap_q = k_gap * atr_rel * z_q
    ret_q = k_ret * atr_rel * z_q
    up_q  =  k_hl * atr_rel * h_q      up = ln(H/c0),假設 H 距 c0 的幅度為半常態
    dn_q  = -k_hl * atr_rel * h_(1-q)  dn = ln(L/c0),與 up 鏡像
明示的假設:
- gap、ret 為零中位數的對稱常態分佈,尺度正比於 ATR。
- 上下影幅度為半常態,尺度正比於 ATR。
- k_gap、k_ret、k_hl 預設 1.0 只是「未校準的佔位值」,不是估計出來的數字。
  必須在 walk-forward 內校準(模組 9),否則此 baseline 的區間可能系統性偏寬或偏窄。
- atr_rel 為相對值,用 T 之前最後一根 bar 的 close 當分母,與 c0 一致。
無 look-ahead:merge_asof 且不含相等日,只用 target_date 之前的 bar。
樣本不足(bar 數 < window)或最近 bar 距 T 超過 max_stale_days,輸出 NaN 並標記 degraded。
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

import numpy as np
import pandas as pd

from .. import config
from ..labels.ohlc import _to_naive_date
from ..labels.validity import ValidityConfig, bar_invalid_reason
from .interval import KEY, make_long

_ND = NormalDist()
_BAR_COLS = ["symbol", "market_date", "open", "high", "low", "close", "volume",
             "source_type", "quality_flag"]


@dataclass(frozen=True)
class AtrConfig:
    """window=14 與各 k=1.0 是未校準的暫定值。"""
    window: int = 14
    smoothing: Literal["sma", "wilder"] = "sma"
    k_gap: float = 1.0
    k_ret: float = 1.0
    k_hl: float = 1.0
    max_stale_days: int = 10


def atr_rel_series(bars: pd.DataFrame, window: int,
                   smoothing: str = "sma") -> pd.Series:
    """單一標的、按日期排序的 bar,回傳 ATR / close。前 window-1 根為 NaN。

    TR = max(H-L, |H-prev_close|, |L-prev_close|);第一根無 prev_close,只用 H-L。
    wilder:ewm(alpha=1/window, adjust=False),以第一根 TR 為種子。
    Raises: ValueError(smoothing 不合法)。
    """
    prev_c = bars["close"].shift(1)
    tr = pd.concat([bars["high"] - bars["low"],
                    (bars["high"] - prev_c).abs(),
                    (bars["low"] - prev_c).abs()], axis=1).max(axis=1)
    if smoothing == "sma":
        atr = tr.rolling(window, min_periods=window).mean()
    elif smoothing == "wilder":
        atr = tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    else:
        raise ValueError(f"unknown smoothing: {smoothing}")
    return atr / bars["close"]


def atr_quantile_array(atr_rel: np.ndarray, cfg: AtrConfig = AtrConfig(),
                       qs=config.QUANTILES) -> np.ndarray:
    """atr_rel 形狀 (n,),回傳 (n, 4, nq),target 順序為 config.TARGETS。NaN 會傳遞。"""
    a = np.asarray(atr_rel, dtype=float)[:, None]
    z = np.array([_ND.inv_cdf(q) for q in qs])[None, :]
    h_up = np.array([_ND.inv_cdf((1 + q) / 2) for q in qs])[None, :]
    h_dn = np.array([_ND.inv_cdf((1 + (1 - q)) / 2) for q in qs])[None, :]
    by_target = {
        "gap": cfg.k_gap * a * z,
        "ret": cfg.k_ret * a * z,
        "up": cfg.k_hl * a * h_up,
        "dn": -cfg.k_hl * a * h_dn,
    }
    return np.stack([by_target[t] for t in config.TARGETS], axis=1)


def atr_predict(market: pd.DataFrame, keys: pd.DataFrame, cfg: AtrConfig = AtrConfig(),
                vcfg: ValidityConfig = ValidityConfig(), qs=config.QUANTILES,
                model_version: str = "atr") -> pd.DataFrame:
    """market:market_daily 長表;keys:要預測的 (symbol, target_date)。

    只使用通過 bar_invalid_reason 的 bar(指數 volume 為 0 時須設 volume_exempt_symbols)。
    Raises: ValueError(缺欄位)。副作用:無。
    """
    missing = [c for c in _BAR_COLS if c not in market.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    keys = keys[KEY].reset_index(drop=True).copy()
    keys["target_date"] = pd.to_datetime(keys["target_date"])

    m = market[_BAR_COLS].copy()
    for c in ("open", "high", "low", "close"):
        m[c] = pd.to_numeric(m[c], errors="coerce").astype("float64")
    m["_date"] = _to_naive_date(m["market_date"])
    m = m[bar_invalid_reason(m, vcfg).isna()]

    atr = np.full(len(keys), np.nan)
    for sym, idx in keys.groupby("symbol").groups.items():
        g = m[m["symbol"] == sym].sort_values("_date").reset_index(drop=True)
        if g.empty:
            continue
        stats = pd.DataFrame({"target_date": g["_date"], "_bar_date": g["_date"],
                              "_atr": atr_rel_series(g, cfg.window, cfg.smoothing)})
        k = keys.loc[idx, ["target_date"]].copy()
        k["_i"] = np.asarray(idx)
        k = k.sort_values("target_date")
        r = pd.merge_asof(k, stats, on="target_date", direction="backward",
                          allow_exact_matches=False)
        age = (r["target_date"] - r["_bar_date"]).dt.days
        ok = (r["_atr"].notna() & (age <= cfg.max_stale_days)).to_numpy()
        atr[r["_i"].to_numpy()[ok]] = r["_atr"].to_numpy()[ok]

    arr = atr_quantile_array(atr, cfg, qs)
    return make_long(keys, arr, qs, model_version, degraded=np.isnan(atr))
