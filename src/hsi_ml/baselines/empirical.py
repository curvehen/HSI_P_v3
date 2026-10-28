"""Empirical baseline:用過去 N 根「有效標籤」的經驗分位數。

無 look-ahead:對目標日 T,只用 target_date < T 的標籤(merge_asof 且不含相等日)。
標籤 T-1 在 T-1 收市後才產生,早於 T 日 07:00 HKT,所以可用。
樣本不足(< min_obs),或最近一筆有效標籤距 T 超過 max_stale_days,
輸出 NaN 並標記 degraded_flag=True。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from .interval import KEY, make_long


def empirical_predict(labels: pd.DataFrame, keys: pd.DataFrame, window: int = 250,
                      min_obs: int = 60, max_stale_days: int = 10,
                      qs=config.QUANTILES,
                      model_version: str = "empirical") -> pd.DataFrame:
    """labels:模組 3 labels 表;keys:要預測的 (symbol, target_date)。

    window/min_obs/max_stale_days 是暫定值,未經驗證。
    Raises: ValueError(window < min_obs)。副作用:無。
    """
    if window < min_obs:
        raise ValueError("window must be >= min_obs")
    keys = keys[KEY].reset_index(drop=True).copy()
    keys["target_date"] = pd.to_datetime(keys["target_date"])
    v = labels.loc[labels["label_valid"], ["symbol", "target_date", *config.TARGETS]].copy()
    v["target_date"] = pd.to_datetime(v["target_date"])
    arr = np.full((len(keys), len(config.TARGETS), len(qs)), np.nan)

    for sym, idx in keys.groupby("symbol").groups.items():
        g = v[v["symbol"] == sym].sort_values("target_date").reset_index(drop=True)
        if g.empty:
            continue
        stats = pd.DataFrame({"target_date": g["target_date"],
                              "_stat_date": g["target_date"]})
        for t in config.TARGETS:
            r = g[t].rolling(window, min_periods=min_obs)
            for qi, q in enumerate(qs):
                stats[f"{t}_{qi}"] = r.quantile(q)
        k = keys.loc[idx, ["target_date"]].copy()
        k["_i"] = np.asarray(idx)
        k = k.sort_values("target_date")
        m = pd.merge_asof(k, stats, on="target_date", direction="backward",
                          allow_exact_matches=False)
        age = (m["target_date"] - m["_stat_date"]).dt.days
        ok = (m["_stat_date"].notna() & (age <= max_stale_days)).to_numpy()
        rows = m["_i"].to_numpy()[ok]
        for ti, t in enumerate(config.TARGETS):
            for qi in range(len(qs)):
                arr[rows, ti, qi] = m.loc[ok, f"{t}_{qi}"].to_numpy()

    degraded = np.isnan(arr).any(axis=(1, 2))
    return make_long(keys, arr, qs, model_version, degraded)
