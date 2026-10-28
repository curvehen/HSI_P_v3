"""預測長表的建構、轉換與後處理。

PRED_COLUMNS 是全局 predictions 表的子集。
pred_id、run_id、created_utc、universe_version、feature_set_version 由管線寫入時補上。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config

KEY = ["symbol", "target_date"]
PRED_COLUMNS = KEY + ["target", "quantile", "value", "model_version", "degraded_flag"]


def make_long(keys: pd.DataFrame, arr: np.ndarray, qs=config.QUANTILES,
              model_version: str = "", degraded=False) -> pd.DataFrame:
    """arr 形狀 (n, 4, nq),target 順序為 config.TARGETS。

    degraded 可為 bool 或長度 n 的陣列。Raises: ValueError(形狀不符)。
    """
    n, T, nq = len(keys), len(config.TARGETS), len(qs)
    arr = np.asarray(arr, dtype=float)
    if arr.shape != (n, T, nq):
        raise ValueError(f"arr shape {arr.shape} != {(n, T, nq)}")
    deg = np.broadcast_to(np.asarray(degraded, dtype=bool), (n,))
    rows = np.repeat(np.arange(n), T * nq)
    out = keys[KEY].reset_index(drop=True).iloc[rows].reset_index(drop=True)
    out["target"] = np.tile(np.repeat(np.array(config.TARGETS), nq), n)
    out["quantile"] = np.tile(np.asarray(qs, dtype=float), n * T)
    out["value"] = arr.reshape(-1)
    out["model_version"] = model_version
    out["degraded_flag"] = np.repeat(deg, T * nq)
    return out[PRED_COLUMNS]


def to_array(long: pd.DataFrame, qs=config.QUANTILES) -> tuple[pd.DataFrame, np.ndarray]:
    """長表轉 (keys, arr)。keys 按 (symbol, target_date) 排序;arr 形狀 (n, 4, nq)。

    Raises: ValueError(組合重複或缺少)。
    """
    qs = tuple(round(float(q), 6) for q in qs)
    d = long[KEY + ["target", "quantile", "value"]].copy()
    d["quantile"] = d["quantile"].astype(float).round(6)
    if d.duplicated(KEY + ["target", "quantile"]).any():
        raise ValueError("duplicate (symbol, target_date, target, quantile)")
    wide = d.pivot(index=KEY, columns=["target", "quantile"], values="value")
    cols = [(t, q) for t in config.TARGETS for q in qs]
    if [c for c in cols if c not in wide.columns]:
        raise ValueError("missing target/quantile columns")
    wide = wide[cols]
    arr = wide.to_numpy(dtype=float).reshape(len(wide), len(config.TARGETS), len(qs))
    return wide.index.to_frame(index=False), arr


def postprocess_array(arr: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """對 (n, 4, nq) 陣列做約束,回傳 (新陣列, 報告)。

    1. 各 target 沿 quantile 軸排序(不交叉)。NaN 保持 NaN(排在最後)。
    2. 逐 quantile 位置:up = max(up, gap, ret)、dn = min(dn, gap, ret),
       對應 H >= max(O,C)、L <= min(O,C)(log ratio 單調,不等式方向不變)。
    近似說明:逐 quantile 取 max/min 不等於 max(O,C) 的真實分位數,
    它是保守(區間略寬)的一致化處理。排序後再取 max/min,單調性保持。
    報告:crossing_rate(發生交叉的 (列, target) 比例)、hl_fix_rate(up 或 dn 被改動的 (列, quantile) 比例)。
    """
    a = np.asarray(arr, dtype=float).copy()
    ti = {t: i for i, t in enumerate(config.TARGETS)}
    srt = np.sort(a, axis=-1)
    with np.errstate(invalid="ignore"):
        crossed = (np.diff(a, axis=-1) < 0).any(axis=-1)
    a = srt
    gap, up, dn, ret = (a[:, ti[t], :] for t in ("gap", "up", "dn", "ret"))
    new_up = np.fmax(np.fmax(up, gap), ret)
    new_dn = np.fmin(np.fmin(dn, gap), ret)
    with np.errstate(invalid="ignore"):
        fixed = (new_up != up) | (new_dn != dn)
    fixed &= ~(np.isnan(up) | np.isnan(dn) | np.isnan(gap) | np.isnan(ret))
    a[:, ti["up"], :] = new_up
    a[:, ti["dn"], :] = new_dn
    return a, {"crossing_rate": float(crossed.mean()) if crossed.size else 0.0,
               "hl_fix_rate": float(fixed.mean()) if fixed.size else 0.0}


def postprocess(long: pd.DataFrame, qs=config.QUANTILES) -> tuple[pd.DataFrame, dict[str, float]]:
    """長表版後處理。model_version 取第一列;degraded_flag 按 key 取第一列。

    Raises: ValueError(長表不完整)。
    """
    keys, arr = to_array(long, qs)
    new, rep = postprocess_array(arr)
    deg = (long.groupby(KEY)["degraded_flag"].first()
           .reindex(pd.MultiIndex.from_frame(keys)).to_numpy())
    mv = str(long["model_version"].iloc[0]) if len(long) else ""
    return make_long(keys, new, qs, mv, deg), rep
