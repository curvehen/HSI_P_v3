"""Random walk baseline:價格不變。

所有 quantile 皆為 0(gap、ret、up、dn 都等於 c0)。
這是退化分佈(零寬度),只適合比較點預測;區間指標(覆蓋率、Winkler)會很差,
那是此 baseline 的本質,不是錯誤。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from .interval import KEY, make_long


def random_walk_predict(keys: pd.DataFrame, qs=config.QUANTILES,
                        model_version: str = "random_walk") -> pd.DataFrame:
    """對每個 (symbol, target_date) 輸出全 0 的 quantile 長表。副作用:無。"""
    k = keys[KEY].reset_index(drop=True)
    arr = np.zeros((len(k), len(config.TARGETS), len(qs)))
    return make_long(k, arr, qs, model_version, degraded=False)
