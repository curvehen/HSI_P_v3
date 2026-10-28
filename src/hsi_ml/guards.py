"""守門函數:阻止 simulated 數據進入真實管線;警告會被 git 追蹤的寫入路徑。"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config

log = logging.getLogger("hsi_ml.guards")

TEST_MODE_ENV = "HSI_TEST_MODE"
_FALLBACK_IGNORED_DIRS = {"data", "models", "logs", ".venv"}
_FALLBACK_IGNORED_SUFFIXES = {".parquet", ".duckdb", ".pkl", ".pt", ".pth"}


class SimulatedDataError(RuntimeError):
    """非測試模式下碰到 source_type == 'simulated'。"""


class RepoPathError(RuntimeError):
    """strict 模式下,目標路徑會被 git 追蹤。"""


def is_test_mode() -> bool:
    """環境變數 HSI_TEST_MODE == '1' 時為測試模式。生產環境不可設定此變數。"""
    return os.environ.get(TEST_MODE_ENV, "").strip() == "1"


def assert_no_simulated(df: pd.DataFrame, test_mode: Optional[bool] = None,
                        column: str = "source_type") -> None:
    """非測試模式下,df 含 'simulated' 就拋 SimulatedDataError。

    test_mode=None 時讀環境變數。
    Raises:
      SimulatedDataError:含 simulated 且非測試模式。
      ValueError:缺少 source_type 欄位,或含未知的 source_type 值。
    副作用:無。訊息只含筆數,不含數據內容。
    """
    if column not in df.columns:
        raise ValueError(f"missing required column: {column}")
    vals = df[column].dropna()
    unknown = set(vals.unique()) - set(config.SOURCE_TYPES)
    if unknown:
        raise ValueError(f"unknown {column} values: {sorted(map(str, unknown))}")
    n_sim = int((vals == "simulated").sum())
    if n_sim and not (is_test_mode() if test_mode is None else test_mode):
        raise SimulatedDataError(
            f"{n_sim} simulated rows found outside test mode")


def training_mask(df: pd.DataFrame, column: str = "source_type") -> pd.Series:
    """回傳布林遮罩:只有 real / proxy 的列可用於訓練與績效統計。

    filled 與 simulated 都排除;缺值視為不可用。Raises: ValueError(缺欄位)。
    """
    if column not in df.columns:
        raise ValueError(f"missing required column: {column}")
    return df[column].isin(config.TRAINING_SOURCE_TYPES)


def _git_check_ignore(path: Path, root: Path) -> Optional[bool]:
    """True=被 .gitignore 忽略,False=未被忽略,None=無法判斷(沒有 git 或非 repo)。"""
    try:
        r = subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=root,
                           capture_output=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode == 0:
        return True
    if r.returncode == 1:
        return False
    return None


def assert_path_not_in_repo_tracked(path: str | Path, strict: bool = False) -> bool:
    """檢查寫入路徑是否安全(在 repo 外,或被 .gitignore 忽略)。

    回傳 True=安全。不安全時記錄警告並回傳 False;strict=True 則拋 RepoPathError。
    判斷順序:repo 外 → 安全;git check-ignore;無 git 時用後備規則(目錄名、副檔名)。
    副作用:可能呼叫 git 子程序(逾時 5 秒);日誌只含相對路徑。
    """
    root = config.project_root()
    p = Path(path).resolve()
    try:
        rel = p.relative_to(root)
    except ValueError:
        return True  # repo 外
    ignored = _git_check_ignore(p, root)
    if ignored is None:
        ignored = (bool(rel.parts) and rel.parts[0] in _FALLBACK_IGNORED_DIRS) \
            or p.suffix in _FALLBACK_IGNORED_SUFFIXES or p.name == ".env"
    if ignored:
        return True
    msg = f"path would be tracked by git: {rel.as_posix()}"
    if strict:
        raise RepoPathError(msg)
    log.warning(msg)
    return False
