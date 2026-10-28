"""全域常數與環境變數輔助。不含任何 secret 值。"""
from __future__ import annotations

import os
from pathlib import Path

QUANTILES: tuple[float, ...] = (0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95)
TARGETS: tuple[str, ...] = ("gap", "up", "dn", "ret")
TZ_HK = "Asia/Hong_Kong"
TZ_NY = "America/New_York"

SOURCE_TYPES: tuple[str, ...] = ("real", "proxy", "filled", "simulated")
TRAINING_SOURCE_TYPES: tuple[str, ...] = ("real", "proxy")

REQUIRED_SECRETS: tuple[str, ...] = (
    "GDRIVE_CLIENT_ID",
    "GDRIVE_CLIENT_SECRET",
    "GDRIVE_REFRESH_TOKEN",
    "DEEPSEEK_API_KEY",
    "TAVILY_API_KEY",
    "GMAIL_USER",
    "GMAIL_APP_PASSWORD",
)

DRIVE_ROOT_NAME = "hsi_ml_state"
DRIVE_UNIVERSE_PATH = "universe/universe.yaml"
DRIVE_UNIVERSE_HISTORY_DIR = "universe/history"


def project_root() -> Path:
    """回傳專案根目錄(src/hsi_ml/config.py 往上三層)。"""
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """本地數據目錄。可用環境變數 HSI_DATA_DIR 覆蓋。副作用:無(不建立目錄)。"""
    override = os.environ.get("HSI_DATA_DIR")
    return Path(override) if override else project_root() / "data"


def universe_template_path() -> Path:
    """回傳 repo 內的 universe 範本路徑。"""
    return project_root() / "config" / "universe.template.yaml"


def missing_secrets(names: tuple[str, ...] = REQUIRED_SECRETS) -> list[str]:
    """回傳未設定(或為空)的環境變數名稱。只回傳名稱,不回傳值。"""
    return [n for n in names if not os.environ.get(n)]
