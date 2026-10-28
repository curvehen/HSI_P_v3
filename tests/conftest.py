import time

import pytest

from hsi_ml import config, logging_setup


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """每個測試前清除相關環境變數與額外登記的 secret,避免本機設定影響結果。"""
    names = (*config.SECRET_NAMES, *config.VARIABLE_NAMES, "HSI_TEST_MODE", "HSI_DATA_DIR")
    for n in names:
        monkeypatch.delenv(n, raising=False)
    logging_setup._extra_secrets.clear()
    yield
    logging_setup._extra_secrets.clear()


@pytest.fixture
def no_sleep(monkeypatch):
    """讓重試測試不真的等待。"""
    monkeypatch.setattr(time, "sleep", lambda s: None)
