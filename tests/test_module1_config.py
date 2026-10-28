from pathlib import Path

import pytest

from hsi_ml import config
from hsi_ml.config import ConfigError, load_gmail_settings, load_llm_settings


def test_missing_secrets_returns_names_only(monkeypatch):
    monkeypatch.setenv("GMAIL_USER", "a@example.com")
    missing = config.missing_secrets()
    assert "GMAIL_USER" not in missing
    assert "DEEPSEEK_API_KEY" in missing


def test_data_dir_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HSI_DATA_DIR", str(tmp_path))
    assert config.data_dir() == Path(tmp_path)


def _llm_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123456")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.example.invalid/")
    monkeypatch.setenv("DEEPSEEK_MODEL", "some-model")


def test_llm_ok_and_defaults(monkeypatch):
    _llm_env(monkeypatch)
    s = load_llm_settings()
    assert s.base_url == "https://api.example.invalid"
    assert s.model == "some-model"
    assert s.timeout_s == 120 and s.max_retries == 2 and s.max_concurrency == 2


def test_llm_repr_hides_key(monkeypatch):
    _llm_env(monkeypatch)
    assert "sk-test-123456" not in repr(load_llm_settings())


def test_llm_missing_lists_names_not_values(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123456")
    with pytest.raises(ConfigError) as ei:
        load_llm_settings()
    msg = str(ei.value)
    assert "DEEPSEEK_BASE_URL" in msg and "DEEPSEEK_MODEL" in msg
    assert "sk-test" not in msg


def test_llm_requires_https(monkeypatch):
    _llm_env(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "http://api.example.invalid")
    with pytest.raises(ConfigError):
        load_llm_settings()


@pytest.mark.parametrize("bad", ["abc", "0", "-3"])
def test_llm_bad_int(monkeypatch, bad):
    _llm_env(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_TIMEOUT_S", bad)
    with pytest.raises(ConfigError):
        load_llm_settings()


def _gmail_env(monkeypatch):
    monkeypatch.setenv("GMAIL_USER", "me@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "abcd efgh ijkl mnop")


def test_gmail_defaults(monkeypatch):
    _gmail_env(monkeypatch)
    s = load_gmail_settings()
    assert s.password == "abcdefghijklmnop"       # 空白已移除
    assert s.to == "me@example.com"               # 預設寄給自己
    assert (s.host, s.port, s.mode) == ("smtp.gmail.com", 465, "ssl")


def test_gmail_repr_hides_password(monkeypatch):
    _gmail_env(monkeypatch)
    assert "abcdefghijklmnop" not in repr(load_gmail_settings())


def test_gmail_starttls_default_port(monkeypatch):
    _gmail_env(monkeypatch)
    monkeypatch.setenv("GMAIL_SMTP_MODE", "starttls")
    assert load_gmail_settings().port == 587


def test_gmail_explicit_port_and_to(monkeypatch):
    _gmail_env(monkeypatch)
    monkeypatch.setenv("GMAIL_SMTP_MODE", "starttls")
    monkeypatch.setenv("GMAIL_SMTP_PORT", "2525")
    monkeypatch.setenv("GMAIL_TO", "other@example.com")
    s = load_gmail_settings()
    assert s.port == 2525 and s.to == "other@example.com"


def test_gmail_bad_mode_and_user(monkeypatch):
    _gmail_env(monkeypatch)
    monkeypatch.setenv("GMAIL_SMTP_MODE", "plain")
    with pytest.raises(ConfigError):
        load_gmail_settings()
    monkeypatch.setenv("GMAIL_SMTP_MODE", "ssl")
    monkeypatch.setenv("GMAIL_USER", "not-an-email")
    with pytest.raises(ConfigError):
        load_gmail_settings()


def test_gmail_missing(monkeypatch):
    with pytest.raises(ConfigError) as ei:
        load_gmail_settings()
    assert "GMAIL_USER" in str(ei.value)
