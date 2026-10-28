import io
import logging

import numpy as np
import pandas as pd

from hsi_ml import logging_setup
from hsi_ml.logging_setup import (
    SecretMaskFilter, current_secrets, mask_text, register_secret, setup_logging,
)

SECRET = "abcdef123456"


def make_logger(name="hsi_ml.t"):
    buf = io.StringIO()
    lg = logging.getLogger(name)
    lg.handlers = []
    lg.propagate = False
    lg.setLevel(logging.DEBUG)
    h = logging.StreamHandler(buf)
    h.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    h.addFilter(SecretMaskFilter())
    lg.addHandler(h)
    return lg, buf


def test_mask_plain_text(monkeypatch):
    monkeypatch.setenv("MY_API_KEY", SECRET)
    assert SECRET not in mask_text(f"token={SECRET}")
    assert "***" in mask_text(f"token={SECRET}")


def test_short_value_and_non_sensitive_names_not_masked(monkeypatch):
    monkeypatch.setenv("MY_KEY", "abc12")                 # 長度 < 6
    monkeypatch.setenv("GMAIL_USER", "me@example.com")    # 名稱不含敏感字
    out = mask_text("abc12 me@example.com")
    assert out == "abc12 me@example.com"


def test_overlapping_secrets_longest_first(monkeypatch):
    monkeypatch.setenv("A_KEY", "abcdef")
    monkeypatch.setenv("B_SECRET", SECRET)
    out = mask_text(f"{SECRET} and abcdef")
    assert "abcdef" not in out
    assert out == "*** and ***"


def test_app_password_with_spaces_masked_both_forms(monkeypatch):
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "abcd efgh ijkl mnop")
    out = mask_text("abcd efgh ijkl mnop / abcdefghijklmnop")
    assert "abcd" not in out and "mnop" not in out


def test_message_and_args_masked(monkeypatch):
    monkeypatch.setenv("MY_API_KEY", SECRET)
    lg, buf = make_logger()
    lg.info("token=%s", SECRET)
    lg.info(f"direct {SECRET}")
    assert SECRET not in buf.getvalue()
    assert buf.getvalue().count("***") == 2


def test_dict_args_masked(monkeypatch):
    monkeypatch.setenv("MY_API_KEY", SECRET)
    lg, buf = make_logger()
    lg.info("v=%(a)s", {"a": SECRET})
    assert SECRET not in buf.getvalue()


def test_secret_set_after_logger_creation_is_masked(monkeypatch):
    lg, buf = make_logger()
    monkeypatch.setenv("LATE_TOKEN", SECRET)
    lg.info("x %s", SECRET)
    assert SECRET not in buf.getvalue()


def test_exception_traceback_masked(monkeypatch):
    monkeypatch.setenv("MY_API_KEY", SECRET)
    lg, buf = make_logger()
    secret = SECRET
    try:
        raise ValueError(f"bad {secret}")
    except ValueError:
        lg.exception("failed")
    out = buf.getvalue()
    assert "ValueError" in out
    assert SECRET not in out


def test_dataframe_arg_logs_shape_only():
    lg, buf = make_logger()
    df = pd.DataFrame({"close": [12345.678, 2.0]})
    lg.info("df=%s arr=%s", df, np.array([98765.4321]))
    out = buf.getvalue()
    assert "shape=(2, 1)" in out and "shape=(1,)" in out
    assert "12345" not in out and "98765" not in out


def test_register_secret(monkeypatch):
    register_secret("runtime-token-xyz")
    register_secret("abc")  # 太短,忽略
    assert "runtime-token-xyz" in current_secrets()
    assert "abc" not in current_secrets()
    assert "runtime-token-xyz" not in mask_text("a runtime-token-xyz b")


def test_filter_error_suppresses_instead_of_leaking():
    lg, buf = make_logger()
    lg.info("%d", "not-an-int")     # 格式化會失敗
    assert "suppressed" in buf.getvalue()


def test_setup_logging_is_idempotent_and_masks_third_party(monkeypatch):
    monkeypatch.setenv("MY_API_KEY", SECRET)
    root = logging.getLogger()
    old_handlers, old_level = root.handlers[:], root.level
    root.handlers = []
    try:
        buf = io.StringIO()
        setup_logging(stream=buf)
        setup_logging(stream=buf)
        flagged = [h for h in root.handlers if getattr(h, logging_setup._HANDLER_FLAG, False)]
        assert len(flagged) == 1
        logging.getLogger("some.third.party").warning("leak? %s", SECRET)
        assert SECRET not in buf.getvalue()
        assert "leak?" in buf.getvalue()
    finally:
        root.handlers = old_handlers
        root.setLevel(old_level)
