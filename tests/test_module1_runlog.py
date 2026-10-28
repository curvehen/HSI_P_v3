import numpy as np
import pandas as pd
import pytest

from hsi_ml.runlog import COLUMNS, MAX_ERR_LEN, RunLog

SECRET = "supersecret123"


def test_ok_soft_hard_statuses_in_order():
    rl = RunLog(run_id="r1", universe_version="u1")
    with rl.step("a") as ctx:
        ctx["rows"] = 5
    with rl.step("b", soft=True):
        raise ValueError("boom")
    assert not rl.has_failure() and rl.has_soft_failure()
    with pytest.raises(RuntimeError):
        with rl.step("c"):
            raise RuntimeError("fail")
    assert rl.has_failure()
    assert [r["status"] for r in rl.rows] == ["ok", "failed_soft", "failed"]
    assert [r["step"] for r in rl.rows] == ["a", "b", "c"]
    assert all(r["run_id"] == "r1" and r["universe_version"] == "u1" for r in rl.rows)
    assert rl.rows[0]["n_rows"] == 5
    assert rl.rows[1]["error_type"] == "ValueError"


def test_keyboard_interrupt_recorded_and_reraised():
    rl = RunLog()
    with pytest.raises(KeyboardInterrupt):
        with rl.step("x", soft=True):   # soft 也不可吞掉 BaseException
            raise KeyboardInterrupt()
    assert rl.rows[0]["status"] == "failed"


def test_error_message_masked_and_truncated(monkeypatch):
    monkeypatch.setenv("MY_API_TOKEN", SECRET)
    rl = RunLog()
    with rl.step("s", soft=True):
        raise ValueError(f"{SECRET} " + "x" * 1000)
    msg = rl.rows[0]["error_msg"]
    assert SECRET not in msg
    assert len(msg) <= MAX_ERR_LEN


def test_n_rows_types():
    rl = RunLog()
    with rl.step("a") as ctx:
        ctx["rows"] = np.int64(7)
    with rl.step("b") as ctx:
        ctx["rows"] = "many"
    with rl.step("c"):
        pass
    assert [r["n_rows"] for r in rl.rows] == [7, None, None]


def test_meta_never_contains_dataframe_content():
    rl = RunLog()
    df = pd.DataFrame({"close": [999.5, 1000.5]})
    with rl.step("s", frame=df, n=3, label="ok"):
        pass
    meta = rl.rows[0]["meta"]
    assert "shape=(2, 1)" in meta
    assert "999.5" not in meta
    assert "n" in meta


def test_meta_secret_masked(monkeypatch):
    monkeypatch.setenv("MY_API_TOKEN", SECRET)
    rl = RunLog()
    with rl.step("s", note=SECRET):
        pass
    assert SECRET not in rl.rows[0]["meta"]


def test_empty_frame_has_schema():
    df = RunLog().to_frame()
    assert list(df.columns) == COLUMNS
    assert len(df) == 0


def test_parquet_roundtrip(tmp_path):
    rl = RunLog(universe_version="u")
    with rl.step("a") as ctx:
        ctx["rows"] = 3
    with rl.step("b", soft=True):
        raise ValueError("x")
    with rl.step("c"):
        pass
    p = rl.to_parquet(tmp_path / "sub" / "run.parquet")
    back = pd.read_parquet(p)
    assert len(back) == 3
    assert list(back.columns) == COLUMNS
    assert back["status"].tolist() == ["ok", "failed_soft", "ok"]


def test_summary_has_no_error_text_or_meta():
    rl = RunLog(run_id="r")
    with rl.step("a", note="private"):
        pass
    with rl.step("b", soft=True):
        raise ValueError("boom-message")
    s = rl.summary()
    assert s["n_steps"] == 2 and s["n_ok"] == 1 and s["n_failed_soft"] == 1
    assert s["n_failed"] == 0
    text = str(s)
    assert "boom-message" not in text and "private" not in text
    assert s["steps"][0] == {"step": "a", "status": "ok", "n_rows": None}
