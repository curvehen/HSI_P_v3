import pandas as pd
import pytest

from hsi_ml import config
from hsi_ml.guards import (
    RepoPathError, SimulatedDataError, assert_no_simulated,
    assert_path_not_in_repo_tracked, is_test_mode, training_mask,
)


def _df(vals):
    return pd.DataFrame({"source_type": vals, "close": [123.456] * len(vals)})


def test_simulated_blocked_outside_test_mode():
    with pytest.raises(SimulatedDataError):
        assert_no_simulated(_df(["real", "simulated"]), test_mode=False)


def test_simulated_allowed_in_test_mode():
    assert_no_simulated(_df(["simulated"]), test_mode=True)


def test_env_var_controls_test_mode(monkeypatch):
    assert not is_test_mode()
    with pytest.raises(SimulatedDataError):
        assert_no_simulated(_df(["simulated"]))
    monkeypatch.setenv("HSI_TEST_MODE", "1")
    assert is_test_mode()
    assert_no_simulated(_df(["simulated"]))


def test_clean_data_passes():
    assert_no_simulated(_df(["real", "proxy", "filled"]), test_mode=False)


def test_missing_column_and_unknown_value():
    with pytest.raises(ValueError):
        assert_no_simulated(pd.DataFrame({"close": [1.0]}), test_mode=False)
    with pytest.raises(ValueError):
        assert_no_simulated(_df(["real", "fake"]), test_mode=False)


def test_error_message_has_count_only():
    with pytest.raises(SimulatedDataError) as ei:
        assert_no_simulated(_df(["simulated", "simulated"]), test_mode=False)
    msg = str(ei.value)
    assert "2" in msg and "123.456" not in msg


def test_training_mask_only_real_and_proxy():
    m = training_mask(_df(["real", "proxy", "filled", "simulated", None]))
    assert m.tolist() == [True, True, False, False, False]
    with pytest.raises(ValueError):
        training_mask(pd.DataFrame({"close": [1.0]}))


def test_path_outside_repo_is_safe(tmp_path):
    assert assert_path_not_in_repo_tracked(tmp_path / "x.parquet") is True


def test_ignored_paths_inside_repo_are_safe():
    root = config.project_root()
    assert assert_path_not_in_repo_tracked(root / "data" / "raw" / "a.parquet") is True
    assert assert_path_not_in_repo_tracked(root / "models" / "m.pkl") is True


def test_tracked_path_warns_or_raises_in_strict(caplog):
    p = config.project_root() / "src" / "hsi_ml" / "leak.py"
    with caplog.at_level("WARNING", logger="hsi_ml.guards"):
        assert assert_path_not_in_repo_tracked(p) is False
    assert "would be tracked" in caplog.text
    with pytest.raises(RepoPathError):
        assert_path_not_in_repo_tracked(p, strict=True)
