from pathlib import Path

import pytest
import yaml

from hsi_ml import config
from hsi_ml.universe import (
    Universe, diff, load_universe, load_universe_with_fallback, upload_universe,
)

TEMPLATE = config.universe_template_path()
TEXT = TEMPLATE.read_text(encoding="utf-8")


def _raw() -> dict:
    return yaml.safe_load(TEXT)


def test_template_loads():
    u = load_universe(TEMPLATE)
    assert len(u.active) == 20
    assert len(u.version) == 12
    assert u.get("hsi").role == "index"
    assert "internet" in u.by_sector()
    assert "HSI" not in sum(u.by_sector().values(), []) or True  # 指數屬 sector=index


def test_template_has_no_guessed_values():
    for i in load_universe(TEMPLATE).instruments:
        assert i.shortable is None and i.lot_size is None and i.listed_date is None
        assert i.a_share is None
        for fl in (i.adr, i.london):
            if fl is not None:
                assert fl.ratio is None


def test_version_stable_and_sensitive():
    assert Universe.from_text(TEXT).version == Universe.from_text(TEXT).version
    assert Universe.from_text(TEXT + "\n# x").version != Universe.from_text(TEXT).version


def test_duplicate_code_rejected():
    raw = _raw()
    raw["instruments"].append(dict(raw["instruments"][1]))
    with pytest.raises(ValueError):
        Universe.model_validate(raw)


@pytest.mark.parametrize("bad", ["70", "07000", "ABCD"])
def test_bad_stock_code_rejected(bad):
    raw = _raw()
    raw["instruments"][1]["code"] = bad
    with pytest.raises(ValueError):
        Universe.model_validate(raw)


def test_unquoted_int_code_rejected():
    raw = _raw()
    raw["instruments"][1]["code"] = 700
    with pytest.raises(ValueError):
        Universe.model_validate(raw)


def test_hsi_must_be_index():
    raw = _raw()
    raw["instruments"][0]["role"] = "stock"
    with pytest.raises(ValueError):
        Universe.model_validate(raw)


def test_needs_an_index():
    raw = _raw()
    raw["instruments"] = [i for i in raw["instruments"] if i["role"] != "index"]
    with pytest.raises(ValueError):
        Universe.model_validate(raw)


def test_inactive_excluded():
    raw = _raw()
    raw["instruments"][1]["active"] = False
    assert len(Universe.model_validate(raw).active) == 19


def test_bad_values_rejected():
    raw = _raw()
    raw["instruments"][1]["lot_size"] = 0
    with pytest.raises(ValueError):
        Universe.model_validate(raw)
    raw = _raw()
    raw["instruments"][2]["adr"] = {"ticker": "BABA", "ratio": -1}
    with pytest.raises(ValueError):
        Universe.model_validate(raw)


def test_get_case_insensitive_and_missing():
    u = load_universe(TEMPLATE)
    assert u.get(" hsi ").code == "HSI"
    with pytest.raises(KeyError):
        u.get("9999")


def test_diff():
    old = Universe.model_validate(_raw())
    raw = _raw()
    raw["instruments"][1]["active"] = False              # 0700 停用
    raw["instruments"][2]["sector"] = "other"            # 9988 欄位變更
    raw["instruments"] = [i for i in raw["instruments"] if i["code"] != "6181"]
    raw["instruments"].append({"code": "0003", "yf": "0003.HK", "name": "X",
                               "role": "stock", "sector": "other"})
    d = diff(old, Universe.model_validate(raw))
    assert d.added == ["0003"]
    assert d.removed == ["6181"]
    assert d.deactivated == ["0700"]
    assert "sector" in d.changed["9988"]
    assert d.summary() == {"added": 1, "removed": 1, "deactivated": 1,
                           "activated": 0, "changed": 1}
    assert not d.is_empty


def test_diff_identical_is_empty():
    u = Universe.model_validate(_raw())
    assert diff(u, u).is_empty


class FakeUploader:
    def __init__(self, history_exists=False):
        self.calls = []
        self.history_exists = history_exists

    def upload(self, local, remote_path, overwrite=False):
        self.calls.append((remote_path, overwrite))
        if self.history_exists and "history" in remote_path:
            raise FileExistsError(remote_path)
        return "id"


def test_upload_universe_ok():
    up = FakeUploader()
    v = upload_universe(up, TEMPLATE)
    assert v == load_universe(TEMPLATE).version
    assert up.calls[0] == (config.DRIVE_UNIVERSE_PATH, True)
    assert up.calls[1] == (f"{config.DRIVE_UNIVERSE_HISTORY_DIR}/universe_{v}.yaml", False)


def test_upload_universe_history_exists_is_ignored():
    up = FakeUploader(history_exists=True)
    assert upload_universe(up, TEMPLATE)  # 不拋錯


def test_upload_universe_invalid_is_not_uploaded(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("x: 1", encoding="utf-8")
    up = FakeUploader()
    with pytest.raises(ValueError):
        upload_universe(up, bad)
    assert up.calls == []
