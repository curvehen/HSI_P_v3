"""Universe:標的名單的驗證、載入(Drive 優先、本地回退)與變更偵測。

version = 檔案原始文字的 SHA256 前 12 位。
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal, Optional, Protocol

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from . import config

log = logging.getLogger("hsi_ml.universe")


class ForeignListing(BaseModel):
    """海外上市證券。ratio = 1 份海外證券相當於幾股港股;未核實前必須為 None。"""
    ticker: str
    ratio: Optional[float] = None

    @field_validator("ratio")
    @classmethod
    def _ratio_pos(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError("ratio must be > 0")
        return v


class AShare(BaseModel):
    ticker: str


class Instrument(BaseModel):
    code: str
    yf: str
    name: str
    role: Literal["index", "stock"]
    sector: str
    active: bool = True
    adr: Optional[ForeignListing] = None
    london: Optional[ForeignListing] = None
    a_share: Optional[AShare] = None
    shortable: Optional[bool] = None
    lot_size: Optional[int] = None
    listed_date: Optional[date] = None
    min_history_days: int = 500

    @field_validator("code", mode="before")
    @classmethod
    def _norm_code(cls, v: Any) -> str:
        if not isinstance(v, str):
            # YAML 把 0700 解析成整數時會到這裡,明確拒絕而不是猜測補零
            raise ValueError(f"code must be a string (quote it in YAML): {v!r}")
        return v.strip().upper()

    @field_validator("yf", "sector", "name")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v.strip()

    @field_validator("lot_size", "min_history_days")
    @classmethod
    def _positive(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("must be > 0")
        return v

    @model_validator(mode="after")
    def _role_code(self) -> "Instrument":
        if self.role == "stock" and not (self.code.isdigit() and len(self.code) == 4):
            raise ValueError(f"stock code must be 4 digits: {self.code}")
        if self.code == "HSI" and self.role != "index":
            raise ValueError("HSI must have role=index")
        return self


@dataclass
class UniverseDiff:
    """兩個 Universe 的差異。changed: code -> {field: (old, new)}。"""
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    deactivated: list[str] = field(default_factory=list)
    activated: list[str] = field(default_factory=list)
    changed: dict[str, dict[str, tuple[Any, Any]]] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.deactivated
                    or self.activated or self.changed)

    def summary(self) -> dict[str, int]:
        """只回傳計數,可安全寫入日誌。"""
        return {"added": len(self.added), "removed": len(self.removed),
                "deactivated": len(self.deactivated),
                "activated": len(self.activated), "changed": len(self.changed)}


class Universe(BaseModel):
    version_note: str = ""
    instruments: list[Instrument]
    version: str = Field(default="", description="SHA256 前 12 位,載入時自動填入")

    @model_validator(mode="after")
    def _validate(self) -> "Universe":
        codes = [i.code for i in self.instruments]
        if len(codes) != len(set(codes)):
            dups = sorted({c for c in codes if codes.count(c) > 1})
            raise ValueError(f"duplicate codes: {dups}")
        if not any(i.role == "index" for i in self.instruments):
            raise ValueError("universe needs at least one index")
        return self

    @property
    def active(self) -> list[Instrument]:
        """所有 active=True 的標的(保持檔案順序)。"""
        return [i for i in self.instruments if i.active]

    def get(self, code: str) -> Instrument:
        """按代碼取標的。找不到拋 KeyError。"""
        c = code.strip().upper()
        for i in self.instruments:
            if i.code == c:
                return i
        raise KeyError(code)

    def by_sector(self) -> dict[str, list[str]]:
        """active 標的按板塊分組,回傳 sector -> [code]。"""
        out: dict[str, list[str]] = {}
        for i in self.active:
            out.setdefault(i.sector, []).append(i.code)
        return out

    @classmethod
    def from_text(cls, text: str) -> "Universe":
        """由 YAML 文字建立並填入 version。

        Raises: ValueError(格式錯誤)、pydantic.ValidationError(欄位驗證失敗,
        它是 ValueError 的子類)。
        """
        raw = yaml.safe_load(text)
        if not isinstance(raw, dict) or "instruments" not in raw:
            raise ValueError("universe yaml must be a mapping with 'instruments'")
        raw.pop("version", None)
        u = cls.model_validate(raw)
        u.version = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        return u


def load_universe(path: str | Path) -> Universe:
    """由本地檔案載入。Raises: FileNotFoundError、ValueError。"""
    return Universe.from_text(Path(path).read_text(encoding="utf-8"))


def diff(old: Universe, new: Universe) -> UniverseDiff:
    """比較兩個 Universe(忽略 version 與 version_note)。"""
    o = {i.code: i for i in old.instruments}
    n = {i.code: i for i in new.instruments}
    d = UniverseDiff()
    d.added = sorted(set(n) - set(o))
    d.removed = sorted(set(o) - set(n))
    for code in sorted(set(o) & set(n)):
        a, b = o[code], n[code]
        if a.active and not b.active:
            d.deactivated.append(code)
        elif not a.active and b.active:
            d.activated.append(code)
        da, db = a.model_dump(mode="json"), b.model_dump(mode="json")
        ch = {k: (da[k], db[k]) for k in da if k != "active" and da[k] != db[k]}
        if ch:
            d.changed[code] = ch
    return d


class _Downloader(Protocol):
    def download(self, remote_path: str, local: str | Path) -> bool: ...


class _Uploader(Protocol):
    def upload(self, local: str | Path, remote_path: str, overwrite: bool = False) -> str: ...


@dataclass
class UniverseLoad:
    universe: Universe
    source: Literal["drive", "local_template"]


def load_universe_with_fallback(
    drive: Optional[_Downloader],
    cache_path: str | Path,
    remote_path: str = config.DRIVE_UNIVERSE_PATH,
    template_path: str | Path | None = None,
) -> UniverseLoad:
    """Drive 優先,失敗(未連線、檔案不存在、下載錯誤、驗證失敗)回退到本地範本。

    副作用:成功時把 Drive 檔案寫到 cache_path;日誌記錄來源(不記錄內容)。
    Raises: 僅當範本本身也無法載入時拋出。
    """
    if drive is not None:
        try:
            if drive.download(remote_path, cache_path):
                u = load_universe(cache_path)
                log.info("universe source=drive version=%s n_active=%d",
                         u.version, len(u.active))
                return UniverseLoad(u, "drive")
            log.warning("universe not found on drive; falling back to template")
        except Exception as e:  # noqa: BLE001
            log.warning("universe drive load failed (%s); falling back to template",
                        type(e).__name__)
    else:
        log.warning("drive unavailable; using local template universe")
    u = load_universe(template_path or config.universe_template_path())
    log.info("universe source=local_template version=%s n_active=%d",
             u.version, len(u.active))
    return UniverseLoad(u, "local_template")


def upload_universe(drive: _Uploader, local_path: str | Path,
                    remote_path: str = config.DRIVE_UNIVERSE_PATH,
                    history_dir: str = config.DRIVE_UNIVERSE_HISTORY_DIR) -> str:
    """驗證後上傳 universe。drive.file scope 看不到手動放入 Drive 的檔案,
    所以 universe 必須經此函數上傳。

    - 當前版本 `universe/universe.yaml`:覆蓋。
    - 歷史版本 `universe/history/universe_<version>.yaml`:append-only,已存在則略過。
    回傳 version。Raises: ValueError(驗證失敗,不上傳)。
    """
    u = load_universe(local_path)
    drive.upload(local_path, remote_path, overwrite=True)
    try:
        drive.upload(local_path, f"{history_dir}/universe_{u.version}.yaml", overwrite=False)
    except FileExistsError:
        log.info("universe history version=%s already exists", u.version)
    return u.version
