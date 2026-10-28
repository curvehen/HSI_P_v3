"""對真實 Google Drive 的手動測試(需要 GDRIVE_* 環境變數)。

用法:
    export GDRIVE_CLIENT_ID=... GDRIVE_CLIENT_SECRET=... GDRIVE_REFRESH_TOKEN=...
    python scripts/drive_manual_test.py

- 使用獨立資料夾 hsi_ml_manual_test,不碰正式的 hsi_ml_state。
- 每次運行用新的子資料夾 run_<時間戳>,可重複運行。
- 程式沒有刪除功能。測試完請在 Drive 網頁手動刪除 hsi_ml_manual_test。
- 結束碼:0 全部通過,1 有失敗,2 無法連線。
"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hsi_ml.logging_setup import setup_logging  # noqa: E402
from hsi_ml.storage import DriveStore, PendingUploads, upload_or_queue  # noqa: E402

results: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    results.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")


def main() -> int:
    setup_logging()
    drive = DriveStore.try_connect("hsi_ml_manual_test")
    if drive is None:
        print("無法連線 Drive(憑證缺失、token 失效或網絡問題)。看上方日誌的錯誤類型。")
        return 2
    check("connect", True)

    base = f"run_{int(time.time())}"
    tmp = Path(tempfile.mkdtemp())
    f = tmp / "a.txt"
    f.write_bytes(b"v1")

    # 1 上傳與存在
    check("exists before upload is False", not drive.exists(f"{base}/a.txt"))
    drive.upload(f, f"{base}/a.txt")
    check("exists after upload", drive.exists(f"{base}/a.txt"))
    check("list_names contains file", "a.txt" in drive.list_names(base))

    # 2 append-only
    f.write_bytes(b"v2")
    try:
        drive.upload(f, f"{base}/a.txt")
        check("duplicate upload refused", False)
    except FileExistsError:
        check("duplicate upload refused", True)

    # 3 下載仍是舊內容
    out = tmp / "out" / "a.txt"
    check("download returns True", drive.download(f"{base}/a.txt", out))
    check("content unchanged after refused upload", out.read_bytes() == b"v1")

    # 4 overwrite=True 更新
    drive.upload(f, f"{base}/a.txt", overwrite=True)
    out2 = tmp / "out" / "a2.txt"
    drive.download(f"{base}/a.txt", out2)
    check("overwrite=True updates content", out2.read_bytes() == b"v2")
    check("no duplicate after overwrite", drive.list_names(base).count("a.txt") == 1)

    # 5 不存在的檔案
    check("download missing returns False",
          drive.download(f"{base}/none.txt", tmp / "x.txt") is False)

    # 6 檔名含單引號(查詢轉義)
    q = tmp / "it's.txt"
    q.write_bytes(b"q")
    drive.upload(q, f"{base}/it's.txt")
    check("quote in file name", drive.exists(f"{base}/it's.txt"))

    # 7 待補傳:先排隊,再用真實 Drive 補傳
    pend = PendingUploads(tmp / ".pending" / "pending.json")
    p = tmp / "pending.txt"
    p.write_bytes(b"p")
    check("upload_or_queue queues when drive None",
          upload_or_queue(None, p, f"{base}/pending.txt", pend) == "queued")
    stats = pend.flush(drive)
    check("flush uploads queued file", stats["uploaded"] == 1 and stats["remaining"] == 0)
    check("queued file exists on drive", drive.exists(f"{base}/pending.txt"))

    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    if failed:
        print("failed:", ", ".join(failed))
        return 1
    print("請到 Drive 手動刪除資料夾 hsi_ml_manual_test。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
