"""一次性取得 Google Drive refresh token。

警告:
- 只在你自己的電腦運行,不要在 GitHub Actions、共用機器或錄影時運行。
- token 只印在終端機。不要貼進聊天、issue、commit 或任何公開位置。
- 取得後立即存入 GitHub Secrets (GDRIVE_REFRESH_TOKEN),並清除終端機捲動紀錄。
- OAuth consent screen 若停留在 Testing,refresh token 約 7 日失效(未核實),請發佈為 In production。

用法:
    export GDRIVE_CLIENT_ID=...
    export GDRIVE_CLIENT_SECRET=...
    python scripts/get_refresh_token.py
"""
import os
import sys

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def main() -> int:
    cid = os.environ.get("GDRIVE_CLIENT_ID", "").strip()
    secret = os.environ.get("GDRIVE_CLIENT_SECRET", "").strip()
    if not cid or not secret:
        print("請先設定環境變數 GDRIVE_CLIENT_ID 與 GDRIVE_CLIENT_SECRET", file=sys.stderr)
        return 2
    cfg = {"installed": {
        "client_id": cid,
        "client_secret": secret,
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://localhost"],
    }}
    flow = InstalledAppFlow.from_client_config(cfg, SCOPES)
    # access_type / prompt 會傳給授權網址,確保回傳 refresh token
    creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
    if not creds.refresh_token:
        print("沒有取得 refresh token。請到 Google 帳戶撤銷此應用的授權後重試。",
              file=sys.stderr)
        return 1
    print("\n" + "=" * 60)
    print("警告:以下是機密。只存入 GitHub Secrets,不要分享。")
    print("=" * 60)
    print("GDRIVE_REFRESH_TOKEN =", creds.refresh_token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
