"""YouTube Shorts via YouTube Data API v3 (resumable upload).

A vertical video < 60s with #Shorts in the title/description is auto-classified
as a Short. Default API quota (10,000 units/day) allows ~6 uploads/day
(1,600 units each) — matches the default daily cap in config.yaml.

One-time auth:  python -m clipper.publish.youtube --auth
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def _creds():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    token_file = Path(os.environ.get("YOUTUBE_TOKEN_FILE", "secrets/youtube_token.json"))
    secret_file = Path(os.environ.get("YOUTUBE_CLIENT_SECRET_FILE", "secrets/youtube_client_secret.json"))
    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_file.write_text(creds.to_json())
    if not creds or not creds.valid:
        if "--auth" not in sys.argv and not sys.stdin.isatty():
            raise RuntimeError("YouTube token missing/invalid — run `python -m clipper.publish.youtube --auth`")
        flow = InstalledAppFlow.from_client_secrets_file(str(secret_file), SCOPES)
        creds = flow.run_local_server(port=0)
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(creds.to_json())
    return creds


def upload(cfg, video_path: Path, title: str, description: str, hashtags: list[str]) -> tuple[str, str]:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    yt = build("youtube", "v3", credentials=_creds(), cache_discovery=False)
    y = cfg.get("publish", "youtube", default={}) or {}
    if "#shorts" not in title.lower():
        title = (title[:92] + " #Shorts")[:100]
    tags = [h.lstrip("#") for h in hashtags][:15]
    body = {
        "snippet": {"title": title, "description": description + "\n\n#Shorts",
                    "tags": tags, "categoryId": str(y.get("category_id", "24"))},
        "status": {"privacyStatus": y.get("privacy", "public"),
                   "selfDeclaredMadeForKids": bool(y.get("made_for_kids", False))},
    }
    media = MediaFileUpload(str(video_path), chunksize=8 * 1024 * 1024, resumable=True, mimetype="video/mp4")
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = None
    while resp is None:
        _, resp = req.next_chunk()
    vid = resp["id"]
    return vid, f"https://youtube.com/shorts/{vid}"


if __name__ == "__main__":
    if "--auth" in sys.argv:
        from ..config import load_config
        load_config()
        _creds()
        print("YouTube auth OK — token cached.")
