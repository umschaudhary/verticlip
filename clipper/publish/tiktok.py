"""TikTok via the Content Posting API (Direct Post, FILE_UPLOAD).

IMPORTANT: until TikTok audits your app, it may only post with
privacy_level = SELF_ONLY (private to you). Submit the app for audit in the
developer portal, then flip publish.tiktok.privacy to PUBLIC_TO_EVERYONE.

Tokens expire every 24h; refresh tokens last 365 days — we refresh
automatically and write the new pair back to .env.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import requests

API = "https://open.tiktokapis.com/v2"
CHUNK = 10 * 1024 * 1024  # 10 MB


def _persist_env(updates: dict[str, str]) -> None:
    env = Path(".env")
    if not env.exists():
        return
    txt = env.read_text()
    for k, v in updates.items():
        if re.search(rf"^{k}=", txt, flags=re.M):
            txt = re.sub(rf"^{k}=.*$", f"{k}={v}", txt, flags=re.M)
        else:
            txt += f"\n{k}={v}"
    env.write_text(txt)


def _refresh() -> str:
    r = requests.post(f"{API}/oauth/token/", data={
        "client_key": os.environ["TIKTOK_CLIENT_KEY"],
        "client_secret": os.environ["TIKTOK_CLIENT_SECRET"],
        "grant_type": "refresh_token",
        "refresh_token": os.environ["TIKTOK_REFRESH_TOKEN"],
    }, headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=30)
    r.raise_for_status()
    j = r.json()
    os.environ["TIKTOK_ACCESS_TOKEN"] = j["access_token"]
    os.environ["TIKTOK_REFRESH_TOKEN"] = j.get("refresh_token", os.environ["TIKTOK_REFRESH_TOKEN"])
    _persist_env({"TIKTOK_ACCESS_TOKEN": j["access_token"],
                  "TIKTOK_REFRESH_TOKEN": os.environ["TIKTOK_REFRESH_TOKEN"]})
    return j["access_token"]


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8"}


def upload(cfg, video_path: Path, title: str, description: str, hashtags: list[str]) -> tuple[str, str]:
    token = os.environ.get("TIKTOK_ACCESS_TOKEN") or _refresh()
    privacy = cfg.get("publish", "tiktok", "privacy", default="SELF_ONLY")
    size = video_path.stat().st_size
    chunks = max(1, -(-size // CHUNK))
    caption = (description if description else title)[:2200]

    body = {
        "post_info": {"title": caption, "privacy_level": privacy,
                      "disable_duet": False, "disable_comment": False, "disable_stitch": False,
                      "video_cover_timestamp_ms": 1000},
        "source_info": {"source": "FILE_UPLOAD", "video_size": size,
                        "chunk_size": CHUNK if chunks > 1 else size, "total_chunk_count": chunks},
    }
    r = requests.post(f"{API}/post/publish/video/init/", json=body, headers=_headers(token), timeout=30)
    if r.status_code == 401:
        token = _refresh()
        r = requests.post(f"{API}/post/publish/video/init/", json=body, headers=_headers(token), timeout=30)
    r.raise_for_status()
    j = r.json()
    if j.get("error", {}).get("code") not in (None, "ok"):
        raise RuntimeError(f"TikTok init error: {j['error']}")
    publish_id, upload_url = j["data"]["publish_id"], j["data"]["upload_url"]

    with video_path.open("rb") as f:
        offset = 0
        chunk_size = CHUNK if chunks > 1 else size
        while offset < size:
            data = f.read(chunk_size if offset + chunk_size < size else size - offset)
            end = offset + len(data) - 1
            u = requests.put(upload_url, data=data, headers={
                "Content-Type": "video/mp4",
                "Content-Range": f"bytes {offset}-{end}/{size}",
                "Content-Length": str(len(data)),
            }, timeout=300)
            if u.status_code not in (200, 201, 206):
                raise RuntimeError(f"TikTok chunk upload failed: {u.status_code} {u.text[:200]}")
            offset += len(data)

    # poll until published
    for _ in range(30):
        s = requests.post(f"{API}/post/publish/status/fetch/", json={"publish_id": publish_id},
                          headers=_headers(token), timeout=30).json()
        st = s.get("data", {}).get("status")
        if st == "PUBLISH_COMPLETE":
            ids = s["data"].get("publicaly_available_post_id") or s["data"].get("publicly_available_post_id") or []
            vid = str(ids[0]) if ids else publish_id
            return vid, (f"https://www.tiktok.com/@me/video/{vid}" if ids else f"tiktok:{publish_id}")
        if st == "FAILED":
            raise RuntimeError(f"TikTok publish failed: {s['data'].get('fail_reason')}")
        time.sleep(5)
    return publish_id, f"tiktok:{publish_id}"
