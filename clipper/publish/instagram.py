"""Instagram Reels via the Instagram Graph API (content publishing).

The API only accepts a *public URL* for the video, so we first push the file
to Cloudflare R2 (S3-compatible, free tier) and hand Instagram the public link.
Flow: upload to R2 → POST /{ig-user-id}/media (REELS) → poll status → POST /media_publish
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import requests

GRAPH = "https://graph.facebook.com/v21.0"


def _public_upload(video_path: Path) -> str:
    import boto3  # lazy

    acct = os.environ["R2_ACCOUNT_ID"]
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{acct}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    bucket = os.environ.get("R2_BUCKET", "clips")
    key = f"reels/{video_path.name}"
    s3.upload_file(str(video_path), bucket, key, ExtraArgs={"ContentType": "video/mp4"})
    return f"{os.environ['R2_PUBLIC_BASE_URL'].rstrip('/')}/{key}"


def upload(cfg, video_path: Path, title: str, description: str, hashtags: list[str]) -> tuple[str, str]:
    ig_user = os.environ["IG_USER_ID"]
    token = os.environ["IG_ACCESS_TOKEN"]
    video_url = _public_upload(video_path)

    r = requests.post(f"{GRAPH}/{ig_user}/media", data={
        "media_type": "REELS", "video_url": video_url,
        "caption": description[:2200], "share_to_feed": "true", "access_token": token,
    }, timeout=60)
    r.raise_for_status()
    container = r.json()["id"]

    # Instagram transcodes asynchronously — poll until FINISHED (usually 30-90s)
    for _ in range(40):
        s = requests.get(f"{GRAPH}/{container}", params={
            "fields": "status_code,status", "access_token": token}, timeout=30).json()
        code = s.get("status_code")
        if code == "FINISHED":
            break
        if code == "ERROR":
            raise RuntimeError(f"IG container error: {s}")
        time.sleep(5)
    else:
        raise RuntimeError("IG container never finished processing")

    p = requests.post(f"{GRAPH}/{ig_user}/media_publish",
                      data={"creation_id": container, "access_token": token}, timeout=60)
    p.raise_for_status()
    media_id = p.json()["id"]
    link = requests.get(f"{GRAPH}/{media_id}", params={"fields": "permalink", "access_token": token},
                        timeout=30).json().get("permalink", f"https://www.instagram.com/reel/{media_id}")
    return media_id, link
