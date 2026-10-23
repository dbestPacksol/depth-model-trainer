"""S3 discovery and resumable downloads for the camera-server snapshot layout."""

import hashlib
import json
import re
import shutil
from pathlib import Path

def create_s3_client(access_key: str, secret_key: str):
    """Connect with the credentials selected by the download script."""
    import boto3
    from botocore.config import Config

    return boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    ).client(
        "s3", config=Config(retries={"mode": "standard", "max_attempts": 5}, connect_timeout=15, read_timeout=60)
    )


def find_snapshots(client, bucket: str, prefix: str = "") -> list[dict]:
    """Read every listing page, ignoring event/session recordings and unrelated PNGs."""
    samples = {}
    paginator = client.get_paginator("list_objects_v2")
    for page_number, page in enumerate(paginator.paginate(Bucket=bucket, Prefix=prefix), start=1):
        for item in page.get("Contents", []):
            parts = item["Key"].split("/")
            if len(parts) < 4 or parts[-4] != "snapshots" or not re.fullmatch(r"snapshot_\d+", parts[-3]):
                continue
            if parts[-1] not in {"rgb.png", "depth.png", "frame.json"}:
                continue
            camera_prefix = "/".join(parts[:-1])
            sample = samples.setdefault(camera_prefix, {
                "camera_prefix": camera_prefix,
                "capture": "/".join(parts[:-2]),
                "snapshot": parts[-3],
                "camera": parts[-2],
                "objects": {},
            })
            sample["objects"][parts[-1]] = {"key": item["Key"], "etag": item["ETag"], "size": item["Size"]}
        if page_number % 10 == 0:
            print(f"Scanned {page_number} S3 listing pages; found {len(samples)} camera folders...", flush=True)
    return [sample for _, sample in sorted(samples.items()) if {"rgb.png", "depth.png"} & sample["objects"].keys()]


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Publish a complete JSON file even if the process is interrupted."""
    temporary_path = path.with_name(path.name + ".part")
    temporary_path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary_path.replace(path)


def download_object(client, bucket: str, source: dict, destination: Path) -> None:
    """Reuse a verified local copy; otherwise download the listed object version."""
    signature = {"bucket": bucket, **source}
    receipt_path = destination.with_name(destination.name + ".s3.json")
    if destination.is_file() and receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt["source"] == signature and file_hash(destination) == receipt["sha256"]:
                return
        except (ValueError, KeyError):
            pass  # An incomplete receipt is safe to replace by downloading again.

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(destination.name + ".part")
    try:
        # ETag is an opaque version fingerprint, not necessarily an MD5 checksum.
        response = client.get_object(Bucket=bucket, Key=source["key"], IfMatch=source["etag"])
        with response["Body"] as body, temporary_path.open("wb") as handle:
            shutil.copyfileobj(body, handle)
        if temporary_path.stat().st_size != source["size"]:
            raise OSError(f"Incomplete download: {source['key']}")
        temporary_path.replace(destination)
        write_json(receipt_path, {"source": signature, "sha256": file_hash(destination)})
    finally:
        temporary_path.unlink(missing_ok=True)
